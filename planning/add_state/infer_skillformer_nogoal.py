#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_skillformer_nogoal.py
推理版本（No-Goal）：
  - 输入：instruction + start_image + proprio state（可选）
  - 输出：预测 skill 序列
支持：
  A) --episode_npz  (直接读取 prep_plan_data.py 输出)
  B) --text + --img_start (+ --state_json)
  + 支持 transition.npy 约束解码
  + 使用 HuggingFace PaliGemma 编码文本与图像
"""

import os, json, argparse, hashlib
import numpy as np
from typing import Optional
from PIL import Image
import torch, torch.nn as nn
from transformers import AutoTokenizer, AutoProcessor, AutoModelForVision2Seq

# ------------------------------------------------------------
# Pi0Adapter (with HuggingFace PaliGemma)
# ------------------------------------------------------------
class Pi0Adapter:
    def __init__(self, d_model=1024, hf_model="google/paligemma-3b-pt-224", device=None):
        self.d_model = d_model
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"[Pi0Adapter] Using HuggingFace PaliGemma ({hf_model}) for encoding...")
        self.processor = AutoProcessor.from_pretrained(hf_model)
        self.model = AutoModelForVision2Seq.from_pretrained(hf_model, torch_dtype=torch.float16).to(self.device).eval()
        self.hf_tokenizer = self.processor.tokenizer
        with torch.no_grad():
            dummy_ids = torch.zeros(1, 4, dtype=torch.long, device=self.device)
            txt_emb = self.model.language_model.model.embed_tokens(dummy_ids)
        self.D_encoder = int(txt_emb.shape[-1])
        if self.D_encoder != d_model:
            self.proj = nn.Linear(self.D_encoder, d_model, bias=False).to(self.device, dtype=torch.float32)
            torch.nn.init.orthogonal_(self.proj.weight)
            for p in self.proj.parameters(): p.requires_grad_(False)
            self.proj = self.proj.to(torch.float16)
            print(f"[Pi0Adapter] projection: {self.D_encoder} → {d_model}")
        else:
            self.proj = None

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        tok = self.hf_tokenizer(text, return_tensors="pt", add_special_tokens=True).to(self.device)
        emb = self.model.language_model.model.embed_tokens(tok["input_ids"])
        if self.proj is not None:
            emb = self.proj(emb.to(self.proj.weight.dtype))
        return emb.squeeze(0).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode_image_rgb(self, img_uint8_hw3: np.ndarray) -> np.ndarray:
        img = Image.fromarray(img_uint8_hw3)
        proc = self.processor(images=img, return_tensors="pt").to(self.device)
        vis_out = self.model.vision_tower(pixel_values=proc["pixel_values"]).last_hidden_state
        if self.proj is not None:
            vis_out = self.proj(vis_out.to(self.proj.weight.dtype))
        return vis_out.squeeze(0).cpu().numpy().astype(np.float32)

# ------------------------------------------------------------
# Model (No-Goal version)
# ------------------------------------------------------------
class CondProject(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.pos = nn.Parameter(torch.randn(1, 2048, d) * 0.01)
    def forward(self, H_txt, H_vs, H_state):
        mem = torch.cat([H_txt, H_vs, H_state], dim=1)
        Tm = mem.size(1)
        return self.ln(mem + self.pos[:, :Tm, :])

class StateMLP(nn.Module):
    def __init__(self, d_state, d, hidden=2):
        super().__init__()
        layers=[]
        in_dim=d_state
        for _ in range(hidden):
            layers += [nn.Linear(in_dim, d), nn.GELU()]
            in_dim = d
        self.net = nn.Sequential(*layers)
    def forward(self, s):
        return self.net(s).unsqueeze(1)

class SkillFormer(nn.Module):
    def __init__(self, K, d=1024, n_layers=4, n_heads=4, d_ff=1024, max_len=64, dropout=0.1, d_state=0):
        super().__init__()
        self.K = K; self.d = d
        self.embed_out = nn.Embedding(K+3, d)
        self.pos_out   = nn.Parameter(torch.randn(1, max_len, d)*0.01)
        layer = nn.TransformerDecoderLayer(d, n_heads, d_ff, dropout=dropout, batch_first=True)
        self.dec = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.fc  = nn.Linear(d, K+3)
        self.cond= CondProject(d)
        self.state_mlp = StateMLP(d_state, d) if d_state>0 else None
    def forward(self, H_txt, H_vs, Yin, S=None):
        B, L = Yin.size()
        if self.state_mlp is not None and S is not None:
            H_state = self.state_mlp(S)
        else:
            H_state = torch.zeros((B,1,self.d), device=Yin.device, dtype=torch.float32)
        mem = self.cond(H_txt, H_vs, H_state)
        x   = self.embed_out(Yin) + self.pos_out[:,:L,:]
        causal = torch.triu(torch.full((L,L), float("-inf"), device=Yin.device), diagonal=1)
        out = self.dec(x, mem, tgt_mask=causal)
        return self.fc(out)

# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def load_episode_npz(path: str):
    z = np.load(path, allow_pickle=True)
    return {
        "H_txt": z["H_txt"].astype(np.float32),
        "H_vs":  z["H_vis_start"].astype(np.float32),
        "state": z["state"].astype(np.float32) if "state" in z else np.array([], dtype=np.float32),
        "skill_seq": z["skill_seq"].astype(np.int64).tolist() if "skill_seq" in z else None,
        "episode": str(z.get("episode","")),
        "instruction": str(z.get("instruction","")),
    }

def load_image_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))

def to_tensor_pad1(x: np.ndarray, device):
    t = torch.from_numpy(x).to(device)
    return t.unsqueeze(0)

def build_trans_mask(transition_path: Optional[str], K: int, device):
    if transition_path and os.path.exists(transition_path):
        trans = np.load(transition_path)
        if trans.shape[0] != K:
            raise ValueError(f"transition.npy K mismatch: {trans.shape} vs {K}")
        mask = (trans > 0)
        return torch.from_numpy(mask).to(device)
    return torch.ones((K,K), dtype=torch.bool, device=device)

@torch.no_grad()
def greedy_decode(model, H_txt, H_vs, S, K, trans_mask, max_len=32, temperature=1.0):
    BOS, EOS, PAD = K, K+1, K+2
    device = H_txt.device
    Yin = torch.full((1,1), BOS, dtype=torch.long, device=device)
    prev = None
    seq = []
    for _ in range(max_len):
        logits = model(H_txt, H_vs, Yin, S)[0, -1, :] / max(1e-6, temperature)
        logits[PAD] = -1e9
        if prev is not None and prev<K:
            mask = trans_mask[prev]
            logits[:K][~mask] = -1e9
        probs = torch.softmax(logits, dim=-1)
        token = int(torch.argmax(probs).item())
        Yin = torch.cat([Yin, torch.tensor([[token]], device=device)], dim=1)
        if token == EOS: break
        if token < K:
            seq.append(token)
            prev = token
        else:
            prev = None
    return seq

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main(args):
    # Load model checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu")
    K   = int(ckpt["K"])
    d   = int(ckpt["cfg"]["d"])
    max_out_len = int(ckpt["cfg"]["max_out_len"])
    d_state = int(ckpt["cfg"].get("state_dim", 0))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Load meta (for state normalization)
    meta = None
    meta_path = args.meta
    if meta_path is None and args.transition:
        cand = os.path.join(os.path.dirname(args.transition), "meta.json")
        if os.path.exists(cand): meta_path = cand
    if meta_path:
        meta = json.load(open(meta_path, "r"))
    state_mean = np.array(meta["state_mean"], dtype=np.float32) if meta and meta.get("state_mean") is not None else None
    state_std  = np.array(meta["state_std"],  dtype=np.float32) if meta and meta.get("state_std")  is not None else None
    state_dim_meta = int(meta.get("state_dim", d_state)) if meta else d_state

    # Build model
    model = SkillFormer(K, d=d, n_layers=4, n_heads=4, d_ff=4*d, max_len=max_out_len, dropout=0.1, d_state=d_state).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # -------- Input A: episode_npz --------
    if args.episode_npz:
        sample = load_episode_npz(args.episode_npz)
        H_txt = to_tensor_pad1(sample["H_txt"], device)
        H_vs  = to_tensor_pad1(sample["H_vs"], device)
        # state
        if d_state > 0:
            s = sample["state"]
            s_pad = np.zeros(d_state, dtype=np.float32)
            if s.size>0:
                m = min(d_state, s.shape[0])
                s_pad[:m] = s[:m]
            if state_mean is not None and state_std is not None:
                s_pad = (s_pad - state_mean[:d_state]) / (state_std[:d_state] + 1e-6)
            S = torch.from_numpy(s_pad[None,:]).to(device)
        else:
            S = None
        meta_out = {"source": "episode_npz", "episode": sample.get("episode",""), "instruction": sample.get("instruction","")}

    # -------- Input B: text + img_start (+state) --------
    else:
        assert args.text and args.img_start, "需提供 --text 与 --img_start，或使用 --episode_npz"
        adapter = Pi0Adapter(d_model=d)
        H_txt = to_tensor_pad1(adapter.encode_text(args.text), device)
        img0 = load_image_rgb(args.img_start)
        H_vs  = to_tensor_pad1(adapter.encode_image_rgb(img0), device)
        if d_state > 0:
            if args.state_json and os.path.exists(args.state_json):
                st_obj = json.load(open(args.state_json))
                if isinstance(st_obj, dict) and "state" in st_obj:
                    vec = np.array(st_obj["state"], dtype=np.float32)
                elif isinstance(st_obj, list):
                    vec = np.array(st_obj, dtype=np.float32)
                else:
                    vec = np.zeros(d_state, dtype=np.float32)
            else:
                print("[warn] no --state_json provided; using zero state.")
                vec = np.zeros(d_state, dtype=np.float32)
            s_pad = np.zeros(d_state, dtype=np.float32)
            m = min(d_state, vec.shape[0])
            s_pad[:m] = vec[:m]
            if state_mean is not None and state_std is not None:
                s_pad = (s_pad - state_mean[:d_state]) / (state_std[:d_state] + 1e-6)
            S = torch.from_numpy(s_pad[None,:]).to(device)
        else:
            S = None
        meta_out = {"source": "text+image", "instruction": args.text, "img_start": args.img_start}

    # transition constraints
    trans_mask = build_trans_mask(args.transition, K, device)

    # decode
    seq = greedy_decode(model, H_txt, H_vs, S, K, trans_mask, max_len=args.max_len, temperature=args.temp)

    out = {
        "pred_skill_seq": seq,
        "K": K,
        "max_len": args.max_len,
        "temperature": args.temp,
        "constrained": bool(args.transition and os.path.exists(args.transition)),
        "meta": meta_out,
    }

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[OK] wrote -> {args.out}")
    else:
        print(json.dumps(out, ensure_ascii=False, indent=2))

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser("SkillFormer Inference (No-Goal, +proprio, HuggingFace Pi0Adapter)")
    p.add_argument("--ckpt", required=True, help="train_skillformer_nogoal.py 导出的 best.pt/last.pt")
    # A) episode npz
    p.add_argument("--episode_npz", default=None)
    # B) text + start image
    p.add_argument("--text", default=None)
    p.add_argument("--img_start", default=None)
    p.add_argument("--state_json", default=None, help='JSON 文件，{"state":[...]} 或直接数组')
    # meta / transition
    p.add_argument("--meta", default=None, help="meta.json（用于 state 标准化）")
    p.add_argument("--transition", default=None, help="transition.npy 用于约束解码（可选）")
    # decode
    p.add_argument("--max_len", type=int, default=32)
    p.add_argument("--temp", type=float, default=1.0)
    # others
    p.add_argument("--out", default=None)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    main(args)
