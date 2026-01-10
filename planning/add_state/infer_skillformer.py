#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SkillFormer 推理（支持 proprio 状态）
两种输入：
A) --episode_npz  直接读取预处理产物（推荐）
B) --text + --img_start + --img_goal (+ 可选 --state_json)

若提供 --transition（transition.npy），则进行约束解码。
若未显式提供 --meta，会尝试从 --transition 的目录推断 meta.json。
"""
import os, json, argparse
import numpy as np
from typing import Optional
from PIL import Image
import torch
import torch.nn as nn

# ------------------- Pi0 Adapter -------------------
class Pi0Adapter:
    def __init__(self, d_model=256):
        self.d_model = d_model
        # TODO: 在此加载你本地的 pi0 文本/视觉编码器

    def encode_text(self, text: str) -> np.ndarray:
        # ----- PLACEHOLDER（请替换为真实实现） -----
        import hashlib
        rng = np.random.default_rng(int(hashlib.md5(text.encode()).hexdigest(), 16) % (10**9))
        T_txt, D = 16, self.d_model
        return rng.standard_normal((T_txt, D), dtype=np.float32)

    def encode_image_rgb(self, img: np.ndarray) -> np.ndarray:
        # ----- PLACEHOLDER（请替换为真实实现） -----
        T_vis, D = 64, self.d_model
        seed = int(img[:8,:8,:].sum())
        rng = np.random.default_rng(seed)
        return rng.standard_normal((T_vis, D), dtype=np.float32)

# ------------------- Model（与训练一致） -------------------
class CondProject(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.pos = nn.Parameter(torch.randn(1, 2048, d) * 0.01)
    def forward(self, H_txt, H_vs, H_vg, H_state):
        mem = torch.cat([H_txt, H_vs, H_vg, H_state], dim=1)
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
    def forward(self, s): # [B,d_state]
        return self.net(s).unsqueeze(1)

class SkillFormer(nn.Module):
    def __init__(self, K, d=256, n_layers=4, n_heads=4, d_ff=1024, max_len=64, dropout=0.1, d_state=0):
        super().__init__()
        self.K = K; self.d = d
        self.embed_out = nn.Embedding(K+3, d)
        self.pos_out   = nn.Parameter(torch.randn(1, max_len, d)*0.01)
        layer = nn.TransformerDecoderLayer(d, n_heads, d_ff, dropout=dropout, batch_first=True)
        self.dec = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.fc  = nn.Linear(d, K+3)
        self.cond= CondProject(d)
        self.state_mlp = StateMLP(d_state, d) if d_state>0 else None
    def forward(self, H_txt, H_vs, H_vg, Yin, S=None):
        B, L = Yin.size()
        if self.state_mlp is not None and S is not None:
            H_state = self.state_mlp(S)
        else:
            H_state = torch.zeros((B,1,self.d), device=Yin.device, dtype=torch.float32)
        mem = self.cond(H_txt, H_vs, H_vg, H_state)
        x   = self.embed_out(Yin) + self.pos_out[:,:L,:]
        causal = torch.triu(torch.full((L,L), float("-inf"), device=Yin.device), diagonal=1)
        out = self.dec(x, mem, tgt_mask=causal)
        return self.fc(out)

# ------------------- Utils -------------------
def load_episode_npz(path: str):
    z = np.load(path, allow_pickle=True)
    return {
        "H_txt": z["H_txt"].astype(np.float32),
        "H_vs":  z["H_vis_start"].astype(np.float32),
        "H_vg":  z["H_vis_goal"].astype(np.float32),
        "skill_seq": z["skill_seq"].astype(np.int64).tolist() if "skill_seq" in z else None,
        "state": z["state"].astype(np.float32) if "state" in z else np.array([], dtype=np.float32),
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
def greedy_decode(model: SkillFormer, H_txt, H_vs, H_vg, S, K: int,
                  trans_mask: torch.Tensor, max_len=32, temperature=1.0):
    BOS, EOS, PAD = K, K+1, K+2
    device = H_txt.device
    Yin = torch.full((1,1), BOS, dtype=torch.long, device=device)
    prev = None
    seq = []
    for _ in range(max_len):
        logits = model(H_txt, H_vs, H_vg, Yin, S)[0, -1, :] / max(1e-6, temperature)
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

def main(args):
    # 载入 ckpt
    ckpt = torch.load(args.ckpt, map_location="cpu")
    K   = int(ckpt["K"])
    d   = int(ckpt["cfg"]["d"])
    max_out_len = int(ckpt["cfg"]["max_out_len"])
    d_state = int(ckpt["cfg"].get("state_dim", 0))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # meta（为 state 标准化）
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

    # 模型
    model = SkillFormer(K, d=d, n_layers=4, n_heads=4, d_ff=4*d, max_len=max_out_len, dropout=0.1, d_state=d_state).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # 输入
    if args.episode_npz:
        sample = load_episode_npz(args.episode_npz)
        H_txt = to_tensor_pad1(sample["H_txt"], device)
        H_vs  = to_tensor_pad1(sample["H_vs"], device)
        H_vg  = to_tensor_pad1(sample["H_vg"], device)
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
    else:
        assert args.text and args.img_start and args.img_goal, "需同时提供 --text --img_start --img_goal，或使用 --episode_npz"
        adapter = Pi0Adapter(d_model=d)
        H_txt = to_tensor_pad1(adapter.encode_text(args.text), device)
        img0 = load_image_rgb(args.img_start); H_vs = to_tensor_pad1(adapter.encode_image_rgb(img0), device)
        img1 = load_image_rgb(args.img_goal);  H_vg = to_tensor_pad1(adapter.encode_image_rgb(img1), device)
        # state：来自 --state_json 或全零
        if d_state > 0:
            if args.state_json and os.path.exists(args.state_json):
                st_obj = json.load(open(args.state_json))
                # 期望传入 {"state": [float,...]} 或直接 [float,...]
                if isinstance(st_obj, dict) and "state" in st_obj:
                    vec = np.array(st_obj["state"], dtype=np.float32)
                elif isinstance(st_obj, list):
                    vec = np.array(st_obj, dtype=np.float32)
                else:
                    vec = np.zeros(d_state, dtype=np.float32)
            else:
                print("[warn] no --state_json provided; using zero state.")
                vec = np.zeros(d_state, dtype=np.float32)
            s_pad = np.zeros(d_state, dtype=np.float32); m = min(d_state, vec.shape[0]); s_pad[:m] = vec[:m]
            if state_mean is not None and state_std is not None:
                s_pad = (s_pad - state_mean[:d_state]) / (state_std[:d_state] + 1e-6)
            S = torch.from_numpy(s_pad[None,:]).to(device)
        else:
            S = None
        meta_out = {"source": "text+images", "instruction": args.text, "img_start": args.img_start, "img_goal": args.img_goal}

    # 约束
    trans_mask = build_trans_mask(args.transition, K, device)

    # 解码
    seq = greedy_decode(model, H_txt, H_vs, H_vg, S, K, trans_mask, max_len=args.max_len, temperature=args.temp)

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

if __name__ == "__main__":
    p = argparse.ArgumentParser("SkillFormer Inference (+proprio)")
    p.add_argument("--ckpt", required=True, help="train_skillformer.py 导出的 best.pt/last.pt")
    # A) episode npz
    p.add_argument("--episode_npz", default=None)
    # B) text + images
    p.add_argument("--text", default=None)
    p.add_argument("--img_start", default=None)
    p.add_argument("--img_goal", default=None)
    p.add_argument("--state_json", default=None, help='JSON 文件，{"state":[...]} 或直接数组')
    # meta / transition
    p.add_argument("--meta", default=None, help="meta.json（用于 state 标准化）；若缺省则尝试从 transition 同目录推断")
    p.add_argument("--transition", default=None, help="transition.npy 用于约束解码（可选）")
    # 解码
    p.add_argument("--max_len", type=int, default=32)
    p.add_argument("--temp", type=float, default=1.0)
    # 其它
    p.add_argument("--out", default=None)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    main(args)
