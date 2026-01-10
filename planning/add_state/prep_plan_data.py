#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prep_plan_data.py (suffix-slicing, frame-accurate)
为 Planning Head 准备输入数据（后缀样本）：
  - 每条 episode 读取 skills.jsonl 的 segments（含 t0,t1,skill）
  - 合并相邻相同 skill，保留每个合并后段的起帧 t0
  - 生成后缀样本：对每个合并段 i，保存
        instruction, H_txt, H_vis_start@t0_i, state@t0_i, skill_seq[i:]
  - 仅保存起帧图像编码，不保存 goal
  - 支持 HF PaliGemma 或本地 Pi-0 权重（safetensors）
  - 兼容多种 TFRecord 写法（图像/状态的时间序列 vs 扁平/逐帧 bytes）
"""

import os, json, glob, re, io, hashlib, sys
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from PIL import Image

# --------------------------------------------------------------
# Torch / HF / OpenPI
# --------------------------------------------------------------
import torch
import safetensors.torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForVision2Seq

sys.path.append("/home/lulsong/WorkTask/chain-of-action/openpi/src/")
from openpi.shared.image_tools import resize_with_pad_torch
import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel


# ============================ Pi0Adapter ============================
class Pi0Adapter:
    def __init__(
        self,
        d_model: int = 1024,
        device: Optional[str] = None,
        hf_text_tokenizer: str = "google/paligemma-3b-pt-224",
        image_resolution: Tuple[int, int] = (224, 224),
        pi0_ckpt_path: Optional[str] = None,
        precision: str = "bfloat16",
        seed: int = 20240517,
    ):
        self.d_model = d_model
        self.image_resolution = image_resolution
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.precision = precision

        if pi0_ckpt_path:
            print(f"[Pi0Adapter] Loading Pi-0 safetensors weights from: {pi0_ckpt_path}")
            paligemma_config = _gemma.get_config("gemma_2b")
            action_expert_config = _gemma.get_config("gemma_300m")
            self.model = PaliGemmaWithExpertModel(
                paligemma_config, action_expert_config, use_adarms=[False, False], precision="bfloat16"
            ).to(self.device).eval()
            state = safetensors.torch.load_file(os.path.join(pi0_ckpt_path, "model.safetensors"))
            miss, unexp = self.model.load_state_dict(state, strict=False)
            print(f"[Pi0Adapter] loaded Pi0 ckpt (missing={len(miss)}, unexpected={len(unexp)})")
            self.hf_tokenizer = AutoTokenizer.from_pretrained(hf_text_tokenizer, use_fast=True, local_files_only=True)
            with torch.no_grad():
                dummy_ids = torch.zeros(1, 4, dtype=torch.long, device=self.device)
                txt_emb = self.model.paligemma.language_model.embed_tokens(dummy_ids)
            self.D_encoder = int(txt_emb.shape[-1])
        else:
            print("[Pi0Adapter] Using HuggingFace PaliGemma encoder (no Pi-0 weights).")
            self.processor = AutoProcessor.from_pretrained(hf_text_tokenizer)
            self.model = AutoModelForVision2Seq.from_pretrained(
                hf_text_tokenizer, torch_dtype=torch.float16
            ).to(self.device).eval()
            self.hf_tokenizer = self.processor.tokenizer
            with torch.no_grad():
                dummy_ids = torch.zeros(1, 4, dtype=torch.long, device=self.device)
                txt_emb = self.model.language_model.model.embed_tokens(dummy_ids)
            self.D_encoder = int(txt_emb.shape[-1])

        self.proj = None
        if self.D_encoder != self.d_model:
            self.proj = torch.nn.Linear(self.D_encoder, self.d_model, bias=False).to(self.device, dtype=torch.float32)
            torch.manual_seed(seed)
            torch.nn.init.orthogonal_(self.proj.weight)
            self.proj = self.proj.to(dtype=torch.bfloat16 if self.precision == "bfloat16" else torch.float32)
            for p in self.proj.parameters():
                p.requires_grad_(False)
            print(f"[Pi0Adapter] projection: {self.D_encoder} → {self.d_model}")

        for p in self.model.parameters():
            p.requires_grad_(False)

    def encode_text(self, text: str) -> np.ndarray:
        with torch.no_grad():
            tok = self.hf_tokenizer(text, return_tensors="pt", add_special_tokens=True).to(self.device)
            if hasattr(self.model, "paligemma"):
                emb = self.model.paligemma.language_model.embed_tokens(tok["input_ids"])
            else:
                emb = self.model.language_model.model.embed_tokens(tok["input_ids"])
            if self.proj is not None:
                emb = self.proj(emb.to(self.proj.weight.dtype))
            emb = emb.to(torch.float32)
            return emb.squeeze(0).cpu().numpy().astype(np.float32)

    def encode_image_rgb(self, img_uint8_hw3: np.ndarray) -> np.ndarray:
        assert img_uint8_hw3.ndim == 3 and img_uint8_hw3.shape[2] == 3
        with torch.no_grad():
            if hasattr(self, "processor"):  # HF
                img = Image.fromarray(img_uint8_hw3)
                proc = self.processor(images=img, return_tensors="pt").to(self.device)
                vis_out = self.model.vision_tower(pixel_values=proc["pixel_values"]).last_hidden_state
            else:  # Pi-0
                img = torch.from_numpy(img_uint8_hw3.astype(np.float32) / 255.0).unsqueeze(0).to(self.device)
                img = img * 2.0 - 1.0
                img_resized = resize_with_pad_torch(img, *self.image_resolution)
                if img_resized.dim() == 3:
                    img_resized = img_resized.unsqueeze(0)
                img_bchw = img_resized.permute(0, 3, 1, 2).contiguous()
                vis_out = self.model.paligemma.model.get_image_features(img_bchw)

            Dv = vis_out.shape[-1]
            if Dv != self.D_encoder:
                if not hasattr(self, "_proj_vis") or getattr(self, "_proj_vis_in", None) != Dv:
                    self._proj_vis_in = Dv
                    self._proj_vis = torch.nn.Linear(Dv, self.D_encoder, bias=False).to(self.device)
                    torch.nn.init.xavier_uniform_(self._proj_vis.weight)
                    for p in self._proj_vis.parameters():
                        p.requires_grad_(False)
                    print(f"[Pi0Adapter] visual proj created: {Dv} → {self.D_encoder}")
                vis_out = self._proj_vis(vis_out)

            if self.proj is not None:
                vis_out = self.proj(vis_out)

            vis_out = vis_out.to(torch.float32)
            return vis_out.squeeze(0).cpu().numpy().astype(np.float32)


# ============================ utils ============================
def hsh(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:16]

def try_get_byteslist(f, key) -> Optional[List[bytes]]:
    return f[key].bytes_list.value if (key in f and len(f[key].bytes_list.value)) else None

def pick_instruction(f) -> Optional[str]:
    # 取最长的一条指令文本
    for key in ["steps/language_instruction", "steps/language_instruction_2", "steps/language_instruction_3"]:
        if key not in f:
            continue
        vals = f[key].bytes_list.value
        if not vals:
            continue
        texts = [b.decode(errors="ignore").strip() for b in vals if len(b) > 1]
        if texts:
            import re as _re
            return _re.sub(r"\s+", " ", max(texts, key=len))
    return None

# ---- 从 features 中按帧索引 t 提取图像（RGB）
def decode_image_at_t(f, t: int) -> Optional[np.ndarray]:
    for key in [
        "steps/observation/wrist_image_left",
        "steps/observation/exterior_image_1_left",
        "steps/observation/exterior_image_2_left",
    ]:
        bl = try_get_byteslist(f, key)
        if not bl:
            continue
        # 多帧：直接取第 t 个
        if len(bl) > t:
            try:
                return np.array(Image.open(io.BytesIO(bl[t])).convert("RGB"))
            except Exception:
                continue
        # 单帧：退化场景，取唯一帧
        if len(bl) == 1:
            try:
                return np.array(Image.open(io.BytesIO(bl[0])).convert("RGB"))
            except Exception:
                continue
    return None

# ---- 从 features 中按帧索引 t 提取 state（jp 7 + gp 1）
def extract_state_at(f, t: int, jp_dim: int = 7, gp_dim: int = 1) -> Optional[np.ndarray]:
    def pull(name, dim):
        if name not in f:
            return None
        # 优先 bytes_list：可能每帧一块
        bl = f[name].bytes_list.value
        if bl:
            if len(bl) > t:
                try:
                    arr = np.frombuffer(bl[t], dtype=np.float32)
                    # 有些来源每帧就 dim 个，有些可能多于 dim，这里截断
                    return arr[:dim] if arr.size >= dim else None
                except Exception:
                    pass
            # 单块 bytes：扁平 [T*dim] 或 [dim]
            if len(bl) == 1:
                arr = np.frombuffer(bl[0], dtype=np.float32)
                if arr.size >= (t + 1) * dim:
                    return arr[t * dim : (t + 1) * dim]
                if arr.size == dim:
                    return arr  # 常量
                return None

        # 其次 float_list：常见是扁平 [T*dim] 或 [dim]
        vals = f[name].float_list.value
        if vals:
            arr = np.array(vals, dtype=np.float32)
            if arr.size >= (t + 1) * dim:
                return arr[t * dim : (t + 1) * dim]
            if arr.size == dim:
                return arr
            # 部分脏数据：长度不足，退化为前 dim 截断（谨慎）
            if arr.size > dim:
                return arr[:dim]
        return None

    jp = pull("steps/observation/joint_position", jp_dim)
    gp = pull("steps/observation/gripper_position", gp_dim)
    if jp is None or gp is None:
        return None
    return np.concatenate([jp, gp], axis=0).astype(np.float32)  # 期望 8 维


# ============================ skills reader ============================
def read_skills_with_t0(skills_jsonl: str) -> Dict[str, Dict[str, Any]]:
    """
    返回:
      ep2info[episode] = {
        "skills": [k0, k1, ...]        # 合并相邻相同后的技能序列
        "starts": [t0_0, t0_1, ...]    # 对应每段的起帧 t0
      }
    """
    ep2info: Dict[str, Dict[str, Any]] = {}
    with open(skills_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            ep = str(o["episode"])
            segs = o.get("segments", [])
            skills, starts = [], []
            last_skill = None
            for seg in segs:
                k = int(seg["skill"])
                t0 = int(seg["t0"])
                if last_skill is None or k != last_skill:
                    skills.append(k)
                    starts.append(t0)
                    last_skill = k
                else:
                    # 相邻相同 skill 合并：保持第一次出现的 t0，不追加
                    continue
            ep2info[ep] = {"skills": skills, "starts": starts}
    return ep2info


# ============================ main ============================
def main(
    tfrecord_glob: str,
    skills_jsonl: str,
    out_dir: str,
    episode_key: str = "episode_metadata/file_path",
    d_model: int = 1024,
    tokenizer_name_or_path: str = "google/paligemma-3b-pt-224",
    pi0_ckpt_path: Optional[str] = None,
):
    os.makedirs(out_dir, exist_ok=True)
    epi_dir = os.path.join(out_dir, "episodes")
    os.makedirs(epi_dir, exist_ok=True)

    ep2info = read_skills_with_t0(skills_jsonl)
    print(f"[skills] {len(ep2info)} episodes (deduped)")

    adapter = Pi0Adapter(d_model=d_model, hf_text_tokenizer=tokenizer_name_or_path, pi0_ckpt_path=pi0_ckpt_path)

    files = sorted(glob.glob(tfrecord_glob))
    kept = miss_inst = miss_img = miss_skill = no_state_cnt = 0
    state_sum = state_sumsq = None
    state_cnt = 0
    state_dim = None

    # 估计 K
    K = 0
    for v in ep2info.values():
        if v["skills"]:
            K = max(K, max(v["skills"]) + 1)
    trans = np.zeros((K, K), dtype=np.int64) if K > 0 else np.zeros((1, 1), dtype=np.int64)

    def normalize_ep_key(ep_raw: str) -> str:
        return re.sub(r"/recordings/[^/]+$", "/trajectory.h5", ep_raw)

    # 为了不重复加转移，只在每条 episode 的去重技能序列上统计一次
    def add_transitions_once(skills: List[int]):
        for a, b in zip(skills[:-1], skills[1:]):
            trans[a, b] += 1

    for tfrec in files:
        ds = tf.data.TFRecordDataset(tfrec)
        for rec in tqdm(ds, desc=os.path.basename(tfrec)):
            ex = tf.train.Example.FromString(bytes(rec.numpy()))
            f = ex.features.feature

            # 取 episode 路径
            ep = None
            for k in [episode_key, "episode_metadata/file_path", "episode_metadata/recording_folderpath"]:
                if k in f and len(f[k].bytes_list.value):
                    ep = f[k].bytes_list.value[0].decode()
                    break
            if not ep:
                continue

            ep_norm = normalize_ep_key(ep)
            ep_key = ep_norm if ep_norm in ep2info else ep
            if ep_key not in ep2info:
                miss_skill += 1
                continue

            info = ep2info[ep_key]
            skills = info["skills"]
            starts = info["starts"]
            if not skills or not starts or len(skills) != len(starts):
                continue

            # 指令与文本编码（每样本共用同一 instruction）
            inst = pick_instruction(f)
            if not inst:
                miss_inst += 1
                continue
            H_txt = adapter.encode_text(inst)

            # 仅对该 episode 的去重序列统计一次转移
            add_transitions_once(skills)

            ep_id = hsh(ep_key)

            for i, (k, t0) in enumerate(zip(skills, starts)):
                # 按 t0 提取图像（严格当前帧）
                img = decode_image_at_t(f, t0)
                if img is None:
                    miss_img += 1
                    continue
                H_vs = adapter.encode_image_rgb(img)  # H_vis_start

                # 按 t0 提取状态（严格当前帧）
                st = extract_state_at(f, t0, jp_dim=7, gp_dim=1)
                if st is None:
                    no_state_cnt += 1
                    # 可以选择跳过，也可用零向量占位，这里保守跳过
                    continue

                # 统计 state 均值方差
                if state_dim is None:
                    state_dim = int(st.shape[0])
                    state_sum = np.zeros(state_dim, dtype=np.float64)
                    state_sumsq = np.zeros(state_dim, dtype=np.float64)
                st_c = st[:state_dim].astype(np.float32)
                state_sum += st_c
                state_sumsq += (st_c ** 2)
                state_cnt += 1

                # 后缀技能序列
                skill_suffix = np.array(skills[i:], dtype=np.int16)

                # 保存样本
                np.savez(
                    os.path.join(epi_dir, f"{ep_id}_suffix_{i:02d}.npz"),
                    episode=ep_key,
                    instruction=inst,
                    H_txt=H_txt.astype(np.float32),
                    H_vis_start=H_vs.astype(np.float32),
                    skill_seq=skill_suffix,
                    state=st_c,
                    t0=int(t0),   # 记录起帧索引，便于追溯
                )
                kept += 1

    # 保存转移
    np.save(os.path.join(out_dir, "transition.npy"), trans)

    # 统计 meta
    state_mean = state_std = None
    if state_cnt > 0 and state_dim is not None:
        state_mean = (state_sum / max(1, state_cnt)).astype(np.float32)
        var = (state_sumsq / max(1, state_cnt)) - state_mean.astype(np.float64) ** 2
        var[var < 1e-12] = 1e-12
        state_std = np.sqrt(var).astype(np.float32)

    meta = dict(
        d_model=d_model,
        K=int(K),
        kept=kept,
        miss_inst=miss_inst,
        miss_img=miss_img,
        miss_skill=miss_skill,
        no_state_cnt=no_state_cnt,
        state_dim=int(state_dim or 0),
        state_mean=state_mean.tolist() if state_mean is not None else None,
        state_std=state_std.tolist() if state_std is not None else None,
        episode_dir="episodes",
        tokenizer=tokenizer_name_or_path,
        pi0_ckpt_path=pi0_ckpt_path,
        image_resolution=list(adapter.image_resolution),
        suffix_format="{ep_hash}_suffix_{i:02d}.npz",
        note="Each sample is a suffix from t0_i to episode end; H_vis_start/state taken at exact t0_i.",
    )
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[done] kept={kept}, miss_inst={miss_inst}, miss_img={miss_img}, "
          f"miss_skill={miss_skill}, no_state={no_state_cnt}")
    print(f"[out] episodes/*.npz, transition.npy, meta.json -> {out_dir}")


# ============================ CLI ============================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("Prep planning data (suffix slicing, frame-accurate HF/Pi-0)")
    ap.add_argument("--tfrecord_glob", required=True)
    ap.add_argument("--skills_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--episode_key", default="episode_metadata/file_path")
    ap.add_argument("--d_model", type=int, default=1024)
    ap.add_argument("--tokenizer_name_or_path", type=str, default="google/paligemma-3b-pt-224")
    ap.add_argument("--pi0_ckpt_path", type=str, default=None)
    args = ap.parse_args()
    main(**vars(args))
