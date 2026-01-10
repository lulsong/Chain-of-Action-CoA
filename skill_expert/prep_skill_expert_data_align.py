#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prep_skill_expert_data_from_tfrecord.py
从原始 DROID TFRecord + skills.jsonl 生成 per-skill expert 数据 (.npz)
输入:
  --tfrecord_glob   e.g. "/path/to/r2d2_faceblur-train.tfrecord-*"
  --skills_jsonl    e.g. "./segmentation/droid_out_k9_stream_success/skills.jsonl"
  --out_dir         e.g. "./skill_expert_data"
每个输出 .npz 含字段:
  images[T,H,W,3], states[T,Ds], actions[T,Da], done[T], skill_id, instruction
"""

import os, re, json, glob
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from typing import Dict, List, Any, Optional, Tuple

# --------------------- 工具函数 ---------------------

def read_skills(skills_jsonl: str) -> Dict[str, List[Dict[str, Any]]]:
    """读取 skills.jsonl -> {episode_path: [{"start":, "end":, "skill":}, ...]}"""
    ep2segs = {}
    with open(skills_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            ep = str(o["episode"])
            segs = o["segments"]
            ep2segs[ep] = segs
    print(f"[skills] loaded {len(ep2segs)} episodes from {skills_jsonl}")
    return ep2segs

def pick_instruction(f: Any) -> Optional[str]:
    """从 TFRecord example 选取非空 language_instruction（取最长非空）"""
    for key in ["steps/language_instruction", "steps/language_instruction_2", "steps/language_instruction_3"]:
        if key not in f:
            continue
        vals = f[key].bytes_list.value
        if not vals:
            continue
        texts = [b.decode(errors="ignore").strip() for b in vals if len(b) > 0]
        texts = [t for t in texts if len(t) > 0]
        if texts:
            s = max(texts, key=len)
            return re.sub(r"\s+", " ", s)
    return None

def decode_image(img_bytes: bytes) -> np.ndarray:
    """JPEG/PNG -> np.uint8[H,W,3]"""
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    return img.numpy().astype(np.uint8)

def _extract_series_from_floatlist(f, key: str, T_hint: Optional[int]) -> Optional[np.ndarray]:
    """把 float_list 扁平数组转为 [T,D]；优先用 T_hint（来自图像帧数）推断 T。"""
    if key not in f:
        return None
    vals = f[key].float_list.value
    if not vals:
        return None
    arr = np.array(vals, dtype=np.float32)
    if T_hint and T_hint > 0 and arr.size % T_hint == 0:
        D = arr.size // T_hint
        return arr.reshape(T_hint, D)
    return arr.reshape(-1, 1)

def _extract_series_from_byteslist(f, key: str) -> Optional[np.ndarray]:
    """尝试从 bytes_list 解析序列（常见做法：每步一个 bytes）。若不是这种格式则返回 None。"""
    if key not in f:
        return None
    bl = f[key].bytes_list.value
    if not bl:
        return None
    out = []
    for b in bl:
        try:
            x = np.frombuffer(b, dtype=np.float32)
            if x.size == 0:
                return None
            out.append(x[None, :])  # [1,D]
            continue
        except Exception:
            return None
    if out:
        return np.concatenate(out, axis=0)  # [T,D]
    return None

def extract_multichannel_series(
    f: Any,
    keys_priority: List[str],
    T_hint: Optional[int],
) -> Optional[np.ndarray]:
    """
    从多个候选 key 中抽取时间序列，并在特征维拼接：
      - 每个 key 独立解析为 [T_k, D_k]
      - 对齐时间长度 T = min(T_hint, *T_k>0)
      - 截断每个通道到 T，并在特征维 concat -> [T, sum D_k]
    若一个都取不到，返回 None。
    """
    series_list: List[np.ndarray] = []

    # 先尝试 bytes_list（每步一个 bytes）
    for key in keys_priority:
        sr = _extract_series_from_byteslist(f, key)
        if sr is not None and sr.ndim == 2 and sr.shape[0] > 0:
            series_list.append(sr)

    # 再尝试 float_list（扁平一长串）
    for key in keys_priority:
        sr = _extract_series_from_floatlist(f, key, T_hint=T_hint)
        if sr is not None and sr.ndim == 2 and sr.shape[0] > 0:
            series_list.append(sr)

    if not series_list:
        return None

    # 时间对齐（尽量与图像长度一致）
    T_candidates = [s.shape[0] for s in series_list]
    if T_hint and T_hint > 0:
        T = min([T_hint] + T_candidates)
    else:
        T = min(T_candidates)

    if T <= 0:
        return None

    aligned = []
    for s in series_list:
        if s.shape[0] < T:
            T_use = s.shape[0]
        else:
            T_use = T
        aligned.append(s[:T_use])

    for i, s in enumerate(aligned):
        if s.shape[0] < T:
            pad = np.zeros((T - s.shape[0], s.shape[1]), dtype=s.dtype)
            aligned[i] = np.concatenate([s, pad], axis=0)

    return np.concatenate(aligned, axis=1)  # [T, sum D_k]

def parse_example(ex) -> Dict[str, Any]:
    """解析一条 TFRecord Example -> dict"""
    f = ex.features.feature
    out = {}

    # episode path
    ep = None
    for k in ["episode_metadata/file_path", "episode_metadata/recording_folderpath"]:
        if k in f and len(f[k].bytes_list.value):
            ep = f[k].bytes_list.value[0].decode()
            break
    out["episode"] = ep

    # instruction
    out["instruction"] = pick_instruction(f)

    # observation image（优先 wrist，再次 general image）
    img_key = None
    for k in [
        "steps/observation/wrist_image_left",
        "steps/observation/wrist_image_right",
        "steps/observation/exterior_image_1_left",
        "steps/observation/image",
    ]:
        if k in f and len(f[k].bytes_list.value):
            img_key = k
            break

    if img_key is not None:
        imgs_bytes = f[img_key].bytes_list.value
        imgs = [decode_image(b) for b in imgs_bytes]
        images = np.stack(imgs, axis=0)  # [T,H,W,3]
    else:
        images = np.zeros((0, 64, 64, 3), dtype=np.uint8)
    out["images"] = images
    T_hint = images.shape[0] if images.ndim == 4 else None

    # states（按特征维 concat）
    state_keys = [
        "steps/observation/joint_position",
        "steps/observation/gripper_position",
    ]
    states = extract_multichannel_series(f, state_keys, T_hint=T_hint)
    if states is None:
        states = np.zeros((T_hint or 0, 1), dtype=np.float32)  # 没有就给占位
    out["states"] = states  # [T, Ds]

    # actions（按特征维 concat）
    action_keys = [
        "steps/action_dict/joint_position",
        "steps/action_dict/gripper_position",
    ]
    actions = extract_multichannel_series(f, action_keys, T_hint=T_hint)
    if actions is None:
        actions = np.zeros((T_hint or 0, 1), dtype=np.float32)
    out["actions"] = actions  # [T, Da]

    return out

# --------------------- 主流程 ---------------------
def main(tfrecord_glob: str, skills_jsonl: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    ep2segs = read_skills(skills_jsonl)
    files = sorted(glob.glob(tfrecord_glob))
    print(f"[data] found {len(files)} TFRecord shards")

    kept = miss_skill = miss_img = miss_inst = 0

    for tfrec in files:
        ds = tf.data.TFRecordDataset(tfrec)
        for rec in tqdm(ds, desc=os.path.basename(tfrec)):
            ex = tf.train.Example.FromString(bytes(rec.numpy()))
            data = parse_example(ex)

            ep = data["episode"]
            if ep is None:
                continue

            # 标准化路径以匹配 skills.jsonl
            ep_norm = re.sub(r"/recordings/[^/]+$", "/trajectory.h5", ep)
            if ep in ep2segs:
                segs = ep2segs[ep]
            elif ep_norm in ep2segs:
                segs = ep2segs[ep_norm]
            else:
                miss_skill += 1
                continue

            images: np.ndarray = data["images"]  # [T,H,W,3]
            if images.shape[0] == 0:
                miss_img += 1
                continue

            instr = data["instruction"] or ""
            if not instr.strip():
                miss_inst += 1
                continue

            states: np.ndarray = data["states"]  # [T, Ds]
            actions: np.ndarray = data["actions"]  # [T, Da]
            T = images.shape[0]

            # 与图像对齐（保险起见）
            T_series = min(T, states.shape[0], actions.shape[0])
            images = images[:T_series]
            states = states[:T_series]
            actions = actions[:T_series]
            T = T_series

            # 为每个 skill 段保存 .npz
            for seg in segs:
                s, e, skill = int(seg["t0"]), int(seg["t1"]), int(seg["skill"])
                if e > T:
                    e = T
                if e - s < 2:
                    continue

                seg_imgs = images[s:e]  # [Ts,H,W,3]
                seg_states = states[s:e]  # [Ts, Ds]
                seg_actions = actions[s:e]  # [Ts, Da]
                done = np.zeros(e - s, dtype=np.float32)
                done[-1] = 1.0

                name = os.path.basename((ep if ep in ep2segs else ep_norm)).replace(".h5", "")
                outp = os.path.join(out_dir, f"{name}_{s:04d}_{e:04d}_k{skill}.npz")
                np.savez_compressed(
                    outp,
                    images=seg_imgs,
                    states=seg_states,
                    actions=seg_actions,
                    done=done,
                    skill_id=int(skill),
                    instruction=instr,
                )
                kept += 1

    print(f"\n✅ Done: {kept} skill segments saved")
    print(f"  miss_skill={miss_skill}, miss_img={miss_img}, miss_inst={miss_inst}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("Prepare per-skill expert data from TFRecord + skills.jsonl")
    ap.add_argument("--tfrecord_glob", required=True)
    ap.add_argument("--skills_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    main(**vars(args))
