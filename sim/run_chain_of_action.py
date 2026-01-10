#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_chain_of_action.py

整合 Planning → Skill Experts：
 - 输入：planning 输出的技能序列 + 当前指令 + 帧流 + 状态流
 - 对每个技能按顺序执行：逐帧动作预测 & 混合式中止
 - 将动作序列拼接成 full trajectory（或在线下发到控制器）

演示版：从一个帧目录（全局时间）和状态.npy 切段执行；真实部署时，替换为实时摄像头+传感器。
"""

import os, json
import numpy as np
from PIL import Image
import torch

from skill_expert.skill_expert_base import Pi0Backbone, SkillExpertHead, MixedStopper


def slice_stream(frames, states, t0, t1):
    return frames[t0:t1], states[t0:t1]


@torch.no_grad()
def run_chain(
    planners_seq,         # List[int] 规划出来的技能ID序列
    instruction: str,
    frames: np.ndarray,   # [T,H,W,3]
    states: np.ndarray,   # [T,Ds]
    head_ckpt: str,
    tau: float = 0.6,
    max_steps_per_skill: int = 96,
):
    # 载入技能头
    ckpt = torch.load(head_ckpt, map_location="cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bb_cfg = ckpt["backbone"]
    backbone = Pi0Backbone(
        d_out=bb_cfg["d_out"], device=device,
        hf_path_or_name=bb_cfg["hf_path"], pi0_ckpt_dir=bb_cfg["pi0_ckpt_dir"],
        precision=bb_cfg.get("precision","bfloat16")
    )
    Ds = states.shape[-1]
    head_cfg = ckpt["cfg"]
    head = SkillExpertHead(
        d_in=bb_cfg["d_out"]+bb_cfg["d_out"]+ Ds + 64,
        d_hidden=head_cfg["d_hidden"],
        action_dim=head_cfg["action_dim"],
        use_gru=head_cfg["use_gru"],
    ).to(device)
    head.load_state_dict(ckpt["head"], strict=True)
    head.eval()

    T = frames.shape[0]
    t_cursor = 0
    all_actions = []

    for k in planners_seq:
        # 从当前光标起，拿一个窗口（这里演示用固定窗口；实际可在线滚动）
        t1 = min(T, t_cursor + max_steps_per_skill)
        if t1 <= t_cursor:
            break
        sub_frames = [frames[t] for t in range(t_cursor, t1)]
        sub_states = states[t_cursor:t1]
        # 推理
        ret = infer_one(backbone, head, instruction, sub_frames, sub_states, k, tau=tau, max_steps=max_steps_per_skill)
        acts = ret["actions"]
        all_actions.extend(acts)
        t_cursor += (ret["stop_idx"] + 1)
        if t_cursor >= T:
            break

    return np.asarray(all_actions, dtype=np.float32)


def infer_one(backbone, head, instruction, frames_list, states, skill_id, tau, max_steps):
    # 与 infer_skill_expert.py 相同逻辑，写成内部函数避免重复导入
    tctx = backbone.encode_text(instruction)
    if tctx.shape[0] > 8: tctx = tctx[:8]
    tctx = tctx.unsqueeze(0)

    feats = []
    for img in frames_list:
        v = backbone.encode_image(img)
        feats.append(v.mean(dim=0))
    img_seq = torch.stack(feats, dim=0).unsqueeze(0)   # [1,T,d]
    state_seq = torch.from_numpy(states.astype(np.float32)).to(backbone.device).unsqueeze(0)
    skills = torch.tensor([skill_id], device=backbone.device, dtype=torch.long)

    out = head(tctx, img_seq, state_seq, skills)
    a = out["action"][0]
    p = out["p_done"][0]

    heur = torch.zeros_like(p)
    if a.shape[0]>1:
        diff = torch.norm(a[1:]-a[:-1], dim=-1)
        thr = torch.quantile(diff, 0.8) if diff.numel()>0 else torch.tensor(0.0, device=diff.device)
        heur[1:][diff>thr] = 1.0

    stopper = MixedStopper(tau=tau)
    stop, stop_idx, p_mix = stopper.stopping_mask(p, heur)
    if max_steps is not None:
        stop_idx = min(stop_idx, max_steps-1)

    return {"actions": a[:stop_idx+1].cpu().numpy().tolist(),
            "stop_idx": int(stop_idx), "stopped": bool(stop)}


def main(planner_json: str, head_ckpt: str, frame_dir: str, states_npy: str,
         instruction: str, tau: float=0.6, max_steps_per_skill: int=96, out: str="./run_actions.npy"):
    # 规划输出格式示例：{"pred_skill_seq":[k1,k2,...], "K":..., ...}
    plan = json.load(open(planner_json, "r"))
    seq = plan.get("pred_skill_seq", [])
    if not seq:
        raise RuntimeError("planner sequence empty.")

    # 载入帧与状态（演示版：从目录/文件）
    frames = []
    for fn in sorted([x for x in os.listdir(frame_dir) if x.lower().endswith((".jpg",".png",".jpeg"))]):
        frames.append(np.array(Image.open(os.path.join(frame_dir, fn)).convert("RGB")))
    frames = np.stack(frames, axis=0)
    states = np.load(states_npy)

    acts = run_chain(seq, instruction, frames, states, head_ckpt, tau=tau, max_steps_per_skill=max_steps_per_skill)
    np.save(out, acts)
    print(f"[OK] total {acts.shape[0]} actions -> {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("Chain-of-Action: planning → skill experts (mixed stopping)")
    ap.add_argument("--planner_json", required=True)
    ap.add_argument("--head_ckpt", required=True)
    ap.add_argument("--frame_dir", required=True)
    ap.add_argument("--states_npy", required=True)
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--max_steps_per_skill", type=int, default=96)
    ap.add_argument("--out", default="./run_actions.npy")
    args = ap.parse_args()
    main(**vars(args))
