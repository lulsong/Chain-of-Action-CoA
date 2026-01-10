#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_skill_expert.py

从 prep_skill_expert_data_from_tfrecord.py 生成的 .npz 段训练 SkillExpert：
  - 冻结 Pi0Backbone，用其编码文本 & 图像；
  - 动作头预测 a_t、p_done_t；
  - 混合式中止损失 (label + heuristic + prior)；
  - 动作监督：L1 或 MSE（默认 SmoothL1）。
"""

import os, glob, json, random
import numpy as np
from typing import Optional, Dict, Any, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from skill_expert_base import Pi0Backbone, SkillExpertHead, MixedStopper


def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ---------------- Dataset ----------------
class SkillSegDataset(Dataset):
    """
    每个样本是一段技能片段：images[T,H,W,3], states[T,Ds], actions[T,Da], done[T], skill_id, instruction
    """
    def __init__(self, npz_dir: str, max_len: int = 60):
        self.paths = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
        self.max_len = max_len

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        z = np.load(self.paths[i], allow_pickle=True)
        img = z["images"]    # [T,H,W,3] uint8
        st  = z["states"].astype(np.float32)  # [T,Ds]
        act = z["actions"].astype(np.float32) # [T,Da]
        if act.shape[-1] < 32:
            act = np.pad(act, ((0, 0), (0, 32 - act.shape[-1])), mode='constant')
        dn  = z["done"].astype(np.float32)    # [T]
        skill = int(z["skill_id"])
        instr = str(z["instruction"])

        T = img.shape[0]
        if T > self.max_len:
            # 简单中心裁剪
            s = (T - self.max_len)//2
            e = s + self.max_len
            img, st, act, dn = img[s:e], st[s:e], act[s:e], dn[s:e]
            T = img.shape[0]

        return {
            "images": img, "states": st, "actions": act, "done": dn,
            "skill": skill, "instruction": instr, "T": T
        }


def collate(batch: List[Dict[str,Any]], device, backbone: Pi0Backbone):
    B = len(batch)
    # 编码文本（每个样本1次；推后到GPU）
    txt_list = []
    for b in batch:
        txt = backbone.encode_text(b["instruction"])  # [Ttxt, d_out]
        txt_list.append(txt)
    # 编码图像（每帧）
    img_list = []
    for b in batch:
        feats = []
        for t in range(b["T"]):
            feats.append(backbone.encode_image(b["images"][t]))  # [Tvis,d_out] -> mean
        # 对每帧图像特征做 mean 池化（SigLIP 输出多个 patch token）
        # 这里对 encode_image 的输出是 [T_vis, d]; 取均值作为帧向量
        f = torch.stack([x.mean(dim=0) for x in feats], dim=0)  # [T, d_out]
        img_list.append(f)

    # pad 到同长度
    T_max = max(b["T"] for b in batch)
    d_out = img_list[0].shape[-1]
    d_state = batch[0]["states"].shape[-1]
    da = batch[0]["actions"].shape[-1]

    img_pad = torch.zeros(B, T_max, d_out, device=backbone.device, dtype=backbone.proj.weight.dtype)
    state_pad = torch.zeros(B, T_max, d_state, device=backbone.device, dtype=torch.float32)
    act_pad = torch.zeros(B, T_max, da, device=backbone.device, dtype=torch.float32)
    done_pad = torch.zeros(B, T_max, device=backbone.device, dtype=torch.float32)
    mask = torch.zeros(B, T_max, device=backbone.device, dtype=torch.bool)
    txt_ctx = torch.zeros(B, 8, d_out, device=backbone.device, dtype=backbone.proj.weight.dtype)  # 截断8个token上下文

    skill_ids = torch.tensor([b["skill"] for b in batch], device=backbone.device, dtype=torch.long)

    for i,b in enumerate(batch):
        T = b["T"]
        img_pad[i,:T] = img_list[i]
        state_pad[i,:T] = torch.from_numpy(b["states"]).to(backbone.device)
        act_pad[i,:T] = torch.from_numpy(b["actions"]).to(backbone.device)
        done_pad[i,:T] = torch.from_numpy(b["done"]).to(backbone.device)
        mask[i,:T] = True
        tctx = txt_list[i]
        if tctx.shape[0] >= 8:
            txt_ctx[i] = tctx[:8]
        else:
            txt_ctx[i,:tctx.shape[0]] = tctx

    return {
        "txt_ctx": txt_ctx,           # [B, Ttxt(=8), d]
        "img_seq": img_pad,           # [B, T, d]
        "state_seq": state_pad,       # [B, T, Ds]
        "actions": act_pad,           # [B, T, Da]
        "done": done_pad,             # [B, T]
        "mask": mask,                 # [B, T]
        "skill_ids": skill_ids,       # [B]
        "T_max": T_max
    }


# ---------------- 训练主逻辑 ----------------
def train(
    data_dir: str,
    out_dir: str,
    pi0_ckpt_dir: Optional[str],
    hf_path_or_name: str,
    action_dim: int,
    d_backbone_out: int = 1024,
    d_hidden: int = 1024,
    batch_size: int = 4,
    epochs: int = 10,
    lr: float = 1e-4,
    max_len: int = 60,
    use_gru: bool = True,
    seed: int = 42,
):
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 冻结骨干
    backbone = Pi0Backbone(
        d_out=d_backbone_out, device=device,
        hf_path_or_name=hf_path_or_name, pi0_ckpt_dir=pi0_ckpt_dir, precision="bfloat16"
    )

    # head 输入维：txt(d) + img(d) + state(Ds) + skill(d_skill)
    # Ds、Da 从数据首个样本推断
    train_ds = SkillSegDataset(data_dir, max_len=max_len)
    if len(train_ds) == 0:
        raise RuntimeError(f"No .npz found under {data_dir}")
    # Peek one sample to get dims
    tmp = train_ds[0]
    Ds = tmp["states"].shape[-1]
    Da = tmp["actions"].shape[-1]
    assert Da == action_dim, f"action_dim mismatch: npz={Da}, arg={action_dim}"

    head = SkillExpertHead(
        d_in=d_backbone_out + d_backbone_out + Ds + 64,
        d_hidden=d_hidden, action_dim=action_dim, use_gru=use_gru,
        skill_vocab=9, d_skill=64, dropout=0.1
    ).to(device)

    head.init_from_pi0(backbone.action_proj)

    # dataloader
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                        collate_fn=lambda b: collate(b, device, backbone), num_workers=0)

    # 损失
    stopper = MixedStopper(lambda_label=1.0, lambda_heur=0.25, lambda_prior=0.2,
                           alpha=0.7, beta=0.15, gamma=0.15, tau=0.6)

    act_loss_fn = nn.SmoothL1Loss(reduction="none")
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)

    best = {"loss": 1e9}

    for ep in range(1, epochs+1):
        head.train()
        loss_meter = act_meter = done_meter = 0.0
        n_tok = 0

        for batch in loader:
            txt_ctx = batch["txt_ctx"]
            mask = batch["mask"]
            out = head(txt_ctx, batch["img_seq"], batch["state_seq"], batch["skill_ids"])
            a_pred = out["action"]   # [B,T,Da]
            p_done = out["p_done"]   # [B,T]

            # 动作损失（只在有效时间步）
            act_loss = act_loss_fn(a_pred[..., :8], batch["actions"][..., :8]).mean(dim=-1)  # [B,T]
            act_loss = (act_loss * mask.float()).sum() / mask.float().sum().clamp(min=1)

            # 构造启发式 done（速度/像素势能缺省，这里用Δa阈值近似）
            with torch.no_grad():
                diff = torch.norm(batch["actions"][:,1:]-batch["actions"][:,:-1], dim=-1)  # [B,T-1]
                thr = torch.quantile(diff[batch["mask"][:,1:]], 0.8) if diff[batch["mask"][:,1:]].numel() > 0 else torch.tensor(0.0, device=diff.device)
                heur = torch.zeros_like(p_done)
                heur[:,1:][diff > thr] = 1.0  # 粗糙的启发式
            # 中止损失
            d_loss, parts = stopper.loss(
                p_done=p_done[mask],                 # flatten
                done_label=batch["done"][mask],
                done_heur=heur[mask]
            )

            loss = act_loss + d_loss

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()

            nt = int(mask.float().sum().item())
            n_tok += nt
            loss_meter += float(loss.item()) * nt
            act_meter += float(act_loss.item()) * nt
            done_meter += float(d_loss.item()) * nt

        print(f"[ep {ep:02d}] loss={(loss_meter/max(1,n_tok)):.4f}  act={(act_meter/max(1,n_tok)):.4f}  done={(done_meter/max(1,n_tok)):.4f}")

        # 保存
        ckpt = {
            "head": head.state_dict(),
            "backbone": {
                "d_out": d_backbone_out,
                "hf_path": hf_path_or_name,
                "pi0_ckpt_dir": pi0_ckpt_dir,
                "precision": "bfloat16",
            },
            "cfg": {
                "d_hidden": d_hidden,
                "action_dim": action_dim,
                "use_gru": use_gru,
            }
        }
        torch.save(ckpt, os.path.join(out_dir, "last.pt"))
        if (loss_meter/max(1,n_tok)) < best["loss"]:
            best = {"loss": (loss_meter/max(1,n_tok))}
            torch.save(ckpt, os.path.join(out_dir, "best.pt"))
            print("  ↳ new best saved")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("Train Skill Expert on per-skill segments with mixed stopping")
    ap.add_argument("--data_dir", required=True, help="dir of *.npz from prep_skill_expert_data_from_tfrecord.py")
    ap.add_argument("--out_dir", default="./skill_expert_ckpt")
    ap.add_argument("--pi0_ckpt_dir", default="/home/lulsong/Downloads/pi0_droid_pytorch")
    ap.add_argument("--hf_path_or_name", default="/home/lulsong/Downloads/paligemma-3b-pt-224")
    ap.add_argument("--action_dim", type=int, required=True)
    ap.add_argument("--d_backbone_out", type=int, default=1024)
    ap.add_argument("--d_hidden", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_len", type=int, default=60)
    ap.add_argument("--use_gru", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    train(**vars(args))
