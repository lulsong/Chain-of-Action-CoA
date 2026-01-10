#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
skill_expert_base.py

- Pi0Backbone: 冻结的多模态编码（PaliGemma 文本+图像），适配本地 safetensors 权重或 HF PaliGemma。
- SkillExpertHead: 轻量动作头 (可选 GRU)，输出动作 a_t 和 p_done_t。
- MixedStopper: 训练与推理共享的混合式中止逻辑（显式标签/启发式/时长先验）。
"""

import os
import math
from typing import Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 你本地 openpi 路径
import sys
sys.path.append("/home/lulsong/WorkTask/chain-of-action/openpi/src/")

from transformers import AutoTokenizer, AutoProcessor, AutoModelForVision2Seq
from PIL import Image

import safetensors.torch
from openpi.shared.image_tools import resize_with_pad_torch
import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel


# -----------------------------
# 冻结的 Pi0 / HF 编码骨干
# -----------------------------
class Pi0Backbone(nn.Module):
    """
    负责把 (instruction, image, optional state) 编码成固定维度特征。
    - 若 pi0_ckpt_dir 提供，则加载你转换好的 PaliGemmaWithExpertModel safetensors 权重 (bfloat16)；
    - 否则退化为 HF PaliGemma 前端（仅编码 text / image）。
    """

    def __init__(
        self,
        d_out: int = 1024,
        device: Optional[str] = None,
        hf_path_or_name: str = "/home/lulsong/Downloads/paligemma-3b-pt-224",
        pi0_ckpt_dir: Optional[str] = None,
        precision: str = "bfloat16",
        seed: int = 20240517,
    ):
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.precision = precision
        self.hf_path = hf_path_or_name
        self.pi0_ckpt_dir = pi0_ckpt_dir
        self.d_out = d_out
        self.action_proj = None

        if pi0_ckpt_dir:
            print(f"[Pi0Backbone] Load Pi-0 (safetensors) from: {pi0_ckpt_dir}")
            paligemma_cfg = _gemma.get_config("gemma_2b")
            action_expert_cfg = _gemma.get_config("gemma_300m")
            self.model = PaliGemmaWithExpertModel(
                paligemma_cfg, action_expert_cfg, use_adarms=[False, False],
                precision=("bfloat16" if precision == "bfloat16" else "float32")
            ).to(self.device).eval()

            st_path = os.path.join(pi0_ckpt_dir, "model.safetensors")
            state = safetensors.torch.load_file(st_path, device="cpu")
            miss, unexp = self.model.load_state_dict(state, strict=False)
            print(f"[Pi0Backbone] ckpt loaded (missing={len(miss)}, unexpected={len(unexp)})")

            # 分词器：直接本地路径加载，避免拉网
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_path, use_fast=True, local_files_only=True)
            # 估计嵌入维
            with torch.no_grad():
                dummy = torch.zeros(1, 4, dtype=torch.long, device=self.device)
                txt_emb = self.model.paligemma.language_model.embed_tokens(dummy)
            self.D_enc = int(txt_emb.shape[-1])

            self._load_action_proj(pi0_ckpt_dir)

            self.dtype_t = torch.bfloat16 if precision == "bfloat16" else torch.float32

        else:
            print("[Pi0Backbone] Using HF PaliGemma (no Pi-0 weights).")
            self.processor = AutoProcessor.from_pretrained(self.hf_path, local_files_only=True)
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.hf_path, torch_dtype=(torch.float16 if precision == "float16" else torch.float32),
                local_files_only=True
            ).to(self.device).eval()
            self.tokenizer = self.processor.tokenizer
            with torch.no_grad():
                dummy = torch.zeros(1, 4, dtype=torch.long, device=self.device)
                txt_emb = self.model.language_model.model.embed_tokens(dummy)
            self.D_enc = int(txt_emb.shape[-1])
            self.dtype_t = self.model.dtype if hasattr(self.model, "dtype") else torch.float16

        # 固定映射到 d_out（冻结）
        torch.manual_seed(seed)
        self.proj = nn.Linear(self.D_enc, d_out, bias=False).to(self.device, dtype=torch.float32)
        nn.init.orthogonal_(self.proj.weight)
        self.proj = self.proj.to(dtype=self.dtype_t)
        for p in self.parameters():
            p.requires_grad_(False)

        # 视觉异常维度时的单次投影
        self._proj_vis: Optional[nn.Linear] = None
        self._proj_vis_in: Optional[int] = None

    def _load_action_proj(self, ckpt_dir):
        """从原始 safetensors 提取 action_out_proj 权重"""
        try:
            sd = safetensors.torch.load_file(os.path.join(ckpt_dir, "model.safetensors"), device="cpu")
            weight_key = [k for k in sd.keys() if "action_out_proj.weight" in k]
            bias_key = [k for k in sd.keys() if "action_out_proj.bias" in k]
            if weight_key:
                w = sd[weight_key[0]]
                b = sd[bias_key[0]] if bias_key else torch.zeros(w.shape[0])
                self.action_proj = nn.Linear(w.shape[1], w.shape[0])
                self.action_proj.weight.data.copy_(w)
                self.action_proj.bias.data.copy_(b)
                print(f"[Pi0Backbone] ✔ Loaded action_out_proj ({list(w.shape)})")
            else:
                print("[Pi0Backbone] ⚠ No action_out_proj found in checkpoint.")
        except Exception as e:
            print(f"[Pi0Backbone] ⚠ Failed to load action_out_proj: {e}")

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        tok = self.tokenizer(text, return_tensors="pt", add_special_tokens=True).to(self.device)
        if hasattr(self.model, "paligemma"):  # Pi0
            emb = self.model.paligemma.language_model.embed_tokens(tok["input_ids"])
        else:  # HF
            emb = self.model.language_model.model.embed_tokens(tok["input_ids"])
        emb = self.proj(emb.to(self.proj.weight.dtype))
        return emb.squeeze(0)  # [T, d_out]

    @torch.no_grad()
    def encode_image(self, img_uint8: np.ndarray) -> torch.Tensor:
        assert img_uint8.ndim == 3 and img_uint8.shape[2] == 3
        if hasattr(self, "processor"):  # HF
            img = Image.fromarray(img_uint8)
            proc = self.processor(images=img, return_tensors="pt").to(self.device)
            vis = self.model.vision_tower(pixel_values=proc["pixel_values"]).last_hidden_state
        else:  # Pi0
            img = torch.from_numpy(img_uint8.astype(np.float32) / 255.0).to(self.device)
            if img.dim() == 3:
                img = img.unsqueeze(0)  # 1,H,W,3
            if img.shape[-1] != 3:
                img = img.permute(0, 2, 3, 1)
            img = img * 2.0 - 1.0
            img_resize = resize_with_pad_torch(img, 224, 224)  # [1,H,W,3]
            if img_resize.dim() == 3:
                img_resize = img_resize.unsqueeze(0)
            bchw = img_resize.permute(0, 3, 1, 2).contiguous()
            vis = self.model.paligemma.model.get_image_features(bchw)

        Dv = vis.shape[-1]
        if Dv != self.D_enc:
            if (self._proj_vis is None) or (self._proj_vis_in != Dv):
                self._proj_vis_in = Dv
                self._proj_vis = nn.Linear(Dv, self.D_enc, bias=False).to(self.device, dtype=self.dtype_t)
                nn.init.xavier_uniform_(self._proj_vis.weight)
                for p in self._proj_vis.parameters(): p.requires_grad_(False)
                print(f"[Pi0Backbone] create visual proj: {Dv} -> {self.D_enc}")
            vis = self._proj_vis(vis.to(self.dtype_t))
        out = self.proj(vis.to(self.proj.weight.dtype))  # [1,T,d_out]
        return out.squeeze(0)  # [T, d_out]


# -----------------------------
# 技能专家头
# -----------------------------
class SkillExpertHead(nn.Module):
    """
    简洁但强的动作头：
     - 输入: 每帧融合特征 F_t = fuse([txt ctx, img feat_t, state_t, skill_emb])
     - 编码器: GRU (可关) + MLP
     - 输出: 动作向量 a_t (Da) + p_done_t

    仅训练此头（Pi0Backbone 冻结）。
    """

    def __init__(
        self,
        d_in: int,           # d_text + d_img + d_state + d_skill
        d_hidden: int,
        action_dim: int,
        use_gru: bool = True,
        skill_vocab: int = 9,
        d_skill: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_gru = use_gru
        self.skill_emb = nn.Embedding(skill_vocab, d_skill)
        self.fuse = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if use_gru:
            self.gru = nn.GRU(input_size=d_hidden, hidden_size=d_hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, action_dim),
        )
        self.done_head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, 1),
        )

        # 轻量技能特化器（每个技能一个 adapter）
        self.skill_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_hidden, d_hidden),
                nn.GELU(),
                nn.Linear(d_hidden, d_hidden)
            ) for _ in range(skill_vocab)
        ])

    def init_from_pi0(self, pi0_action_proj: Optional[nn.Linear]):
        """用 Pi0 原 action_out_proj 初始化输出层"""
        if pi0_action_proj is None:
            print("[SkillExpertHead] ⚠ No Pi0 action_out_proj provided.")
            return
        my_proj = self.head[-1]
        if (my_proj.weight.shape == pi0_action_proj.weight.shape):
            my_proj.weight.data.copy_(pi0_action_proj.weight.data)
            my_proj.bias.data.copy_(pi0_action_proj.bias.data)
            print(f"[SkillExpertHead] ✔ Initialized from Pi0 action_out_proj ({list(my_proj.weight.shape)})")
        else:
            print(f"[SkillExpertHead] ⚠ Shape mismatch: {my_proj.weight.shape} vs {pi0_action_proj.weight.shape}")

    def forward(
        self,
        txt_ctx: torch.Tensor,   # [B, Ttxt, d_txt]
        img_seq: torch.Tensor,   # [B, T,    d_img]
        state_seq: torch.Tensor, # [B, T,    d_state]
        skill_ids: torch.Tensor, # [B] or [B,1]
    ) -> Dict[str, torch.Tensor]:
        B, T, d_img = img_seq.shape
        # 文本上下文池化（mean/cls）
        tctx = txt_ctx.mean(dim=1, keepdim=True).expand(B, T, -1)  # [B,T,d_txt]
        # 状态 + 技能嵌入
        skill = self.skill_emb(skill_ids.view(B))  # [B, d_skill]
        skill = skill.unsqueeze(1).expand(B, T, -1)
        # 融合
        x = torch.cat([tctx, img_seq, state_seq, skill], dim=-1)  # [B,T,d_in]
        x = self.fuse(x)  # [B,T,d_hidden]
        if self.use_gru:
            x, _ = self.gru(x)  # [B,T,d_hidden]

        # 使用 Adapter 为每个技能添加专门的特化操作
        skill_specific_x = []
        for i in range(B):
            # 获取当前样本的技能 ID，并通过对应的 Adapter
            adapter = self.skill_adapters[skill_ids[i].item()]
            skill_specific_x.append(adapter(x[i]))  # [T,d_hidden]
        x = torch.stack(skill_specific_x, dim=0)  # [B,T,d_hidden]

        a = self.head(x)            # [B,T,Da]
        p = torch.sigmoid(self.done_head(x)).squeeze(-1)  # [B,T]
        return {"action": a, "p_done": p}


# -----------------------------
# 混合式中止逻辑（训练/推理共用）
# -----------------------------
class MixedStopper:
    """
    训练阶段：
      Loss = λ1·BCE(p_done, done_label) + λ2·BCE(p_done, done_heur) + λ3·KL(p_done || prior(t;T))
    推理阶段：
      p_mix = α·p_done + β·p_heur + γ·p_prior(t;T)
      触发条件：p_mix > τ 或 达到最大步数
    """

    def __init__(
        self,
        lambda_label: float = 1.0,
        lambda_heur: float = 0.3,
        lambda_prior: float = 0.2,
        alpha: float = 0.7,
        beta: float = 0.15,
        gamma: float = 0.15,
        tau: float = 0.6,
        prior_type: str = "linear",   # "linear" | "beta"
        beta_ab: Tuple[float, float] = (2.0, 2.0),
    ):
        self.lambda_label = lambda_label
        self.lambda_heur = lambda_heur
        self.lambda_prior = lambda_prior
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.prior_type = prior_type
        self.beta_a, self.beta_b = beta_ab

    def _build_prior(self, T: int, device) -> torch.Tensor:
        # 0..T-1 -> (0,1] 线性上升 / Beta CDF
        t = torch.arange(T, device=device, dtype=torch.float32)
        x = (t + 1) / T
        if self.prior_type == "linear":
            return x  # 简单上升
        else:
            # 近似 Beta(a,b) CDF，简单用正则化不严格数值：用 x^a / (x^a + (1-x)^b)
            a, b = self.beta_a, self.beta_b
            num = x.clamp(1e-6, 1-1e-6).pow(a)
            den = num + (1 - x.clamp(1e-6, 1-1e-6)).pow(b)
            return (num / den).clamp(1e-3, 1-1e-3)

    def loss(
        self,
        p_done: torch.Tensor,      # [B,T]
        done_label: torch.Tensor,  # [B,T] 0/1
        done_heur: Optional[torch.Tensor] = None,  # [B,T] 0/1
    ):
        # --- 支持 flatten 输入 ---
        if p_done.dim() == 1:
            # 无法展开成 B,T 结构时，只用 label + heur，跳过 prior
            loss_label = F.binary_cross_entropy(p_done, done_label, reduction="mean")
            loss_heur = 0.0
            if done_heur is not None:
                loss_heur = F.binary_cross_entropy(p_done, done_heur, reduction="mean")
            total = self.lambda_label * loss_label + self.lambda_heur * loss_heur
            return total, {
                "loss_label": float(loss_label.item()),
                "loss_heur": float(loss_heur if isinstance(loss_heur, float) else loss_heur.item()),
                "loss_prior": 0.0,
            }
        B, T = p_done.shape
        device = p_done.device
        prior = self._build_prior(T, device)[None, :].expand(B, T)

        loss_label = F.binary_cross_entropy(p_done, done_label, reduction="mean")

        loss_heur = 0.0
        if done_heur is not None:
            loss_heur = F.binary_cross_entropy(p_done, done_heur, reduction="mean")

        # KL(p||prior) 约束 p 随时间上升（避免过早触发）
        p = p_done.clamp(1e-4, 1-1e-4)
        q = prior.clamp(1e-4, 1-1e-4)
        loss_prior = torch.mean(p * (torch.log(p) - torch.log(q)) + (1-p) * (torch.log(1-p) - torch.log(1-q)))

        total = self.lambda_label * loss_label + self.lambda_heur * loss_heur + self.lambda_prior * loss_prior
        return total, {
            "loss_label": float(loss_label.item()),
            "loss_heur": float(loss_heur if isinstance(loss_heur, float) else loss_heur.item()),
            "loss_prior": float(loss_prior.item()),
        }

    @torch.no_grad()
    def stopping_mask(
        self,
        p_done: torch.Tensor,             # [T]
        heur_seq: Optional[torch.Tensor], # [T]
    ):
        T = p_done.shape[0]
        device = p_done.device
        prior = self._build_prior(T, device)
        p_mix = self.alpha * p_done
        if heur_seq is not None:
            p_mix = p_mix + self.beta * heur_seq
        p_mix = p_mix + self.gamma * prior
        stop_idx = (p_mix > self.tau).nonzero().flatten()
        if len(stop_idx) == 0:
            return False, T-1, p_mix  # 不触发，走到末帧
        return True, int(stop_idx[0].item()), p_mix
