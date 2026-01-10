#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_chain_of_action_mujoco.py (final aligned)
- 规划链路(Planner): 1024维特征 —— 与 SkillFormer 训练一致
- 技能链路(Skill expert): 1024维特征 —— 与 SkillExpert 训练一致
- 状态处理：规划链路做标准化(与 plan collate 一致)；技能链路用原始/pad截断(与 expert 训练一致)
- 图像特征：对每帧做 patch-token mean pooling（与 expert 训练一致）
- 规划解码：greedy_with_constraints（与训练一致）
"""

import os, json, cv2
import numpy as np
import torch
import mujoco
from tqdm import tqdm

from planning.add_state.train_skillformer_nogoal import SkillFormer, greedy_with_constraints
from skill_expert.skill_expert_base import SkillExpertHead, Pi0Backbone

import sys
sys.path.append('/home/lulsong/WorkTask/chain-of-action/openpi/src/openpi')
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel

# =========================
# Config
# =========================
class Config:
    # ---- 环境 ----
    xml_path = "/home/lulsong/WorkTask/chain-of-action/sim/panda.xml"
    width, height = 640, 480

    # ---- Planning ----
    meta_json = "/home/lulsong/WorkTask/chain-of-action/planning/plan_data_pi0/meta.json"
    plan_ckpt = "/home/lulsong/WorkTask/chain-of-action/planning/plan_ckpt_nogoal/best.pt"
    trans_npy = "/home/lulsong/WorkTask/chain-of-action/planning/plan_data_pi0/transition.npy"

    # ---- Skill Expert ----
    skill_ckpt = "../skill_expert/skill_expert_ckpt/best.pt"

    # ---- Pi0 / Tokenizer ----
    pi0_ckpt_path = "/home/lulsong/Downloads/pi0_droid_pytorch"
    tokenizer_dir = "/home/lulsong/Downloads/paligemma-3b-pt-224"
    precision = "bfloat16"

    # ---- 推理参数 ----
    tau = 0.6
    max_steps = 96
    ctrl_clip = 1.0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 输出 ----
    video_out = "./coa_demo.mp4"
    result_json = "./coa_result.json"

    # ---- 文本指令（示例）----
    instruction = "pick up the red cube"



# =========================
# MuJoCo env (Renderer)
# =========================
class MujocoEnv:
    def __init__(self, xml_path, width, height):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Model not found: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.width, self.height = width, height
        self.nu = self.model.nu
        print(f"[Env] Loaded {xml_path}, nu={self.nu}")

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)

    def step(self, ctrl):
        if ctrl.shape[0] != self.nu:
            raise ValueError(f"ctrl shape mismatch: {ctrl.shape[0]} vs {self.nu}")
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data)

    def get_state(self):
        return np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()])

    def render_rgb(self):
        self.renderer.update_scene(self.data)
        img = self.renderer.render()
        return img[:, :, :3]





# =========================
# Utils: state preprocessing
# =========================
def plan_preprocess_state(env_state: np.ndarray, target_dim: int,
                          mean: np.ndarray | None, std: np.ndarray | None) -> np.ndarray:
    """
    规划链路的状态处理，与 train_skillformer_nogoal.collate 完全一致：
      - 截断/零填充到 target_dim
      - 标准化 (x - mean) / (std + 1e-6)
    返回 shape [1, target_dim]
    """
    s = np.zeros(target_dim, dtype=np.float32)
    n = min(target_dim, env_state.shape[0])
    s[:n] = env_state[:n]
    if mean is not None and std is not None:
        s = (s - mean) / (std + 1e-6)
    return s[None, :]

def expert_pad_state(env_state: np.ndarray, target_dim: int) -> np.ndarray:
    """
    技能链路的状态处理，与 train_skill_expert.py 一致：**不做标准化**，只截断/零填充。
    返回 shape [target_dim]
    """
    s = np.zeros(target_dim, dtype=np.float32)
    n = min(target_dim, env_state.shape[0])
    s[:n] = env_state[:n]
    return s


# =========================
# Load Planning Head
# =========================
def load_planning_head(plan_ckpt, meta_json, device):
    meta = json.load(open(meta_json))
    K = int(meta["K"])
    d_model = int(meta["d_model"])            # = 1024 （你的训练）
    state_dim = int(meta.get("state_dim", 0))
    max_len = int(meta.get("max_len", 32))
    state_mean = np.array(meta["state_mean"], np.float32) if meta.get("state_mean") is not None else None
    state_std  = np.array(meta["state_std"],  np.float32) if meta.get("state_std")  is not None else None

    model = SkillFormer(K=K, d=d_model, d_state=state_dim, max_len=max_len).to(device)
    ckpt = torch.load(plan_ckpt, map_location="cpu")
    sd = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[PlanningHead] {plan_ckpt} | missing={len(missing)}, unexpected={len(unexpected)}")
    return model.eval(), meta, K, d_model, state_dim, state_mean, state_std


# =========================
# Load Skill Expert
# =========================
def load_skill_expert(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # 我们在训练时把 head 和部分 meta 存了
    if "head" in ckpt:         # 你训练脚本的存法
        head_sd = ckpt["head"]
        bb_cfg  = ckpt.get("backbone", {})
        head_cfg= ckpt.get("cfg", {})
        bb_d_out   = int(bb_cfg.get("d_out", 1024))
        d_hidden   = int(head_cfg.get("d_hidden", 1024))
        action_dim = int(head_cfg.get("action_dim", 32))
        use_gru    = bool(head_cfg.get("use_gru", True))
        skill_vocab= 256
        d_skill    = 64
        # 反推 Ds：d_in = 2*bb_d_out + Ds + d_skill
        d_in = 2*bb_d_out + d_skill + int(  # 暂占位，Ds 训练时来自数据，无法存；这里由 head_sd 形状推断更稳
            # 尝试从第一层线性权重反推输入维
            next(v for k,v in head_sd.items() if k.startswith("fuse.0.weight")).shape[1] - (2*bb_d_out + d_skill)
        )
    else:                       # 兼容其它保存格式
        head_sd = ckpt if isinstance(ckpt, dict) else ckpt.state_dict()
        # 尽可能地从权重形状推断
        fuse_w = next(v for k,v in head_sd.items() if k.endswith("fuse.0.weight"))
        d_in   = fuse_w.shape[1]
        # 猜测参数（不完美，但能跑）
        bb_d_out = 1024; d_hidden=1024; action_dim=32; use_gru=True; skill_vocab=256; d_skill=64

    head = SkillExpertHead(
        d_in=d_in, d_hidden=d_hidden, action_dim=action_dim,
        use_gru=use_gru, skill_vocab=skill_vocab, d_skill=d_skill, dropout=0.1
    ).to(device)
    missing, unexpected = head.load_state_dict(head_sd, strict=False)
    print(f"[SkillExpertHead] {ckpt_path} | missing={len(missing)}, unexpected={len(unexpected)}, d_in={d_in}")
    return head.eval(), bb_d_out, d_in, d_skill


# =========================
# Run one skill (streaming, keep history)
# =========================
@torch.no_grad()
def run_one_skill(env, head, act_bb, txt_ctx_expert, Ds_expert, skill_id, cfg):
    """
    - 连续收集序列（与训练形态一致），每步把到当前的整段序列送入（GRU有益）
    - 图像特征：每帧 mean-pool
    - 状态：截断/零填充到 Ds_expert（不做标准化）
    """
    frames, actions, p_dones = [], [], []
    img_feats = []   # 每帧 [d_backbone_out]
    st_feats  = []   # 每帧 [Ds_expert]

    for t in tqdm(range(cfg.max_steps), desc=f"Skill {skill_id}"):
        state = env.get_state()
        rgb   = env.render_rgb()

        # 视觉：patch-token -> mean pool -> [d]
        v_tokens = act_bb.encode_image(rgb)            # [Tpatch, d_act]
        v_frame  = v_tokens.mean(dim=0)                # [d_act]
        img_feats.append(v_frame)

        # 状态：pad/trunc，不标准化
        st_frame = torch.from_numpy(expert_pad_state(state, Ds_expert))
        st_feats.append(st_frame)

        # 堆叠成序列
        img_seq   = torch.stack(img_feats, dim=0)[None, ...].to(cfg.device)      # [1, T, d_act]
        state_seq = torch.stack(st_feats, dim=0)[None, ...].to(cfg.device)       # [1, T, Ds]
        txt_ctx   = txt_ctx_expert                                              # [1, Ttxt, d_act]

        out = head(
            txt_ctx=txt_ctx,
            img_seq=img_seq,
            state_seq=state_seq,
            skill_ids=torch.tensor([skill_id], device=cfg.device)
        )
        # 取最后一步
        a_last = out["action"][:, -1, :].cpu().numpy()[0]
        p_last = float(out["p_done"][:, -1].cpu().numpy()[0])

        env.step(np.clip(a_last[:8], -cfg.ctrl_clip, cfg.ctrl_clip))
        frames.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        actions.append(a_last.tolist())
        p_dones.append(p_last)

        if p_last > cfg.tau:
            print(f"  ↳ early stop t={t}, p_done={p_last:.3f}")
            break

    return dict(skill_id=skill_id, frames=frames, actions=actions, p_dones=p_dones)


# =========================
# Main
# =========================
@torch.no_grad()
def main():
    cfg = Config()
    env = MujocoEnv(cfg.xml_path, cfg.width, cfg.height)
    env.reset()

    # ---- 规划模型 + 元信息 ----
    plan_head, meta, K, d_plan, state_dim, state_mean, state_std = load_planning_head(
        cfg.plan_ckpt, cfg.meta_json, cfg.device
    )
    trans_mask = torch.from_numpy((np.load(cfg.trans_npy) > 0)).to(cfg.device)

    # ---- 技能头 + 其所需的骨干输出维 ----
    skill_head, d_act, d_in_head, d_skill = load_skill_expert(cfg.skill_ckpt, cfg.device)

    # ---- 两条骨干：规划用 1024维；技能用 1024维（与你训练一致）----
    plan_bb = Pi0Backbone(
        d_out=d_plan, device=cfg.device,
        hf_path_or_name=cfg.tokenizer_dir, pi0_ckpt_dir=cfg.pi0_ckpt_path, precision=cfg.precision
    )
    act_bb = Pi0Backbone(
        d_out=d_act, device=cfg.device,
        hf_path_or_name=cfg.tokenizer_dir, pi0_ckpt_dir=cfg.pi0_ckpt_path, precision=cfg.precision
    )

    # ---- 初始观测 ----
    img0 = env.render_rgb()
    st0  = env.get_state()

    # 规划链路特征（全部是 d_plan=1024）
    H_txt0_plan = plan_bb.encode_text(cfg.instruction).unsqueeze(0).float().to(cfg.device)   # [1, Ttxt, 1024]
    H_v0_plan   = plan_bb.encode_image(img0).unsqueeze(0).float().to(cfg.device)             # [1, Tvis, 1024]
    S0_plan     = torch.from_numpy(plan_preprocess_state(st0, state_dim, state_mean, state_std)).float().to(cfg.device)  # [1, state_dim]
    print(f"[Planner feats] H_txt={H_txt0_plan.shape}, H_vs={H_v0_plan.shape}, S={S0_plan.shape}, d={d_plan}")

    # ---- 规划解码（与训练完全一致的约束贪心）----
    skill_seqs = greedy_with_constraints(
        plan_head, H_txt0_plan, H_v0_plan, S0_plan, K, trans_mask, max_len=meta.get("max_len", 32)
    )
    skill_seq = skill_seqs[0]
    print(f"[Planning] Pred skill seq: {skill_seq}")

    # ---- 技能链路文本上下文（d_act 维；与训练一致）----
    H_txt_expert = act_bb.encode_text(cfg.instruction)
    if H_txt_expert.size(0) > 32:
        H_txt_expert = H_txt_expert[:32]
    H_txt_expert = H_txt_expert.unsqueeze(0).float().to(cfg.device)  # [1, ≤8, d_act]

    # 反推出 expert 训练时的 Ds： d_in = 2*d_act + Ds + d_skill
    Ds_expert = int(d_in_head - (2*d_act + d_skill))
    if Ds_expert <= 0:
        raise RuntimeError(f"Invalid Ds_expert={Ds_expert} from d_in={d_in_head}, d_act={d_act}, d_skill={d_skill}")
    print(f"[Expert dims] d_act={d_act}, d_in_head={d_in_head}, d_skill={d_skill}, Ds_expert={Ds_expert}")

    # ---- 顺序执行技能 ----
    all_results = []
    for sid in skill_seq:
        res = run_one_skill(env, skill_head, act_bb, H_txt_expert, Ds_expert, sid, cfg)
        all_results.append(res)

    # ---- 保存结果 ----
    summary = [{"skill_id": r["skill_id"], "steps": len(r["actions"]),
                "p_done_last": r["p_dones"][-1] if r["p_dones"] else 0.0} for r in all_results]
    os.makedirs(os.path.dirname(cfg.result_json) or ".", exist_ok=True)
    with open(cfg.result_json, "w") as f:
        json.dump({"summary": summary, "skill_seq": skill_seq}, f, indent=2)
    print(f"[Save] JSON -> {cfg.result_json}")

    # ---- 保存视频 ----
    if cfg.video_out and all_results and all_results[0]["frames"]:
        h, w, _ = all_results[0]["frames"][0].shape
        vw = cv2.VideoWriter(cfg.video_out, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
        for r in all_results:
            for fr in r["frames"]:
                vw.write(fr)
        vw.release()
        print(f"[Video] Saved -> {cfg.video_out}")


if __name__ == "__main__":
    main()
