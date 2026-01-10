#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pi0_mujoco.py (final Pi0 only)
----------------------------------
只支持 Pi0 模型推理。
使用 Pi0 Backbone 和 Pi0 模型进行动作预测。

Author: llsong
"""

import os, cv2, torch
import numpy as np
import mujoco
from tqdm import tqdm
from torch import nn
from transformers import AutoTokenizer

# 引入 Pi0 模型
import sys

sys.path.append('/home/lulsong/WorkTask/chain-of-action/openpi/src')
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


# =====================================================
# =============== CONFIG ===============================
# =====================================================
class Config:
    # 模式选择
    use_pi0_direct = True  # True: Pi0端到端动作
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctrl_clip = 1.0
    env_nu = 8
    max_steps = 60

    # === 模型路径 ===
    pi0_ckpt_path = "/home/lulsong/Downloads/pi0_droid_pytorch"
    tokenizer_dir = "/home/lulsong/Downloads/paligemma-3b-pt-224"

    # === 特征维度 ===
    d_backbone = 1024
    action_dim = 32

    # === 任务 ===
    instruction = "pick up the red cube"
    tau = 0.9
    xml_path = "/home/lulsong/WorkTask/chain-of-action/sim/panda.xml"
    out_video = "coa_pi0_test.mp4"


# =====================================================
# =============== ENV WRAPPER =========================
# =====================================================
import mujoco

class MujocoEnv:
    def __init__(self, xml_path, width=640, height=480):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.nu = self.model.nu
        print(f"[Env] Loaded model: {xml_path}, nu={self.nu}")

    def step(self, ctrl):
        if ctrl.shape[0] != self.nu:
            raise ValueError(f"ctrl shape mismatch: {ctrl.shape[0]} vs {self.nu}")
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data)

    def render_rgb(self, camera_name=None):
        if camera_name is not None:
            # 获取摄像头的 ID
            camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            if camera_id == -1:
                raise ValueError(f"Camera '{camera_name}' not found.")
            # 渲染指定摄像头视角的图像
            self.renderer.update_scene(self.data, camera=camera_id)
            return self.renderer.render()[:, :, :3]
        else:
            # Default rendering
            self.renderer.update_scene(self.data)
            return self.renderer.render()[:, :, :3]

    def get_state(self):
        return np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()])

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)




# =====================================================
# =============== UTIL FUNCTIONS ======================
# =====================================================
def save_video(frames_external, frames_wrist, path, fps=30):
    if not frames_external or not frames_wrist:
        print("No frames to save!")
        return

    h, w, _ = frames_external[0].shape
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * 2, h))

    # Combine frames from both cameras into a single frame
    for fr_external, fr_wrist in zip(frames_external, frames_wrist):
        # Horizontally concatenate both frames (external and wrist)
        combined_frame = np.hstack((fr_external, fr_wrist))
        vw.write(combined_frame)

    vw.release()
    print(f"[Video] Saved -> {path}")


# =====================================================
# =============== LOADERS ==============================
# =====================================================
import safetensors.torch


def load_pi0(cfg):
    class DummyCfg:
        paligemma_variant = "gemma_2b"
        action_expert_variant = "gemma_300m"
        dtype = "bfloat16"
        pi05 = False
        action_dim = 32
        action_horizon = 10

    # Safetensors格式路径，假设权重文件存放在 `cfg.pi0_ckpt_path` 目录下
    pi0_ckpt_path = cfg.pi0_ckpt_path  # The path to the directory containing the safetensors files

    # Initialize the Pi0 model using the dummy configuration
    pi0_model = PI0Pytorch(DummyCfg()).to(cfg.device).eval()
    pi0_ckpt_path = os.path.join(cfg.pi0_ckpt_path, "model.safetensors")

    # Safetensors的加载方法
    # safetensors 加载权重
    checkpoint = safetensors.torch.load_file(pi0_ckpt_path, device=cfg.device)

    # 加载权重到模型
    pi0_model.load_state_dict(checkpoint, strict=False)
    print(f"[Load] PI0 model weights from {pi0_ckpt_path}")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_dir, local_files_only=True)

    return pi0_model, tokenizer


# =====================================================
# =============== PREPROCESS OBSERVATION ===============
# =====================================================
class SimpleObservation:
    def __init__(self, images, image_masks, tokenized_prompt, tokenized_prompt_mask, state):
        self.images = images
        self.image_masks = image_masks
        self.tokenized_prompt = tokenized_prompt
        self.tokenized_prompt_mask = tokenized_prompt_mask
        self.state = state


def preprocess_observation(env, tokenizer, instruction, device):
    # 获取外部视角和腕部视角图像
    rgb_external = env.render_rgb(camera_name=None)  # 外部视角摄像头名称
    rgb_wrist = env.render_rgb(camera_name='wrist1')  # 腕部视角摄像头名称

    # 生成全零的图像用于填充外部视角和其他视角
    rgb_zero = np.zeros_like(rgb_wrist)  # 使用与腕部相机相同尺寸的全零图像

    # 处理腕部视角图像
    img_wrist = torch.from_numpy(rgb_wrist.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

    # 处理零值图像（外部视角和其他视角）
    img_zero = torch.from_numpy(rgb_zero.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

    # 获取模型的状态
    state = torch.from_numpy(env.get_state()).unsqueeze(0).to(device)

    # 创建掩码
    batch_shape = state.shape[:-1]
    img_mask = torch.ones(1, 3, rgb_wrist.shape[0], rgb_wrist.shape[1], device=device, dtype=torch.bool)

    # 对于零值图像的掩码，设置为 False
    img_mask_zero = torch.zeros_like(img_mask, dtype=torch.bool)

    # 令外部视角和其他视角的掩码为 False，表示它们无效
    image_masks = {
        "base_0_rg": img_mask_zero,  # 外部视角掩码
        "left_wrist_0_rg": img_mask,  # 腕部视角掩码
        "right_wrist_0_rg": img_mask_zero,  # 右手腕视角掩码
    }

    # 用零值图像填充其他视角
    images = {
        "base_0_rgb": img_zero,  # 外部视角图像
        "left_wrist_0_rgb": img_wrist,  # 腕部视角图像
        "right_wrist_0_rgb": img_zero,  # 右手腕视角图像
    }

    # 处理文本输入
    tok = tokenizer(instruction, return_tensors="pt", add_special_tokens=True).to(device)

    return SimpleObservation(
        images=images,
        image_masks=image_masks,
        tokenized_prompt=tok["input_ids"],
        tokenized_prompt_mask=tok["attention_mask"],
        state=state,
    )


# =====================================================
# =============== MODE: PI0 DIRECT ====================
# =====================================================
@torch.no_grad()
def run_pi0_direct(env, pi0_model, tokenizer, cfg):
    print("[Mode] Pi0 Direct end-to-end inference")
    frames_external = []
    frames_wrist = []

    # # 获取并处理图像
    obs = preprocess_observation(env, tokenizer, cfg.instruction, cfg.device)

    # 使用 Pi0 模型生成动作
    acts = pi0_model.sample_actions(cfg.device, obs, num_steps=10)  # [1, horizon, 32]

    print(f"[Pi0] Actions shape: {acts.shape}")

    for t in tqdm(range(min(cfg.max_steps, acts.shape[1]))):
        a = acts[0, t, :cfg.env_nu].cpu().numpy()
        env.step(np.clip(a, -cfg.ctrl_clip, cfg.ctrl_clip))

        # 获取外部视角和腕部视角
        rgb_external = env.render_rgb(camera_name='camera_name_external')
        rgb_wrist = env.render_rgb(camera_name='wrist1')

        frames_external.append(cv2.cvtColor(rgb_external, cv2.COLOR_RGB2BGR))
        frames_wrist.append(cv2.cvtColor(rgb_wrist, cv2.COLOR_RGB2BGR))

    return frames_external, frames_wrist


# =====================================================
# =================== MAIN =============================
# =====================================================
def main():
    cfg = Config()
    env = MujocoEnv(cfg.xml_path)
    env.reset()
    ctrl = np.zeros(env.nu)  # 或者使用合理的控制信号
    env.step(ctrl)

    pi0_model, tokenizer = load_pi0(cfg)

    frames_external, frames_wrist = run_pi0_direct(env, pi0_model, tokenizer, cfg)

    # 保存视频，外部视角和腕部视角都保存
    save_video(frames_external, frames_wrist, cfg.out_video)


if __name__ == "__main__":
    main()
