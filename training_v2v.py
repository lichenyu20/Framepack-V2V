"""
FramePack v2v training script (flow-matching).

This script fine-tunes the packed HunyuanVideo transformer on next-frame-section prediction
using *video-to-video* style conditioning (history frames -> future frames).

Dataset format (JSONL):
Each line is a JSON object with at least:
  - "video": path to an mp4 (or any format decord can read)
  - "prompt": text prompt for the clip (string; can be empty)

Example:
{"video": "/data/clips/0001.mp4", "prompt": "The man dances energetically, leaping mid-air."}
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import decord
from accelerate import Accelerator
from accelerate.utils import set_seed

from diffusers import AutoencoderKLHunyuanVideo
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    LlamaModel,
    LlamaTokenizerFast,
    SiglipImageProcessor,
    SiglipVisionModel,
)

from diffusers_helper.bucket_tools import find_nearest_bucket
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import calculate_flux_mu, flux_time_shift
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.utils import crop_or_pad_yield_mask, resize_and_center_crop

try:
    from safetensors.torch import save_file as safetensors_save_file
except Exception:
    safetensors_save_file = None


# -------------------------
# LoRA (no external deps)
# -------------------------


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear expects nn.Linear")
        if r <= 0:
            raise ValueError("LoRA rank r must be > 0")

        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()

        in_features = base.in_features
        out_features = base.out_features

        # A: down projection, B: up projection
        self.lora_A = nn.Linear(in_features, self.r, bias=False)
        self.lora_B = nn.Linear(self.r, out_features, bias=False)

        # init: A random, B zero (common LoRA init)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


def _set_module_by_name(root: nn.Module, name: str, new_module: nn.Module) -> None:
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def inject_lora(
    model: nn.Module,
    *,
    r: int,
    alpha: float,
    dropout: float,
    target_keywords: List[str],
) -> int:
    """
    Replace selected nn.Linear layers with LoRALinear.
    target_keywords: if any keyword is a substring of the module's full name -> inject.
    Returns number of injected modules.
    """
    injected = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(k in name for k in target_keywords):
            continue
        # avoid double wrapping
        if isinstance(module, LoRALinear):
            continue
        _set_module_by_name(model, name, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
        injected += 1
    return injected


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    sd: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            sd[f"{name}.lora_A.weight"] = module.lora_A.weight.detach().cpu()
            sd[f"{name}.lora_B.weight"] = module.lora_B.weight.detach().cpu()
    return sd


# -------------------------
# Dataset / preprocessing
# -------------------------


def _read_jsonl(path: str) -> List[dict]:
    items: List[dict] = []
    with open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _frames_to_bcthw(frames_np: np.ndarray) -> torch.Tensor:
    """
    frames_np: [T, H, W, 3], uint8 RGB
    return: [1, 3, T, H, W], float32 in [-1, 1]
    """
    x = torch.from_numpy(frames_np).float() / 127.5 - 1.0
    x = x.permute(3, 0, 1, 2)[None]  # [1, 3, T, H, W]
    return x


def _tail_sample_indices(n_total: int, n_take: int, tail_ratio: float = 0.8) -> np.ndarray:
    if n_total <= 0:
        raise ValueError("n_total must be > 0")
    if n_take <= 0:
        raise ValueError("n_take must be > 0")
    if n_total == 1:
        return np.zeros((n_take,), dtype=np.int64)

    start = int(max(0, math.floor((n_total - 1) * (1.0 - tail_ratio))))
    candidate = np.arange(start, n_total, dtype=np.int64)
    if len(candidate) == 1:
        return np.repeat(candidate, n_take)

    idx = np.linspace(0, len(candidate) - 1, n_take).round().astype(np.int64)
    return candidate[idx]


def _make_condition_sets_from_history(
    history_frames: np.ndarray,
    target_width: int,
    target_height: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      processed_history_frames: [T, H, W, 3]
      frames_pre:  [1, H, W, 3]   (-> latent T ~ 1)
      frames_post: [1, H, W, 3]   (-> latent T ~ 1)
      frames_2x:   [8, H, W, 3]   (-> latent T ~ 2)
      frames_4x:   [64, H, W, 3]  (-> latent T ~ 16)
    """
    processed = np.stack(
        [resize_and_center_crop(f, target_width=target_width, target_height=target_height) for f in history_frames],
        axis=0,
    )
    T = int(processed.shape[0])
    if T <= 0:
        raise ValueError("History video has no frames")

    idx_pre = np.array([T - 1], dtype=np.int64)
    idx_post = np.array([max(0, T - 2)], dtype=np.int64)
    idx_2x = _tail_sample_indices(T, 8, tail_ratio=0.40)
    idx_4x = _tail_sample_indices(T, 64, tail_ratio=0.90)

    frames_pre = processed[idx_pre]
    frames_post = processed[idx_post]
    frames_2x = processed[idx_2x]
    frames_4x = processed[idx_4x]
    return processed, frames_pre, frames_post, frames_2x, frames_4x


@dataclass
class TrainBatch:
    # diffusion variable and target
    x_noisy: torch.Tensor  # [B, 16, T, h, w] float32
    v_target: torch.Tensor  # [B, 16, T, h, w] float32
    sigma: torch.Tensor  # [B] float32 in (0,1]

    # text conditioning
    llama_vec: torch.Tensor
    llama_mask: torch.Tensor
    clip_pooler: torch.Tensor

    # v2v conditioning
    latent_indices: torch.Tensor
    clean_latents: torch.Tensor
    clean_latent_indices: torch.Tensor
    clean_latents_2x: torch.Tensor
    clean_latent_2x_indices: torch.Tensor
    clean_latents_4x: torch.Tensor
    clean_latent_4x_indices: torch.Tensor
    image_embeddings: Optional[torch.Tensor]
    latent_padding_size: torch.Tensor  # [B] int64 (debug/analysis)


class JsonlVideoDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        resolution: int,
        latent_window_size: int,
        train_future_sections: int,
        min_history_frames: int = 72,
        max_video_frames: Optional[int] = None,
        allow_short_videos: bool = True,
    ) -> None:
        self.items = _read_jsonl(jsonl_path)
        if len(self.items) == 0:
            raise ValueError(f"Dataset is empty: {jsonl_path}")
        self.resolution = int(resolution)
        self.latent_window_size = int(latent_window_size)
        self.train_future_sections = int(train_future_sections)
        if self.train_future_sections <= 0:
            raise ValueError("train_future_sections must be > 0")
        self.min_history_frames = int(min_history_frames)
        self.max_video_frames = None if max_video_frames is None else int(max_video_frames)
        self.allow_short_videos = bool(allow_short_videos)

    def __len__(self) -> int:
        return len(self.items)

    def _load_video(self, path: str) -> np.ndarray:
        vr = decord.VideoReader(str(Path(path).expanduser().resolve()))
        n = len(vr)
        if self.max_video_frames is not None:
            n = min(n, self.max_video_frames)
        frames = vr.get_batch(range(n)).asnumpy()  # (T, H, W, C) RGB
        return frames

    def __getitem__(self, idx: int) -> Dict[str, object]:
        item = self.items[idx]
        video_path = item.get("video")
        prompt = item.get("prompt", "")
        if not isinstance(video_path, str):
            raise ValueError(f"Bad item at idx={idx}: missing string 'video'")
        if not isinstance(prompt, str):
            prompt = str(prompt)

        frames = self._load_video(video_path)
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"Unexpected frames shape for {video_path}: {frames.shape}")

        T, H, W, _ = frames.shape
        target_height, target_width = find_nearest_bucket(H, W, resolution=self.resolution)

        # We need enough frames:
        # - history: at least ~64 frames so 4x context can be built
        # - future: latent_window_size*4*train_future_sections frames so future latents cover multiple sections
        need_target_frames = self.latent_window_size * 4 * self.train_future_sections

        if T < 2:
            raise ValueError(f"Video too short to use: {video_path} has {T} frames")

        # If too short, we still train by padding/repeating frames.
        if (T < self.min_history_frames + need_target_frames + 2) and (not self.allow_short_videos):
            raise ValueError(
                f"Video too short: {video_path} has {T} frames, need >= {self.min_history_frames + need_target_frames + 2}"
            )

        # Pick a split point such that:
        # history_frames = [0:split)
        # future_frames  = [split:split+need_target_frames) (will be padded if needed)
        # split should be >= min_history_frames when possible, otherwise use a smaller split.
        max_split = max(1, T - 1)
        min_split = min(self.min_history_frames, max_split)
        if min_split >= max_split:
            split = max_split
        else:
            split = random.randint(min_split, max_split)

        history_frames = frames[:split]
        future_frames = frames[split:split + need_target_frames]

        # Pad history/future by repeating last frame if needed
        if history_frames.shape[0] < max(1, self.min_history_frames):
            pad_n = max(1, self.min_history_frames) - history_frames.shape[0]
            pad = np.repeat(history_frames[-1:], pad_n, axis=0)
            history_frames = np.concatenate([history_frames, pad], axis=0)

        if future_frames.shape[0] < need_target_frames:
            pad_n = need_target_frames - future_frames.shape[0]
            # If split is at end, repeat last available frame from the whole video
            last = future_frames[-1:] if future_frames.shape[0] > 0 else frames[-1:]
            pad = np.repeat(last, pad_n, axis=0)
            future_frames = np.concatenate([future_frames, pad], axis=0)

        # Resize/crop both history and target to bucket
        history_frames = np.stack(
            [resize_and_center_crop(f, target_width=target_width, target_height=target_height) for f in history_frames],
            axis=0,
        )
        future_frames = np.stack(
            [resize_and_center_crop(f, target_width=target_width, target_height=target_height) for f in future_frames],
            axis=0,
        )

        return {
            "prompt": prompt,
            "history_frames": history_frames,  # [Th, H, W, 3]
            "future_frames": future_frames,  # [Tf, H, W, 3]
            "bucket_hw": (target_height, target_width),
            "video": video_path,
        }


def _collate_identity(batch: List[Dict[str, object]]) -> List[Dict[str, object]]:
    # Videos are variable length at source; we sample fixed-size windows in __getitem__,
    # so a simple list collate is fine and avoids large CPU tensors being concatenated prematurely.
    return batch


# -------------------------
# Training helpers
# -------------------------


def _build_indices(latent_window_size: int, latent_padding_size: int = 0) -> Dict[str, torch.Tensor]:
    """
    Match the inference indexing scheme (see `v2v_cli.py`).

    We build a 1D index sequence and split into:
      pre(1), blank(padding), latent(window), post(1), 2x(2), 4x(16)
    """
    total = 1 + latent_padding_size + latent_window_size + 1 + 2 + 16
    indices = torch.arange(0, total).unsqueeze(0)  # [1, total]
    (
        clean_latent_indices_pre,
        _blank_indices,
        latent_indices,
        clean_latent_indices_post,
        clean_latent_2x_indices,
        clean_latent_4x_indices,
    ) = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)

    clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)
    return {
        "latent_indices": latent_indices,
        "clean_latent_indices": clean_latent_indices,
        "clean_latent_2x_indices": clean_latent_2x_indices,
        "clean_latent_4x_indices": clean_latent_4x_indices,
    }


def _pad_time_repeat_last(x: torch.Tensor, target_t: int) -> torch.Tensor:
    """
    Pads along time dimension (dim=2) by repeating last frame.
    Expects x shape [B, C, T, H, W] or [B, T] for indices (handled separately).
    """
    if x.dim() == 2:
        # indices: [B, T]
        t = int(x.shape[1])
        if t >= target_t:
            return x[:, :target_t]
        pad = x[:, -1:].repeat(1, target_t - t)
        return torch.cat([x, pad], dim=1)

    t = int(x.shape[2])
    if t >= target_t:
        return x[:, :, :target_t]
    pad = x[:, :, -1:, :, :].repeat(1, 1, target_t - t, 1, 1)
    return torch.cat([x, pad], dim=2)


@torch.no_grad()
def _encode_conditions_and_target(
    batch_items: List[Dict[str, object]],
    *,
    vae: AutoencoderKLHunyuanVideo,
    feature_extractor: SiglipImageProcessor,
    image_encoder: SiglipVisionModel,
    text_encoder: LlamaModel,
    text_encoder_2: CLIPTextModel,
    tokenizer: LlamaTokenizerFast,
    tokenizer_2: CLIPTokenizer,
    device: torch.device,
    use_image_conditioning: bool,
    prompt_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    latent_window_size: int,
    distilled_guidance_scale: float,
    num_clean_frames: int,
    train_future_sections: int,
) -> TrainBatch:
    B = len(batch_items)
    if B <= 0:
        raise ValueError("Empty batch")

    # Encode prompts (cache by exact string)
    llama_vec_list = []
    llama_mask_list = []
    clip_pooler_list = []
    for it in batch_items:
        prompt = str(it["prompt"])
        cached = prompt_cache.get(prompt)
        if cached is None:
            llama_vec, clip_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
            llama_vec, llama_mask = crop_or_pad_yield_mask(llama_vec, length=512)
            prompt_cache[prompt] = (llama_vec.cpu(), llama_mask.cpu(), clip_pooler.cpu())
            cached = prompt_cache[prompt]
        llama_vec_list.append(cached[0])
        llama_mask_list.append(cached[1])
        clip_pooler_list.append(cached[2])

    llama_vec = torch.cat(llama_vec_list, dim=0).to(device)
    llama_mask = torch.cat(llama_mask_list, dim=0).to(device)
    clip_pooler = torch.cat(clip_pooler_list, dim=0).to(device)

    # Encode conditioning latents and target latents via VAE
    vae = vae.to(device)

    x0_list: List[torch.Tensor] = []
    clean_latents_list: List[torch.Tensor] = []
    clean_2x_list: List[torch.Tensor] = []
    clean_4x_list: List[torch.Tensor] = []
    latent_indices_list: List[torch.Tensor] = []
    clean_latent_indices_list: List[torch.Tensor] = []
    clean_2x_indices_list: List[torch.Tensor] = []
    clean_4x_indices_list: List[torch.Tensor] = []
    latent_padding_size_list: List[int] = []
    image_emb_list: List[Optional[torch.Tensor]] = []

    for it in batch_items:
        history_frames = np.asarray(it["history_frames"], dtype=np.uint8)
        future_frames = np.asarray(it["future_frames"], dtype=np.uint8)

        # Encode history + future to latents
        # history latents act like `video_latents` in inference_v2v.py (past)
        # future latents provide teacher-forced "already generated" content in `history_latents`.
        video_lat = vae_encode(_frames_to_bcthw(history_frames).to(device), vae)  # [1,16,Th_lat,h,w]
        future_lat = vae_encode(_frames_to_bcthw(future_frames).to(device), vae)  # [1,16,Tf_lat,h,w]

        # enforce expected future latent length = latent_window_size * train_future_sections
        expected_future_lat = int(latent_window_size) * int(train_future_sections)
        if future_lat.shape[2] < expected_future_lat:
            rep = future_lat[:, :, -1:].repeat(1, 1, expected_future_lat - future_lat.shape[2], 1, 1)
            future_lat = torch.cat([future_lat, rep], dim=2)
        elif future_lat.shape[2] > expected_future_lat:
            future_lat = future_lat[:, :, :expected_future_lat]

        # Pick a random section (like latent_padding schedule in inference_v2v.py)
        total_latent_sections = int(train_future_sections)
        latent_paddings: List[int] = list(reversed(range(total_latent_sections)))
        if total_latent_sections > 4:
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]
        latent_padding = random.choice(latent_paddings)
        section_start = int(latent_padding) * int(latent_window_size)
        section_start = min(section_start, max(0, future_lat.shape[2] - int(latent_window_size)))
        latent_padding_size = int(latent_padding) * int(latent_window_size)
        latent_padding_size_list.append(latent_padding_size)

        # x0 target is the latent window for this section
        x0 = future_lat[:, :, section_start:section_start + int(latent_window_size)]

        # Build teacher-forced "history_latents" the same way inference uses:
        # [already_generated_future, past_video_latents]
        # Here, "already_generated_future" is future from *this section onward*.
        history_lat = torch.cat([future_lat[:, :, section_start:], video_lat], dim=2).detach()

        # Dynamic context allocation and clean/context extraction (copy inference_v2v.py logic)
        available_frames = history_lat.shape[2]

        effective_clean_frames = max(0, int(num_clean_frames) - 1) if int(num_clean_frames) > 1 else 1
        clean_latent_pre_frames = effective_clean_frames

        if available_frames > clean_latent_pre_frames + 1:
            num_2x_frames = min(2, max(1, available_frames - clean_latent_pre_frames - 1))
        else:
            num_2x_frames = 1

        if available_frames > clean_latent_pre_frames + num_2x_frames:
            num_4x_frames = min(16, max(1, available_frames - clean_latent_pre_frames - num_2x_frames))
        else:
            num_4x_frames = 1

        total_context_frames = num_2x_frames + num_4x_frames
        total_context_frames = min(total_context_frames, available_frames - clean_latent_pre_frames)
        post_frames = effective_clean_frames

        # Indices (match inference_v2v.py exactly for RoPE positions)
        total_index_length = (
            clean_latent_pre_frames
            + latent_padding_size
            + int(latent_window_size)
            + post_frames
            + num_2x_frames
            + num_4x_frames
        )
        indices = torch.arange(0, total_index_length, device=device).unsqueeze(0)  # [1, L]
        (
            clean_latent_indices_pre,
            _blank_indices,
            latent_indices,
            clean_latent_indices_post,
            clean_latent_2x_indices,
            clean_latent_4x_indices,
        ) = indices.split(
            [
                clean_latent_pre_frames,
                latent_padding_size,
                int(latent_window_size),
                post_frames,
                num_2x_frames,
                num_4x_frames,
            ],
            dim=1,
        )
        clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

        # clean pre: tail of past video_lat (closest to boundary)
        clean_latents_pre = video_lat[:, :, -min(effective_clean_frames, video_lat.shape[2]):]
        if clean_latents_pre.shape[2] < clean_latent_pre_frames:
            repeat_factor = math.ceil(clean_latent_pre_frames / clean_latents_pre.shape[2])
            clean_latents_pre = clean_latents_pre.repeat(1, 1, repeat_factor, 1, 1)[:, :, :clean_latent_pre_frames]

        # clean post: head of teacher-forced future/history_lat
        clean_latents_post = history_lat[:, :, :min(effective_clean_frames, history_lat.shape[2]), :, :]
        if clean_latents_post.shape[2] < post_frames:
            repeat_factor = math.ceil(post_frames / clean_latents_post.shape[2])
            clean_latents_post = clean_latents_post.repeat(1, 1, repeat_factor, 1, 1)[:, :, :post_frames]

        clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)  # [1,16,pre+post,...]

        # context frames from teacher-forced history_lat
        context_frames = (
            history_lat[:, :, -(total_context_frames + clean_latent_pre_frames):-clean_latent_pre_frames, :, :]
            if total_context_frames > 0
            else history_lat[:, :, :1, :, :]
        )

        split_sizes = [num_4x_frames, num_2x_frames]
        split_sizes = [s for s in split_sizes if s > 0]

        if split_sizes and context_frames.shape[2] >= sum(split_sizes):
            splits = context_frames.split(split_sizes, dim=2)
            split_idx = 0
            if num_4x_frames > 0:
                lat_4x = splits[split_idx]
                split_idx += 1
            else:
                lat_4x = history_lat[:, :, :1, :, :]

            if num_2x_frames > 0 and split_idx < len(splits):
                lat_2x = splits[split_idx]
            else:
                lat_2x = history_lat[:, :, :1, :, :]
        else:
            lat_4x = history_lat[:, :, :1, :, :]
            lat_2x = history_lat[:, :, :1, :, :]

        # Ensure shapes match indices temporal lengths (in case context alloc returns shorter)
        clean_latents = _pad_time_repeat_last(clean_latents, int(clean_latent_indices.shape[1]))
        lat_2x = _pad_time_repeat_last(lat_2x, int(clean_latent_2x_indices.shape[1]))
        lat_4x = _pad_time_repeat_last(lat_4x, int(clean_latent_4x_indices.shape[1]))

        x0_list.append(x0)
        clean_latents_list.append(clean_latents)
        clean_2x_list.append(lat_2x)
        clean_4x_list.append(lat_4x)
        latent_indices_list.append(latent_indices.detach().cpu())
        clean_latent_indices_list.append(clean_latent_indices.detach().cpu())
        clean_2x_indices_list.append(clean_latent_2x_indices.detach().cpu())
        clean_4x_indices_list.append(clean_latent_4x_indices.detach().cpu())

        if use_image_conditioning:
            last_frame = history_frames[-1]
            image_encoder_output = hf_clip_vision_encode(last_frame, feature_extractor, image_encoder)
            image_emb_list.append(image_encoder_output.last_hidden_state)
        else:
            image_emb_list.append(None)

    x0 = torch.cat(x0_list, dim=0).to(dtype=torch.float32)  # train in fp32 for targets
    # Pad variable-length conditioning across batch (repeat-last) so batch_size > 1 can work.
    max_clean_t = max(int(t.shape[2]) for t in clean_latents_list)
    max_2x_t = max(int(t.shape[2]) for t in clean_2x_list)
    max_4x_t = max(int(t.shape[2]) for t in clean_4x_list)
    max_clean_idx = max(int(t.shape[1]) for t in clean_latent_indices_list)
    max_2x_idx = max(int(t.shape[1]) for t in clean_2x_indices_list)
    max_4x_idx = max(int(t.shape[1]) for t in clean_4x_indices_list)

    clean_latents = torch.cat([_pad_time_repeat_last(t, max_clean_t) for t in clean_latents_list], dim=0).to(dtype=torch.float32)
    clean_latents_2x = torch.cat([_pad_time_repeat_last(t, max_2x_t) for t in clean_2x_list], dim=0).to(dtype=torch.float32)
    clean_latents_4x = torch.cat([_pad_time_repeat_last(t, max_4x_t) for t in clean_4x_list], dim=0).to(dtype=torch.float32)

    latent_indices = torch.cat([_pad_time_repeat_last(t, int(latent_window_size)) for t in latent_indices_list], dim=0).to(device)
    clean_latent_indices = torch.cat([_pad_time_repeat_last(t, max_clean_idx) for t in clean_latent_indices_list], dim=0).to(device)
    clean_latent_2x_indices = torch.cat([_pad_time_repeat_last(t, max_2x_idx) for t in clean_2x_indices_list], dim=0).to(device)
    clean_latent_4x_indices = torch.cat([_pad_time_repeat_last(t, max_4x_idx) for t in clean_4x_indices_list], dim=0).to(device)

    image_embeddings = None
    if use_image_conditioning:
        image_embeddings = torch.cat([x for x in image_emb_list if x is not None], dim=0).to(device)

    # Sample sigma with the same "flux time shift" family used in inference.
    # The shift depends on context length; approximate it with latent sequence length (T*h*w/4).
    _, _, T_lat, h_lat, w_lat = x0.shape
    seq_len = int(T_lat * h_lat * w_lat // 4)
    mu = calculate_flux_mu(seq_len, exp_max=7.0)
    u = torch.rand((B,), device=device).clamp_(1e-4, 1.0)  # avoid exactly 0
    sigma = torch.tensor([flux_time_shift(float(t), mu=mu) for t in u], device=device, dtype=torch.float32)

    noise = torch.randn_like(x0)
    x_noisy = x0 * (1.0 - sigma.view(B, 1, 1, 1, 1)) + noise * sigma.view(B, 1, 1, 1, 1)
    v_target = (x_noisy - x0) / sigma.view(B, 1, 1, 1, 1)  # == noise - x0 (for this mixing)

    return TrainBatch(
        x_noisy=x_noisy,
        v_target=v_target,
        sigma=sigma,
        llama_vec=llama_vec,
        llama_mask=llama_mask,
        clip_pooler=clip_pooler,
        latent_indices=latent_indices,
        clean_latents=clean_latents,
        clean_latent_indices=clean_latent_indices,
        clean_latents_2x=clean_latents_2x,
        clean_latent_2x_indices=clean_latent_2x_indices,
        clean_latents_4x=clean_latents_4x,
        clean_latent_4x_indices=clean_latent_4x_indices,
        image_embeddings=image_embeddings,
        latent_padding_size=torch.tensor(latent_padding_size_list, dtype=torch.int64, device=device),
    )


def _save_checkpoint(
    output_dir: str,
    step: int,
    *,
    accelerator: Accelerator,
    transformer: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    extra: Optional[dict] = None,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Model weights
    model_path = out / f"transformer_step_{step}.safetensors"
    state_dict = accelerator.get_state_dict(transformer)
    if safetensors_save_file is not None:
        safetensors_save_file(state_dict, str(model_path))
    else:
        torch.save(state_dict, str(model_path.with_suffix(".pt")))

    # Optimizer + accelerator state (for resume)
    accel_state_dir = out / f"accelerate_state_step_{step}"
    accelerator.save_state(str(accel_state_dir))

    # Metadata
    meta = {
        "step": int(step),
        "time": time.time(),
    }
    if extra:
        meta.update(extra)
    with open(out / f"meta_step_{step}.json", "wt", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _save_lora_checkpoint(output_dir: str, step: int, *, accelerator: Accelerator, transformer: torch.nn.Module) -> None:
    if safetensors_save_file is None:
        return
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = accelerator.unwrap_model(transformer)
    sd = lora_state_dict(model)
    if len(sd) == 0:
        return
    safetensors_save_file(sd, str(out / f"lora_step_{step}.safetensors"))


# -------------------------
# Main
# -------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="FramePack v2v training (flow-matching)")

    # data
    parser.add_argument("--dataset_jsonl", type=str, required=True, help="Path to JSONL dataset")
    parser.add_argument("--resolution", type=int, default=640, help="Max bucket side (same as inference)")
    parser.add_argument("--latent_window_size", type=int, default=9)
    parser.add_argument("--train_future_sections", type=int, default=2, help="How many future sections to teacher-force from each sample")
    parser.add_argument("--min_history_frames", type=int, default=30, help="Min raw frames in history segment")
    parser.add_argument("--max_video_frames", type=int, default=None, help="Optional cap when loading videos")
    parser.add_argument("--allow_short_videos", action="store_true", help="Allow short videos by padding frames (repeat-last)")

    # train
    parser.add_argument("--output_dir", type=str, default="./training_outputs_v2v")
    parser.add_argument("--max_steps", type=int, default=10_000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable transformer gradient checkpointing to save VRAM")

    # LoRA
    parser.add_argument("--use_lora", action="store_true", help="Train LoRA adapters instead of full model (recommended)")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora_targets",
        type=str,
        default="to_q,to_k,to_v,to_out,add_q_proj,add_k_proj,add_v_proj,to_add_out,proj_mlp,proj_out,proj_in",
        help="Comma-separated substrings of nn.Linear module names to inject LoRA into",
    )

    # conditioning
    parser.add_argument("--use_image_conditioning", action="store_true", help="Use SigLIP image embeddings from last history frame")
    parser.add_argument("--distilled_guidance_scale", type=float, default=10.0, help="Matches inference default gs=10; internally multiplied by 1000")
    parser.add_argument("--num_clean_frames", type=int, default=5, help="Matches inference_v2v.py num_clean_frames (in latent steps)")

    # checkpointing
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--resume_from", type=str, default=None, help="Path to an accelerate_state_step_* directory")

    # model ids (override if needed)
    parser.add_argument("--hunyuan_repo", type=str, default="hunyuanvideo-community/HunyuanVideo")
    parser.add_argument("--framepack_repo", type=str, default="lllyasviel/FramePackI2V_HY")
    parser.add_argument("--flux_redux_repo", type=str, default="lllyasviel/flux_redux_bfl")

    args = parser.parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
    )
    set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "args.json"), "wt", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)

    # Models
    # Note: we freeze VAE/text/image encoders; only train transformer.
    with accelerator.main_process_first():
        text_encoder = LlamaModel.from_pretrained(args.hunyuan_repo, subfolder="text_encoder", torch_dtype=torch.float16).cpu()
        text_encoder_2 = CLIPTextModel.from_pretrained(args.hunyuan_repo, subfolder="text_encoder_2", torch_dtype=torch.float16).cpu()
        tokenizer = LlamaTokenizerFast.from_pretrained(args.hunyuan_repo, subfolder="tokenizer")
        tokenizer_2 = CLIPTokenizer.from_pretrained(args.hunyuan_repo, subfolder="tokenizer_2")
        vae = AutoencoderKLHunyuanVideo.from_pretrained(args.hunyuan_repo, subfolder="vae", torch_dtype=torch.float16).cpu()

        feature_extractor = SiglipImageProcessor.from_pretrained(args.flux_redux_repo, subfolder="feature_extractor")
        image_encoder = SiglipVisionModel.from_pretrained(args.flux_redux_repo, subfolder="image_encoder", torch_dtype=torch.float16).cpu()

        transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(args.framepack_repo, torch_dtype=torch.bfloat16).cpu()

    text_encoder.eval().requires_grad_(False)
    text_encoder_2.eval().requires_grad_(False)
    image_encoder.eval().requires_grad_(False)
    vae.eval().requires_grad_(False)

    transformer.train().requires_grad_(True)
    transformer.high_quality_fp32_output_for_inference = False

    if args.gradient_checkpointing:
        # Uses the model's internal checkpointing wrapper (non-reentrant)
        transformer.enable_gradient_checkpointing()

    if args.use_lora:
        # Freeze everything, then inject LoRA and train only LoRA params
        transformer.requires_grad_(False)
        targets = [t.strip() for t in str(args.lora_targets).split(",") if t.strip()]
        n_injected = inject_lora(
            transformer,
            r=int(args.lora_rank),
            alpha=float(args.lora_alpha),
            dropout=float(args.lora_dropout),
            target_keywords=targets,
        )
        print(f"[lora] injected {n_injected} Linear modules")
        # Make sure LoRA params are trainable
        for p in transformer.parameters():
            if p.requires_grad:
                # keep
                continue
        # (LoRALinear sets requires_grad on A/B by default)

    # For v2v conditioning, the packed model must have both projections installed.
    # If checkpoint was trained without them, install here.
    if transformer.clean_x_embedder is None:
        transformer.install_clean_x_embedder()
    if transformer.image_projection is None and args.use_image_conditioning:
        transformer.install_image_projection(in_channels=transformer.config.get("image_proj_dim", 1152))

    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters (did you mean to set --use_lora?)")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    dataset = JsonlVideoDataset(
        jsonl_path=args.dataset_jsonl,
        resolution=args.resolution,
        latent_window_size=args.latent_window_size,
        train_future_sections=int(args.train_future_sections),
        min_history_frames=args.min_history_frames,
        max_video_frames=args.max_video_frames,
        allow_short_videos=bool(args.allow_short_videos),
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=_collate_identity,
        drop_last=True,
    )

    transformer, optimizer, loader = accelerator.prepare(transformer, optimizer, loader)

    # Move frozen encoders to the training device for fast encoding
    device = accelerator.device
    text_encoder.to(device)
    text_encoder_2.to(device)
    vae.to(device)
    image_encoder.to(device)

    if args.resume_from is not None:
        accelerator.load_state(args.resume_from)

    prompt_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    global_step = 0
    running_loss = 0.0
    running_count = 0

    while global_step < args.max_steps:
        for batch_items in loader:
            if global_step >= args.max_steps:
                break

            # Robustness: if a video is too short or decoding fails, skip and continue.
            try:
                train_batch = _encode_conditions_and_target(
                    batch_items,
                    vae=vae,
                    feature_extractor=feature_extractor,
                    image_encoder=image_encoder,
                    text_encoder=text_encoder,
                    text_encoder_2=text_encoder_2,
                    tokenizer=tokenizer,
                    tokenizer_2=tokenizer_2,
                    device=device,
                    use_image_conditioning=bool(args.use_image_conditioning),
                    prompt_cache=prompt_cache,
                    latent_window_size=args.latent_window_size,
                    distilled_guidance_scale=float(args.distilled_guidance_scale),
                    num_clean_frames=int(args.num_clean_frames),
                    train_future_sections=int(args.train_future_sections),
                )
            except Exception as e:
                if accelerator.is_main_process:
                    print(f"[skip] batch due to error: {e}")
                continue

            with accelerator.accumulate(transformer):
                # Model expects dtype-specific timestep/guidance; keep sigma in fp32 but cast inputs later.
                timestep = (train_batch.sigma * 1000.0).to(dtype=transformer.dtype)
                guidance = torch.full(
                    (train_batch.x_noisy.shape[0],),
                    float(args.distilled_guidance_scale) * 1000.0,
                    device=device,
                    dtype=transformer.dtype,
                )

                pred = transformer(
                    hidden_states=train_batch.x_noisy.to(dtype=torch.float32),  # wrapper uses fp32 at sampling; keep stable here
                    timestep=timestep,
                    encoder_hidden_states=train_batch.llama_vec.to(dtype=transformer.dtype),
                    encoder_attention_mask=train_batch.llama_mask,
                    pooled_projections=train_batch.clip_pooler.to(dtype=transformer.dtype),
                    guidance=guidance,
                    latent_indices=train_batch.latent_indices,
                    clean_latents=train_batch.clean_latents,
                    clean_latent_indices=train_batch.clean_latent_indices,
                    clean_latents_2x=train_batch.clean_latents_2x,
                    clean_latent_2x_indices=train_batch.clean_latent_2x_indices,
                    clean_latents_4x=train_batch.clean_latents_4x,
                    clean_latent_4x_indices=train_batch.clean_latent_4x_indices,
                    image_embeddings=train_batch.image_embeddings.to(dtype=transformer.dtype) if train_batch.image_embeddings is not None else None,
                    return_dict=False,
                )[0]

                loss = F.mse_loss(pred.float(), train_batch.v_target.float(), reduction="mean")

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # logging
            running_loss += float(loss.detach().cpu())
            running_count += 1
            global_step += 1

            if accelerator.is_main_process and (global_step % 10 == 0):
                avg = running_loss / max(1, running_count)
                print(f"step={global_step} loss={avg:.6f}")
                running_loss = 0.0
                running_count = 0

            if accelerator.is_main_process and (global_step % args.save_every == 0):
                _save_checkpoint(
                    args.output_dir,
                    global_step,
                    accelerator=accelerator,
                    transformer=transformer,
                    optimizer=optimizer,
                )
                if args.use_lora:
                    _save_lora_checkpoint(args.output_dir, global_step, accelerator=accelerator, transformer=transformer)


    if accelerator.is_main_process:
        _save_checkpoint(
            args.output_dir,
            global_step,
            accelerator=accelerator,
            transformer=transformer,
            optimizer=optimizer,
        )
        if args.use_lora:
            _save_lora_checkpoint(args.output_dir, global_step, accelerator=accelerator, transformer=transformer)


if __name__ == "__main__":
    main()

