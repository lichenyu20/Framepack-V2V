import argparse
import os
from pathlib import Path

import decord
import numpy as np
import torch

from diffusers import AutoencoderKLHunyuanVideo
from transformers import (
    LlamaModel,
    CLIPTextModel,
    LlamaTokenizerFast,
    CLIPTokenizer,
    SiglipImageProcessor,
    SiglipVisionModel,
)

from diffusers_helper.bucket_tools import find_nearest_bucket
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.hunyuan import encode_prompt_conds, vae_encode, vae_decode
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.utils import resize_and_center_crop, crop_or_pad_yield_mask, save_bcthw_as_mp4, soft_append_bcthw

from training_v2v import LoRALinear, inject_lora


def _frames_to_bcthw(frames_np: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(frames_np).float() / 127.5 - 1.0
    x = x.permute(3, 0, 1, 2)[None]
    return x


def load_model_with_lora(
    hunyuan_repo: str,
    framepack_repo: str,
    flux_redux_repo: str,
    lora_path: str,
    device: torch.device,
) -> tuple:
    text_encoder = LlamaModel.from_pretrained(hunyuan_repo, subfolder="text_encoder", torch_dtype=torch.float16).to(device)
    text_encoder_2 = CLIPTextModel.from_pretrained(hunyuan_repo, subfolder="text_encoder_2", torch_dtype=torch.float16).to(device)
    tokenizer = LlamaTokenizerFast.from_pretrained(hunyuan_repo, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(hunyuan_repo, subfolder="tokenizer_2")
    vae = AutoencoderKLHunyuanVideo.from_pretrained(hunyuan_repo, subfolder="vae", torch_dtype=torch.float16).to(device)

    feature_extractor = SiglipImageProcessor.from_pretrained(flux_redux_repo, subfolder="feature_extractor")
    image_encoder = SiglipVisionModel.from_pretrained(flux_redux_repo, subfolder="image_encoder", torch_dtype=torch.float16).to(device)

    transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(framepack_repo, torch_dtype=torch.bfloat16).to(device)
    transformer.high_quality_fp32_output_for_inference = True
    transformer.eval()

    # Inject LoRA structure and load weights
    _ = inject_lora(
        transformer,
        r=16,
        alpha=16.0,
        dropout=0.0,
        target_keywords=[
            "to_q",
            "to_k",
            "to_v",
            "to_out",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_add_out",
            "proj_mlp",
            "proj_out",
            "proj_in",
        ],
    )
    if lora_path is not None:
        sd = torch.load(lora_path, map_location="cpu")
        missing = transformer.load_state_dict(sd, strict=False)
        print(f"[lora] loaded from {lora_path}, missing={missing}")

    return (
        text_encoder,
        text_encoder_2,
        tokenizer,
        tokenizer_2,
        vae,
        feature_extractor,
        image_encoder,
        transformer,
    )


@torch.no_grad()
def run_v2v_eval(
    video_path: str,
    prompt: str,
    negative_prompt: str,
    output_path: str,
    *,
    total_second_length: float = 5.0,
    latent_window_size: int = 9,
    steps: int = 25,
    cfg: float = 1.0,
    gs: float = 10.0,
    rs: float = 0.0,
    num_clean_frames: int = 5,
    eval_history_max_frames: int = 256,
    resolution: int = 640,
    mp4_crf: int = 16,
    device: torch.device,
    models: tuple,
) -> None:
    (
        text_encoder,
        text_encoder_2,
        tokenizer,
        tokenizer_2,
        vae,
        feature_extractor,
        image_encoder,
        transformer,
    ) = models

    vr = decord.VideoReader(str(Path(video_path).expanduser().resolve()))
    fps = float(vr.get_avg_fps())
    n = min(len(vr), int(eval_history_max_frames))
    n = (n // 4) * 4
    if n <= 0:
        raise ValueError(f"Video too short after truncation: {video_path}")

    raw_frames = vr.get_batch(range(n)).asnumpy()
    T, H, W, _ = raw_frames.shape
    target_height, target_width = find_nearest_bucket(H, W, resolution=resolution)

    frames = np.stack(
        [resize_and_center_crop(f, target_width=target_width, target_height=target_height) for f in raw_frames],
        axis=0,
    )

    input_image_np = frames[0]
    input_pixels = _frames_to_bcthw(frames).to(device)

    # Encode full video to latents
    video_pt = _frames_to_bcthw(frames).to(device)
    vae_batch = 16
    latents_parts = []
    for i in range(0, video_pt.shape[2], vae_batch):
        batch = video_pt[:, :, i : i + vae_batch]
        latents_parts.append(vae_encode(batch, vae))
    video_latents = torch.cat(latents_parts, dim=2)  # [1,16,T_lat,h,w]

    # Text conds
    llama_vec, clip_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
    llama_vec, llama_mask = crop_or_pad_yield_mask(llama_vec, length=512)

    if cfg == 1.0:
        llama_vec_n = torch.zeros_like(llama_vec)
        clip_pooler_n = torch.zeros_like(clip_pooler)
    else:
        llama_vec_n, clip_pooler_n = encode_prompt_conds(negative_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
        llama_vec_n, _ = crop_or_pad_yield_mask(llama_vec_n, length=512)

    llama_vec = llama_vec.to(device=device, dtype=transformer.dtype)
    llama_mask = llama_mask.to(device=device)
    clip_pooler = clip_pooler.to(device=device, dtype=transformer.dtype)
    llama_vec_n = llama_vec_n.to(device=device, dtype=transformer.dtype)
    clip_pooler_n = clip_pooler_n.to(device=device, dtype=transformer.dtype)

    image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
    image_embeddings = image_encoder_output.last_hidden_state.to(device=device, dtype=transformer.dtype)

    # Progressive sampling (简化版，直接用 training_v2v 的 eval 逻辑)
    total_latent_sections = (float(total_second_length) * fps) / (int(latent_window_size) * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    rnd = torch.Generator("cpu").manual_seed(31337)

    history_latents = video_latents
    history_pixels = None
    total_generated_latent_frames = 0

    latent_paddings = list(reversed(range(total_latent_sections)))
    if total_latent_sections > 4:
        latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

    for latent_padding in latent_paddings:
        is_start_of_video = latent_padding == 0
        latent_padding_size = latent_padding * int(latent_window_size)

        available_frames = video_latents.shape[2] if is_start_of_video else history_latents.shape[2]

        effective_clean_frames = max(0, int(num_clean_frames) - 1) if int(num_clean_frames) > 1 else 1
        if is_start_of_video:
            effective_clean_frames = 1

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

        total_index_length = (
            clean_latent_pre_frames
            + latent_padding_size
            + int(latent_window_size)
            + post_frames
            + num_2x_frames
            + num_4x_frames
        )
        indices = torch.arange(0, total_index_length, device=device).unsqueeze(0)
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

        context_frames = (
            history_latents[:, :, -(total_context_frames + clean_latent_pre_frames) : -clean_latent_pre_frames, :, :]
            if total_context_frames > 0
            else history_latents[:, :, :1, :, :]
        )

        split_sizes = [num_4x_frames, num_2x_frames]
        split_sizes = [s for s in split_sizes if s > 0]

        if split_sizes and context_frames.shape[2] >= sum(split_sizes):
            splits = context_frames.split(split_sizes, dim=2)
            split_idx = 0
            if num_4x_frames > 0:
                clean_latents_4x = splits[split_idx]
                split_idx += 1
            else:
                clean_latents_4x = history_latents[:, :, :1, :, :]

            if num_2x_frames > 0 and split_idx < len(splits):
                clean_latents_2x = splits[split_idx]
            else:
                clean_latents_2x = history_latents[:, :, :1, :, :]
        else:
            clean_latents_4x = history_latents[:, :, :1, :, :]
            clean_latents_2x = history_latents[:, :, :1, :, :]

        clean_latents_pre = video_latents[:, :, -min(effective_clean_frames, video_latents.shape[2]) :].to(history_latents)
        clean_latents_post = history_latents[:, :, : min(effective_clean_frames, history_latents.shape[2]), :, :]

        if clean_latents_pre.shape[2] < clean_latent_pre_frames:
            repeat_factor = int(np.ceil(clean_latent_pre_frames / clean_latents_pre.shape[2]))
            clean_latents_pre = clean_latents_pre.repeat(1, 1, repeat_factor, 1, 1)[:, :, :clean_latent_pre_frames]

        if clean_latents_post.shape[2] < post_frames:
            repeat_factor = int(np.ceil(post_frames / clean_latents_post.shape[2]))
            clean_latents_post = clean_latents_post.repeat(1, 1, repeat_factor, 1, 1)[:, :, :post_frames]

        clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

        max_frames = min(int(latent_window_size) * 4 - 3, history_latents.shape[2] * 4)

        generated_latents = sample_hunyuan(
            transformer=transformer,
            sampler="unipc",
            width=int(target_width),
            height=int(target_height),
            frames=int(max_frames),
            real_guidance_scale=cfg,
            distilled_guidance_scale=gs,
            guidance_rescale=rs,
            num_inference_steps=int(steps),
            generator=rnd,
            prompt_embeds=llama_vec,
            prompt_embeds_mask=llama_mask,
            prompt_poolers=clip_pooler,
            negative_prompt_embeds=llama_vec_n,
            negative_prompt_embeds_mask=llama_mask_n,
            negative_prompt_poolers=clip_pooler_n,
            device=device,
            dtype=torch.bfloat16,
            image_embeddings=image_embeddings,
            latent_indices=latent_indices,
            clean_latents=clean_latents,
            clean_latent_indices=clean_latent_indices,
            clean_latents_2x=clean_latents_2x,
            clean_latent_2x_indices=clean_latent_2x_indices,
            clean_latents_4x=clean_latents_4x,
            clean_latent_4x_indices=clean_latent_4x_indices,
        )

        if is_start_of_video:
            generated_latents = torch.cat([video_latents[:, :, -1:].to(generated_latents), generated_latents], dim=2)

        total_generated_latent_frames += int(generated_latents.shape[2])
        history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

        real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

        if history_pixels is None:
            history_pixels = vae_decode(real_history_latents.to(device), vae).cpu()
        else:
            section_latent_frames = int(latent_window_size) * 2 + (1 if is_start_of_video else 0)
            overlapped_frames = int(latent_window_size) * 4 - 3
            current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames].to(device), vae).cpu()
            history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

        if is_start_of_video:
            break

    if history_pixels is None:
        raise RuntimeError("Eval produced no pixels")

    history_pixels = torch.cat([input_pixels, history_pixels], dim=2)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    save_bcthw_as_mp4(history_pixels.cpu(), output_path, fps=fps, crf=mp4_crf)


def main():
    parser = argparse.ArgumentParser(description="Eval v2v with LoRA on a single video")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--lora_path", type=str, required=True, help="Path to lora_step_*.safetensors")

    parser.add_argument("--total_second_length", type=float, default=5.0)
    parser.add_argument("--latent_window_size", type=int, default=9)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--gs", type=float, default=10.0)
    parser.add_argument("--rs", type=float, default=0.0)
    parser.add_argument("--num_clean_frames", type=int, default=5)
    parser.add_argument("--eval_history_max_frames", type=int, default=256)
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--mp4_crf", type=int, default=16)

    parser.add_argument("--hunyuan_repo", type=str, default="hunyuanvideo-community/HunyuanVideo")
    parser.add_argument("--framepack_repo", type=str, default="lllyasviel/FramePackI2V_HY")
    parser.add_argument("--flux_redux_repo", type=str, default="lllyasviel/flux_redux_bfl")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = load_model_with_lora(
        args.hunyuan_repo,
        args.framepack_repo,
        args.flux_redux_repo,
        args.lora_path,
        device,
    )

    run_v2v_eval(
        video_path=args.video,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        output_path=args.output,
        total_second_length=args.total_second_length,
        latent_window_size=args.latent_window_size,
        steps=args.steps,
        cfg=args.cfg,
        gs=args.gs,
        rs=args.rs,
        num_clean_frames=args.num_clean_frames,
        eval_history_max_frames=args.eval_history_max_frames,
        resolution=args.resolution,
        mp4_crf=args.mp4_crf,
        device=device,
        models=models,
    )


if __name__ == "__main__":
    main()

