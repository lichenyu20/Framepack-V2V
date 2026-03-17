from diffusers_helper.hf_login import login

import os
import argparse
import math
import traceback

os.environ["HF_HOME"] = "/work/hdd/bbsg/cli34/cli34/models"

import cv2
import torch
import einops
import numpy as np

from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from transformers import SiglipImageProcessor, SiglipVisionModel

from diffusers_helper.hunyuan import (
    encode_prompt_conds,
    vae_decode,
    vae_encode,
    vae_decode_fake,
)
from diffusers_helper.utils import (
    save_bcthw_as_mp4,
    crop_or_pad_yield_mask,
    soft_append_bcthw,
    resize_and_center_crop,
    generate_timestamp,
)
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import (
    cpu,
    gpu,
    get_cuda_free_memory_gb,
    move_model_to_device_with_memory_preservation,
    offload_model_from_device_for_memory_preservation,
    fake_diffusers_current_device,
    DynamicSwapInstaller,
    unload_complete_models,
    load_model_as_complete,
)
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket


# -------------------------
# Argument Parser
# -------------------------
parser = argparse.ArgumentParser(description="FramePack CLI inference with image or history-video conditioning")

parser.add_argument("--image", type=str, default=None, help="Path to input image")
parser.add_argument("--history_video", type=str, default=None, help="Path to history video for future generation")
parser.add_argument("--prompt", type=str, required=True, help="Positive prompt")
parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt")
parser.add_argument("--seed", type=int, default=31337)

parser.add_argument("--total_second_length", type=float, default=5.0, help="Generated future video length (seconds)")
parser.add_argument("--latent_window_size", type=int, default=9)
parser.add_argument("--steps", type=int, default=25)
parser.add_argument("--cfg", type=float, default=1.0)
parser.add_argument("--gs", type=float, default=10.0, help="Distilled CFG Scale")
parser.add_argument("--rs", type=float, default=0.0, help="CFG Re-Scale")

parser.add_argument("--gpu_memory_preservation", type=float, default=6.0)
parser.add_argument("--use_teacache", action="store_true")
parser.add_argument("--mp4_crf", type=int, default=16)
parser.add_argument("--output_dir", type=str, default="./outputs")

parser.add_argument("--save_input_image", action="store_true")
parser.add_argument("--save_condition_frames", action="store_true")
parser.add_argument("--prepend_history", action="store_true", help="Also save a combined history+future video")
parser.add_argument("--history_max_frames", type=int, default=256, help="Max frames to load from history video")

args = parser.parse_args()

if (args.image is None and args.history_video is None) or (args.image is not None and args.history_video is not None):
    raise ValueError("Please provide exactly one of --image or --history_video")

print("Args:")
print(args)


# -------------------------
# Env / Memory
# -------------------------
free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60

print(f"Free VRAM: {free_mem_gb} GB")
print(f"High-VRAM Mode: {high_vram}")


# -------------------------
# Model Loading
# -------------------------
print("Loading models ...")

text_encoder = LlamaModel.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo",
    subfolder="text_encoder",
    torch_dtype=torch.float16
).cpu()

text_encoder_2 = CLIPTextModel.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo",
    subfolder="text_encoder_2",
    torch_dtype=torch.float16
).cpu()

tokenizer = LlamaTokenizerFast.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo",
    subfolder="tokenizer"
)

tokenizer_2 = CLIPTokenizer.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo",
    subfolder="tokenizer_2"
)

vae = AutoencoderKLHunyuanVideo.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo",
    subfolder="vae",
    torch_dtype=torch.float16
).cpu()

feature_extractor = SiglipImageProcessor.from_pretrained(
    "lllyasviel/flux_redux_bfl",
    subfolder="feature_extractor"
)

image_encoder = SiglipVisionModel.from_pretrained(
    "lllyasviel/flux_redux_bfl",
    subfolder="image_encoder",
    torch_dtype=torch.float16
).cpu()

transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
    "lllyasviel/FramePackI2V_HY",
    torch_dtype=torch.bfloat16
).cpu()

vae.eval()
text_encoder.eval()
text_encoder_2.eval()
image_encoder.eval()
transformer.eval()

if not high_vram:
    vae.enable_slicing()
    vae.enable_tiling()

transformer.high_quality_fp32_output_for_inference = True
print("transformer.high_quality_fp32_output_for_inference = True")

transformer.to(dtype=torch.bfloat16)
vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)
text_encoder.to(dtype=torch.float16)
text_encoder_2.to(dtype=torch.float16)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)
image_encoder.requires_grad_(False)
transformer.requires_grad_(False)

if not high_vram:
    DynamicSwapInstaller.install_model(transformer, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    text_encoder.to(gpu)
    text_encoder_2.to(gpu)
    image_encoder.to(gpu)
    vae.to(gpu)
    transformer.to(gpu)

print("Models loaded.")


# -------------------------
# Utils
# -------------------------
def load_image_as_numpy(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    return np.array(img)


def load_video_as_numpy_frames(video_path: str, max_frames: int = 256) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

        if max_frames is not None and len(frames) >= max_frames:
            break

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames found in video: {video_path}")

    return np.stack(frames, axis=0)  # [T, H, W, C]


def preprocess_frame(frame: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    return resize_and_center_crop(frame, target_width=target_width, target_height=target_height)


def frames_to_bcthw(frames_np: np.ndarray) -> torch.Tensor:
    """
    frames_np: [T, H, W, 3], uint8 RGB
    return: [1, 3, T, H, W], float32 in [-1, 1]
    """
    x = torch.from_numpy(frames_np).float() / 127.5 - 1.0
    x = x.permute(3, 0, 1, 2)[None]   # [1, 3, T, H, W]
    return x


@torch.no_grad()
def encode_frames_to_latents(frames_np: np.ndarray, vae) -> torch.Tensor:
    """
    frames_np: [T, H, W, 3]
    return: latent [1, C, T_lat, h, w]
    """
    video_pt = frames_to_bcthw(frames_np)
    latents = vae_encode(video_pt, vae)
    return latents


def tail_sample_indices(n_total: int, n_take: int, tail_ratio: float = 0.8) -> np.ndarray:
    """
    Sample indices with preference toward the tail of the video.
    If n_take > available frames, this will repeat indices by interpolation.
    """
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


def make_condition_sets_from_history(history_frames: np.ndarray, target_width: int, target_height: int):
    """
    Build condition frame groups from history video.

    IMPORTANT:
    - pre/post each need to encode to 1 latent step -> 1 pixel frame is fine
    - clean_latents_2x expects 2 latent steps -> use ~8 pixel frames
    - clean_latents_4x expects 16 latent steps -> use ~64 pixel frames
    """
    processed = np.stack(
        [preprocess_frame(f, target_width, target_height) for f in history_frames],
        axis=0
    )  # [T, H, W, 3]

    T = len(processed)
    if T == 0:
        raise ValueError("History video has no frames")

    # Closest to the future boundary
    idx_pre = np.array([T - 1], dtype=np.int64)

    # Neighbor frame near the boundary
    idx_post = np.array([max(0, T - 2)], dtype=np.int64)

    # 2 latent steps -> 8 pixel frames
    idx_2x = tail_sample_indices(T, 8, tail_ratio=0.40)

    # 16 latent steps -> 64 pixel frames
    idx_4x = tail_sample_indices(T, 64, tail_ratio=0.90)

    frames_pre = processed[idx_pre]     # [1, H, W, 3]
    frames_post = processed[idx_post]   # [1, H, W, 3]
    frames_2x = processed[idx_2x]       # [8, H, W, 3]   -> latent T ~ 2
    frames_4x = processed[idx_4x]       # [64, H, W, 3]  -> latent T ~ 16

    return processed, frames_pre, frames_post, frames_2x, frames_4x


def bcthw_cat_time(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    both [B, C, T, H, W]
    """
    return torch.cat([a, b], dim=2)


# -------------------------
# Core Inference
# -------------------------
@torch.no_grad()
def run_inference(
    input_image: np.ndarray = None,
    history_video_frames: np.ndarray = None,
    prompt: str = "",
    n_prompt: str = "",
    seed: int = 31337,
    total_second_length: float = 5.0,
    latent_window_size: int = 9,
    steps: int = 25,
    cfg: float = 1.0,
    gs: float = 10.0,
    rs: float = 0.0,
    gpu_memory_preservation: float = 6.0,
    use_teacache: bool = False,
    mp4_crf: int = 16,
    output_dir: str = "./outputs",
    save_input_image: bool = False,
    save_condition_frames: bool = False,
    prepend_history: bool = False,
):
    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    job_id = generate_timestamp()
    os.makedirs(output_dir, exist_ok=True)

    print(f"[{job_id}] Starting inference ...")
    print(f"[{job_id}] total_latent_sections = {total_latent_sections}")

    try:
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

        # -------------------------
        # Text encoding
        # -------------------------
        print(f"[{job_id}] Text encoding ...")

        if not high_vram:
            fake_diffusers_current_device(text_encoder, gpu)
            load_model_as_complete(text_encoder_2, target_device=gpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(
            prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2
        )

        if cfg == 1:
            llama_vec_n = torch.zeros_like(llama_vec)
            clip_l_pooler_n = torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(
                n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2
            )

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # -------------------------
        # Input processing
        # -------------------------
        print(f"[{job_id}] Input processing ...")

        if input_image is not None:
            H, W, _ = input_image.shape
            height, width = find_nearest_bucket(H, W, resolution=640)

            input_image_np = resize_and_center_crop(
                input_image,
                target_width=width,
                target_height=height
            )

            if save_input_image:
                input_png = os.path.join(output_dir, f"{job_id}_input.png")
                Image.fromarray(input_image_np).save(input_png)
                print(f"[{job_id}] Saved processed input image to: {input_png}")

            if not high_vram:
                load_model_as_complete(vae, target_device=gpu)

            input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
            input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]  # [1, 3, 1, H, W]

            start_latent = vae_encode(input_image_pt, vae)  # [1, 16, 1, h, w]
            clean_post_latent = torch.zeros_like(start_latent)

            clean_latents_2x = torch.zeros(
                (1, start_latent.shape[1], 2, start_latent.shape[3], start_latent.shape[4]),
                dtype=start_latent.dtype,
                device=start_latent.device,
            )
            clean_latents_4x = torch.zeros(
                (1, start_latent.shape[1], 16, start_latent.shape[3], start_latent.shape[4]),
                dtype=start_latent.dtype,
                device=start_latent.device,
            )

            if not high_vram:
                load_model_as_complete(image_encoder, target_device=gpu)

            image_encoder_output = hf_clip_vision_encode(
                input_image_np, feature_extractor, image_encoder
            )
            image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

            processed_history_pixels = None

        else:
            first_frame = history_video_frames[0]
            H, W, _ = first_frame.shape
            height, width = find_nearest_bucket(H, W, resolution=640)

            processed_history_frames, frames_pre, frames_post, frames_2x_np, frames_4x_np = make_condition_sets_from_history(
                history_video_frames, width, height
            )

            if save_condition_frames:
                Image.fromarray(frames_pre[0]).save(os.path.join(output_dir, f"{job_id}_cond_pre.png"))
                Image.fromarray(frames_post[0]).save(os.path.join(output_dir, f"{job_id}_cond_post.png"))
                Image.fromarray(frames_2x_np[-1]).save(os.path.join(output_dir, f"{job_id}_cond_2x_last.png"))
                Image.fromarray(frames_4x_np[-1]).save(os.path.join(output_dir, f"{job_id}_cond_4x_last.png"))

            if not high_vram:
                load_model_as_complete(vae, target_device=gpu)

            start_latent = encode_frames_to_latents(frames_pre, vae)        # expect [1, 16, 1, h, w]
            clean_post_latent = encode_frames_to_latents(frames_post, vae)  # expect [1, 16, 1, h, w]
            clean_latents_2x = encode_frames_to_latents(frames_2x_np, vae)  # expect [1, 16, 2, h, w]
            clean_latents_4x = encode_frames_to_latents(frames_4x_np, vae)  # expect [1, 16, 16, h, w]

            print(f"[{job_id}] start_latent shape:      {tuple(start_latent.shape)}")
            print(f"[{job_id}] clean_post_latent shape: {tuple(clean_post_latent.shape)}")
            print(f"[{job_id}] clean_latents_2x shape:  {tuple(clean_latents_2x.shape)}")
            print(f"[{job_id}] clean_latents_4x shape:  {tuple(clean_latents_4x.shape)}")

            # Safety checks
            if start_latent.shape[2] != 1:
                raise ValueError(f"Expected start_latent temporal length 1, got {start_latent.shape}")
            if clean_post_latent.shape[2] != 1:
                raise ValueError(f"Expected clean_post_latent temporal length 1, got {clean_post_latent.shape}")
            if clean_latents_2x.shape[2] != 2:
                raise ValueError(f"Expected clean_latents_2x temporal length 2, got {clean_latents_2x.shape}")
            if clean_latents_4x.shape[2] != 16:
                raise ValueError(f"Expected clean_latents_4x temporal length 16, got {clean_latents_4x.shape}")

            if not high_vram:
                load_model_as_complete(image_encoder, target_device=gpu)

            image_encoder_output = hf_clip_vision_encode(
                processed_history_frames[-1], feature_extractor, image_encoder
            )
            image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

            if prepend_history:
                hist_pt = frames_to_bcthw(processed_history_frames)     # [-1,1]
                processed_history_pixels = (hist_pt.cpu() + 1.0) / 2.0  # [0,1]
            else:
                processed_history_pixels = None

        # -------------------------
        # Dtype conversion
        # -------------------------
        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # -------------------------
        # Sampling
        # -------------------------
        print(f"[{job_id}] Start sampling ...")

        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3

        history_latents = torch.zeros(
            size=(1, 16, 1 + 2 + 16, height // 8, width // 8),
            dtype=torch.float32
        ).cpu()

        history_latents[:, :, 0:1, :, :] = clean_post_latent.to(history_latents)
        history_latents[:, :, 1:3, :, :] = clean_latents_2x.to(history_latents)
        history_latents[:, :, 3:19, :, :] = clean_latents_4x.to(history_latents)

        history_pixels = None
        total_generated_latent_frames = 0

        latent_paddings = reversed(range(total_latent_sections))
        if total_latent_sections > 4:
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        latest_output_filename = None

        for section_idx, latent_padding in enumerate(latent_paddings, start=1):
            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size

            print(
                f"[{job_id}] Section {section_idx}: "
                f"latent_padding_size = {latent_padding_size}, "
                f"is_last_section = {is_last_section}"
            )

            indices = torch.arange(
                0,
                sum([1, latent_padding_size, latent_window_size, 1, 2, 16])
            ).unsqueeze(0)

            (
                clean_latent_indices_pre,
                blank_indices,
                latent_indices,
                clean_latent_indices_post,
                clean_latent_2x_indices,
                clean_latent_4x_indices,
            ) = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)

            clean_latent_indices = torch.cat(
                [clean_latent_indices_pre, clean_latent_indices_post], dim=1
            )

            clean_latents_pre = start_latent.to(history_latents)
            clean_latents_post_from_hist, clean_latents_2x_from_hist, clean_latents_4x_from_hist = history_latents[:, :, :1 + 2 + 16, :, :].split(
                [1, 2, 16], dim=2
            )
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post_from_hist], dim=2)

            if not high_vram:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(
                    transformer,
                    target_device=gpu,
                    preserved_memory_gb=gpu_memory_preservation
                )

            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            def callback(d):
                current_step = d["i"] + 1
                percentage = int(100.0 * current_step / steps)

                if current_step == 1 or current_step == steps or current_step % 5 == 0:
                    preview = d["denoised"]
                    preview = vae_decode_fake(preview)
                    preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                    preview = einops.rearrange(preview, "b c t h w -> (b h) (t w) c")
                    print(
                        f"[{job_id}]   step {current_step}/{steps} "
                        f"({percentage}%) | preview shape = {preview.shape}"
                    )
                else:
                    print(f"[{job_id}]   step {current_step}/{steps} ({percentage}%)")

            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler="unipc",
                width=width,
                height=height,
                frames=num_frames,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=gpu,
                dtype=torch.bfloat16,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x_from_hist,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x_from_hist,
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )

            # image mode keeps original behavior
            if is_last_section and input_image is not None:
                generated_latents = torch.cat(
                    [start_latent.to(generated_latents), generated_latents],
                    dim=2
                )

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

            if not high_vram:
                offload_model_from_device_for_memory_preservation(
                    transformer,
                    target_device=gpu,
                    preserved_memory_gb=8
                )
                load_model_as_complete(vae, target_device=gpu)

            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                overlapped_frames = latent_window_size * 4 - 3

                current_pixels = vae_decode(
                    real_history_latents[:, :, :section_latent_frames],
                    vae
                ).cpu()
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

            if not high_vram:
                unload_complete_models()

            latest_output_filename = os.path.join(
                output_dir,
                f"{job_id}_{total_generated_latent_frames}.mp4"
            )

            save_bcthw_as_mp4(history_pixels, latest_output_filename, fps=30, crf=mp4_crf)

            current_total_frames = int(max(0, total_generated_latent_frames * 4 - 3))
            current_seconds = max(0, current_total_frames / 30)

            print(
                f"[{job_id}] Decoded and saved: {latest_output_filename}\n"
                f"[{job_id}] Current latent shape: {real_history_latents.shape}\n"
                f"[{job_id}] Current pixel shape: {history_pixels.shape}\n"
                f"[{job_id}] Total generated future frames: {current_total_frames}\n"
                f"[{job_id}] Future video length: {current_seconds:.2f} sec"
            )

            if is_last_section:
                break

        if prepend_history and processed_history_pixels is not None and history_pixels is not None:
            combined_pixels = bcthw_cat_time(processed_history_pixels, history_pixels)
            combined_output_filename = os.path.join(output_dir, f"{job_id}_history_plus_future.mp4")
            save_bcthw_as_mp4(combined_pixels, combined_output_filename, fps=30, crf=mp4_crf)
            print(f"[{job_id}] Saved combined history+future video: {combined_output_filename}")

        print(f"[{job_id}] Done.")
        if latest_output_filename is not None:
            print(f"[{job_id}] Final future video: {latest_output_filename}")
        return latest_output_filename

    except Exception as e:
        traceback.print_exc()

        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

        raise e


def main():
    input_image = None
    history_video_frames = None

    if args.image is not None:
        input_image = load_image_as_numpy(args.image)

    if args.history_video is not None:
        history_video_frames = load_video_as_numpy_frames(
            args.history_video,
            max_frames=args.history_max_frames
        )
        print(f"Loaded history video: {history_video_frames.shape}")

    final_video = run_inference(
        input_image=input_image,
        history_video_frames=history_video_frames,
        prompt=args.prompt,
        n_prompt=args.negative_prompt,
        seed=args.seed,
        total_second_length=args.total_second_length,
        latent_window_size=args.latent_window_size,
        steps=args.steps,
        cfg=args.cfg,
        gs=args.gs,
        rs=args.rs,
        gpu_memory_preservation=args.gpu_memory_preservation,
        use_teacache=args.use_teacache,
        mp4_crf=args.mp4_crf,
        output_dir=args.output_dir,
        save_input_image=args.save_input_image,
        save_condition_frames=args.save_condition_frames,
        prepend_history=args.prepend_history,
    )

    print(f"Final output video saved to: {final_video}")


if __name__ == "__main__":
    main()