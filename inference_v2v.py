from diffusers_helper.hf_login import login

import os
import argparse
import math
import traceback
import pathlib

os.environ['HF_HOME'] = "/work/hdd/bbsg/cli34/cli34/models"

import torch
import einops
import numpy as np
import decord

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
parser = argparse.ArgumentParser(description="FramePack CLI video extension inference")

parser.add_argument("--video", type=str, required=True, help="Path to input video")
parser.add_argument("--prompt", type=str, required=True, help="Positive prompt")
parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt")
parser.add_argument("--seed", type=int, default=31337)
parser.add_argument("--total_second_length", type=float, default=5.0, help="Additional seconds to generate")
parser.add_argument("--latent_window_size", type=int, default=9)
parser.add_argument("--steps", type=int, default=25)
parser.add_argument("--cfg", type=float, default=1.0)
parser.add_argument("--gs", type=float, default=10.0, help="Distilled CFG Scale")
parser.add_argument("--rs", type=float, default=0.0, help="CFG Re-Scale")
parser.add_argument("--gpu_memory_preservation", type=float, default=6.0)
parser.add_argument("--use_teacache", action="store_true")
parser.add_argument("--mp4_crf", type=int, default=16)
parser.add_argument("--output_dir", type=str, default="./outputs")

parser.add_argument("--resolution", type=int, default=640, help="Max width or height bucket")
parser.add_argument("--no_resize", action="store_true", help="Use original video resolution directly")
parser.add_argument("--vae_batch", type=int, default=16, help="VAE batch size for video encoding")
parser.add_argument("--num_clean_frames", type=int, default=5, help="Number of context frames")
parser.add_argument("--save_input_frame", action="store_true")

args = parser.parse_args()

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
# Video Encode
# -------------------------
@torch.no_grad()
def video_encode(
    video_path: str,
    resolution: int,
    no_resize: bool,
    vae,
    vae_batch_size: int = 16,
    device: str = "cuda",
):
    """
    Encode the full input video into latents.
    Returns:
        start_latent: [1, C, 1, H', W']
        input_image_np: first frame for CLIP vision
        history_latents: [1, C, T, H', W']
        fps: original video fps
        target_height, target_width
        input_video_pixels: [1, 3, T, H, W] normalized pixels (for final concat)
    """
    video_path = str(pathlib.Path(video_path).resolve())
    print(f"Processing video: {video_path}")

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, fallback to CPU")
        device = "cpu"

    vr = decord.VideoReader(video_path)
    fps = vr.get_avg_fps()
    num_real_frames = len(vr)

    print(f"Video loaded: {num_real_frames} frames, FPS={fps}")

    latent_size_factor = 4
    num_frames = (num_real_frames // latent_size_factor) * latent_size_factor
    if num_frames != num_real_frames:
        print(f"Truncating frames from {num_real_frames} to {num_frames} for latent compatibility")
    num_real_frames = num_frames

    frames = vr.get_batch(range(num_real_frames)).asnumpy()  # (T, H, W, C)
    native_height, native_width = frames.shape[1], frames.shape[2]
    print(f"Native resolution: {native_width}x{native_height}")

    target_height = native_height
    target_width = native_width

    if not no_resize:
        target_height, target_width = find_nearest_bucket(
            target_height, target_width, resolution=resolution
        )
        print(f"Using resized bucket: {target_width}x{target_height}")
    else:
        print(f"Using original resolution without resizing: {target_width}x{target_height}")

    processed_frames = []
    for i, frame in enumerate(frames):
        frame_np = resize_and_center_crop(
            frame,
            target_width=target_width,
            target_height=target_height
        )
        processed_frames.append(frame_np)
    processed_frames = np.stack(processed_frames, axis=0)

    input_image_np = processed_frames[0]

    frames_pt = torch.from_numpy(processed_frames).float() / 127.5 - 1.0
    frames_pt = frames_pt.permute(0, 3, 1, 2)          # (T, C, H, W)
    frames_pt = frames_pt.unsqueeze(0)                 # (1, T, C, H, W)
    frames_pt = frames_pt.permute(0, 2, 1, 3, 4)      # (1, C, T, H, W)

    input_video_pixels = frames_pt.cpu()
    frames_pt = frames_pt.to(device)

    print(f"Video tensor shape: {frames_pt.shape}")

    vae.to(device)
    latents = []

    with torch.no_grad():
        for i in range(0, frames_pt.shape[2], vae_batch_size):
            batch = frames_pt[:, :, i:i + vae_batch_size]
            batch_latent = vae_encode(batch, vae)
            if device == "cuda":
                torch.cuda.synchronize()
            latents.append(batch_latent)

    history_latents = torch.cat(latents, dim=2)
    start_latent = history_latents[:, :, :1]

    if device == "cuda":
        vae.to(cpu)
        torch.cuda.empty_cache()

    print(f"Encoded history_latents shape: {history_latents.shape}")
    print(f"Start latent shape: {start_latent.shape}")

    return (
        start_latent,
        input_image_np,
        history_latents.cpu(),
        fps,
        target_height,
        target_width,
        input_video_pixels,
    )


# -------------------------
# Inference
# -------------------------
@torch.no_grad()
def run_inference(
    input_video_path: str,
    prompt: str,
    n_prompt: str,
    seed: int,
    total_second_length: float,
    latent_window_size: int,
    steps: int,
    cfg: float,
    gs: float,
    rs: float,
    gpu_memory_preservation: float,
    use_teacache: bool,
    mp4_crf: int,
    output_dir: str,
    resolution: int,
    no_resize: bool,
    vae_batch: int,
    num_clean_frames: int,
    save_input_frame: bool = False,
):
    job_id = generate_timestamp()
    os.makedirs(output_dir, exist_ok=True)

    print(f"[{job_id}] Starting inference ...")

    if high_vram and (no_resize or resolution > 640):
        print(f"[{job_id}] Disabling high_vram mode due to no_resize/high resolution")
        # 注意：这里只是模仿原始 gradio 逻辑的想法，但不重写全局状态
        # 你的环境如果确实是高显存，也还是可以先保持原逻辑运行

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
        # Video processing
        # -------------------------
        print(f"[{job_id}] Video processing ...")

        (
            start_latent,
            input_image_np,
            video_latents,
            fps,
            height,
            width,
            input_video_pixels,
        ) = video_encode(
            input_video_path,
            resolution=resolution,
            no_resize=no_resize,
            vae=vae,
            vae_batch_size=vae_batch,
            device=gpu,
        )

        if save_input_frame:
            input_png = os.path.join(output_dir, f"{job_id}_input_first_frame.png")
            Image.fromarray(input_image_np).save(input_png)
            print(f"[{job_id}] Saved first frame to: {input_png}")

        # -------------------------
        # CLIP Vision
        # -------------------------
        print(f"[{job_id}] CLIP Vision encoding ...")

        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)

        image_encoder_output = hf_clip_vision_encode(
            input_image_np,
            feature_extractor,
            image_encoder
        )
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        # -------------------------
        # Dtype conversion
        # -------------------------
        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # -------------------------
        # Number of sections
        # -------------------------
        total_latent_sections = (total_second_length * fps) / (latent_window_size * 4)
        total_latent_sections = int(max(round(total_latent_sections), 1))

        print(f"[{job_id}] total_latent_sections = {total_latent_sections}")
        print(f"[{job_id}] fps = {fps}")

        rnd = torch.Generator("cpu").manual_seed(seed)

        history_latents = video_latents.cpu()
        history_pixels = None
        total_generated_latent_frames = 0

        latent_paddings = list(reversed(range(total_latent_sections)))
        if total_latent_sections > 4:
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        latest_output_filename = None

        for section_idx, latent_padding in enumerate(latent_paddings, start=1):
            is_start_of_video = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size

            print(
                f"[{job_id}] Section {section_idx}/{len(latent_paddings)} | "
                f"latent_padding={latent_padding} | "
                f"latent_padding_size={latent_padding_size} | "
                f"is_start_of_video={is_start_of_video}"
            )

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

            # -------------------------
            # Dynamic context allocation
            # -------------------------
            available_frames = video_latents.shape[2] if is_start_of_video else history_latents.shape[2]

            effective_clean_frames = max(0, num_clean_frames - 1) if num_clean_frames > 1 else 1
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
                clean_latent_pre_frames +
                latent_padding_size +
                latent_window_size +
                post_frames +
                num_2x_frames +
                num_4x_frames
            )
            indices = torch.arange(0, total_index_length).unsqueeze(0)

            (
                clean_latent_indices_pre,
                blank_indices,
                latent_indices,
                clean_latent_indices_post,
                clean_latent_2x_indices,
                clean_latent_4x_indices,
            ) = indices.split(
                [
                    clean_latent_pre_frames,
                    latent_padding_size,
                    latent_window_size,
                    post_frames,
                    num_2x_frames,
                    num_4x_frames,
                ],
                dim=1
            )

            clean_latent_indices = torch.cat(
                [clean_latent_indices_pre, clean_latent_indices_post],
                dim=1
            )

            context_frames = (
                history_latents[:, :, -(total_context_frames + clean_latent_pre_frames):-clean_latent_pre_frames, :, :]
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

            clean_latents_pre = video_latents[:, :, -min(effective_clean_frames, video_latents.shape[2]):].to(history_latents)
            clean_latents_post = history_latents[:, :, :min(effective_clean_frames, history_latents.shape[2]), :, :]

            if clean_latents_pre.shape[2] < clean_latent_pre_frames:
                repeat_factor = math.ceil(clean_latent_pre_frames / clean_latents_pre.shape[2])
                clean_latents_pre = clean_latents_pre.repeat(1, 1, repeat_factor, 1, 1)[:, :, :clean_latent_pre_frames]

            if clean_latents_post.shape[2] < post_frames:
                repeat_factor = math.ceil(post_frames / clean_latents_post.shape[2])
                clean_latents_post = clean_latents_post.repeat(1, 1, repeat_factor, 1, 1)[:, :, :post_frames]

            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            max_frames = min(latent_window_size * 4 - 3, history_latents.shape[2] * 4)

            def callback(d):
                current_step = d["i"] + 1
                percentage = int(100.0 * current_step / steps)

                if current_step == 1 or current_step == steps or current_step % 5 == 0:
                    preview = d["denoised"]
                    preview = vae_decode_fake(preview)
                    preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                    preview = einops.rearrange(preview, "b c t h w -> (b h) (t w) c")
                    print(
                        f"[{job_id}]   step {current_step}/{steps} ({percentage}%) | "
                        f"preview shape = {preview.shape}"
                    )
                else:
                    print(f"[{job_id}]   step {current_step}/{steps} ({percentage}%)")

            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler="unipc",
                width=width,
                height=height,
                frames=max_frames,
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
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )

            if is_start_of_video:
                generated_latents = torch.cat(
                    [video_latents[:, :, -1:].to(generated_latents), generated_latents],
                    dim=2
                )

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat(
                [generated_latents.to(history_latents), history_latents],
                dim=2
            )

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
                section_latent_frames = (
                    latent_window_size * 2 + 1 if is_start_of_video else latent_window_size * 2
                )
                overlapped_frames = latent_window_size * 4 - 3
                current_pixels = vae_decode(
                    real_history_latents[:, :, :section_latent_frames],
                    vae
                ).cpu()
                history_pixels = soft_append_bcthw(
                    current_pixels,
                    history_pixels,
                    overlapped_frames
                )

            if not high_vram:
                unload_complete_models()

            latest_output_filename = os.path.join(
                output_dir,
                f"{job_id}_{total_generated_latent_frames}.mp4"
            )

            save_bcthw_as_mp4(
                history_pixels,
                latest_output_filename,
                fps=fps,
                crf=mp4_crf
            )

            current_total_frames = int(max(0, total_generated_latent_frames * 4 - 3))
            current_seconds = max(0, current_total_frames / fps)

            print(
                f"[{job_id}] Decoded and saved: {latest_output_filename}\n"
                f"[{job_id}] Current latent shape: {real_history_latents.shape}\n"
                f"[{job_id}] Current pixel shape: {history_pixels.shape}\n"
                f"[{job_id}] Total generated frames: {current_total_frames}\n"
                f"[{job_id}] Video length: {current_seconds:.2f} sec"
            )

            if is_start_of_video:
                break

        # prepend original input video pixels
        history_pixels = torch.cat([input_video_pixels, history_pixels], dim=2)

        final_output_filename = os.path.join(
            output_dir,
            f"{job_id}_final.mp4"
        )
        save_bcthw_as_mp4(
            history_pixels,
            final_output_filename,
            fps=fps,
            crf=mp4_crf
        )

        print(f"[{job_id}] Done.")
        print(f"[{job_id}] Final video: {final_output_filename}")
        return final_output_filename

    except Exception as e:
        traceback.print_exc()

        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )
        raise e


def main():
    final_video = run_inference(
        input_video_path=args.video,
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
        resolution=args.resolution,
        no_resize=args.no_resize,
        vae_batch=args.vae_batch,
        num_clean_frames=args.num_clean_frames,
        save_input_frame=args.save_input_frame,
    )

    print(f"Final output video saved to: {final_video}")


if __name__ == "__main__":
    main()