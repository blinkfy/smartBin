#!/usr/bin/env python3
"""用固定 prompt/seed 对比 SDXS Diffusers 与 ONNX 单步结果。"""

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sdxs_text_embedding import get_sdxs_prompt_embedding


MODEL_DIR = BACKEND_DIR / "models" / "sdxs-512-dreamshaper"
DEFAULT_OUTPUT = BACKEND_DIR / "temp" / "sdxs-validation"
DEFAULT_PROMPT = "a transparent plastic water bottle with a red cap, simple product photo, white background"


def mulberry32(seed):
    value = seed & 0xFFFFFFFF

    def random():
        nonlocal value
        value = (value + 0x6D2B79F5) & 0xFFFFFFFF
        result = ((value ^ (value >> 15)) * (1 | value)) & 0xFFFFFFFF
        result ^= (result + (((result ^ (result >> 7)) * (61 | result)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((result ^ (result >> 14)) & 0xFFFFFFFF) / 4294967296.0

    return random


def seeded_noise(shape, sigma, seed):
    random = mulberry32(seed)
    output = np.empty(int(np.prod(shape)), dtype=np.float32)
    for index in range(output.size):
        first = max(random(), 1e-7)
        second = random()
        output[index] = math.sqrt(-2 * math.log(first)) * math.cos(2 * math.pi * second) * sigma
    return output.reshape(shape)


def image_from_tensor(values):
    from PIL import Image

    pixels = np.clip(values[0].transpose(1, 2, 0) / 2 + 0.5, 0, 1)
    return Image.fromarray(np.round(pixels * 255).astype(np.uint8), "RGB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    import onnxruntime as ort
    import torch
    from diffusers import AutoencoderTiny, EulerDiscreteScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    unet_onnx = MODEL_DIR / "unet" / "model.fp16.onnx"
    vae_onnx = MODEL_DIR / "vae_decoder" / "model.fp16.onnx"
    if not unet_onnx.exists() or not vae_onnx.exists():
        raise FileNotFoundError("请先运行 convert_sdxs_to_onnx.py")

    tokenizer = CLIPTokenizer.from_pretrained("IDKiro/sdxs-512-dreamshaper", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        "IDKiro/sdxs-512-dreamshaper", subfolder="text_encoder", torch_dtype=torch.float32
    ).eval()
    tokens = tokenizer(
        args.prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.inference_mode():
        official_embedding = text_encoder(tokens.input_ids)[0].numpy().astype(np.float32)
    streamed_embedding = get_sdxs_prompt_embedding(args.prompt)
    embedding_delta = np.abs(official_embedding - streamed_embedding)
    del text_encoder

    scheduler = EulerDiscreteScheduler.from_pretrained(
        "IDKiro/sdxs-512-dreamshaper", subfolder="scheduler"
    )
    scheduler.set_timesteps(1)
    timestep = float(scheduler.timesteps[0])
    sigma = float(scheduler.sigmas[0])
    sigma_next = float(scheduler.sigmas[1])
    latents = seeded_noise((1, 4, 64, 64), sigma, args.seed)
    model_input = scheduler.scale_model_input(torch.from_numpy(latents.copy()), scheduler.timesteps[0]).numpy()

    export_cache = MODEL_DIR / ".export-cache"
    unet = UNet2DConditionModel.from_pretrained(
        export_cache / "unet", torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True
    ).eval()
    with torch.inference_mode():
        torch_noise = unet(
            torch.from_numpy(model_input).half(),
            torch.tensor([timestep], dtype=torch.float32),
            encoder_hidden_states=torch.from_numpy(streamed_embedding).half(),
            return_dict=False,
        )[0].float().numpy()
    del unet

    unet_session = ort.InferenceSession(str(unet_onnx), providers=["CPUExecutionProvider"])
    onnx_noise = unet_session.run(
        ["out_sample"],
        {
            "sample": model_input.astype(np.float32),
            "timestep": np.array([timestep], dtype=np.float32),
            "encoder_hidden_states": streamed_embedding.astype(np.float32),
        },
    )[0]
    del unet_session
    unet_delta = np.abs(torch_noise - onnx_noise)

    torch_latents = scheduler.step(
        torch.from_numpy(torch_noise), scheduler.timesteps[0], torch.from_numpy(latents.copy()), return_dict=False
    )[0].numpy()
    onnx_latents = latents + onnx_noise * (sigma_next - sigma)
    scheduler_delta = np.abs(torch_latents - onnx_latents)

    vae = AutoencoderTiny.from_pretrained(export_cache / "vae", torch_dtype=torch.float32, use_safetensors=True).half().eval()
    with torch.inference_mode():
        torch_image = vae.decode(torch.from_numpy(torch_latents).half(), return_dict=False)[0].float().numpy()
    del vae
    vae_session = ort.InferenceSession(str(vae_onnx), providers=["CPUExecutionProvider"])
    onnx_image = vae_session.run(["sample"], {"latent_sample": onnx_latents.astype(np.float32)})[0]
    del vae_session
    image_delta = np.abs(torch_image - onnx_image)

    reference_path = output_dir / "diffusers.png"
    onnx_path = output_dir / "onnx.png"
    image_from_tensor(torch_image).save(reference_path)
    image_from_tensor(onnx_image).save(onnx_path)
    pixel_mse = float(np.mean((torch_image - onnx_image) ** 2))
    pixel_psnr = float("inf") if pixel_mse == 0 else 10 * math.log10(4.0 / pixel_mse)
    report = {
        "prompt": args.prompt,
        "seed": args.seed,
        "timestep": timestep,
        "sigma": sigma,
        "sigmaNext": sigma_next,
        "embeddingMaxAbs": float(embedding_delta.max()),
        "embeddingMeanAbs": float(embedding_delta.mean()),
        "unetMaxAbs": float(unet_delta.max()),
        "unetMeanAbs": float(unet_delta.mean()),
        "schedulerMaxAbs": float(scheduler_delta.max()),
        "schedulerMeanAbs": float(scheduler_delta.mean()),
        "imageMaxAbs": float(image_delta.max()),
        "imageMeanAbs": float(image_delta.mean()),
        "imagePsnrDb": pixel_psnr,
        "diffusersImage": str(reference_path),
        "onnxImage": str(onnx_path),
    }
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
