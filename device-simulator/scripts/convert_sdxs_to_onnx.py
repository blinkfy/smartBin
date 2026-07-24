#!/usr/bin/env python3
"""将 IDKiro/sdxs-512-dreamshaper 的 UNet 与 TAESD Decoder 导出为 WebGPU ONNX。"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys

import numpy as np


REPO_ID = "IDKiro/sdxs-512-dreamshaper"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "models" / "sdxs-512-dreamshaper"
TOKENIZER_FILES = (
    "tokenizer/vocab.json",
    "tokenizer/merges.txt",
    "tokenizer/tokenizer_config.json",
    "tokenizer/special_tokens_map.json",
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_safetensors_header(path):
    handle = open(path, "rb")
    header_size = struct.unpack("<Q", handle.read(8))[0]
    header = json.loads(handle.read(header_size))
    return handle, header, 8 + header_size


def stream_fp16_safetensors(source_path, target_path):
    """逐张量将 F32 safetensors 转为 F16，避免同时占用数 GB 内存。"""
    source, header, source_data_start = read_safetensors_header(source_path)
    tensor_entries = [(name, value) for name, value in header.items() if name != "__metadata__"]
    converted_header = {}
    if "__metadata__" in header:
        converted_header["__metadata__"] = header["__metadata__"]

    offset = 0
    for name, descriptor in tensor_entries:
        start, end = descriptor["data_offsets"]
        source_size = end - start
        dtype = descriptor["dtype"]
        target_size = source_size // 2 if dtype == "F32" else source_size
        converted_header[name] = {
            "dtype": "F16" if dtype == "F32" else dtype,
            "shape": descriptor["shape"],
            "data_offsets": [offset, offset + target_size],
        }
        offset += target_size

    encoded_header = json.dumps(converted_header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    encoded_header += b" " * ((8 - len(encoded_header) % 8) % 8)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "wb") as target:
        target.write(struct.pack("<Q", len(encoded_header)))
        target.write(encoded_header)
        for name, descriptor in tensor_entries:
            start, end = descriptor["data_offsets"]
            source.seek(source_data_start + start)
            remaining = end - start
            dtype = descriptor["dtype"]
            while remaining:
                item_size = 4 if dtype == "F32" else 1
                chunk_size = min(16 * 1024 * 1024, remaining)
                chunk_size -= chunk_size % item_size
                raw = source.read(chunk_size)
                if dtype == "F32":
                    raw = np.frombuffer(raw, dtype=np.float32).astype(np.float16).tobytes()
                target.write(raw)
                remaining -= chunk_size
    source.close()


def export_unet(source_config, source_weights, output_path, export_cache):
    import torch
    from diffusers import UNet2DConditionModel
    from diffusers.models.attention_processor import AttnProcessor

    fp16_dir = export_cache / "unet"
    fp16_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_config, fp16_dir / "config.json")
    fp16_weights = fp16_dir / "diffusion_pytorch_model.safetensors"
    if not fp16_weights.exists():
        print("[convert] streaming UNet weights F32 -> F16")
        stream_fp16_safetensors(source_weights, fp16_weights)

    print("[convert] loading FP16 SDXS UNet")
    unet = UNet2DConditionModel.from_pretrained(
        fp16_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).eval()
    unet.set_attn_processor(AttnProcessor())

    class UNetWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, sample, timestep, encoder_hidden_states):
            result = self.model(
                sample.to(torch.float16),
                timestep,
                encoder_hidden_states=encoder_hidden_states.to(torch.float16),
                return_dict=False,
            )[0]
            return result.float()

    wrapper = UNetWrapper(unet).eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[convert] exporting UNet -> {output_path}")
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (
                torch.zeros((1, 4, 64, 64), dtype=torch.float32),
                torch.tensor([999.0], dtype=torch.float32),
                torch.zeros((1, 77, 768), dtype=torch.float32),
            ),
            output_path,
            input_names=["sample", "timestep", "encoder_hidden_states"],
            output_names=["out_sample"],
            opset_version=17,
            do_constant_folding=True,
        )
    del wrapper, unet


def export_vae_decoder(source_dir, output_path):
    import torch
    from diffusers import AutoencoderTiny

    print("[convert] loading TAESD decoder")
    vae = AutoencoderTiny.from_pretrained(source_dir, torch_dtype=torch.float32, use_safetensors=True).half().eval()

    class DecoderWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, latent_sample):
            return self.model.decode(latent_sample.to(torch.float16), return_dict=False)[0].float()

    wrapper = DecoderWrapper(vae).eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[convert] exporting TAESD decoder -> {output_path}")
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            torch.zeros((1, 4, 8, 8), dtype=torch.float32),
            output_path,
            input_names=["latent_sample"],
            output_names=["sample"],
            dynamic_axes={
                "latent_sample": {2: "latent_height", 3: "latent_width"},
                "sample": {2: "image_height", 3: "image_width"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
    del wrapper, vae


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-export-cache", action="store_true")
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError("请先安装 huggingface_hub、diffusers、torch、onnx") from error

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    export_cache = output_dir / ".export-cache"
    export_cache.mkdir(parents=True, exist_ok=True)

    print(f"[download] source model: {REPO_ID}")
    unet_config = Path(hf_hub_download(REPO_ID, "unet/config.json"))
    unet_weights = Path(hf_hub_download(REPO_ID, "unet/diffusion_pytorch_model.safetensors"))
    vae_config = Path(hf_hub_download(REPO_ID, "vae/config.json"))
    vae_weights = Path(hf_hub_download(REPO_ID, "vae/diffusion_pytorch_model.safetensors"))
    scheduler_config = Path(hf_hub_download(REPO_ID, "scheduler/scheduler_config.json"))
    text_encoder_weights = Path(hf_hub_download(REPO_ID, "text_encoder/model.safetensors"))

    for relative_path in TOKENIZER_FILES:
        source = Path(hf_hub_download(REPO_ID, relative_path))
        target = output_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    text_encoder_target = output_dir / "text_encoder" / "model.safetensors"
    text_encoder_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(text_encoder_weights, text_encoder_target)
    shutil.copy2(scheduler_config, output_dir / "scheduler_config.json")

    vae_source = export_cache / "vae"
    vae_source.mkdir(parents=True, exist_ok=True)
    shutil.copy2(vae_config, vae_source / "config.json")
    shutil.copy2(vae_weights, vae_source / "diffusion_pytorch_model.safetensors")

    unet_onnx = output_dir / "unet" / "model.fp16.onnx"
    vae_onnx = output_dir / "vae_decoder" / "model.fp16.onnx"
    if args.force or not unet_onnx.exists():
        export_unet(unet_config, unet_weights, unet_onnx, export_cache)
    if args.force or not vae_onnx.exists():
        export_vae_decoder(vae_source, vae_onnx)

    manifest = {
        "model": REPO_ID,
        "revision": "76f720262bb051da75666b22c902a78c8e16c763",
        "format": "ONNX FP16 weights with FP32 inputs/outputs",
        "promptEmbeddingShape": [1, 77, 768],
        "scheduler": "EulerDiscreteScheduler, trailing, one step",
        "timestep": 999,
        "vae": "AutoencoderTiny (TAESD), scaling_factor=1.0",
        "files": {},
    }
    for path in (unet_onnx, vae_onnx, text_encoder_target):
        manifest["files"][str(path.relative_to(output_dir)).replace("\\", "/")] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.keep_export_cache:
        shutil.rmtree(export_cache, ignore_errors=True)
    print("[done] SDXS browser assets ready")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[error] {error}", file=sys.stderr)
        raise
