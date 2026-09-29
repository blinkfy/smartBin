#!/usr/bin/env python3
"""下载 LCM-Tiny-SD FP16 ONNX，并复用现有 TAESD（保留 SDXS）。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
MODEL_REPO = "akameswa/lcm-tiny-sd"
ONNX_REPO = "akameswa/lcm-tiny-sd-onnx-fp16"


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def download(repo, revision, filename, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return target
    partial = target.with_suffix(target.suffix + ".part")
    print(f"[download] {filename}", flush=True)
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                expected = response.headers.get("Content-Length")
            if expected and partial.stat().st_size != int(expected):
                raise IOError(f"Incomplete download: {filename}")
            partial.replace(target)
            return target
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models" / "lcm-tiny-sd")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    revisions = (get_json_from_file(manifest_path)["revisions"] if manifest_path.exists() else
                 {repo: get_json(f"https://huggingface.co/api/models/{repo}")["sha"]
                  for repo in (MODEL_REPO, ONNX_REPO)})
    # Downloads stay separate from the live graph until preparation succeeds.
    target = output / "unet" / "model.fp16.onnx"
    external = output / "unet" / "model.onnx_data"
    tasks = [(ONNX_REPO, f"unet/{name}", output / ".download" / "unet" / name)
             for name, prepared in (("model.onnx", target), ("model.onnx_data", external))
             if not prepared.exists()]
    tasks += [(MODEL_REPO, name, output / name) for name in (
        "unet/config.json", "text_encoder/config.json", "text_encoder/model.safetensors",
        "tokenizer/vocab.json", "tokenizer/merges.txt", "tokenizer/tokenizer_config.json",
        "tokenizer/special_tokens_map.json", "scheduler/scheduler_config.json",
    )]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(download, repo, revisions[repo], name, target)
                   for repo, name, target in tasks]
        for future in futures:
            future.result()
    scheduler = get_json_from_file(output / "scheduler" / "scheduler_config.json")
    for key, expected in {"_class_name": "LCMScheduler", "beta_schedule": "scaled_linear",
                          "prediction_type": "epsilon", "thresholding": False,
                          "clip_sample": False, "rescale_betas_zero_snr": False}.items():
        if scheduler.get(key) != expected:
            raise ValueError(f"Unsupported LCM scheduler option: {key}={scheduler.get(key)}")
    # Store only the supported numeric parameters; upstream contains -Infinity.
    supported = {key: scheduler[key] for key in (
        "beta_start", "beta_end", "num_train_timesteps", "original_inference_steps",
        "timestep_scaling", "prediction_type", "beta_schedule",
    )}
    (output / "scheduler_config.json").write_text(json.dumps(supported, indent=2), encoding="utf-8")
    if not target.exists():
        import onnx

        print("[prepare] FP16 ONNX graph and external weights", flush=True)
        model = onnx.load(str(output / ".download" / "unet" / "model.onnx"), load_external_data=False)
        if [item.name for item in model.graph.input] != ["sample", "timestep", "encoder_hidden_states"]:
            raise ValueError("Unexpected LCM UNet inputs")
        shutil.copy2(output / ".download" / "unet" / "model.onnx", target)
        del model
    if not external.exists():
        (output / ".download" / "unet" / "model.onnx_data").replace(external)
    decoder = ROOT / "models" / "sdxs-512-dreamshaper" / "vae_decoder" / "model.fp16.onnx"
    if not decoder.exists():
        raise FileNotFoundError(f"Missing shared TAESD decoder: {decoder}")
    manifest = {
        "model": MODEL_REPO, "revisions": revisions,
        "format": "ONNX FP16 weights with FP32 inputs/outputs",
        "scheduler": "LCMScheduler", "steps": 4, "guidanceScale": 1.0,
        "sharedDecoder": "sdxs-512-dreamshaper/vae_decoder/model.fp16.onnx",
        "latentScalingFactor": 1.0,
        "files": {str(path.relative_to(output)).replace("\\", "/"): {
            "bytes": path.stat().st_size, "sha256": sha256(path)
        } for path in (target, external, output / "text_encoder" / "model.safetensors")},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # Only remove the two known staging assets after the complete graph is saved.
    for name in ("model.onnx", "model.onnx_data"):
        (output / ".download" / "unet" / name).unlink(missing_ok=True)
    print(f"[done] LCM UNet {(target.stat().st_size + external.stat().st_size) / 1024**2:.2f} MiB; SDXS retained", flush=True)


def get_json_from_file(path):
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
