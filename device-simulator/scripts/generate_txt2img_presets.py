#!/usr/bin/env python3
"""生成设备模拟器内置 SDXS prompt embedding（二进制 float32）。"""

from pathlib import Path
import sys


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sdxs_text_embedding import get_sdxs_prompt_embedding


OUTPUT_DIR = BACKEND_DIR / "text-to-image-presets"
PRESETS = {
    "plastic-bottle": "single empty plastic bottle without any others, pure white background",
    "medicine-blister": "pill transparent tablet with pills, without any others, pure white background",
    "vegetable": "a vegetable, without any other objects, pure white background",
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for preset_id, prompt in PRESETS.items():
        embedding = get_sdxs_prompt_embedding(prompt)
        if embedding.shape != (1, 77, 768):
            raise RuntimeError(f"{preset_id} embedding shape error: {embedding.shape}")
        target = OUTPUT_DIR / f"{preset_id}.f32"
        target.write_bytes(embedding.tobytes(order="C"))
        print(f"generated {target.name}: {target.stat().st_size} bytes")

    stale_battery = OUTPUT_DIR / "used-battery.f32"
    if stale_battery.exists():
        stale_battery.unlink()
        print(f"removed stale {stale_battery.name}")


if __name__ == "__main__":
    main()
