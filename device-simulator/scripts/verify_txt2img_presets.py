#!/usr/bin/env python3
"""Verify simulator preset prompts, UI mapping, cache keys, and embedding files."""

from pathlib import Path
import re

import numpy as np

from generate_txt2img_presets import PRESETS


BACKEND_DIR = Path(__file__).resolve().parents[1]
EXPECTED = {
    "plastic-bottle": (
        "塑料瓶",
        "single empty plastic bottle without any others, pure white background",
        "plastic-bottle.f32",
    ),
    "medicine-blister": (
        "废药板",
        "pill transparent tablet with pills, without any others, pure white background",
        "medicine-blister.f32",
    ),
    "vegetable": (
        "蔬菜",
        "a vegetable, without any other objects, pure white background",
        "vegetable.f32",
    ),
}
CACHE_VERSION = "sdxs-simple-white-v3"


def main():
    text_to_image = (BACKEND_DIR / "textToImage.js").read_text(encoding="utf-8")
    ui = (BACKEND_DIR / "device-simulator-ui.html").read_text(encoding="utf-8")
    browser_presets = {}
    for preset_id in EXPECTED:
        block_match = re.search(
            rf"id: '{re.escape(preset_id)}',(?P<body>.*?)(?=\n\s*\}}\s*,?)",
            text_to_image,
            re.DOTALL,
        )
        assert block_match, f"{preset_id}: browser preset block missing"
        body = block_match.group("body")
        label = re.search(r"label: '([^']+)'", body).group(1)
        prompt = re.search(r"prompt: '([^']+)'", body).group(1)
        url = re.search(r"embeddingUrl: '([^']+)'", body).group(1)
        browser_presets[preset_id] = (label, prompt, url)

    assert set(PRESETS) == set(EXPECTED), f"generator preset IDs differ: {set(PRESETS)}"
    assert set(browser_presets) == set(EXPECTED), f"browser preset IDs differ: {set(browser_presets)}"

    for preset_id, (label, prompt, filename) in EXPECTED.items():
        assert PRESETS[preset_id] == prompt, f"{preset_id}: generator prompt differs"
        assert browser_presets[preset_id] == (
            label,
            prompt,
            f"/txt2img-presets/{filename}?v={CACHE_VERSION}",
        ), f"{preset_id}: browser preset differs"
        assert f'data-preset="{preset_id}">{label}</button>' in ui, f"{preset_id}: UI button differs"

        embedding_path = BACKEND_DIR / "text-to-image-presets" / filename
        values = np.fromfile(embedding_path, dtype=np.float32)
        assert embedding_path.stat().st_size == 77 * 768 * 4, f"{preset_id}: byte size differs"
        assert values.size == 77 * 768 and np.isfinite(values).all(), f"{preset_id}: invalid embedding"

    assert not (BACKEND_DIR / "text-to-image-presets" / "used-battery.f32").exists(), "stale battery embedding"
    print("TXT2IMG_PRESETS_OK")


if __name__ == "__main__":
    main()
