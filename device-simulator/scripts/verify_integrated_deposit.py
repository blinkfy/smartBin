#!/usr/bin/env python3
"""Verify the generated-image, YOLO recognition, and virtual-deposit integration."""

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_simulator_module():
    simulator_path = next(path for path in ROOT.glob("*.py") if not path.name.isascii())
    spec = importlib.util.spec_from_file_location("device_simulator", simulator_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    simulator = load_simulator_module()
    expected = {
        "bottle": ("recyclable", "塑料瓶"),
        "drug": ("hazardous", "废药板"),
        "battery": ("hazardous", "废电池"),
        "mooli": ("kitchen", "萝卜"),
        "tomato": ("kitchen", "番茄"),
        "stone": ("residual", "石块"),
    }
    for class_name, (category, item) in expected.items():
        mapped = simulator.map_yolo_detection(class_name)
        assert mapped == {"category": category, "item": item}, (class_name, mapped)
        assert item in simulator.WASTE_CATEGORIES[category]["items"], (class_name, item)

    text_to_image = (ROOT / "textToImage.js").read_text(encoding="utf-8")
    assert "expectedYoloClasses" not in text_to_image
    assert "category: 'recyclable'" in text_to_image
    assert "category: 'hazardous'" in text_to_image
    assert "category: 'kitchen'" in text_to_image

    page = (ROOT / "device-simulator-ui.html").read_text(encoding="utf-8")
    assert 'id="depositVision"' in page
    assert 'id="recognitionResult"' in page
    assert 'id="recognitionDetail"' in page
    assert "YOLO · 自定义启用" in page
    assert '<section class="panel text-image-panel"' not in page
    assert "async function recognizeGeneratedImage" in page
    assert "fetch('/yolo-recognize'" in page
    assert "source: 'preset-label'" in page
    assert "applyPresetLabel(preset)" in page
    assert "preset-conflict" not in page
    assert "compressImageForUpload(appState.textImageBlob)" in page
    assert "maxDimension: 384, quality: 0.58" in page
    assert "maxDimension: 256, quality: 0.42" in page
    assert "image_Base64: uploadImage.dataUrl" in page
    assert "window.setInterval(fetchStatus, 1000)" not in page
    assert "scheduleStatusRefresh(delay)" in page
    assert "new AbortController()" in page
    assert "二维码尚未生成" in page
    assert "recognitionConflict" in page

    simulator_source = next(path for path in ROOT.glob("*.py") if not path.name.isascii()).read_text(encoding="utf-8")
    assert 'if path == "/yolo-recognize"' in simulator_source
    assert '"recognition_source"' in simulator_source
    assert '"detected_class"' in simulator_source
    assert '"image_Base64": image_base64 or None' in simulator_source
    print("INTEGRATED_DEPOSIT_OK")


if __name__ == "__main__":
    main()
