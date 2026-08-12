"""Freeze the standardized scale and overlap diagnostic before inference."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from evaluate_standardized_instance_strata import (
    IOU_THRESHOLDS,
    REPORT_ROOT,
    load_target_contract,
    tree_binding,
)
from mmfc.model import PROJECT_ROOT


PROTOCOL_NAME = "standardized_instance_scale_overlap_diagnostic_v1"
DEFAULT_OUTPUT = REPORT_ROOT / "PROTOCOL.json"
MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "yolo-seg-unique-linked"
    / "manifests"
    / "test.jsonl"
)
DATA_ROOT = PROJECT_ROOT / "data" / "yolo-seg-unique-linked"
MODELS = (
    {
        "name": "rgb_yolo",
        "mode": "rgb_detector",
        "checkpoint": (
            PROJECT_ROOT
            / "checkpoints"
            / "yolo-seg"
            / "yolo11s-seg-qwen-split"
            / "weights"
            / "best.pt"
        ),
    },
    {
        "name": "v5_rgb_fallback",
        "mode": "rgb_fallback",
        "checkpoint": (
            PROJECT_ROOT
            / "checkpoints"
            / "joint-multimodal-seg-spatial"
            / "yolo11s-seg-joint-spatial-contrastive-rule-dev-seed42"
            / "weights"
            / "best.pt"
        ),
    },
    {
        "name": "v6_rgb_fallback",
        "mode": "rgb_fallback",
        "checkpoint": (
            PROJECT_ROOT
            / "checkpoints"
            / "joint-multimodal-seg-patent"
            / "yolo11s-seg-joint-patent-stack-rule-dev-seed42"
            / "weights"
            / "best.pt"
        ),
    },
)
SOURCE_PATHS = {
    "protocol_builder": Path(__file__),
    "evaluator": PROJECT_ROOT / "evaluate_standardized_instance_strata.py",
    "test_manifest": MANIFEST,
    "dataset_export": DATA_ROOT / "export_report.json",
    "dataset_yaml": DATA_ROOT / "dataset.yaml",
    "rgb_checkpoint": MODELS[0]["checkpoint"],
    "v5_checkpoint": MODELS[1]["checkpoint"],
    "v6_checkpoint": MODELS[2]["checkpoint"],
    "v5_protocol": (
        PROJECT_ROOT
        / "reports"
        / "qwen-material"
        / "joint-multimodal-seg-spatial"
        / "yolo11s-seg-joint-spatial-contrastive-rule-dev-seed42"
        / "PROTOCOL.json"
    ),
    "v6_protocol": (
        PROJECT_ROOT
        / "reports"
        / "qwen-material"
        / "joint-multimodal-seg-patent"
        / "yolo11s-seg-joint-patent-stack-rule-dev-seed42"
        / "PROTOCOL.json"
    ),
    "previous_relative_strata_report": (
        PROJECT_ROOT
        / "reports"
        / "qwen-material"
        / "yolo-seg"
        / "integrated"
        / "stratified"
        / "evaluation.json"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze standardized instance-strata protocol"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return (
        path.resolve()
        if path.is_absolute()
        else (PROJECT_ROOT / path).resolve()
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.relative_to(PROJECT_ROOT).as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def build_protocol() -> dict[str, Any]:
    missing = [
        name for name, path in SOURCE_PATHS.items() if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "standardized-strata inputs missing: " + ", ".join(missing)
        )
    _, target_contract = load_target_contract(MANIFEST)
    if (
        target_contract["source_image_count"] != 570
        or target_contract["target_count"] != 825
    ):
        raise ValueError("test target counts differ from the frozen dataset")
    return {
        "schema_version": 1,
        "protocol": PROTOCOL_NAME,
        "status": "frozen_before_new_prediction_run",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "formal_paper_metric": False,
        "evidence_role": "posthoc_standardized_strata_diagnostic",
        "purpose": (
            "Add a standardized visible-mask-area analysis and a stronger "
            "bbox-overlap severity proxy without replacing the previously "
            "frozen relative-quartile analysis."
        ),
        "dataset": {
            "manifest": MANIFEST.relative_to(PROJECT_ROOT).as_posix(),
            "split": "test",
            "source_grouped": True,
            "previously_inspected_internal_test": True,
        },
        "target_contract": target_contract,
        "strata": {
            "coco_visible_mask_area": {
                "measure": (
                    "rasterized visible ground-truth polygon area in "
                    "original-image pixels"
                ),
                "thresholds": {
                    "small": "area < 32^2",
                    "medium": "32^2 <= area < 96^2",
                    "large": "area >= 96^2",
                },
                "terminology_boundary": (
                    "COCO area thresholds are borrowed only for strata; "
                    "the reported target-centric recall is not COCOeval AP."
                ),
            },
            "bbox_overlap_proxy": {
                "measure": (
                    "maximum intersection area with another GT bbox divided "
                    "by the current target bbox area"
                ),
                "thresholds": {
                    "isolated": "ratio == 0",
                    "overlap_low": "0 < ratio < 0.25",
                    "overlap_high": "ratio >= 0.25",
                },
                "terminology_boundary": (
                    "This is a crowding or potential-occlusion proxy, not a "
                    "human occlusion label or amodal visibility estimate."
                ),
            },
        },
        "models": [
            {
                "name": str(model["name"]),
                "mode": str(model["mode"]),
                "checkpoint": model["checkpoint"]
                .relative_to(PROJECT_ROOT)
                .as_posix(),
                "checkpoint_sha256": _sha256(model["checkpoint"]),
            }
            for model in MODELS
        ],
        "inference": {
            "image_size": 640,
            "confidence": 0.25,
            "nms_iou": 0.5,
            "max_detections": 100,
            "batch_size": 32,
            "half": False,
            "retina_masks": True,
            "sidecar_inputs": False,
        },
        "iou_thresholds": list(IOU_THRESHOLDS),
        "metrics": {
            "matching": (
                "class-aware one-to-one greedy matching by descending IoU"
            ),
            "reported": [
                "target recall at IoU 0.50",
                "target recall at IoU 0.75",
                "mean target recall over IoU 0.50:0.95",
            ],
            "modalities": ["box", "visible instance mask"],
            "false_positive_policy": (
                "No stratum precision or AP is claimed because predictions "
                "outside a target stratum do not have an unambiguous GT "
                "stratum assignment."
            ),
        },
        "bootstrap": {
            "method": "source_image_cluster_percentile_bootstrap",
            "repeats": 2000,
            "confidence": 0.95,
            "seed": 20260729,
            "paired_across_models": True,
        },
        "limitations": [
            (
                "This is a post-hoc diagnostic on an already inspected "
                "internal test set, not a formal blind performance endpoint."
            ),
            (
                "Only 11 targets satisfy the visible-mask small threshold "
                "and only 20 satisfy overlap_high, so those estimates are "
                "sample-limited."
            ),
            (
                "Visible polygons do not annotate hidden object extent; "
                "overlap is not a physical occlusion measurement."
            ),
            (
                "v5 and v6 are evaluated in RGB fallback mode without an "
                "external sidecar."
            ),
            (
                "The fixed confidence threshold excludes lower-confidence "
                "predictions and the output is target recall, not COCO AP."
            ),
        ],
        "source_bindings": {
            name: _binding(path) for name, path in SOURCE_PATHS.items()
        },
        "tree_bindings": {
            "test_images": tree_binding(
                DATA_ROOT / "images" / "test", "*"
            ),
            "test_labels": tree_binding(
                DATA_ROOT / "labels" / "test", "*.txt"
            ),
        },
    }


def write_protocol(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = _resolve(output)
    protocol = build_protocol()
    if output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8-sig"))
        left = dict(existing)
        right = dict(protocol)
        left.pop("frozen_at", None)
        right.pop("frozen_at", None)
        if left != right:
            raise FileExistsError(
                "refusing to replace a drifted frozen strata protocol"
            )
        return existing
    _atomic_text(
        output,
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
    )
    contract = protocol["target_contract"]
    markdown = [
        "# 标准化实例分层诊断协议",
        "",
        "- 状态：新推理前冻结；正式论文主指标：否。",
        (
            "- COCO可见掩膜面积 small/medium/large："
            f"{contract['scale_support']}。"
        ),
        (
            "- 框重叠 isolated/low/high："
            f"{contract['overlap_support']}。"
        ),
        "- 模型：RGB YOLO、v5 RGB fallback、v6 RGB fallback。",
        "- 固定推理：imgsz=640、conf=0.25、NMS IoU=0.5、FP32。",
        "- 指标：box/mask R@0.50、R@0.75、mR@0.50:0.95。",
        "",
        (
            "small和overlap_high支持数很少；框重叠只是潜在遮挡代理，"
            "不能写成真实遮挡标签。"
        ),
        "",
    ]
    _atomic_text(output.with_suffix(".md"), "\n".join(markdown))
    return protocol


def main() -> None:
    protocol = write_protocol(parse_args().output)
    print(
        json.dumps(
            {
                "state": "complete",
                "valid": True,
                "formal_paper_metric": protocol["formal_paper_metric"],
                "protocol": protocol["protocol"],
                "target_contract": protocol["target_contract"],
                "model_count": len(protocol["models"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
