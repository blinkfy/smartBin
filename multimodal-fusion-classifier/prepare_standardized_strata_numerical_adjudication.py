"""Adjudicate the observed batch-sensitive high-IoU mask match."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from mmfc.model import PROJECT_ROOT


ROOT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "standardized-instance-strata"
)
PROTOCOL = ROOT / "PROTOCOL.json"
ADDENDUM1 = ROOT / "RUNTIME_RECOVERY_ADDENDUM.json"
ADDENDUM2 = ROOT / "RUNTIME_CHUNKING_ADDENDUM.json"
BATCH32_RGB = ROOT / "evaluation" / "rgb_yolo.json"
BATCH8_RGB = ROOT / "evaluation-recovery-batch8" / "rgb_yolo.json"
CHUNK_ROOT = ROOT / "evaluation-recovery-batch8-chunk64"
CHUNK_RGB = CHUNK_ROOT / "rgb_yolo.json"
CHUNK_V5 = CHUNK_ROOT / "v5_rgb_fallback.json"
CHUNK_V6 = CHUNK_ROOT / "v6_rgb_fallback.json"
EVALUATOR = PROJECT_ROOT / "evaluate_standardized_instance_strata.py"
DEFAULT_OUTPUT = ROOT / "RUNTIME_NUMERICAL_ADJUDICATION.json"
EXPECTED_HASHES = {
    PROTOCOL: "eab7d649ce0a0b4fddb4e60c58ad9f7060b288f0cee1f842ee5b571826a1fba8",
    ADDENDUM1: "8a0a14333f6facd05d074321fc7d2b6fb008292da00181731e75679621955010",
    ADDENDUM2: "28410e9a4e3d06e3768d9a8a8fdd22954ab8603c454f842c709496623fd7f5f5",
    BATCH32_RGB: "5b1eff4cae2f9c80bdd3b9534f12f5712972382f559e5910e4c33f6f675d4050",
    BATCH8_RGB: "acb8be4455f8149bcbea5ebaf6760e7e85782b344410fb3c4b4cba22a594c8ec",
    CHUNK_RGB: "1e7ff4c7d5b929df75938abe5a9c2dcd57ecc7453fefe16170d3bf626d8f5f55",
    CHUNK_V5: "c034f26b436fe58aabd42799a389afa92df05b96f2e000df3c4d6a703d25ccdc",
    CHUNK_V6: "050d01052937bc59cd9c70beff1587c3933b2b33013d9e800cbd78c9375c7b93",
}
PREVIOUS_EVALUATOR_SHA256 = (
    "1161fb01a64735cb1de8b76a1f47b6b20c88732aa058b4d83d7539421ae4b54e"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze batch-sensitivity adjudication"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


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


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["dimension"]),
        str(row["stratum"]),
        str(row["metric"]),
    )


def observed_sensitivity() -> dict[str, Any]:
    batch32 = _read(BATCH32_RGB)
    batch8 = _read(BATCH8_RGB)
    chunk = _read(CHUNK_RGB)
    if (
        batch32.get("rows") != batch8.get("rows")
        or batch32.get("model", {}).get("prediction_count") != 841
        or batch8.get("model", {}).get("prediction_count") != 841
        or chunk.get("model", {}).get("prediction_count") != 841
    ):
        raise ValueError("un-chunked RGB references are inconsistent")
    reference = {_row_key(row): row for row in batch32["rows"]}
    candidate = {_row_key(row): row for row in chunk["rows"]}
    differences = []
    for key in reference:
        if key not in candidate:
            raise ValueError(f"chunked RGB row is missing: {key}")
        for field in reference[key]:
            if field in {"model"}:
                continue
            left = reference[key][field]
            right = candidate[key][field]
            if left != right:
                differences.append(
                    {
                        "dimension": key[0],
                        "stratum": key[1],
                        "metric": key[2],
                        "field": field,
                        "batch32": left,
                        "chunk64": right,
                        "difference": (
                            float(right) - float(left)
                            if isinstance(left, (int, float))
                            and isinstance(right, (int, float))
                            else None
                        ),
                    }
                )
    allowed = {
        (
            "coco_visible_mask_area",
            "large",
            "mask",
            "mean_recall50_95",
        ),
        (
            "coco_visible_mask_area",
            "large",
            "mask",
            "mean_recall50_95_ci_lower",
        ),
        (
            "coco_visible_mask_area",
            "large",
            "mask",
            "mean_recall50_95_ci_upper",
        ),
        (
            "bbox_overlap_proxy",
            "overlap_high",
            "mask",
            "mean_recall50_95",
        ),
        (
            "bbox_overlap_proxy",
            "overlap_high",
            "mask",
            "mean_recall50_95_ci_lower",
        ),
        (
            "bbox_overlap_proxy",
            "overlap_high",
            "mask",
            "mean_recall50_95_ci_upper",
        ),
    }
    actual = {
        (
            row["dimension"],
            row["stratum"],
            row["metric"],
            row["field"],
        )
        for row in differences
    }
    if actual != allowed:
        raise ValueError("observed batch sensitivity differs from audit")
    point_differences = [
        abs(float(row["difference"]))
        for row in differences
        if row["field"] == "mean_recall50_95"
    ]
    return {
        "prediction_count_equal": True,
        "recall50_rows_exact": all(
            reference[key]["recall50"] == candidate[key]["recall50"]
            for key in reference
        ),
        "recall75_rows_exact": all(
            reference[key]["recall75"] == candidate[key]["recall75"]
            for key in reference
        ),
        "different_field_count": len(differences),
        "differences": differences,
        "maximum_absolute_point_difference": max(point_differences),
        "large_mask_mean_recall_difference": (
            candidate[
                ("coco_visible_mask_area", "large", "mask")
            ]["mean_recall50_95"]
            - reference[
                ("coco_visible_mask_area", "large", "mask")
            ]["mean_recall50_95"]
        ),
        "overlap_high_mask_mean_recall_difference": (
            candidate[("bbox_overlap_proxy", "overlap_high", "mask")][
                "mean_recall50_95"
            ]
            - reference[
                ("bbox_overlap_proxy", "overlap_high", "mask")
            ]["mean_recall50_95"]
        ),
    }


def build_adjudication() -> dict[str, Any]:
    for path, expected in EXPECTED_HASHES.items():
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"adjudication binding drift: {path}")
    revised_hash = _sha256(EVALUATOR)
    if revised_hash == PREVIOUS_EVALUATOR_SHA256:
        raise ValueError("final evaluator adjudication is not implemented")
    sensitivity = observed_sensitivity()
    return {
        "schema_version": 1,
        "status": "frozen_after_batch_sensitivity_before_final_rerun",
        "valid": True,
        "formal_paper_metric": False,
        "observed_batch_sensitivity": sensitivity,
        "adjudication": {
            "selected_runtime": {
                "batch_size": 8,
                "prediction_chunk_size": 64,
            },
            "reason": (
                "The chunked runtime is the only registered runtime that "
                "completed all three models. It preserves prediction count, "
                "R@0.50, and R@0.75, while one high-IoU mask match changes."
            ),
            "exact_batch_invariance_claim": False,
            "required_disclosure": (
                "Report the observed 0.0305 percentage-point large-mask "
                "mR shift and 0.5 percentage-point overlap-high shift."
            ),
            "interpretation_floor": {
                "global_or_large_mask_mean_recall_fraction": abs(
                    sensitivity["large_mask_mean_recall_difference"]
                ),
                "overlap_high_mask_mean_recall_fraction": abs(
                    sensitivity[
                        "overlap_high_mask_mean_recall_difference"
                    ]
                ),
            },
        },
        "source_revision": {
            "previous_evaluator_sha256": PREVIOUS_EVALUATOR_SHA256,
            "revised_evaluator_sha256": revised_hash,
        },
        "bindings": {
            "protocol": _binding(PROTOCOL),
            "runtime_addendum_1": _binding(ADDENDUM1),
            "runtime_addendum_2": _binding(ADDENDUM2),
            "batch32_rgb": _binding(BATCH32_RGB),
            "batch8_rgb": _binding(BATCH8_RGB),
            "chunk64_rgb": _binding(CHUNK_RGB),
            "chunk64_v5": _binding(CHUNK_V5),
            "chunk64_v6": _binding(CHUNK_V6),
            "revised_evaluator": _binding(EVALUATOR),
            "adjudication_builder": _binding(Path(__file__)),
        },
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }


def write_adjudication(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = output.resolve()
    result = build_adjudication()
    if output.is_file():
        existing = _read(output)
        left = dict(existing)
        right = dict(result)
        left.pop("frozen_at", None)
        right.pop("frozen_at", None)
        if left != right:
            raise FileExistsError(
                "refusing to replace a drifted numerical adjudication"
            )
        return existing
    _atomic_text(
        output,
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
    )
    sensitivity = result["observed_batch_sensitivity"]
    _atomic_text(
        output.with_suffix(".md"),
        "\n".join(
            [
                "# 标准化实例分层数值敏感性裁决",
                "",
                "- batch32与batch8非分块结果完全一致。",
                "- chunk64保持预测数、R@0.50和R@0.75完全一致。",
                (
                    "- large-mask mR@0.50:0.95变化："
                    f"{sensitivity['large_mask_mean_recall_difference'] * 100:+.4f}"
                    "个百分点。"
                ),
                (
                    "- overlap-high mask mR变化："
                    f"{sensitivity['overlap_high_mask_mean_recall_difference'] * 100:+.4f}"
                    "个百分点。"
                ),
                "- 最终统一使用batch8/chunk64，并公开上述数值敏感性。",
                "- 不声称批大小逐位不变；本分析仍是描述性、非正式证据。",
                "",
            ]
        ),
    )
    return result


def main() -> None:
    result = write_adjudication(parse_args().output)
    print(
        json.dumps(
            {
                "state": "complete",
                "valid": result["valid"],
                "status": result["status"],
                "maximum_absolute_point_difference": result[
                    "observed_batch_sensitivity"
                ]["maximum_absolute_point_difference"],
                "selected_runtime": result["adjudication"][
                    "selected_runtime"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
