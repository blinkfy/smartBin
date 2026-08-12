"""Audit the final standardized scale and overlap diagnostic evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from evaluate_standardized_instance_strata import (
    DEFAULT_PROTOCOL,
    DEFAULT_RUNTIME_ADDENDUM,
    _validate_protocol,
)
from mmfc.model import PROJECT_ROOT


ROOT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "standardized-instance-strata"
)
DEFAULT_EVALUATION = ROOT / "evaluation-final-chunk64" / "evaluation.json"
DEFAULT_OUTPUT = ROOT / "AUDIT.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit standardized instance-strata evidence"
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--runtime-addendum",
        type=Path,
        default=DEFAULT_RUNTIME_ADDENDUM,
    )
    parser.add_argument(
        "--evaluation", type=Path, default=DEFAULT_EVALUATION
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return (
        path.resolve()
        if path.is_absolute()
        else (PROJECT_ROOT / path).resolve()
    )


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


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


def _paired_lookup(
    rows: list[Mapping[str, Any]],
) -> dict[tuple[str, str, str, str, str], Mapping[str, Any]]:
    return {
        (
            str(row["candidate"]),
            str(row["dimension"]),
            str(row["stratum"]),
            str(row["metric"]),
            str(row["score"]),
        ): row
        for row in rows
    }


def audit(
    protocol_path: Path = DEFAULT_PROTOCOL,
    runtime_addendum_path: Path = DEFAULT_RUNTIME_ADDENDUM,
    evaluation_path: Path = DEFAULT_EVALUATION,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    protocol_path = _resolve(protocol_path)
    runtime_addendum_path = _resolve(runtime_addendum_path)
    evaluation_path = _resolve(evaluation_path)
    output = _resolve(output)
    for path in (
        protocol_path,
        runtime_addendum_path,
        evaluation_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"strata audit input is missing: {path}")
    protocol = _read(protocol_path)
    adjudication = _validate_protocol(
        protocol,
        protocol_path,
        runtime_addendum_path,
    )
    evaluation = _read(evaluation_path)
    errors: list[str] = []
    contract = evaluation.get("target_contract", {})
    models = evaluation.get("models", [])
    rows = evaluation.get("rows", [])
    paired = _paired_lookup(evaluation.get("paired_differences", []))
    batch = evaluation.get("batch_sensitivity", {})
    if (
        evaluation.get("valid") is not True
        or evaluation.get("formal_paper_metric") is not False
        or evaluation.get("evidence_role")
        != "posthoc_standardized_strata_diagnostic"
        or contract.get("source_image_count") != 570
        or contract.get("target_count") != 825
        or contract.get("scale_support")
        != {"small": 11, "medium": 486, "large": 328}
        or contract.get("overlap_support")
        != {"isolated": 717, "overlap_low": 88, "overlap_high": 20}
        or len(models) != 3
        or len(rows) != 36
        or len(evaluation.get("paired_differences", [])) != 72
    ):
        errors.append("final standardized-strata report contract mismatch")
    for model in models:
        if (
            model.get("source_image_count") != 570
            or model.get("target_count") != 825
            or model.get("prediction_chunk_size") != 64
        ):
            errors.append(f"model coverage mismatch: {model.get('model')}")
        if model.get("mode") == "rgb_fallback":
            availability = model.get("sensor_availability", {})
            if (
                availability.get("minimum") != 0.0
                or availability.get("maximum") != 0.0
                or availability.get("mean") != 0.0
            ):
                errors.append(
                    f"fallback availability mismatch: {model.get('model')}"
                )
    observed = adjudication["observed_batch_sensitivity"]
    if (
        batch.get("exact_batch_invariance") is not False
        or batch.get("chunked_metric_rows_reproduced_exactly") is not True
        or batch.get("recall50_rows_exact") is not True
        or batch.get("recall75_rows_exact") is not True
        or batch.get("prediction_count_equal") is not True
        or len(batch.get("point_differences", [])) != 2
        or observed.get("maximum_absolute_point_difference")
        != adjudication["adjudication"]["interpretation_floor"][
            "overlap_high_mask_mean_recall_fraction"
        ]
    ):
        errors.append("batch-sensitivity disclosure mismatch")

    selected = {
        "v5_isolated_mask_mean_recall": paired[
            (
                "v5_rgb_fallback",
                "bbox_overlap_proxy",
                "isolated",
                "mask",
                "mean_recall50_95",
            )
        ],
        "v6_isolated_mask_mean_recall": paired[
            (
                "v6_rgb_fallback",
                "bbox_overlap_proxy",
                "isolated",
                "mask",
                "mean_recall50_95",
            )
        ],
        "v6_small_mask_mean_recall": paired[
            (
                "v6_rgb_fallback",
                "coco_visible_mask_area",
                "small",
                "mask",
                "mean_recall50_95",
            )
        ],
        "v5_overlap_high_mask_mean_recall": paired[
            (
                "v5_rgb_fallback",
                "bbox_overlap_proxy",
                "overlap_high",
                "mask",
                "mean_recall50_95",
            )
        ],
        "v6_overlap_high_mask_mean_recall": paired[
            (
                "v6_rgb_fallback",
                "bbox_overlap_proxy",
                "overlap_high",
                "mask",
                "mean_recall50_95",
            )
        ],
    }
    floor_global = adjudication["adjudication"]["interpretation_floor"][
        "global_or_large_mask_mean_recall_fraction"
    ]
    floor_overlap = adjudication["adjudication"]["interpretation_floor"][
        "overlap_high_mask_mean_recall_fraction"
    ]
    interpretation = {
        "v5_isolated_mask_gain_above_numerical_floor": (
            selected["v5_isolated_mask_mean_recall"]["difference"]
            > floor_global
            and selected["v5_isolated_mask_mean_recall"]["ci_lower"] > 0.0
        ),
        "v6_isolated_mask_gain_above_numerical_floor": (
            selected["v6_isolated_mask_mean_recall"]["difference"]
            > floor_global
            and selected["v6_isolated_mask_mean_recall"]["ci_lower"] > 0.0
        ),
        "v6_small_result_sample_limited_and_inconclusive": (
            selected["v6_small_mask_mean_recall"]["ci_lower"] < 0.0
            < selected["v6_small_mask_mean_recall"]["ci_upper"]
        ),
        "v5_overlap_high_effect_not_above_runtime_floor": (
            abs(
                selected["v5_overlap_high_mask_mean_recall"]["difference"]
            )
            <= floor_overlap
        ),
        "v6_overlap_high_result_sample_limited": True,
    }
    if not all(interpretation.values()):
        errors.append("predefined strata interpretation checks failed")

    result = {
        "schema_version": 1,
        "state": "complete" if not errors else "invalid",
        "valid": not errors,
        "formal_paper_metric": False,
        "evidence_role": "audited_posthoc_standardized_strata_diagnostic",
        "target_contract": contract,
        "selected_paired_results": selected,
        "batch_sensitivity": {
            "large_mask_mean_recall_difference": observed[
                "large_mask_mean_recall_difference"
            ],
            "overlap_high_mask_mean_recall_difference": observed[
                "overlap_high_mask_mean_recall_difference"
            ],
            "exact_batch_invariance": False,
        },
        "interpretation": interpretation,
        "paper_claim_policy": {
            "allowed": [
                (
                    "At fixed conf=0.25 and NMS IoU=0.5, v5/v6 RGB "
                    "fallback improved isolated-instance mask mR relative "
                    "to the RGB checkpoint in this internal diagnostic."
                ),
                (
                    "Only 11 small and 20 high-overlap targets are present; "
                    "their results are sample-limited."
                ),
                (
                    "Chunking changed one high-IoU mask match and its "
                    "numerical sensitivity is disclosed."
                ),
            ],
            "not_allowed": [
                "The system has proven 93.3% small-object accuracy.",
                "The system has proven 95.7% occlusion accuracy.",
                "bbox overlap is a human-annotated occlusion state.",
                "These post-hoc internal diagnostics are formal blind metrics.",
            ],
        },
        "bindings": {
            "protocol": _binding(protocol_path),
            "final_validation_addendum": _binding(
                runtime_addendum_path
            ),
            "evaluation": _binding(evaluation_path),
            "report_markdown": _binding(
                evaluation_path.parent / "REPORT.md"
            ),
            "metrics_csv": _binding(
                evaluation_path.parent / "metrics.csv"
            ),
            "paired_csv": _binding(
                evaluation_path.parent / "paired_differences.csv"
            ),
            "auditor_source": _binding(Path(__file__)),
        },
        "errors": errors,
    }
    _atomic_text(
        output,
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
    )
    v5 = selected["v5_isolated_mask_mean_recall"]
    v6 = selected["v6_isolated_mask_mean_recall"]
    small = selected["v6_small_mask_mean_recall"]
    lines = [
        "# 标准化实例分层证据审计",
        "",
        f"- 审计有效：{'是' if result['valid'] else '否'}。",
        "- 正式论文主指标：否；内部描述性诊断。",
        (
            "- v5/v6相对RGB的isolated mask mR差："
            f"{v5['difference'] * 100:+.3f}"
            f"[{v5['ci_lower'] * 100:+.3f},{v5['ci_upper'] * 100:+.3f}]/"
            f"{v6['difference'] * 100:+.3f}"
            f"[{v6['ci_lower'] * 100:+.3f},{v6['ci_upper'] * 100:+.3f}"
            "]个百分点。"
        ),
        (
            "- v6 small mask mR差："
            f"{small['difference'] * 100:+.3f}"
            f"[{small['ci_lower'] * 100:+.3f},{small['ci_upper'] * 100:+.3f}]"
            "个百分点；仅11个实例。"
        ),
        (
            "- batch/chunk数值敏感性：large/high-overlap mask mR "
            f"{observed['large_mask_mean_recall_difference'] * 100:+.4f}/"
            f"{observed['overlap_high_mask_mean_recall_difference'] * 100:+.4f}"
            "个百分点。"
        ),
        "",
        (
            "框重叠仍只是潜在遮挡代理；这些结果不能替代人工遮挡标签、"
            "外部盲测或原专利性能数字。"
        ),
        "",
    ]
    _atomic_text(output.with_suffix(".md"), "\n".join(lines))
    if errors:
        raise ValueError("; ".join(errors))
    return result


def main() -> None:
    args = parse_args()
    result = audit(
        args.protocol,
        args.runtime_addendum,
        args.evaluation,
        args.output,
    )
    print(
        json.dumps(
            {
                "state": result["state"],
                "valid": result["valid"],
                "formal_paper_metric": result["formal_paper_metric"],
                "target_contract": result["target_contract"],
                "interpretation": result["interpretation"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
