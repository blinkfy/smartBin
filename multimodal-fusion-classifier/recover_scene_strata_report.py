"""Recover the missing scene-strata Markdown report without recomputing metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_ROOT = PROJECT_ROOT / "reports" / "qwen-material"
DEFAULT_OUTPUT_DIR = REPORTS_ROOT / "yolo-seg" / "integrated" / "stratified"
DEFAULT_INCIDENT = (
    REPORTS_ROOT / "yolo-seg" / "SCENE_STRATA_REPORTING_INCIDENT.json"
)
DEFAULT_ADDENDUM = (
    REPORTS_ROOT
    / "yolo-seg"
    / "SCENE_STRATA_REPORTING_PROTOCOL_ADDENDUM.json"
)
DEFAULT_OUTCOME = (
    REPORTS_ROOT
    / "yolo-seg"
    / "SCENE_STRATA_REPORTING_RECOVERY_OUTCOME.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate only the missing scene-strata Markdown report from "
            "the already persisted evaluation JSON"
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--incident", type=Path, default=DEFAULT_INCIDENT)
    parser.add_argument("--addendum", type=Path, default=DEFAULT_ADDENDUM)
    parser.add_argument("--outcome", type=Path, default=DEFAULT_OUTCOME)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


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


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _validate_binding(
    binding: Mapping[str, Any],
    *,
    label: str,
) -> Path:
    path = _resolve(Path(str(binding.get("path", ""))))
    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"{label} binding escapes project root") from exc
    if not path.is_file():
        raise FileNotFoundError(f"{label} binding is missing: {path}")
    if path.stat().st_size != binding.get("size_bytes"):
        raise ValueError(f"{label} binding size mismatch")
    if _sha256(path) != binding.get("sha256"):
        raise ValueError(f"{label} binding hash mismatch")
    return path


def _validate_recovery_inputs(
    *,
    output_dir: Path,
    incident_path: Path,
    addendum_path: Path,
) -> tuple[dict[str, Any], Path, Path, Path]:
    incident = _read_json(incident_path)
    addendum = _read_json(addendum_path)
    if (
        incident.get("schema_version") != 1
        or incident.get("state") != "complete"
        or incident.get("valid") is not True
        or incident.get("failure", {}).get("exception_type") != "KeyError"
        or incident.get("partial_outputs", {}).get("row_count") != 42
    ):
        raise ValueError("scene-strata reporting incident is not valid")
    if (
        addendum.get("schema_version") != 1
        or addendum.get("status")
        != "frozen_after_metrics_complete_before_report_recovery"
        or addendum.get("valid") is not True
        or addendum.get("errors") != []
    ):
        raise ValueError("scene-strata reporting addendum is not valid")
    if addendum.get("incident", {}).get("sha256") != _sha256(incident_path):
        raise ValueError("reporting addendum incident binding mismatch")
    invariants = addendum.get("invariants", {})
    if (
        invariants.get("metric_recomputation") is not False
        or invariants.get("training_rerun") is not False
        or invariants.get("detector_rerun") is not False
        or invariants.get("expected_row_count") != 42
        or invariants.get("allowed_new_files")
        != [
            "REPORT.md",
            "SCENE_STRATA_REPORTING_RECOVERY_OUTCOME.json",
            "SCENE_STRATA_REPORTING_RECOVERY_OUTCOME.md",
        ]
    ):
        raise ValueError("reporting addendum invariants mismatch")
    bindings = addendum.get("source_bindings", {})
    if not isinstance(bindings, Mapping):
        raise ValueError("reporting addendum source bindings are missing")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"reporting addendum binding is invalid: {name}")
        _validate_binding(binding, label=f"addendum.{name}")

    evaluation_path = output_dir / "evaluation.json"
    metrics_path = output_dir / "metrics.csv"
    report_path = output_dir / "REPORT.md"
    for name, path in (
        ("partial_evaluation", evaluation_path),
        ("partial_metrics", metrics_path),
    ):
        binding = bindings.get(name)
        if not isinstance(binding, Mapping):
            raise ValueError(f"reporting addendum lacks {name} binding")
        if _resolve(Path(str(binding["path"]))) != path:
            raise ValueError(f"reporting addendum {name} path mismatch")
        _validate_binding(binding, label=name)
    return addendum, evaluation_path, metrics_path, report_path


def _format_interval(row: Mapping[str, Any], metric: str) -> str:
    value = row[metric]
    if metric == "classification_accuracy_on_localized":
        interval_prefix = "classification_accuracy"
    else:
        interval_prefix = metric
    lower = row[f"{interval_prefix}_ci_lower"]
    upper = row[f"{interval_prefix}_ci_upper"]
    if value is None or lower is None or upper is None:
        return "NA"
    return (
        f"{100.0 * float(value):.2f}% "
        f"[{100.0 * float(lower):.2f}%,"
        f"{100.0 * float(upper):.2f}%]"
    )


def _render_report(report: Mapping[str, Any]) -> str:
    rows = list(report.get("rows", []))
    if len(rows) != 42:
        raise ValueError(f"scene-strata row count mismatch: {len(rows)}")
    lines = [
        "# 整图分层辅助评估",
        "",
        "该分析在 detector 训练前冻结分层定义，只复用整图评估的同一批预测。",
        "",
        "| Conf | 类别来源 | 维度 | 分层 | N | 定位 Recall [95% CI] | "
        "匹配分类 Acc [95% CI] | 端到端 Recall [95% CI] |",
        "|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {float(row['confidence_threshold']):.2f} | "
            f"{row['prediction_source']} | {row['dimension']} | "
            f"{row['stratum']} | {int(row['support'])} | "
            f"{_format_interval(row, 'localization_recall')} | "
            f"{_format_interval(row, 'classification_accuracy_on_localized')} | "
            f"{_format_interval(row, 'end_to_end_recall')} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- `relative_small/medium/large` 是固定 test 标注框面积的四分位分层，"
            "不是 COCO 小目标定义；",
            "- `bbox_overlap` 仅表示标注框有正面积交叠，是拥挤/潜在遮挡代理，"
            "不是真实遮挡标签；",
            "- 单次 detector 检查点和固定内部 test 不支持外部部署外推；",
            "- 三种类别来源共享定位结果，定位 Recall 重复列出仅便于同表阅读。",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_metrics_csv(metrics_path: Path) -> int:
    with metrics_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 42:
        raise ValueError(f"scene-strata CSV row count mismatch: {len(rows)}")
    return len(rows)


def recover_scene_strata_report(
    *,
    output_dir: Path,
    incident_path: Path,
    addendum_path: Path,
    outcome_path: Path,
) -> dict[str, Any]:
    output_dir = _resolve(output_dir)
    incident_path = _resolve(incident_path)
    addendum_path = _resolve(addendum_path)
    outcome_path = _resolve(outcome_path)
    addendum, evaluation_path, metrics_path, report_path = (
        _validate_recovery_inputs(
            output_dir=output_dir,
            incident_path=incident_path,
            addendum_path=addendum_path,
        )
    )
    if report_path.exists() or outcome_path.exists():
        raise FileExistsError(
            "refusing to overwrite an existing reporting recovery artifact"
        )

    before_hashes = {
        "evaluation_json": _sha256(evaluation_path),
        "metrics_csv": _sha256(metrics_path),
    }
    evaluation = _read_json(evaluation_path)
    if (
        evaluation.get("protocol")
        != "preregistered_scene_stratified_analysis"
        or evaluation.get("dataset", {}).get("source_image_count") != 570
        or evaluation.get("dataset", {}).get("ground_truth_object_count") != 825
    ):
        raise ValueError("scene-strata evaluation identity mismatch")
    _validate_metrics_csv(metrics_path)
    _atomic_write(report_path, _render_report(evaluation))
    after_hashes = {
        "evaluation_json": _sha256(evaluation_path),
        "metrics_csv": _sha256(metrics_path),
    }
    if after_hashes != before_hashes:
        raise RuntimeError("metric artifacts changed during report recovery")

    outcome = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "recovery": {
            "scope": "generate_missing_markdown_from_persisted_metrics",
            "metric_recomputation": False,
            "training_rerun": False,
            "detector_rerun": False,
            "row_count": len(evaluation["rows"]),
            "metric_hashes_unchanged": True,
            "before_hashes": before_hashes,
            "after_hashes": after_hashes,
        },
        "incident": {
            "path": incident_path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": _sha256(incident_path),
        },
        "addendum": {
            "path": addendum_path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": _sha256(addendum_path),
        },
        "source_bindings": {
            "evaluation": _binding(evaluation_path),
            "metrics": _binding(metrics_path),
            "report": _binding(report_path),
            "incident": _binding(incident_path),
            "addendum": _binding(addendum_path),
            "recovery_source": _binding(Path(__file__)),
        },
        "errors": [],
    }
    _atomic_write(
        outcome_path,
        json.dumps(outcome, ensure_ascii=False, indent=2) + "\n",
    )
    markdown_path = outcome_path.with_suffix(".md")
    _atomic_write(
        markdown_path,
        "\n".join(
            [
                "# Scene-strata reporting recovery outcome",
                "",
                "- State: complete",
                "- Metric recomputation: no",
                "- Training or detector rerun: no",
                f"- Persisted rows: {len(evaluation['rows'])}",
                "- `evaluation.json` and `metrics.csv` hashes: unchanged",
                "- New presentation artifact: `REPORT.md`",
                "",
            ]
        ),
    )
    return {
        "state": "complete",
        "valid": True,
        "row_count": len(evaluation["rows"]),
        "outcome": outcome_path.as_posix(),
        "report": report_path.as_posix(),
        "addendum_sha256": _sha256(addendum_path),
    }


def main() -> None:
    args = parse_args()
    result = recover_scene_strata_report(
        output_dir=args.output_dir,
        incident_path=args.incident,
        addendum_path=args.addendum,
        outcome_path=args.outcome,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
