"""Build paper-ready tables from a strictly audited evidence bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from plot_paper_tables import build_paper_figures


PROJECT_ROOT = Path(__file__).resolve().parent
MAIN_MODELS = (
    ("full", "旧完整基线", "test/evaluation.json"),
    (
        "full_quality_aware_prior",
        "v2b 方差先验",
        "full_quality_aware_prior/test/evaluation.json",
    ),
    (
        "full_cross_modal_quality",
        "v3 跨模态质量",
        "full_cross_modal_quality/test/evaluation.json",
    ),
    (
        "full_cross_modal_quality_balanced",
        "v3 类别平衡负结果",
        "full_cross_modal_quality_balanced/test/evaluation.json",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build final paper tables after strict evidence audit"
    )
    parser.add_argument(
        "--reports",
        type=Path,
        default=PROJECT_ROOT / "reports" / "qwen-material",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "reports"
            / "qwen-material"
            / "paper-tables"
        ),
    )
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


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty paper table: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def classification_row(
    model_id: str,
    display_name: str,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = report["metrics"]
    intervals = (
        report.get("confidence_intervals", {})
        .get("intervals", {})
    )
    accuracy_interval = intervals.get("accuracy", {})
    macro_f1_interval = intervals.get("macro_f1", {})
    performance = report.get("performance", {})
    checkpoint = report.get("checkpoint", {})
    return {
        "model_id": model_id,
        "display_name": display_name,
        "sample_count": int(metrics["sample_count"]),
        "accuracy": float(metrics["accuracy"]),
        "accuracy_ci_lower": accuracy_interval.get("lower"),
        "accuracy_ci_upper": accuracy_interval.get("upper"),
        "macro_precision": float(metrics["macro_precision"]),
        "macro_recall": float(metrics["macro_recall"]),
        "macro_f1": float(metrics["macro_f1"]),
        "macro_f1_ci_lower": macro_f1_interval.get("lower"),
        "macro_f1_ci_upper": macro_f1_interval.get("upper"),
        "weighted_f1": float(metrics["weighted_f1"]),
        "ece": float(metrics["ece"]),
        "negative_log_likelihood": float(
            metrics["negative_log_likelihood"]
        ),
        "multiclass_brier_score": float(
            metrics["multiclass_brier_score"]
        ),
        "active_parameter_count": int(
            performance.get("active_parameter_count", 0)
        ),
        "forward_ms_per_sample": float(
            performance.get("amortized_forward_ms_per_sample", 0.0)
        ),
        "checkpoint_epoch": checkpoint.get("epoch"),
    }


def flatten_scenarios(
    scenarios: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for threshold, sources in scenarios.items():
        for source, metrics in sources.items():
            rows.append(
                {
                    "confidence_threshold": float(threshold),
                    "prediction_source": source,
                    "source_image_count": int(metrics["source_image_count"]),
                    "ground_truth_count": int(metrics["ground_truth_count"]),
                    "detection_count": int(metrics["detection_count"]),
                    "localization_precision": float(
                        metrics["localization_precision"]
                    ),
                    "localization_recall": float(
                        metrics["localization_recall"]
                    ),
                    "classification_accuracy_on_localized": float(
                        metrics["classification_accuracy_on_localized"]
                    ),
                    "end_to_end_precision": float(
                        metrics["end_to_end_precision"]
                    ),
                    "end_to_end_recall": float(
                        metrics["end_to_end_recall"]
                    ),
                    "macro_f1": float(metrics["macro_f1"]),
                }
            )
    return rows


def paired_comparison_row(
    name: str,
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    differences = comparison["paired_bootstrap"]["differences"]
    accuracy = differences["accuracy"]
    macro_f1 = differences["macro_f1"]
    outcomes = comparison["outcome_changes"]
    return {
        "comparison": name,
        "reference": comparison["reference"]["name"],
        "candidate": comparison["candidate"]["name"],
        "accuracy_delta_pp": 100.0 * float(accuracy["point"]),
        "accuracy_ci_lower_pp": 100.0 * float(accuracy["lower"]),
        "accuracy_ci_upper_pp": 100.0 * float(accuracy["upper"]),
        "macro_f1_delta_pp": 100.0 * float(macro_f1["point"]),
        "macro_f1_ci_lower_pp": 100.0 * float(macro_f1["lower"]),
        "macro_f1_ci_upper_pp": 100.0 * float(macro_f1["upper"]),
        "candidate_only_correct": int(outcomes["candidate_only_correct"]),
        "reference_only_correct": int(outcomes["reference_only_correct"]),
        "mcnemar_p_value": float(
            comparison["mcnemar_exact"]["p_value"]
        ),
    }


def _require_comparison_names(
    comparison: Mapping[str, Any],
    *,
    reference: str,
    candidate: str,
    label: str,
) -> None:
    observed_reference = comparison.get("reference", {}).get("name")
    observed_candidate = comparison.get("candidate", {}).get("name")
    if observed_reference != reference or observed_candidate != candidate:
        raise ValueError(
            f"{label} comparison names mismatch: "
            f"{observed_reference!r} -> {observed_candidate!r}"
        )


def _percent(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * float(value):.2f}%"


def _delta_interval(row: Mapping[str, Any], prefix: str) -> str:
    return (
        f"{float(row[f'{prefix}_delta_pp']):+.2f} "
        f"[{float(row[f'{prefix}_ci_lower_pp']):+.2f},"
        f"{float(row[f'{prefix}_ci_upper_pp']):+.2f}]"
    )


def _build_markdown(
    *,
    main_rows: Sequence[Mapping[str, Any]],
    main_paired_rows: Sequence[Mapping[str, Any]],
    ablation_rows: Sequence[Mapping[str, Any]],
    ablation_statistics: Sequence[Mapping[str, Any]],
    yolo_row: Mapping[str, Any],
    scene_rows: Sequence[Mapping[str, Any]],
    scene_strata_rows: Sequence[Mapping[str, Any]],
    sensor_rows: Sequence[Mapping[str, Any]],
    sensor_statistics: Sequence[Mapping[str, Any]],
    sensor_retraining_rows: Sequence[Mapping[str, Any]],
    sensor_retraining_statistics: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# 论文结果表组",
        "",
        "以下表格由通过严格证据审计的机器可读结果自动生成；CSV 保留原始小数，"
        "本页仅作排版预览。",
        "",
        "## 表 1 对象级主模型",
        "",
        "| 模型 | Accuracy [95% CI] | Macro F1 [95% CI] | ECE | NLL | Brier |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in main_rows:
        lines.append(
            f"| {row['display_name']} | {_percent(row['accuracy'])} "
            f"[{_percent(row['accuracy_ci_lower'])},"
            f"{_percent(row['accuracy_ci_upper'])}] | "
            f"{_percent(row['macro_f1'])} "
            f"[{_percent(row['macro_f1_ci_lower'])},"
            f"{_percent(row['macro_f1_ci_upper'])}] | "
            f"{_percent(row['ece'])} | "
            f"{float(row['negative_log_likelihood']):.4f} | "
            f"{float(row['multiclass_brier_score']):.4f} |"
        )
    lines.extend(
        [
            "",
            "### 主模型配对证据",
            "",
            "| 比较 | Accuracy Δ [95% CI] | Macro F1 Δ [95% CI] | McNemar p |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in main_paired_rows:
        lines.append(
            f"| {row['display_name']} | {_delta_interval(row, 'accuracy')} | "
            f"{_delta_interval(row, 'macro_f1')} | "
            f"{float(row['mcnemar_p_value']):.4g} |"
        )

    lines.extend(
        [
            "",
            "## 表 2 基础独立重训消融（旧 full 参照）",
            "",
            "| 变体 | Accuracy | 相对 full | Macro F1 | 相对 full |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in ablation_rows:
        lines.append(
            f"| {row['experiment']} | {_percent(row['accuracy'])} | "
            f"{float(row['accuracy_delta_pp']):+.2f} pp | "
            f"{_percent(row['macro_f1'])} | "
            f"{float(row['macro_f1_delta_pp']):+.2f} pp |"
        )

    lines.extend(
        [
            "",
            "## 表 3 基础消融配对统计（旧 full 参照）",
            "",
            "| 变体 | Accuracy Δ [95% CI] | Macro F1 Δ [95% CI] | Holm p |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in ablation_statistics:
        lines.append(
            f"| {row['experiment']} | {_delta_interval(row, 'accuracy')} | "
            f"{_delta_interval(row, 'macro_f1')} | "
            f"{float(row['mcnemar_holm_p_value']):.4g} |"
        )

    lines.extend(
        [
            "",
            "## 表 4 YOLO-seg 固定测试",
            "",
            "| Box P | Box R | Box mAP50 | Box mAP50-95 | "
            "Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
            "| "
            + " | ".join(
                _percent(yolo_row[key])
                for key in (
                    "box_precision",
                    "box_recall",
                    "box_map50",
                    "box_map50_95",
                    "mask_precision",
                    "mask_recall",
                    "mask_map50",
                    "mask_map50_95",
                )
            )
            + " |",
            "",
            "## 表 5 整图系统评估",
            "",
            "| Conf | 类别来源 | 定位 P | 定位 R | 匹配分类 Acc | "
            "端到端 P | 端到端 R | Macro F1 |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in scene_rows:
        lines.append(
            f"| {float(row['confidence_threshold']):.2f} | "
            f"{row['prediction_source']} | "
            f"{_percent(row['localization_precision'])} | "
            f"{_percent(row['localization_recall'])} | "
            f"{_percent(row['classification_accuracy_on_localized'])} | "
            f"{_percent(row['end_to_end_precision'])} | "
            f"{_percent(row['end_to_end_recall'])} | "
            f"{_percent(row['macro_f1'])} |"
        )

    lines.extend(
        [
            "",
            "## 表 6 整图分层辅助评估",
            "",
            "| Conf | 类别来源 | 维度 | 分层 | N | 定位 Recall | "
            "端到端 Recall |",
            "|---:|---|---|---|---:|---:|---:|",
        ]
    )
    for row in scene_strata_rows:
        lines.append(
            f"| {float(row['confidence_threshold']):.2f} | "
            f"{row['prediction_source']} | {row['dimension']} | "
            f"{row['stratum']} | {int(row['support'])} | "
            f"{_percent(row['localization_recall'])} | "
            f"{_percent(row['end_to_end_recall'])} |"
        )

    lines.extend(
        [
            "",
            "## 表 7 软传感器来源迁移",
            "",
            "| 来源 | Accuracy | Macro F1 | ECE |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in sensor_rows:
        lines.append(
            f"| {row['sensor_source']} | {_percent(row['accuracy'])} | "
            f"{_percent(row['macro_f1'])} | {_percent(row['ece'])} |"
        )
    lines.extend(
        [
            "",
            "| 配对比较 | Accuracy Δ [95% CI] | Macro F1 Δ [95% CI] | McNemar p |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in sensor_statistics:
        lines.append(
            f"| {row['comparison']} | {_delta_interval(row, 'accuracy')} | "
            f"{_delta_interval(row, 'macro_f1')} | "
            f"{float(row['mcnemar_p_value']):.4g} |"
        )

    lines.extend(
        [
            "",
            "## 表 8 软传感器来源独立重训",
            "",
            "| 训练来源 | Accuracy | Macro F1 | ECE |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in sensor_retraining_rows:
        lines.append(
            f"| {row['training_sensor_source']} | "
            f"{_percent(row['accuracy'])} | "
            f"{_percent(row['macro_f1'])} | {_percent(row['ece'])} |"
        )
    lines.extend(
        [
            "",
            "| 配对比较 | Accuracy Δ [95% CI] | Macro F1 Δ [95% CI] | "
            "McNemar p |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in sensor_retraining_statistics:
        lines.append(
            f"| {row['comparison']} | {_delta_interval(row, 'accuracy')} | "
            f"{_delta_interval(row, 'macro_f1')} | "
            f"{float(row['mcnemar_p_value']):.4g} |"
        )

    lines.extend(
        [
            "",
            "## 使用边界",
            "",
            "- 所有分类结果来自已多次查看的固定内部 test，不是外部盲测；",
            "- Qwen/rule-v1 均为图像条件软传感器，不是独立硬件测量；",
            "- 正式消融是单训练种子，配对区间不包含训练种子方差；",
            "- YOLO-seg 与整图结果不表示 MMFC 特征已经注入检测/分割头；",
            "- 来源迁移是同检查点运行时干预，不是两套独立训练数据消融。",
            "- Qwen/rule-v1 独立来源重训也只有种子 42，不能外推训练种子方差。",
            "",
        ]
    )
    return "\n".join(lines)


def build_paper_tables(
    *,
    reports_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    reports_root = _resolve(reports_root)
    output_dir = _resolve(output_dir)
    source_paths = {
        "final_evidence_audit": (
            reports_root / "FINAL_EVIDENCE_AUDIT.json"
        ),
        **{
            f"main_{model_id}": reports_root / relative_path
            for model_id, _, relative_path in MAIN_MODELS
        },
        "formal_ablation_summary": (
            reports_root / "ablations" / "ablation_summary.json"
        ),
        "formal_ablation_statistics": (
            reports_root
            / "ablations"
            / "ablation_pairwise_statistics.json"
        ),
        "v3_vs_v2b_paired": (
            reports_root
            / "full_cross_modal_quality"
            / "PAIRED_COMPARISON.json"
        ),
        "v3_gate_paired": (
            reports_root
            / "full_cross_modal_quality"
            / "GATE_PAIRED_COMPARISON.json"
        ),
        "yolo_segmentation": (
            reports_root / "yolo-seg" / "evaluation.json"
        ),
        "scene_pipeline": (
            reports_root
            / "yolo-seg"
            / "integrated"
            / "evaluation.json"
        ),
        "scene_strata": (
            reports_root
            / "yolo-seg"
            / "integrated"
            / "stratified"
            / "evaluation.json"
        ),
        "sensor_source_transfer": (
            reports_root
            / "sensor-source-transfer"
            / "evaluation.json"
        ),
        "sensor_retraining_candidate": (
            reports_root
            / "sensor-source-retraining"
            / "rule-v1"
            / "test"
            / "evaluation.json"
        ),
        "sensor_retraining_comparison": (
            reports_root
            / "sensor-source-retraining"
            / "RULE_V1_VS_QWEN_RETRAINED_COMPARISON.json"
        ),
    }
    audit_path = source_paths["final_evidence_audit"]
    if not audit_path.is_file():
        raise FileNotFoundError(audit_path)
    audit = _read_json(audit_path)
    if audit.get("state") != "complete" or audit.get("valid") is not True:
        raise ValueError("strict final evidence audit is not valid and complete")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty paper table directory: {output_dir}"
        )
    main_rows = [
        classification_row(
            model_id,
            display_name,
            _read_json(source_paths[f"main_{model_id}"]),
        )
        for model_id, display_name, relative_path in MAIN_MODELS
    ]
    v3_vs_v2b = _read_json(source_paths["v3_vs_v2b_paired"])
    _require_comparison_names(
        v3_vs_v2b,
        reference="full_quality_aware_prior",
        candidate="full_cross_modal_quality",
        label="v3_vs_v2b",
    )
    v3_gate = _read_json(source_paths["v3_gate_paired"])
    _require_comparison_names(
        v3_gate,
        reference="same_checkpoint_no_gate",
        candidate="full_cross_modal_quality_gate",
        label="v3_gate",
    )
    main_paired_rows = [
        {
            **paired_comparison_row(
                "v3_vs_v2b",
                v3_vs_v2b,
            ),
            "display_name": "v3 - v2b",
        },
        {
            **paired_comparison_row(
                "v3_gate_vs_same_checkpoint_no_gate",
                v3_gate,
            ),
            "display_name": "v3 gate - 同检查点 no-gate",
        },
    ]
    ablation_summary = _read_json(
        source_paths["formal_ablation_summary"]
    )
    if ablation_summary.get("protocol") != "formal_retrained":
        raise ValueError("ablation summary is not formal retraining")
    ablation_rows = list(ablation_summary["summary"])
    ablation_statistics = list(
        _read_json(source_paths["formal_ablation_statistics"])["summary"]
    )

    yolo = _read_json(source_paths["yolo_segmentation"])
    yolo_metrics = yolo["metrics"]
    yolo_row = {
        "test_source_image_count": yolo["dataset"]["test_source_images"],
        "test_object_count": yolo["dataset"]["test_objects"],
        "box_precision": yolo_metrics["metrics/precision(B)"],
        "box_recall": yolo_metrics["metrics/recall(B)"],
        "box_map50": yolo_metrics["metrics/mAP50(B)"],
        "box_map50_95": yolo_metrics["metrics/mAP50-95(B)"],
        "mask_precision": yolo_metrics["metrics/precision(M)"],
        "mask_recall": yolo_metrics["metrics/recall(M)"],
        "mask_map50": yolo_metrics["metrics/mAP50(M)"],
        "mask_map50_95": yolo_metrics["metrics/mAP50-95(M)"],
    }

    scene = _read_json(source_paths["scene_pipeline"])
    scene_rows = flatten_scenarios(scene["scenarios"])
    scene_strata_rows = list(
        _read_json(source_paths["scene_strata"])["rows"]
    )
    sensor = _read_json(source_paths["sensor_source_transfer"])
    sensor_rows = [
        {
            "sensor_source": source,
            **{
                key: values["metrics"][key]
                for key in (
                    "accuracy",
                    "macro_precision",
                    "macro_recall",
                    "macro_f1",
                    "weighted_f1",
                    "ece",
                    "negative_log_likelihood",
                    "multiclass_brier_score",
                )
            },
        }
        for source, values in sensor["sources"].items()
    ]
    sensor_statistics = [
        paired_comparison_row(name, comparison)
        for name, comparison in sensor["paired_comparisons"].items()
    ]
    sensor_retraining_rows = [
        {
            "training_sensor_source": source,
            **classification_row(
                model_id,
                display_name,
                _read_json(path),
            ),
        }
        for source, model_id, display_name, path in (
            (
                "qwen",
                "full_cross_modal_quality",
                "v3 Qwen 重训",
                source_paths["main_full_cross_modal_quality"],
            ),
            (
                "rule-v1",
                "full_cross_modal_quality_rule_v1",
                "v3 rule-v1 重训",
                source_paths["sensor_retraining_candidate"],
            ),
        )
    ]
    sensor_retraining_statistics = [
        paired_comparison_row(
            "rule_v1_retrained_vs_qwen_retrained",
            _read_json(source_paths["sensor_retraining_comparison"]),
        )
    ]

    table_map = {
        "main_classification.csv": main_rows,
        "formal_ablations.csv": ablation_rows,
        "formal_ablation_statistics.csv": ablation_statistics,
        "yolo_segmentation.csv": [yolo_row],
        "scene_pipeline.csv": scene_rows,
        "scene_strata.csv": scene_strata_rows,
        "sensor_source_transfer.csv": sensor_rows,
        "sensor_source_statistics.csv": sensor_statistics,
        "sensor_source_retraining.csv": sensor_retraining_rows,
        "sensor_source_retraining_statistics.csv": (
            sensor_retraining_statistics
        ),
    }
    markdown = _build_markdown(
        main_rows=main_rows,
        main_paired_rows=main_paired_rows,
        ablation_rows=ablation_rows,
        ablation_statistics=ablation_statistics,
        yolo_row=yolo_row,
        scene_rows=scene_rows,
        scene_strata_rows=scene_strata_rows,
        sensor_rows=sensor_rows,
        sensor_statistics=sensor_statistics,
        sensor_retraining_rows=sensor_retraining_rows,
        sensor_retraining_statistics=sensor_retraining_statistics,
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.tmp-",
            dir=output_dir.parent,
        )
    )
    try:
        for filename, rows in table_map.items():
            _write_csv(staging_dir / filename, rows)
        _atomic_write(staging_dir / "TABLES.md", markdown)
        figure_paths = build_paper_figures(
            tables=table_map,
            output_dir=staging_dir,
        )
        artifact_paths = {
            **{
                filename: staging_dir / filename
                for filename in table_map
            },
            "TABLES.md": staging_dir / "TABLES.md",
            **figure_paths,
        }
        manifest = {
            "schema_version": 2,
            "source_audit": audit_path.relative_to(
                reports_root
            ).as_posix(),
            "source_audit_valid": True,
            "source_audit_sha256": _sha256(audit_path),
            "source_reports": {
                name: {
                    "path": path.relative_to(reports_root).as_posix(),
                    "sha256": _sha256(path),
                }
                for name, path in source_paths.items()
            },
            "tables": {
                filename: len(rows)
                for filename, rows in table_map.items()
            },
            "markdown": "TABLES.md",
            "figures": list(figure_paths),
            "artifact_sha256": {
                filename: _sha256(path)
                for filename, path in artifact_paths.items()
            },
            "formal_ablation_scope": (
                "legacy_full_reference_not_v3_component_ablation"
            ),
            "main_paired_evidence_count": len(main_paired_rows),
            "scope": "paper_ready_internal_results_tables",
        }
        _atomic_write(
            staging_dir / "manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        )
        if output_dir.exists():
            output_dir.rmdir()
        staging_dir.replace(output_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    return {
        "state": "complete",
        "output": output_dir.as_posix(),
        **manifest,
    }


def main() -> None:
    args = parse_args()
    result = build_paper_tables(
        reports_root=args.reports,
        output_dir=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
