"""Build machine-derived final claim evidence after strict auditing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent
METRIC_OBJECTIVES = {
    "accuracy": "higher",
    "macro_f1": "higher",
    "ece": "lower",
    "negative_log_likelihood": "lower",
    "multiclass_brier_score": "lower",
}
CLAIM_TABLE_SOURCE_NAMES = {
    "formal_summary": "formal_ablation_summary",
    "formal_statistics": "formal_ablation_statistics",
    "yolo": "yolo_segmentation",
    "scene": "scene_pipeline",
    "scene_strata": "scene_strata",
    "sensor_transfer": "sensor_source_transfer",
    "qwen_v3": "main_full_cross_modal_quality",
    "rule_retrained": "sensor_retraining_candidate",
    "sensor_retraining_comparison": "sensor_retraining_comparison",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build audited final claim evidence without manual copying"
    )
    parser.add_argument(
        "--reports",
        type=Path,
        default=PROJECT_ROOT / "reports" / "qwen-material",
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


def metric_interval_assessment(
    metric: str,
    interval: Mapping[str, Any],
) -> str:
    objective = METRIC_OBJECTIVES[metric]
    lower = float(interval["lower"])
    upper = float(interval["upper"])
    if lower <= 0.0 <= upper:
        return "interval_inconclusive"
    candidate_higher = lower > 0.0
    candidate_better = (
        candidate_higher if objective == "higher" else not candidate_higher
    )
    return "candidate_better" if candidate_better else "candidate_worse"


def paired_evidence(
    name: str,
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    differences = comparison["paired_bootstrap"]["differences"]
    metrics = {}
    for metric, objective in METRIC_OBJECTIVES.items():
        if metric not in differences:
            continue
        interval = differences[metric]
        metrics[metric] = {
            "objective": objective,
            "point": float(interval["point"]),
            "lower": float(interval["lower"]),
            "upper": float(interval["upper"]),
            "assessment": metric_interval_assessment(metric, interval),
        }
    return {
        "name": name,
        "reference": comparison["reference"]["name"],
        "candidate": comparison["candidate"]["name"],
        "sample_count": int(
            comparison["paired_bootstrap"]["sample_count"]
        ),
        "source_image_cluster_count": int(
            comparison["paired_bootstrap"]["cluster_count"]
        ),
        "bootstrap_repeats": int(
            comparison["paired_bootstrap"]["repeats"]
        ),
        "mcnemar_p_value": float(
            comparison["mcnemar_exact"]["p_value"]
        ),
        "metrics": metrics,
    }


def _source_paths(reports_root: Path) -> dict[str, Path]:
    return {
        "final_audit": reports_root / "FINAL_EVIDENCE_AUDIT.json",
        "paper_table_manifest": (
            reports_root / "paper-tables" / "manifest.json"
        ),
        "formal_summary": (
            reports_root / "ablations" / "ablation_summary.json"
        ),
        "formal_statistics": (
            reports_root
            / "ablations"
            / "ablation_pairwise_statistics.json"
        ),
        "yolo": reports_root / "yolo-seg" / "evaluation.json",
        "scene": (
            reports_root / "yolo-seg" / "integrated" / "evaluation.json"
        ),
        "scene_strata": (
            reports_root
            / "yolo-seg"
            / "integrated"
            / "stratified"
            / "evaluation.json"
        ),
        "sensor_transfer": (
            reports_root / "sensor-source-transfer" / "evaluation.json"
        ),
        "qwen_v3": (
            reports_root
            / "full_cross_modal_quality"
            / "test"
            / "evaluation.json"
        ),
        "rule_retrained": (
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


def _classification_metrics(report: Mapping[str, Any]) -> dict[str, float]:
    values = report["metrics"]
    return {
        metric: float(values[metric])
        for metric in METRIC_OBJECTIVES
    }


def _require_table_bound_sources(
    *,
    table_manifest: Mapping[str, Any],
    sources: Mapping[str, Path],
    reports_root: Path,
) -> None:
    table_sources = table_manifest.get("source_reports")
    if not isinstance(table_sources, Mapping):
        raise ValueError("paper table source report manifest is missing")
    for claim_name, table_name in CLAIM_TABLE_SOURCE_NAMES.items():
        entry = table_sources.get(table_name)
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"paper table source report is missing: {table_name}"
            )
        source_path = sources[claim_name].resolve()
        recorded_path = Path(str(entry.get("path", "")))
        if not recorded_path.is_absolute():
            recorded_path = (reports_root / recorded_path).resolve()
        else:
            recorded_path = recorded_path.resolve()
        if recorded_path != source_path:
            raise ValueError(
                f"paper table source path mismatch: {table_name}"
            )
        if entry.get("sha256") != _sha256(source_path):
            raise ValueError(
                f"paper table source hash mismatch: {table_name}"
            )


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _comparison_markdown(
    evidence: Mapping[str, Any],
) -> list[str]:
    lines = [
        "| 比较 | 指标 | 差值 | 95% CI | 区间判定 |",
        "|---|---|---:|---:|---|",
    ]
    for metric, values in evidence["metrics"].items():
        lines.append(
            f"| {evidence['name']} | {metric} | "
            f"{float(values['point']):+.4f} | "
            f"[{float(values['lower']):+.4f},"
            f"{float(values['upper']):+.4f}] | "
            f"{values['assessment']} |"
        )
    return lines


def build_claim_evidence(reports_root: Path) -> dict[str, Any]:
    reports_root = _resolve(reports_root)
    sources = _source_paths(reports_root)
    missing = [path for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "claim evidence inputs are missing: "
            + ", ".join(path.as_posix() for path in missing)
        )
    final_audit = _read_json(sources["final_audit"])
    if (
        final_audit.get("state") != "complete"
        or final_audit.get("valid") is not True
    ):
        raise ValueError("strict final evidence audit is not complete")
    table_manifest = _read_json(sources["paper_table_manifest"])
    if (
        table_manifest.get("source_audit_valid") is not True
        or table_manifest.get("source_audit_sha256")
        != _sha256(sources["final_audit"])
    ):
        raise ValueError("paper tables are not tied to the final audit")
    _require_table_bound_sources(
        table_manifest=table_manifest,
        sources=sources,
        reports_root=reports_root,
    )

    formal = _read_json(sources["formal_summary"])
    formal_statistics = _read_json(sources["formal_statistics"])
    yolo = _read_json(sources["yolo"])
    scene = _read_json(sources["scene"])
    strata = _read_json(sources["scene_strata"])
    transfer = _read_json(sources["sensor_transfer"])
    qwen = _read_json(sources["qwen_v3"])
    rule = _read_json(sources["rule_retrained"])
    retraining_comparison = _read_json(
        sources["sensor_retraining_comparison"]
    )

    transfer_comparisons = {
        name: paired_evidence(name, comparison)
        for name, comparison in transfer["paired_comparisons"].items()
    }
    retraining_evidence = paired_evidence(
        "rule_v1_retrained_vs_qwen_retrained",
        retraining_comparison,
    )
    claims = {
        "formal_ablations": {
            "status": "evidence_complete",
            "experiment_count": len(formal["summary"]),
            "paired_comparison_count": len(
                formal_statistics["summary"]
            ),
            "protocol": formal["protocol"],
            "reference_model": "full",
            "claim_boundary": (
                "legacy full reliability baseline; not a component-wise "
                "ablation of v3 cross-modal quality"
            ),
        },
        "C13_same_checkpoint_sensor_transfer": {
            "status": "evidence_complete",
            "protocol": transfer["protocol"],
            "sample_count": int(transfer["sample_count"]),
            "source_image_count": int(transfer["source_image_count"]),
            "source_metrics": {
                source: _classification_metrics(values)
                for source, values in transfer["sources"].items()
            },
            "comparisons": transfer_comparisons,
            "claim_boundary": (
                "runtime intervention; not independent retraining"
            ),
        },
        "C14_scene_strata": {
            "status": "evidence_complete",
            "protocol": strata["protocol"],
            "row_count": len(strata["rows"]),
            "source_image_count": int(
                strata["dataset"]["source_image_count"]
            ),
            "ground_truth_object_count": int(
                strata["dataset"]["ground_truth_object_count"]
            ),
            "claim_boundary": (
                "relative bbox quartiles and overlap proxy; "
                "not human small-target or occlusion labels"
            ),
        },
        "C15_independent_sensor_retraining": {
            "status": "evidence_complete",
            "qwen_metrics": _classification_metrics(qwen),
            "rule_v1_metrics": _classification_metrics(rule),
            "comparison": retraining_evidence,
            "claim_boundary": (
                "one training seed; sidecars remain image-conditioned"
            ),
        },
        "target_domain_system": {
            "status": "evidence_complete",
            "yolo_metrics": {
                key: float(value)
                for key, value in yolo["metrics"].items()
                if key.startswith("metrics/")
            },
            "scene_scenarios": scene["scenarios"],
            "claim_boundary": (
                "two-stage RGB detector and MMFC reclassification; "
                "no fused detector-head injection"
            ),
        },
    }
    result = {
        "schema_version": 1,
        "state": "complete",
        "source_final_audit_sha256": _sha256(
            sources["final_audit"]
        ),
        "source_paper_table_manifest_sha256": _sha256(
            sources["paper_table_manifest"]
        ),
        "source_hashes": {
            name: {
                "path": path.relative_to(reports_root).as_posix(),
                "sha256": _sha256(path),
            }
            for name, path in sources.items()
        },
        "claims": claims,
        "global_boundaries": [
            "fixed internal test, not external blind validation",
            "soft sensors are not synchronized hardware observations",
            "single-seed retraining intervals omit training-seed variance",
            "target-domain detector is separate from the MMFC feature head",
        ],
    }
    json_path = reports_root / "FINAL_CLAIM_EVIDENCE.json"
    _atomic_write(
        json_path,
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
    )

    lines = [
        "# 最终主张证据表",
        "",
        "本文件由严格审计结果自动生成，不替代静态的训练前主张登记表。",
        "",
        "## 基础正式消融（旧 full 参照）",
        "",
        f"- 独立模型数：{claims['formal_ablations']['experiment_count']}；",
        f"- 相对 full 的配对比较数："
        f"{claims['formal_ablations']['paired_comparison_count']}。",
        "- 边界：这些变体不是 v3 跨模态质量模块的逐组件消融。",
        "",
        "## C13 同检查点来源迁移",
        "",
    ]
    for source, metrics in claims[
        "C13_same_checkpoint_sensor_transfer"
    ]["source_metrics"].items():
        lines.append(
            f"- {source}: Accuracy {_percent(metrics['accuracy'])}, "
            f"Macro F1 {_percent(metrics['macro_f1'])}, "
            f"ECE {_percent(metrics['ece'])}。"
        )
    lines.extend([""])
    for comparison in transfer_comparisons.values():
        lines.extend(_comparison_markdown(comparison))
        lines.append("")
    lines.extend(
        [
            "## C14 整图辅助分层",
            "",
            f"- 共 {claims['C14_scene_strata']['row_count']} 个"
            "阈值/类别来源/分层结果；",
            "- 分层是相对面积、场景密度和框重叠代理，不是人工遮挡标签。",
            "",
            "## C15 独立来源重训",
            "",
            f"- Qwen: Accuracy "
            f"{_percent(claims['C15_independent_sensor_retraining']['qwen_metrics']['accuracy'])}, "
            f"Macro F1 "
            f"{_percent(claims['C15_independent_sensor_retraining']['qwen_metrics']['macro_f1'])}；",
            f"- rule-v1: Accuracy "
            f"{_percent(claims['C15_independent_sensor_retraining']['rule_v1_metrics']['accuracy'])}, "
            f"Macro F1 "
            f"{_percent(claims['C15_independent_sensor_retraining']['rule_v1_metrics']['macro_f1'])}。",
            "",
        ]
    )
    lines.extend(_comparison_markdown(retraining_evidence))
    lines.extend(
        [
            "",
            "## 全局边界",
            "",
            *[
                f"- {boundary}；"
                for boundary in result["global_boundaries"]
            ],
            "",
        ]
    )
    markdown_path = reports_root / "FINAL_CLAIM_EVIDENCE.md"
    _atomic_write(markdown_path, "\n".join(lines))
    return {
        "state": "complete",
        "json": json_path.as_posix(),
        "markdown": markdown_path.as_posix(),
        "claim_group_count": len(claims),
    }


def main() -> None:
    args = parse_args()
    result = build_claim_evidence(args.reports)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
