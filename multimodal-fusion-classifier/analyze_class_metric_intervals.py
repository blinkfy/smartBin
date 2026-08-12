"""Estimate source-image cluster bootstrap intervals for per-class metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCES = {
    "evaluation": (
        "reports/qwen-material/full_cross_modal_quality/test/evaluation.json"
    ),
    "predictions": (
        "reports/qwen-material/full_cross_modal_quality/test/predictions.jsonl"
    ),
}
DEFAULT_EVALUATION = PROJECT_ROOT / DEFAULT_SOURCES["evaluation"]
DEFAULT_PREDICTIONS = PROJECT_ROOT / DEFAULT_SOURCES["predictions"]
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "CLASS_METRIC_INTERVALS.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build descriptive per-class metric intervals from saved predictions"
        )
    )
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"JSONL row must be an object: {path}:{line_number}"
                )
            records.append(value)
    return records


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> np.ndarray:
    result = np.full(numerator.shape, np.nan, dtype=np.float64)
    np.divide(
        numerator,
        denominator,
        out=result,
        where=denominator > 0,
    )
    return result


def _metric_interval(
    values: np.ndarray,
    *,
    confidence: float,
) -> dict[str, float | int | None]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {
            "lower": None,
            "upper": None,
            "valid_repeats": 0,
            "undefined_repeats": int(values.size),
        }
    alpha = 1.0 - confidence
    return {
        "lower": float(np.quantile(finite, alpha / 2.0)),
        "upper": float(np.quantile(finite, 1.0 - alpha / 2.0)),
        "valid_repeats": int(finite.size),
        "undefined_repeats": int(values.size - finite.size),
    }


def _point_metrics(confusion: np.ndarray) -> tuple[np.ndarray, ...]:
    true_positive = np.diag(confusion).astype(np.float64)
    target_support = confusion.sum(axis=1).astype(np.float64)
    predicted_support = confusion.sum(axis=0).astype(np.float64)
    precision = _safe_divide(true_positive, predicted_support)
    recall = _safe_divide(true_positive, target_support)
    f1 = _safe_divide(
        2.0 * true_positive,
        target_support + predicted_support,
    )
    f1[target_support == 0] = np.nan
    return precision, recall, f1, target_support, predicted_support


def _validate_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    class_names: Sequence[str],
) -> None:
    seen: set[str] = set()
    class_count = len(class_names)
    for index, row in enumerate(rows, start=1):
        sample_id = str(row.get("sample_id", ""))
        source_image_id = str(row.get("source_image_id", ""))
        if not sample_id or not source_image_id:
            raise ValueError(
                f"prediction row {index} is missing sample_id/source_image_id"
            )
        if sample_id in seen:
            raise ValueError(f"duplicate prediction sample_id: {sample_id}")
        seen.add(sample_id)
        target_id = int(row["target_id"])
        predicted_id = int(row["predicted_id"])
        if not 0 <= target_id < class_count:
            raise ValueError(f"target_id out of range: {target_id}")
        if not 0 <= predicted_id < class_count:
            raise ValueError(f"predicted_id out of range: {predicted_id}")
        if str(row.get("target", class_names[target_id])) != class_names[target_id]:
            raise ValueError(f"target name/id mismatch: {sample_id}")
        if (
            str(row.get("predicted", class_names[predicted_id]))
            != class_names[predicted_id]
        ):
            raise ValueError(f"predicted name/id mismatch: {sample_id}")


def build_class_metric_intervals(
    *,
    evaluation_path: Path,
    predictions_path: Path,
    repeats: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    evaluation_path = _resolve(evaluation_path)
    predictions_path = _resolve(predictions_path)
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8-sig"))
    class_names = [str(value) for value in evaluation.get("class_names", [])]
    if not class_names:
        raise ValueError("evaluation class_names must be a non-empty list")

    rows = _read_jsonl(predictions_path)
    if not rows:
        raise ValueError("predictions must contain at least one row")
    _validate_prediction_rows(rows, class_names)

    evaluated_samples = int(
        evaluation.get("dataset", {}).get("evaluated_samples", -1)
    )
    if evaluated_samples != len(rows):
        raise ValueError(
            "evaluation/prediction sample count mismatch: "
            f"{evaluated_samples} != {len(rows)}"
        )

    class_count = len(class_names)
    source_ids = sorted({str(row["source_image_id"]) for row in rows})
    source_index = {source_id: index for index, source_id in enumerate(source_ids)}
    cluster_confusions = np.zeros(
        (len(source_ids), class_count, class_count),
        dtype=np.int64,
    )
    class_source_sets = [set() for _ in range(class_count)]
    for row in rows:
        source_id = str(row["source_image_id"])
        target_id = int(row["target_id"])
        predicted_id = int(row["predicted_id"])
        cluster_confusions[
            source_index[source_id],
            target_id,
            predicted_id,
        ] += 1
        class_source_sets[target_id].add(source_id)

    confusion = cluster_confusions.sum(axis=0)
    precision, recall, f1, target_support, predicted_support = _point_metrics(
        confusion
    )

    rng = np.random.default_rng(seed)
    cluster_count = len(source_ids)
    cluster_weights = rng.multinomial(
        cluster_count,
        np.full(cluster_count, 1.0 / cluster_count),
        size=repeats,
    )
    bootstrap_confusions = (
        cluster_weights @ cluster_confusions.reshape(cluster_count, -1)
    ).reshape(repeats, class_count, class_count)
    bootstrap_true_positive = np.diagonal(
        bootstrap_confusions,
        axis1=1,
        axis2=2,
    ).astype(np.float64)
    bootstrap_target_support = bootstrap_confusions.sum(axis=2).astype(
        np.float64
    )
    bootstrap_predicted_support = bootstrap_confusions.sum(axis=1).astype(
        np.float64
    )
    bootstrap_precision = _safe_divide(
        bootstrap_true_positive,
        bootstrap_predicted_support,
    )
    bootstrap_recall = _safe_divide(
        bootstrap_true_positive,
        bootstrap_target_support,
    )
    bootstrap_f1 = _safe_divide(
        2.0 * bootstrap_true_positive,
        bootstrap_target_support + bootstrap_predicted_support,
    )
    bootstrap_f1[bootstrap_target_support == 0] = np.nan

    source_per_class = {
        str(row["label"]): row
        for row in evaluation.get("per_class", [])
        if isinstance(row, dict) and "label" in row
    }
    metrics: list[dict[str, Any]] = []
    point_mismatches: list[str] = []
    for class_id, class_name in enumerate(class_names):
        point = {
            "precision": (
                float(precision[class_id])
                if np.isfinite(precision[class_id])
                else None
            ),
            "recall": (
                float(recall[class_id])
                if np.isfinite(recall[class_id])
                else None
            ),
            "f1": float(f1[class_id]) if np.isfinite(f1[class_id]) else 0.0,
        }
        recorded = source_per_class.get(class_name)
        if recorded is None:
            point_mismatches.append(f"missing evaluation row: {class_name}")
        else:
            for metric_name, point_value in point.items():
                recorded_value = recorded.get(metric_name)
                if (
                    point_value is None
                    or recorded_value is None
                    or not np.isclose(
                        float(recorded_value),
                        point_value,
                        atol=1e-12,
                        rtol=0.0,
                    )
                ):
                    point_mismatches.append(
                        f"{class_name}.{metric_name} does not match evaluation"
                    )

        metrics.append(
            {
                "label_id": class_id,
                "label": class_name,
                "support": int(target_support[class_id]),
                "source_image_count": len(class_source_sets[class_id]),
                "predicted_count": int(predicted_support[class_id]),
                "point": point,
                "intervals": {
                    "precision": _metric_interval(
                        bootstrap_precision[:, class_id],
                        confidence=confidence,
                    ),
                    "recall": _metric_interval(
                        bootstrap_recall[:, class_id],
                        confidence=confidence,
                    ),
                    "f1": _metric_interval(
                        bootstrap_f1[:, class_id],
                        confidence=confidence,
                    ),
                },
            }
        )

    errors = point_mismatches
    return {
        "schema_version": 1,
        "state": "complete" if not errors else "invalid",
        "valid": not errors,
        "scope": "posthoc_fixed_test_per_class_uncertainty",
        "input_sources": {
            "evaluation": {
                "path": _display_path(evaluation_path),
                "sha256": _sha256(evaluation_path),
            },
            "predictions": {
                "path": _display_path(predictions_path),
                "sha256": _sha256(predictions_path),
            },
        },
        "dataset": {
            "sample_count": len(rows),
            "source_image_count": cluster_count,
            "class_count": class_count,
        },
        "method": {
            "name": "source_image_cluster_percentile_bootstrap",
            "resampling_unit": "source_image_id",
            "repeats": repeats,
            "confidence": confidence,
            "seed": seed,
            "undefined_metric_policy": (
                "precision is undefined when predicted support is zero; "
                "recall and F1 are undefined only when target support is zero; "
                "a present class with zero true positives has F1=0"
            ),
        },
        "metrics": metrics,
        "errors": errors,
        "boundaries": [
            (
                "Intervals are descriptive post-hoc uncertainty estimates on "
                "the fixed internal test set, not preregistered hypothesis tests."
            ),
            (
                "Resampling source_image_id accounts for multiple objects from "
                "one image but not unidentified capture-session dependence."
            ),
            (
                "Small-support classes can have wide or unstable intervals; "
                "point estimates must not be read as external performance."
            ),
        ],
    }


def _format_metric(
    point: float | None,
    interval: Mapping[str, Any],
) -> str:
    if point is None or interval.get("lower") is None:
        return "undefined"
    return (
        f"{point:.3f} "
        f"[{float(interval['lower']):.3f}, {float(interval['upper']):.3f}]"
    )


def render_markdown(report: Mapping[str, Any]) -> str:
    method = report["method"]
    dataset = report["dataset"]
    lines = [
        "# 逐类别聚类 Bootstrap 区间",
        "",
        (
            f"固定测试集包含 {dataset['sample_count']} 个对象、"
            f"{dataset['source_image_count']} 张源图和 "
            f"{dataset['class_count']} 个类别。以下区间按 "
            f"`source_image_id` 进行 {method['repeats']} 次百分位 bootstrap，"
            f"置信水平为 {float(method['confidence']):.0%}。"
        ),
        "",
        "| 类别 | support | 源图数 | Precision [CI] | Recall [CI] | F1 [CI] |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["metrics"]:
        lines.append(
            "| {label} | {support} | {source_count} | {precision} | "
            "{recall} | {f1} |".format(
                label=row["label"],
                support=row["support"],
                source_count=row["source_image_count"],
                precision=_format_metric(
                    row["point"]["precision"],
                    row["intervals"]["precision"],
                ),
                recall=_format_metric(
                    row["point"]["recall"],
                    row["intervals"]["recall"],
                ),
                f1=_format_metric(
                    row["point"]["f1"],
                    row["intervals"]["f1"],
                ),
            )
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 这是固定内部测试集上的事后描述性区间，不是预注册显著性检验。",
            "- 聚类单位是源图，能处理同一图中多个对象，但不能处理未知采集批次依赖。",
            "- 少数类区间较宽；点估计不能外推为外部部署性能。",
            "- 分母为零的类别-指标重复会从该指标区间中排除，并在 JSON 中计数。",
            "",
            "输入文件及 SHA-256 见同名 JSON 报告。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    output_path = _resolve(args.output)
    report = build_class_metric_intervals(
        evaluation_path=args.evaluation,
        predictions_path=args.predictions,
        repeats=args.repeats,
        confidence=args.confidence,
        seed=args.seed,
    )
    _atomic_write(
        output_path,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    markdown_path = output_path.with_suffix(".md")
    _atomic_write(markdown_path, render_markdown(report))
    print(
        json.dumps(
            {
                "output": output_path.as_posix(),
                "markdown": markdown_path.as_posix(),
                "valid": report["valid"],
                "sample_count": report["dataset"]["sample_count"],
                "source_image_count": report["dataset"]["source_image_count"],
                "class_count": report["dataset"]["class_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
