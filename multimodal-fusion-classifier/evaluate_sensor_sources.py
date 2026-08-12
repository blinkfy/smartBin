"""Compare Qwen, rule-v1, and missing sensors with one frozen checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from compare_predictions import exact_mcnemar_p_value
from evaluate import _checkpoint_metadata, _runtime_metadata
from mmfc.checkpoints import (
    load_checkpoint,
    load_model_state_compatible,
    validated_checkpoint_class_names,
)
from mmfc.evaluation import (
    compute_classification_metrics,
    evaluate_model,
    grouped_bootstrap_classification_intervals,
    grouped_paired_bootstrap_differences,
    write_evaluation_artifacts,
)
from mmfc.model import build_production_model
from mmfc.robustness import default_sensor_scenarios
from mmfc.training import (
    ManifestDataset,
    PROJECT_ROOT,
    load_config,
    resolve_project_path,
)


SOURCES = ("qwen", "rule-v1", "missing")
COMPARISONS = (
    ("rule_v1_vs_qwen", "qwen", "rule-v1"),
    ("missing_vs_qwen", "qwen", "missing"),
    ("rule_v1_vs_missing", "missing", "rule-v1"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate soft-sensor source transfer with a fixed MMFC"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ablations/full_cross_modal_quality.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/ablations/full_cross_modal_quality/best.pt"),
    )
    parser.add_argument(
        "--rule-sensor-manifest",
        type=Path,
        default=Path("data/generated/sensors.jsonl"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "reports/qwen-material/sensor-source-transfer/PROTOCOL.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/qwen-material/sensor-source-transfer"),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--paired-repeats", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260727)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def verify_protocol_files(
    protocol: Mapping[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
) -> list[dict[str, str]]:
    checks = []
    for name, entry in protocol.get("inputs", {}).items():
        if not isinstance(entry, Mapping) or "sha256" not in entry:
            continue
        path = Path(str(entry["path"]))
        path = path.resolve() if path.is_absolute() else (project_root / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"protocol input {name} is missing: {path}")
        actual = _sha256(path)
        expected = str(entry["sha256"]).lower()
        if actual != expected:
            raise ValueError(
                f"protocol input hash mismatch for {name}: {actual} != {expected}"
            )
        checks.append(
            {"name": str(name), "path": path.as_posix(), "sha256": actual}
        )
    return checks


def _validate_protocol_arguments(
    protocol: Mapping[str, Any],
    args: argparse.Namespace,
) -> None:
    evaluation = protocol["evaluation"]
    checks = {
        "batch_size": (int(args.batch_size), int(evaluation["batch_size"])),
        "num_workers": (int(args.num_workers), int(evaluation["num_workers"])),
        "bootstrap_repeats": (
            int(args.bootstrap_repeats),
            int(evaluation["bootstrap_repeats"]),
        ),
        "paired_repeats": (
            int(args.paired_repeats),
            int(evaluation["paired_repeats"]),
        ),
        "bootstrap_seed": (
            int(args.bootstrap_seed),
            int(evaluation["bootstrap_seed"]),
        ),
    }
    mismatches = [
        f"{name}: {actual!r} != {expected!r}"
        for name, (actual, expected) in checks.items()
        if actual != expected
    ]
    expected_paths = {
        "config": protocol["inputs"]["classifier_config"]["path"],
        "checkpoint": protocol["inputs"]["classifier_checkpoint"]["path"],
        "rule_sensor_manifest": protocol["inputs"]["rule_sensor_manifest"]["path"],
    }
    for name, expected in expected_paths.items():
        actual_path = resolve_project_path(getattr(args, name))
        expected_path = resolve_project_path(Path(str(expected)))
        if actual_path != expected_path:
            mismatches.append(f"{name}: {actual_path} != {expected_path}")
    if mismatches:
        raise ValueError(
            "evaluation arguments differ from frozen protocol: "
            + "; ".join(mismatches)
        )


def _prediction_arrays(
    records: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    targets = np.asarray(
        [int(record["target_id"]) for record in records],
        dtype=np.int64,
    )
    probabilities = np.asarray(
        [record["probabilities"] for record in records],
        dtype=np.float64,
    )
    source_ids = [str(record["source_image_id"]) for record in records]
    sample_ids = [str(record["sample_id"]) for record in records]
    return targets, probabilities, source_ids, sample_ids


def compare_source_predictions(
    reference_records: Sequence[Mapping[str, Any]],
    candidate_records: Sequence[Mapping[str, Any]],
    *,
    class_names: Sequence[str],
    reference_name: str,
    candidate_name: str,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    ref_targets, ref_probabilities, source_ids, ref_ids = _prediction_arrays(
        reference_records
    )
    cand_targets, cand_probabilities, cand_sources, cand_ids = _prediction_arrays(
        candidate_records
    )
    if ref_ids != cand_ids:
        raise ValueError("sensor-source prediction sample order differs")
    if source_ids != cand_sources:
        raise ValueError("sensor-source prediction source-image order differs")
    if not np.array_equal(ref_targets, cand_targets):
        raise ValueError("sensor-source prediction targets differ")

    reference_metrics = compute_classification_metrics(
        ref_targets,
        ref_probabilities,
        class_names,
    )
    candidate_metrics = compute_classification_metrics(
        ref_targets,
        cand_probabilities,
        class_names,
    )
    ref_correct = ref_probabilities.argmax(axis=1) == ref_targets
    cand_correct = cand_probabilities.argmax(axis=1) == ref_targets
    candidate_only = int(np.sum(~ref_correct & cand_correct))
    reference_only = int(np.sum(ref_correct & ~cand_correct))
    return {
        "reference": {
            "name": reference_name,
            "metrics": reference_metrics["metrics"],
        },
        "candidate": {
            "name": candidate_name,
            "metrics": candidate_metrics["metrics"],
        },
        "outcome_changes": {
            "both_correct": int(np.sum(ref_correct & cand_correct)),
            "candidate_only_correct": candidate_only,
            "reference_only_correct": reference_only,
            "both_wrong": int(np.sum(~ref_correct & ~cand_correct)),
        },
        "mcnemar_exact": {
            "discordant_pairs": candidate_only + reference_only,
            "p_value": exact_mcnemar_p_value(candidate_only, reference_only),
        },
        "paired_bootstrap": grouped_paired_bootstrap_differences(
            ref_targets,
            ref_probabilities,
            cand_probabilities,
            source_ids,
            class_names,
            repeats=repeats,
            confidence=0.95,
            seed=seed,
        ),
    }


def _make_loader(
    *,
    config: Mapping[str, Any],
    class_names: Sequence[str],
    sensor_manifest: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[ManifestDataset, DataLoader[Any]]:
    data = config["data"]
    dataset = ManifestDataset(
        data["test_manifest"],
        class_names=class_names,
        image_size=int(data.get("image_size", 640)),
        sensor_manifest=sensor_manifest,
        preprocess=config.get("preprocess", {}),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    return dataset, loader


def _write_summary(
    output_dir: Path,
    report: Mapping[str, Any],
) -> None:
    _atomic_write(
        output_dir / "evaluation.json",
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    lines = [
        "# 软传感器来源迁移测试",
        "",
        "| 来源 | Accuracy | Macro F1 | ECE |",
        "|---|---:|---:|---:|",
    ]
    labels = {
        "qwen": "Qwen sidecar",
        "rule-v1": "rule-v1 sidecar",
        "missing": "缺失回退",
    }
    for source in SOURCES:
        metrics = report["sources"][source]["metrics"]
        lines.append(
            f"| {labels[source]} | {float(metrics['accuracy']):.2%} | "
            f"{float(metrics['macro_f1']):.2%} | {float(metrics['ece']):.2%} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "三种条件使用同一 v3 检查点、同一 826 条固定 test 对象和同一类别顺序；"
            "唯一运行时变化是传感器 sidecar 或缺失回退。该实验衡量来源迁移敏感性，"
            "不是 Qwen 与 rule-v1 各自独立训练后的训练数据消融。",
            "",
            "Qwen 与 rule-v1 都由输入图像条件生成，均不是独立硬件观测。配对区间按"
            "`source_image_id` 聚类，测试集也不是外部盲测集。",
            "",
        ]
    )
    _atomic_write(output_dir / "REPORT.md", "\n".join(lines))


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size <= 0:
        raise ValueError("batch size must be greater than zero")
    if args.num_workers < 0:
        raise ValueError("num workers cannot be negative")
    if args.bootstrap_repeats < 2 or args.paired_repeats < 2:
        raise ValueError("bootstrap repeats must be at least two")
    protocol_path = resolve_project_path(args.protocol)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_source_transfer_test":
        raise ValueError("sensor-source transfer protocol is not frozen")
    _validate_protocol_arguments(protocol, args)
    hash_checks = verify_protocol_files(protocol)

    config_path = resolve_project_path(args.config)
    checkpoint_path = resolve_project_path(args.checkpoint)
    rule_manifest = resolve_project_path(args.rule_sensor_manifest)
    output_dir = resolve_project_path(args.output)
    if output_dir.exists():
        existing = [
            path
            for path in output_dir.iterdir()
            if path.name != "PROTOCOL.json"
        ]
        if existing:
            raise FileExistsError(
                f"refusing to overwrite non-protocol output: {output_dir}"
            )

    config = load_config(config_path)
    qwen_manifest = resolve_project_path(config["data"]["sensor_manifest"])
    device = torch.device(args.device)
    checkpoint = load_checkpoint(checkpoint_path, device)
    class_names = validated_checkpoint_class_names(checkpoint, config)
    model = build_production_model(config).to(device)
    load_model_state_compatible(model, checkpoint["model_state"])
    runtime = _runtime_metadata(device)
    checkpoint_metadata = _checkpoint_metadata(checkpoint, checkpoint_path)

    source_results: dict[str, dict[str, Any]] = {}
    prediction_records: dict[str, list[dict[str, Any]]] = {}
    for index, source in enumerate(SOURCES):
        manifest = rule_manifest if source == "rule-v1" else qwen_manifest
        dataset, loader = _make_loader(
            config=config,
            class_names=class_names,
            sensor_manifest=manifest,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        degradation = (
            default_sensor_scenarios()["missing_100"]
            if source == "missing"
            else None
        )
        result = evaluate_model(
            model,
            loader,
            device=device,
            class_names=class_names,
            ece_bins=15,
            warmup_batches=3 if index == 0 else 0,
            sensor_degradation=degradation,
            degradation_seed=args.bootstrap_seed,
        )
        records = result["prediction_records"]
        targets, probabilities, source_ids, _ = _prediction_arrays(records)
        result["confidence_intervals"] = (
            grouped_bootstrap_classification_intervals(
                targets,
                probabilities,
                source_ids,
                class_names,
                repeats=args.bootstrap_repeats,
                confidence=0.95,
                seed=args.bootstrap_seed + index,
            )
        )
        result["dataset"] = {
            "split": "test",
            "manifest": str(resolve_project_path(config["data"]["test_manifest"])),
            "sensor_manifest": str(manifest),
            "evaluated_samples": len(dataset),
        }
        result["sensor_source"] = {
            "name": source,
            "runtime_intervention": True,
            "missing_degradation": source == "missing",
        }
        result["checkpoint"] = checkpoint_metadata
        result["class_names"] = list(class_names)
        result["runtime"] = runtime
        result["evaluation_config"] = {
            "config_path": str(config_path),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "bootstrap_repeats": args.bootstrap_repeats,
            "bootstrap_seed": args.bootstrap_seed + index,
        }
        write_evaluation_artifacts(
            result,
            output_dir / source,
            class_names=class_names,
            split=f"test / sensor source {source}",
        )
        source_results[source] = {
            "metrics": result["metrics"],
            "confidence_intervals": result["confidence_intervals"],
            "sensor_manifest": str(manifest),
        }
        prediction_records[source] = records

    comparisons = {
        name: compare_source_predictions(
            prediction_records[reference],
            prediction_records[candidate],
            class_names=class_names,
            reference_name=reference,
            candidate_name=candidate,
            repeats=args.paired_repeats,
            seed=args.bootstrap_seed + 100 + index,
        )
        for index, (name, reference, candidate) in enumerate(COMPARISONS)
    }
    report = {
        "schema_version": 1,
        "protocol": "same_checkpoint_sensor_source_transfer",
        "frozen_protocol": {
            "path": str(protocol_path),
            "sha256": _sha256(protocol_path),
            "input_hash_checks": hash_checks,
        },
        "sources": source_results,
        "paired_comparisons": comparisons,
        "sample_count": 826,
        "source_image_count": 570,
        "checkpoint": checkpoint_metadata,
        "limitations": protocol["limitations"],
    }
    _write_summary(output_dir, report)
    return {
        "state": "complete",
        "output": str(output_dir),
        "sources": {
            source: values["metrics"] for source, values in source_results.items()
        },
    }


def main() -> None:
    result = run_evaluation(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
