"""Aggregate every registered v6 Qwen adaptation seed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Mapping

from mmfc.model import PROJECT_ROOT
from prepare_patent_stack_qwen_formal_protocol import (
    DEFAULT_OUTPUT as DEFAULT_PROTOCOL,
    PROTOCOL_NAME,
    RUN_PROTOCOL_NAME,
)


REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "joint-multimodal-seg-patent-qwen"
)
DEFAULT_OUTPUT = REPORT_ROOT / "FORMAL_ADAPTATION_MULTISEED_SUMMARY.json"
PRIMARY_METRICS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "metrics/precision(M)",
    "metrics/recall(M)",
    "metrics/mAP50(M)",
    "metrics/mAP50-95(M)",
)
CONDITIONS = (
    "clean_matched",
    "rgb_fallback",
    "shuffled",
    "missing",
    "noise_0_15",
    "bias_drift_0_25",
    "stuck",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate formal v6 Qwen adaptation seeds"
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return (
        path.resolve()
        if path.is_absolute()
        else (PROJECT_ROOT / path).resolve()
    )


def _read_json(path: Path) -> dict[str, Any]:
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


def _valid_binding(binding: Mapping[str, Any]) -> bool:
    raw_path = binding.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return False
    path = _resolve(Path(raw_path))
    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError:
        return False
    return (
        path.is_file()
        and path.stat().st_size == binding.get("size_bytes")
        and _sha256(path) == binding.get("sha256")
    )


def _valid_named_bindings(bindings: Mapping[str, Any]) -> bool:
    if not bindings:
        return False
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            return False
        candidate = dict(binding)
        candidate.setdefault("path", name)
        if not _valid_binding(candidate):
            return False
    return True


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def summarize_values(values: Iterable[float]) -> dict[str, Any]:
    samples = [float(value) for value in values]
    if not samples:
        raise ValueError("cannot summarize an empty sequence")
    return {
        "values": samples,
        "count": len(samples),
        "mean": mean(samples),
        "sample_standard_deviation": (
            stdev(samples) if len(samples) > 1 else 0.0
        ),
        "minimum": min(samples),
        "maximum": max(samples),
    }


def summarize_conditions(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    records = list(rows)
    if not records:
        raise ValueError("no condition rows supplied")
    summary: dict[str, Any] = {
        "conditions": {},
        "paired_differences": {},
    }
    for condition in CONDITIONS:
        summary["conditions"][condition] = {
            metric: summarize_values(
                row["metrics"][condition][metric] for row in records
            )
            for metric in PRIMARY_METRICS
        }
    for label, reference in (
        ("clean_minus_rgb_fallback", "rgb_fallback"),
        ("clean_minus_shuffled", "shuffled"),
        ("clean_minus_missing", "missing"),
        ("clean_minus_noise_0_15", "noise_0_15"),
        ("clean_minus_bias_drift_0_25", "bias_drift_0_25"),
        ("clean_minus_stuck", "stuck"),
    ):
        summary["paired_differences"][label] = {
            metric: summarize_values(
                float(row["metrics"]["clean_matched"][metric])
                - float(row["metrics"][reference][metric])
                for row in records
            )
            for metric in PRIMARY_METRICS
        }
    return summary


def _normalized_train_args(
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    train_args = dict(protocol.get("train_args", {}))
    train_args.pop("name", None)
    train_args.pop("seed", None)
    return train_args


def aggregate(
    protocol_path: Path = DEFAULT_PROTOCOL,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    protocol_path = _resolve(protocol_path)
    output = _resolve(output)
    protocol = _read_json(protocol_path)
    if (
        protocol.get("protocol") != PROTOCOL_NAME
        or protocol.get("run_protocol") != RUN_PROTOCOL_NAME
        or protocol.get("status")
        != "preregistered_awaiting_api_credentials_and_data"
        or protocol.get("training", {}).get("seeds") != [42, 43, 44]
        or not _valid_named_bindings(protocol.get("source_bindings", {}))
    ):
        raise ValueError("formal extension protocol is invalid or drifted")

    data_root = _resolve(Path(protocol["data_generation"]["output"]))
    data_paths = {
        "qwen_build": data_root / "BUILD_REPORT.json",
        "qwen_validation": data_root / "VALIDATION_REPORT.json",
        "qwen_observations": data_root / "observations.jsonl",
        "qwen_shuffled": data_root / "SHUFFLED_OBSERVATIONS.jsonl",
        "qwen_shuffle_report": data_root / "SHUFFLE_REPORT.json",
    }
    missing = [
        name for name, path in data_paths.items() if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "formal Qwen data evidence missing: " + ", ".join(missing)
        )
    build = _read_json(data_paths["qwen_build"])
    validation = _read_json(data_paths["qwen_validation"])
    shuffle = _read_json(data_paths["qwen_shuffle_report"])
    expected = int(
        protocol["data_generation"]["expected_source_image_count"]
    )
    coverage = float(validation.get("qwen_success_count", -1)) / expected
    observations_sha256 = _sha256(data_paths["qwen_observations"])
    shuffled_sha256 = _sha256(data_paths["qwen_shuffled"])
    errors: list[str] = []
    if (
        build.get("valid") is not True
        or build.get("mode") != "qwen"
        or validation.get("valid") is not True
        or validation.get("mode") != "qwen"
        or validation.get("source_image_count") != expected
        or validation.get("error_count") != 0
        or coverage
        < float(protocol["data_generation"]["minimum_qwen_coverage"])
        or validation.get("observations_sha256")
        != observations_sha256
        or shuffle.get("valid") is not True
        or shuffle.get("self_assignment_count") != 0
        or shuffle.get("split_membership_preserved") is not True
        or shuffle.get("sensor_payload_multiset_preserved") is not True
        or shuffle.get("bindings", {})
        .get("original_observations", {})
        .get("sha256")
        != observations_sha256
        or shuffle.get("bindings", {})
        .get("shuffled_observations", {})
        .get("sha256")
        != shuffled_sha256
    ):
        errors.append("formal Qwen data or derangement gate failed")

    rows: list[dict[str, Any]] = []
    bindings = {
        "aggregator_source": _binding(Path(__file__)),
        "formal_protocol": _binding(protocol_path),
        **{name: _binding(path) for name, path in data_paths.items()},
    }
    normalized_args = None
    parent_hash = protocol["lineage"]["parent_checkpoint"]["sha256"]
    training = protocol["training"]
    fusion = protocol["fusion"]
    for seed in training["seeds"]:
        name = f"yolo11s-seg-joint-patent-qwen-adapt-seed{seed}"
        run_root = REPORT_ROOT / name
        checkpoint = (
            PROJECT_ROOT
            / training["output_project"]
            / name
            / "weights"
            / "best.pt"
        )
        paths = {
            "run_protocol": run_root / "PROTOCOL.json",
            "evaluation": run_root / "evaluation.json",
            "conditions": run_root / "RELIABILITY_CONDITIONS.json",
            "checkpoint": checkpoint,
        }
        absent = [key for key, path in paths.items() if not path.is_file()]
        if absent:
            errors.append(
                f"seed {seed} files missing: {', '.join(absent)}"
            )
            continue
        run_protocol = _read_json(paths["run_protocol"])
        evaluation = _read_json(paths["evaluation"])
        conditions = _read_json(paths["conditions"])
        current_args = _normalized_train_args(run_protocol)
        if normalized_args is None:
            normalized_args = current_args
        completed_epochs = int(
            evaluation.get("training", {}).get("completed_epochs", 0)
        )
        fusion_args = run_protocol.get("fusion_args", {})
        if (
            run_protocol.get("protocol") != RUN_PROTOCOL_NAME
            or run_protocol.get("formal_paper_metric") is not True
            or run_protocol.get("seed") != seed
            or run_protocol.get("sensor_source", {}).get("mode") != "qwen"
            or run_protocol.get("parent_checkpoint", {}).get("sha256")
            != parent_hash
            or run_protocol.get("qwen_formal_gate", {}).get(
                "observations_sha256"
            )
            != observations_sha256
            or current_args != normalized_args
            or current_args.get("workers") != training["workers"]
            or current_args.get("batch") != training["batch_size"]
            or current_args.get("epochs") != training["epochs"]
            or current_args.get("patience") != training["patience"]
            or current_args.get("freeze") != training["freeze_layers"]
            or current_args.get("lr0") != training["learning_rate"]
            or current_args.get("weight_decay") != training["weight_decay"]
            or fusion_args.get("semantic_dim") != fusion["semantic_dim"]
            or fusion_args.get("sensor_hidden_dim")
            != fusion["sensor_hidden_dim"]
            or fusion_args.get("text_temperature")
            != fusion["text_temperature"]
            or fusion_args.get("reliability_calibration_weight")
            != fusion["reliability_calibration_weight"]
            or not 1 <= completed_epochs <= training["epochs"]
            or not _valid_named_bindings(
                run_protocol.get("source_bindings", {})
            )
            or evaluation.get("valid") is not True
            or evaluation.get("formal_paper_metric") is not True
            or evaluation.get("sensor_observations", {}).get("mode")
            != "qwen"
            or evaluation.get("sensor_observations", {}).get("sha256")
            != observations_sha256
            or evaluation.get("checkpoint", {}).get("sha256")
            != _sha256(checkpoint)
            or conditions.get("valid") is not True
            or conditions.get("formal_paper_metric") is not True
            or conditions.get("evidence_role")
            != "formal_qwen_patent_stack_adaptation_conditions"
            or conditions.get("test_source_image_count") != 570
            or conditions.get("bindings", {})
            .get("observations", {})
            .get("sha256")
            != observations_sha256
            or conditions.get("bindings", {})
            .get("shuffled_observations", {})
            .get("sha256")
            != shuffled_sha256
            or not _valid_named_bindings(
                conditions.get("bindings", {})
            )
        ):
            errors.append(f"seed {seed} run contract mismatch")
        metrics = {
            condition: conditions["conditions"][condition]["metrics"]
            for condition in CONDITIONS
        }
        rows.append(
            {
                "seed": seed,
                "run_name": name,
                "completed_epochs": completed_epochs,
                "checkpoint_sha256": _sha256(checkpoint),
                "metrics": metrics,
                "gate_ordering_all_passed": conditions.get(
                    "gate_ordering_all_passed"
                ),
            }
        )
        for label, path in paths.items():
            bindings[f"seed{seed}_{label}"] = _binding(path)
    if len(rows) != len(training["seeds"]):
        errors.append("not all registered adaptation seeds are complete")
    if errors:
        raise ValueError("; ".join(errors))

    summary = summarize_conditions(rows)
    result = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "formal_paper_metric": True,
        "evidence_role": "formal_qwen_patent_stack_adaptation_multiseed",
        "protocol": protocol["protocol"],
        "run_protocol": protocol["run_protocol"],
        "lineage": protocol["lineage"],
        "seeds": training["seeds"],
        "run_count": len(rows),
        "qwen_data": {
            "source_image_count": expected,
            "qwen_success_count": validation["qwen_success_count"],
            "fallback_count": validation["fallback_count"],
            "coverage": coverage,
            "observations_sha256": observations_sha256,
            "shuffled_observations_sha256": shuffled_sha256,
            "annotation_inputs": False,
            "hardware_measurement": False,
        },
        "runs": rows,
        "summary": summary,
        "claim_boundaries": protocol["claim_boundaries"],
        "source_bindings": bindings,
        "errors": [],
    }
    _atomic_text(
        output,
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
    )
    box_fallback = summary["paired_differences"][
        "clean_minus_rgb_fallback"
    ]["metrics/mAP50-95(B)"]
    mask_fallback = summary["paired_differences"][
        "clean_minus_rgb_fallback"
    ]["metrics/mAP50-95(M)"]
    box_shuffled = summary["paired_differences"][
        "clean_minus_shuffled"
    ]["metrics/mAP50-95(B)"]
    mask_shuffled = summary["paired_differences"][
        "clean_minus_shuffled"
    ]["metrics/mAP50-95(M)"]
    markdown = [
        "# v6 Qwen 正式适配三种子汇总",
        "",
        f"- Qwen 覆盖率：{coverage * 100:.2f}%",
        "- 种子：42、43、44；全部纳入，无测试集择优。",
        (
            "- clean 相对 RGB fallback 的 box/mask mAP50-95："
            f"{box_fallback['mean'] * 100:+.4f}±"
            f"{box_fallback['sample_standard_deviation'] * 100:.4f}/"
            f"{mask_fallback['mean'] * 100:+.4f}±"
            f"{mask_fallback['sample_standard_deviation'] * 100:.4f} 个百分点"
        ),
        (
            "- clean 相对 shuffled 的 box/mask mAP50-95："
            f"{box_shuffled['mean'] * 100:+.4f}±"
            f"{box_shuffled['sample_standard_deviation'] * 100:.4f}/"
            f"{mask_shuffled['mean'] * 100:+.4f}±"
            f"{mask_shuffled['sample_standard_deviation'] * 100:.4f} 个百分点"
        ),
        "",
        (
            "三个种子共享固定 v5 父检查点，只描述 Qwen 适配随机性；"
            "Qwen 观测是图像条件软传感器，不是硬件实测。"
        ),
        "",
    ]
    _atomic_text(output.with_suffix(".md"), "\n".join(markdown))
    return result


def main() -> None:
    args = parse_args()
    result = aggregate(args.protocol, args.output)
    print(
        json.dumps(
            {
                "state": result["state"],
                "valid": result["valid"],
                "formal_paper_metric": result["formal_paper_metric"],
                "run_count": result["run_count"],
                "qwen_coverage": result["qwen_data"]["coverage"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
