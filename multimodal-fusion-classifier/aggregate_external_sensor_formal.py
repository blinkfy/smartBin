"""Aggregate preregistered Qwen v2 runs across all formal seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "joint-multimodal-seg-external"
)
DEFAULT_PROTOCOL = REPORT_ROOT / "QWEN_FORMAL_PROTOCOL.json"
DEFAULT_OUTPUT = REPORT_ROOT / "FORMAL_MULTISEED_SUMMARY.json"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate full-image Qwen external-sensor formal seeds"
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


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


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def summarize_values(values: Iterable[float]) -> dict[str, Any]:
    samples = [float(value) for value in values]
    if not samples:
        raise ValueError("cannot summarize an empty sequence")
    return {
        "values": samples,
        "count": len(samples),
        "mean": mean(samples),
        "sample_standard_deviation": stdev(samples) if len(samples) > 1 else 0.0,
        "minimum": min(samples),
        "maximum": max(samples),
    }


def summarize_conditions(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    records = list(rows)
    if not records:
        raise ValueError("no condition rows supplied")
    conditions = ("external_matched", "rgb_fallback", "external_shuffled")
    summary: dict[str, Any] = {"conditions": {}, "paired_differences": {}}
    for condition in conditions:
        summary["conditions"][condition] = {
            metric: summarize_values(
                row["metrics"][condition][metric] for row in records
            )
            for metric in PRIMARY_METRICS
        }
    for comparison, reference in (
        ("matched_minus_fallback", "rgb_fallback"),
        ("matched_minus_shuffled", "external_shuffled"),
    ):
        summary["paired_differences"][comparison] = {
            metric: summarize_values(
                float(row["metrics"]["external_matched"][metric])
                - float(row["metrics"][reference][metric])
                for row in records
            )
            for metric in PRIMARY_METRICS
        }
    return summary


def _validate_binding(binding: Mapping[str, Any]) -> bool:
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


def _validate_named_bindings(bindings: Mapping[str, Any]) -> bool:
    if not bindings:
        return False
    for name, raw_binding in bindings.items():
        if not isinstance(raw_binding, Mapping):
            return False
        raw_path = raw_binding.get("path", name)
        path = _resolve(Path(str(raw_path)))
        try:
            path.relative_to(PROJECT_ROOT)
        except ValueError:
            return False
        if (
            not path.is_file()
            or path.stat().st_size != raw_binding.get("size_bytes")
            or _sha256(path) != raw_binding.get("sha256")
        ):
            return False
    return True


def _normalized_run_args(protocol: Mapping[str, Any]) -> dict[str, Any]:
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
    seeds = protocol.get("training", {}).get("seeds")
    if seeds != [42, 43, 44]:
        raise ValueError("formal protocol does not contain the three frozen seeds")
    if not all(
        _validate_binding(binding)
        for binding in protocol.get("source_bindings", {}).values()
    ):
        raise ValueError("formal protocol source binding drift")

    data_root = _resolve(Path(protocol["data_generation"]["output"]))
    build_path = data_root / "BUILD_REPORT.json"
    validation_path = data_root / "VALIDATION_REPORT.json"
    observations_path = data_root / "observations.jsonl"
    for path in (build_path, validation_path, observations_path):
        if not path.is_file():
            raise FileNotFoundError(f"formal Qwen data evidence missing: {path}")
    build = _read_json(build_path)
    validation = _read_json(validation_path)
    expected = int(protocol["data_generation"]["expected_source_image_count"])
    coverage = float(validation.get("qwen_success_count", -1)) / expected
    errors: list[str] = []
    if (
        build.get("valid") is not True
        or build.get("mode") != "qwen"
        or validation.get("valid") is not True
        or validation.get("source_image_count") != expected
        or validation.get("error_count") != 0
        or coverage
        < float(protocol["data_generation"]["minimum_qwen_coverage"])
        or validation.get("observations_sha256") != _sha256(observations_path)
    ):
        errors.append("formal Qwen data gate failed")

    rows = []
    bindings: dict[str, Any] = {
        "aggregator_source": _binding(Path(__file__)),
        "formal_protocol": _binding(protocol_path),
        "qwen_build": _binding(build_path),
        "qwen_validation": _binding(validation_path),
        "qwen_observations": _binding(observations_path),
    }
    normalized_args = None
    checkpoint_hashes = []
    for seed in seeds:
        name = f"yolo11s-seg-joint-external-qwen-seed{seed}"
        run_root = REPORT_ROOT / name
        run_paths = {
            "protocol": run_root / "PROTOCOL.json",
            "evaluation": run_root / "evaluation.json",
            "conditions": (
                run_root / "conditions" / "CONDITION_DIAGNOSTICS.json"
            ),
        }
        missing = [
            label for label, path in run_paths.items() if not path.is_file()
        ]
        if missing:
            errors.append(f"seed {seed} files missing: {', '.join(missing)}")
            continue
        run_protocol = _read_json(run_paths["protocol"])
        evaluation = _read_json(run_paths["evaluation"])
        conditions = _read_json(run_paths["conditions"])
        current_args = _normalized_run_args(run_protocol)
        leakage = run_protocol.get("leakage_control", {})
        if normalized_args is None:
            normalized_args = current_args
        if (
            run_protocol.get("seed") != seed
            or run_protocol.get("sensor_source", {}).get("mode") != "qwen"
            or current_args != normalized_args
            or current_args.get("workers")
            != protocol["training"]["workers"]
            or current_args.get("batch") != protocol["training"]["batch_size"]
            or current_args.get("epochs") != protocol["training"]["epochs"]
            or current_args.get("patience")
            != protocol["training"]["patience"]
            or current_args.get("freeze")
            != protocol["training"]["freeze_layers"]
            or current_args.get("lr0")
            != protocol["training"]["learning_rate"]
            or current_args.get("weight_decay")
            != protocol["training"]["weight_decay"]
            or run_protocol.get("fusion_args") != protocol["fusion"]
            or run_protocol.get("architecture", {}).get(
                "image_backbone_inputs"
            )
            != ["RGB"]
            or leakage.get("ground_truth_boxes_consumed_by_sensor") is not False
            or leakage.get("ground_truth_masks_consumed_by_sensor") is not False
            or leakage.get("ground_truth_classes_consumed_by_sensor") is not False
            or leakage.get("annotation_conditioned_spatial_projection")
            is not False
            or not _validate_named_bindings(
                run_protocol.get("source_bindings", {})
            )
            or evaluation.get("valid") is not True
            or evaluation.get("sensor_observations", {}).get("mode") != "qwen"
            or evaluation.get("sensor_observations", {}).get("sha256")
            != _sha256(observations_path)
        ):
            errors.append(f"seed {seed} training/evaluation contract mismatch")
        condition_contract = conditions.get("condition", {})
        if (
            conditions.get("valid") is not True
            or conditions.get("evidence_role")
            != "formal_negative_control_candidate"
            or condition_contract.get("self_assignment_count") != 0
            or condition_contract.get("test_source_image_count") != 570
            or condition_contract.get("split_preserved") is not True
            or condition_contract.get("sensor_distribution_preserved") is not True
            or conditions.get("bindings", {})
            .get("original_observations", {})
            .get("sha256")
            != _sha256(observations_path)
        ):
            errors.append(f"seed {seed} shuffled condition contract mismatch")
        condition_bindings = conditions.get("bindings", {})
        if not condition_bindings or not all(
            _validate_binding(binding)
            for binding in condition_bindings.values()
        ):
            errors.append(f"seed {seed} condition binding drift")
        checkpoint = Path(str(evaluation.get("checkpoint", {}).get("path", "")))
        if not checkpoint.is_file():
            errors.append(f"seed {seed} checkpoint missing")
        elif (
            checkpoint.stat().st_size
            != evaluation.get("checkpoint", {}).get("size_bytes")
            or _sha256(checkpoint)
            != evaluation.get("checkpoint", {}).get("sha256")
        ):
            errors.append(f"seed {seed} checkpoint binding drift")
        else:
            checkpoint_hashes.append(_sha256(checkpoint))
            bindings[f"seed{seed}_checkpoint"] = _binding(checkpoint)
        for label, path in run_paths.items():
            bindings[f"seed{seed}_{label}"] = _binding(path)
        rows.append(
            {
                "seed": seed,
                "run_name": name,
                "completed_epochs": evaluation.get("training", {}).get(
                    "completed_epochs"
                ),
                "checkpoint_sha256": evaluation.get("checkpoint", {}).get(
                    "sha256"
                ),
                "metrics": conditions["metrics"],
            }
        )

    if len(rows) != len(seeds):
        errors.append("not all preregistered seeds are complete")
    if len(checkpoint_hashes) != len(seeds):
        errors.append("not all checkpoint bindings are valid")
    if errors:
        raise ValueError("; ".join(errors))

    summary = summarize_conditions(rows)
    result = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "formal_paper_metric": True,
        "protocol": protocol["protocol"],
        "seeds": seeds,
        "run_count": len(rows),
        "qwen_data": {
            "source_image_count": expected,
            "qwen_success_count": validation["qwen_success_count"],
            "fallback_count": validation["fallback_count"],
            "coverage": coverage,
            "observations_sha256": _sha256(observations_path),
            "annotation_inputs": False,
            "hardware_measurement": False,
        },
        "runs": rows,
        "summary": summary,
        "claim_boundaries": [
            "Formal metrics use Qwen image-conditioned soft sensors, not hardware measurements.",
            "The three seeds and all three input conditions are included without test-set selection.",
            "No RGB-D input or synchronized physical sensor is present.",
        ],
        "source_bindings": bindings,
        "errors": [],
    }
    _atomic_write(
        output,
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
    )
    box = summary["paired_differences"]["matched_minus_fallback"][
        "metrics/mAP50-95(B)"
    ]
    mask = summary["paired_differences"]["matched_minus_fallback"][
        "metrics/mAP50-95(M)"
    ]
    shuffled_box = summary["paired_differences"]["matched_minus_shuffled"][
        "metrics/mAP50-95(B)"
    ]
    shuffled_mask = summary["paired_differences"]["matched_minus_shuffled"][
        "metrics/mAP50-95(M)"
    ]
    markdown = [
        "# Qwen 整图外部传感器三种子正式汇总",
        "",
        f"- Qwen 覆盖率：{coverage * 100:.2f}%",
        f"- 种子：{', '.join(str(seed) for seed in seeds)}",
        (
            "- matched 相对 RGB fallback 的 box/mask mAP50-95："
            f"{box['mean'] * 100:+.4f}±"
            f"{box['sample_standard_deviation'] * 100:.4f}/"
            f"{mask['mean'] * 100:+.4f}±"
            f"{mask['sample_standard_deviation'] * 100:.4f} 个百分点"
        ),
        (
            "- matched 相对 shuffled 的 box/mask mAP50-95："
            f"{shuffled_box['mean'] * 100:+.4f}±"
            f"{shuffled_box['sample_standard_deviation'] * 100:.4f}/"
            f"{shuffled_mask['mean'] * 100:+.4f}±"
            f"{shuffled_mask['sample_standard_deviation'] * 100:.4f} 个百分点"
        ),
        "",
        "以上结果覆盖全部预注册种子和三种输入条件；Qwen 观测仍是图像条件软传感器，"
        "不等同于真实硬件测量。",
        "",
    ]
    _atomic_write(output.with_suffix(".md"), "\n".join(markdown))
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
