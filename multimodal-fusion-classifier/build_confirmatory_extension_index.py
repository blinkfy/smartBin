"""Build the submission index for the v3 seed-replication extension."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from audit_v3_seed_replication_recovery import validate_recovery_addendum


PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_ROOT = PROJECT_ROOT / "reports" / "qwen-material"
SEED_ROOT = REPORTS_ROOT / "full_cross_modal_quality" / "seed-replication"
DEFAULT_OUTPUT = REPORTS_ROOT / "CONFIRMATORY_EXTENSION_INDEX.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the confirmatory submission-extension index"
    )
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


def _validate_bound_sources(document: Mapping[str, Any]) -> list[str]:
    errors = []
    bindings = document.get("source_bindings", {})
    if not isinstance(bindings, Mapping) or not bindings:
        return ["document source bindings are missing"]
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            errors.append(f"malformed source binding: {name}")
            continue
        raw_path = binding.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            errors.append(f"malformed source path: {name}")
            continue
        path = _resolve(Path(raw_path))
        try:
            path.relative_to(PROJECT_ROOT)
        except ValueError:
            errors.append(f"source escapes project: {name}")
            continue
        if not path.is_file():
            errors.append(f"source is missing: {name}")
            continue
        if (
            path.stat().st_size != binding.get("size_bytes")
            or _sha256(path) != binding.get("sha256")
        ):
            errors.append(f"source binding drift: {name}")
    return errors


def build_index(output_path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output_path = _resolve(output_path)
    markdown_path = output_path.with_suffix(".md")
    if output_path.exists() or markdown_path.exists():
        raise FileExistsError("confirmatory extension index is immutable")
    paths = {
        "builder_source": Path(__file__),
        "base_submission_index": REPORTS_ROOT / "SUBMISSION_INDEX.json",
        "base_final_audit": REPORTS_ROOT / "FINAL_EVIDENCE_AUDIT.json",
        "base_readiness_audit": REPORTS_ROOT / "PAPER_READINESS_AUDIT.json",
        "base_archive_manifest": (
            REPORTS_ROOT / "submission-evidence" / "archive-manifest.json"
        ),
        "protocol": SEED_ROOT / "PROTOCOL.json",
        "queue_incident": SEED_ROOT / "QUEUE_INCIDENT.json",
        "queue_recovery_addendum": SEED_ROOT / "QUEUE_RECOVERY_ADDENDUM.json",
        "outcome": SEED_ROOT / "OUTCOME.json",
        "paper_addendum": SEED_ROOT / "PAPER_ADDENDUM.json",
        "validation_summary": SEED_ROOT / "VALIDATION_SEED_SUMMARY.json",
        "test_summary": SEED_ROOT / "TEST_SEED_SUMMARY.json",
        "input_benchmark": (
            REPORTS_ROOT / "training" / "INPUT_PIPELINE_BENCHMARK.json"
        ),
        "fast_config": (
            PROJECT_ROOT
            / "configs"
            / "performance"
            / "full_cross_modal_quality_workers4.json"
        ),
        "fast_config_provenance": (
            PROJECT_ROOT
            / "configs"
            / "performance"
            / "full_cross_modal_quality_workers4.provenance.json"
        ),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("extension index inputs missing: " + ", ".join(missing))
    base_index = _read_json(paths["base_submission_index"])
    final_audit = _read_json(paths["base_final_audit"])
    readiness = _read_json(paths["base_readiness_audit"])
    archive_manifest = _read_json(paths["base_archive_manifest"])
    incident = _read_json(paths["queue_incident"])
    outcome = _read_json(paths["outcome"])
    paper = _read_json(paths["paper_addendum"])
    benchmark = _read_json(paths["input_benchmark"])
    fast_provenance = _read_json(paths["fast_config_provenance"])
    recovery = validate_recovery_addendum(
        protocol_path=paths["protocol"],
        addendum_path=paths["queue_recovery_addendum"],
    )
    errors = []
    if (
        base_index.get("state") != "complete"
        or base_index.get("gates", {}).get(
            "internal_paper_evidence_ready"
        )
        is not True
        or final_audit.get("state") != "complete"
        or final_audit.get("valid") is not True
        or readiness.get("state") != "complete"
        or readiness.get("valid") is not True
    ):
        errors.append("base paper-evidence audits are not valid")
    if (
        archive_manifest.get("archive_sha256")
        != _sha256(REPORTS_ROOT / "submission-evidence" / "paper-evidence.zip")
        or archive_manifest.get("checkpoints_included") is not False
        or archive_manifest.get("raw_data_included") is not False
    ):
        errors.append("base paper-evidence archive binding mismatch")
    if (
        recovery["valid"] is not True
        or incident.get("valid") is not True
        or outcome.get("state") != "complete"
        or outcome.get("valid") is not True
        or paper.get("state") != "complete"
        or paper.get("valid") is not True
    ):
        errors.append("confirmatory seed-replication evidence is not valid")
    errors.extend(_validate_bound_sources(outcome))
    errors.extend(_validate_bound_sources(paper))
    if (
        benchmark.get("valid") is not True
        or benchmark.get("recommended_num_workers") != 4
        or benchmark.get("recommended_speedup", 0.0) <= 1.0
        or fast_provenance.get("valid") is not True
    ):
        errors.append("input-equivalent performance evidence is not valid")
    validation = outcome.get("validation_summary", {})
    test = outcome.get("test_summary", {})
    if (
        validation.get("run_count") != 3
        or test.get("run_count") != 3
        or validation.get("sample_count_per_run") != 826
        or test.get("sample_count_per_run") != 826
    ):
        errors.append("three-seed summary identity mismatch")
    if errors:
        raise ValueError("; ".join(errors))

    validation_aggregate = validation["aggregate"]
    test_aggregate = test["aggregate"]
    index = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "scope": "confirmatory_extension_to_existing_paper_evidence",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "base_evidence": {
            "submission_index": _binding(paths["base_submission_index"]),
            "final_audit": _binding(paths["base_final_audit"]),
            "readiness_audit": _binding(paths["base_readiness_audit"]),
            "archive": _binding(
                REPORTS_ROOT / "submission-evidence" / "paper-evidence.zip"
            ),
            "archive_manifest": _binding(paths["base_archive_manifest"]),
        },
        "confirmatory_training_seed_replication": {
            "seeds": [42, 43, 44],
            "epochs_per_seed": 100,
            "validation_samples_per_seed": 826,
            "test_samples_per_seed": 826,
            "seed42_remains_main_model": True,
            "best_seed_replacement_performed": False,
            "validation_accuracy": validation_aggregate["accuracy"],
            "validation_macro_f1": validation_aggregate["macro_f1"],
            "test_accuracy": test_aggregate["accuracy"],
            "test_macro_f1": test_aggregate["macro_f1"],
        },
        "paper_ready_statement": {
            "english": (
                "Across three independent training seeds, the fixed v3 model "
                f"achieved test accuracy "
                f"{test_aggregate['accuracy']['mean'] * 100:.2f}% ± "
                f"{test_aggregate['accuracy']['sample_std'] * 100:.2f} "
                f"percentage points and Macro F1 "
                f"{test_aggregate['macro_f1']['mean'] * 100:.2f}% ± "
                f"{test_aggregate['macro_f1']['sample_std'] * 100:.2f} "
                "percentage points (mean ± sample standard deviation)."
            ),
            "chinese": (
                "固定 v3 配置在 3 个独立训练种子上的测试准确率为 "
                f"{test_aggregate['accuracy']['mean'] * 100:.2f}% ± "
                f"{test_aggregate['accuracy']['sample_std'] * 100:.2f} "
                "个百分点，Macro F1 为 "
                f"{test_aggregate['macro_f1']['mean'] * 100:.2f}% ± "
                f"{test_aggregate['macro_f1']['sample_std'] * 100:.2f} "
                "个百分点（均值 ± 样本标准差）。"
            ),
        },
        "engineering_extension": {
            "recommended_num_workers": benchmark["recommended_num_workers"],
            "measured_input_samples_per_second_baseline": benchmark[
                "baseline_samples_per_second"
            ],
            "measured_input_samples_per_second_recommended": benchmark[
                "recommended_samples_per_second"
            ],
            "measured_input_speedup": benchmark["recommended_speedup"],
            "future_config_only": True,
        },
        "remaining_boundaries": [
            "the fixed internal test was already inspected and is not external blind validation",
            "formal ablation variants still use one training seed",
            "the independent Qwen/rule-v1 retraining comparison still uses one training seed",
            "Qwen sidecars are image-conditioned soft sensors, not hardware measurements",
            "three seeds quantify descriptive training-seed variation, not deployment uncertainty",
            "the extension does not close RGB-D, hardware synchronization, or joint-head patent gaps",
        ],
        "source_bindings": {
            name: _binding(path) for name, path in paths.items()
        },
        "errors": [],
    }
    _atomic_write(
        output_path,
        json.dumps(index, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write(
        markdown_path,
        "\n".join(
            [
                "# Confirmatory paper-evidence extension",
                "",
                "## Training-seed stability",
                "",
                index["paper_ready_statement"]["english"],
                "",
                index["paper_ready_statement"]["chinese"],
                "",
                "| Split | Metric | Mean | Sample SD | Min | Max |",
                "|---|---|---:|---:|---:|---:|",
                (
                    "| validation | Accuracy | "
                    f"{validation_aggregate['accuracy']['mean']:.4f} | "
                    f"{validation_aggregate['accuracy']['sample_std']:.4f} | "
                    f"{validation_aggregate['accuracy']['min']:.4f} | "
                    f"{validation_aggregate['accuracy']['max']:.4f} |"
                ),
                (
                    "| validation | Macro F1 | "
                    f"{validation_aggregate['macro_f1']['mean']:.4f} | "
                    f"{validation_aggregate['macro_f1']['sample_std']:.4f} | "
                    f"{validation_aggregate['macro_f1']['min']:.4f} | "
                    f"{validation_aggregate['macro_f1']['max']:.4f} |"
                ),
                (
                    "| test | Accuracy | "
                    f"{test_aggregate['accuracy']['mean']:.4f} | "
                    f"{test_aggregate['accuracy']['sample_std']:.4f} | "
                    f"{test_aggregate['accuracy']['min']:.4f} | "
                    f"{test_aggregate['accuracy']['max']:.4f} |"
                ),
                (
                    "| test | Macro F1 | "
                    f"{test_aggregate['macro_f1']['mean']:.4f} | "
                    f"{test_aggregate['macro_f1']['sample_std']:.4f} | "
                    f"{test_aggregate['macro_f1']['min']:.4f} | "
                    f"{test_aggregate['macro_f1']['max']:.4f} |"
                ),
                "",
                "Seed 42 remains the main model. No test-based seed "
                "replacement was performed.",
                "",
                "## Runtime extension",
                "",
                (
                    "For future runs on the current machine, four DataLoader "
                    f"workers improved measured input throughput by "
                    f"{benchmark['recommended_speedup']:.2f}x while preserving "
                    "multi-epoch shuffled sample order and input tensor bytes."
                ),
                "",
                "This runtime configuration was not used to rewrite any "
                "completed scientific result.",
                "",
                "## Boundaries",
                "",
                *[f"- {item}" for item in index["remaining_boundaries"]],
                "",
            ]
        ),
    )
    return index


def main() -> None:
    report = build_index(parse_args().output)
    print(
        json.dumps(
            {
                "state": report["state"],
                "valid": report["valid"],
                "test_accuracy": report[
                    "confirmatory_training_seed_replication"
                ]["test_accuracy"],
                "test_macro_f1": report[
                    "confirmatory_training_seed_replication"
                ]["test_macro_f1"],
                "output": DEFAULT_OUTPUT.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
