from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import audit_v3_seed_replication as audit
import audit_v3_seed_replication_recovery as recovery_audit
import build_v3_seed_replication_addendum as addendum_builder
import build_v3_seed_replication_protocol as protocol_builder
from prepare_seed_replicates import build_seed_replicate
from summarize_seed_evaluations import summarize_runs


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _binding(project_root: Path, path: Path) -> dict[str, object]:
    return {
        "path": Path(os.path.relpath(path, project_root)).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _base_config() -> dict:
    return {
        "model": {"feature_dim": 512},
        "loss": {"focal_weight": 1.0},
        "training": {
            "seed": 42,
            "epochs": 100,
            "checkpoint_dir": (
                "checkpoints/ablations/full_cross_modal_quality"
            ),
        },
        "experiment": {
            "name": "full_cross_modal_quality",
            "description": "Frozen v3 model",
        },
    }


def _make_protocol_tree(
    tmp_path: Path,
    monkeypatch,
) -> tuple[Path, dict[str, Path]]:
    project_root = tmp_path / "project"
    project_root.mkdir()
    monkeypatch.setattr(audit, "PROJECT_ROOT", project_root)
    base = _base_config()
    paths: dict[str, Path] = {}
    paths["base_config"] = project_root / "base.json"
    _write_json(paths["base_config"], base)
    for seed in (43, 44):
        replicate = build_seed_replicate(
            base,
            seed=seed,
            checkpoint_root="checkpoints/ablations/v3_seed_replicates",
        )
        path = project_root / f"seed{seed}.json"
        _write_json(path, replicate)
        paths[f"seed{seed}_config"] = path
    stub = project_root / "source.txt"
    stub.write_text("frozen source", encoding="utf-8")
    yolo = tmp_path / "yolo11s-seg.pt"
    yolo.write_bytes(b"weights")
    for name in audit.REQUIRED_PROTOCOL_BINDINGS:
        paths.setdefault(name, stub)
    paths["yolo_initial_weights"] = yolo
    bindings = {
        name: _binding(project_root, paths[name])
        for name in audit.REQUIRED_PROTOCOL_BINDINGS
    }
    document = {
        "schema_version": 1,
        "status": "frozen_before_replicate_training",
        "valid": True,
        "replicates": {
            "seeds": [43, 44],
            "all_training_seeds": [42, 43, 44],
        },
        "source_bindings": bindings,
        "errors": [],
    }
    protocol_path = project_root / "PROTOCOL.json"
    _write_json(protocol_path, document)
    return protocol_path, paths


def test_replicate_validator_allows_only_seed_paths_and_metadata():
    base = _base_config()
    replicate = build_seed_replicate(
        base,
        seed=43,
        checkpoint_root="checkpoints/ablations/v3_seed_replicates",
    )

    valid = protocol_builder.validate_replicate_config(
        base,
        replicate,
        seed=43,
    )
    changed = copy.deepcopy(replicate)
    changed["model"]["feature_dim"] = 256
    invalid = protocol_builder.validate_replicate_config(
        base,
        changed,
        seed=43,
    )

    assert valid["valid"] is True
    assert invalid["valid"] is False
    assert "non-seed scientific configuration differs" in invalid["errors"]


def test_external_initial_weight_binding_is_relative(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    project_root.mkdir()
    initial_weight = tmp_path / "yolo11s-seg.pt"
    initial_weight.write_bytes(b"weights")
    monkeypatch.setattr(protocol_builder, "PROJECT_ROOT", project_root)

    binding = protocol_builder._binding(initial_weight)

    assert binding["path"] == "../yolo11s-seg.pt"


def test_protocol_preflight_accepts_bound_sources(tmp_path, monkeypatch):
    protocol_path, _ = _make_protocol_tree(tmp_path, monkeypatch)

    result = audit.validate_protocol(protocol_path)

    assert result["valid"] is True
    assert result["errors"] == []


def test_protocol_preflight_reports_source_drift(tmp_path, monkeypatch):
    protocol_path, paths = _make_protocol_tree(tmp_path, monkeypatch)
    seed43 = json.loads(paths["seed43_config"].read_text(encoding="utf-8"))
    seed43["model"]["feature_dim"] = 256
    _write_json(paths["seed43_config"], seed43)

    result = audit.validate_protocol(protocol_path)

    assert result["valid"] is False
    assert any("seed43_config" in error for error in result["errors"])
    assert any("scientific configuration differs" in error for error in result["errors"])


def test_protocol_preflight_handles_missing_bindings(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    project_root.mkdir()
    monkeypatch.setattr(audit, "PROJECT_ROOT", project_root)
    protocol_path = project_root / "PROTOCOL.json"
    _write_json(
        protocol_path,
        {
            "schema_version": 1,
            "status": "frozen_before_replicate_training",
            "valid": True,
            "replicates": {
                "seeds": [43, 44],
                "all_training_seeds": [42, 43, 44],
            },
            "source_bindings": {},
            "errors": [],
        },
    )

    result = audit.validate_protocol(protocol_path)

    assert result["valid"] is False
    assert "protocol source binding is missing: base_config" in result["errors"]


def test_seed_replication_queue_is_stage_level_and_uses_best_checkpoint():
    path = (
        Path(__file__).resolve().parents[1]
        / "reports"
        / "qwen-material"
        / "training"
        / "run_v3_seed_replication.ps1"
    )
    source = path.read_text(encoding="utf-8")

    assert "audit_v3_seed_replication.py" in source
    assert "TRAINING_PROCESS_EXIT.json" in source
    assert '"$checkpointDir/best.pt"' in source
    assert "-Wait" in source
    assert "PollSeconds" not in source
    assert "epoch_monitor" not in source


def test_seed_replication_postprocess_waits_without_epoch_polling():
    path = (
        Path(__file__).resolve().parents[1]
        / "reports"
        / "qwen-material"
        / "training"
        / "run_v3_seed_replication_postprocess.ps1"
    )
    source = path.read_text(encoding="utf-8")

    assert "Wait-Process" in source
    assert "build_v3_seed_replication_addendum.py" in source
    assert "tests/test_v3_seed_replication.py" in source
    assert "Get-Content -LiteralPath $Stdout" not in source
    assert "Start-Sleep" not in source
    assert "epoch_monitor" not in source


def test_recovery_audit_reads_history_object_rows():
    rows = [{"epoch": epoch} for epoch in range(1, 101)]

    assert recovery_audit._history_rows({"history": rows}) == rows
    assert recovery_audit._history_rows(rows) == rows
    assert recovery_audit._history_rows({"history": "invalid"}) == []


def test_recovery_queue_avoids_powershell_case_collision():
    path = (
        Path(__file__).resolve().parents[1]
        / "reports"
        / "qwen-material"
        / "training"
        / "run_v3_seed_replication_recovery.ps1"
    )
    source = path.read_text(encoding="utf-8")

    assert "$historyRows = @($historyDocument.history)" in source
    assert "$ReportBase = " in source
    assert "$seedReportDir = Join-Path" in source
    assert '$reportRoot = "$ReportRoot/seed${seed}"' not in source
    assert "REUSE seed43_training" in source
    assert "Invoke-FreshTraining -Seed 44" in source
    assert "audit_v3_seed_replication_recovery.py" in source


def _evaluation(seed: int, split: str, best_epoch: int) -> dict:
    return {
        "checkpoint": {"epoch": best_epoch},
        "dataset": {"split": split},
        "evaluation_config": {"bootstrap_repeats": 2000},
        "class_names": [f"class-{index}" for index in range(19)],
        "metrics": {
            "sample_count": 826,
            **{
                name: 0.5 + seed * 0.001 + index * 0.0001
                for index, name in enumerate(audit.AGGREGATE_METRICS)
            },
        },
    }


def _write_evaluation(
    path: Path,
    *,
    seed: int,
    split: str,
    best_epoch: int,
) -> dict:
    document = _evaluation(seed, split, best_epoch)
    _write_json(path, document)
    (path.parent / "predictions.jsonl").write_text(
        "{}\n" * 826,
        encoding="utf-8",
    )
    return document


def test_completed_replication_audit_closes_three_seed_bundle(
    tmp_path,
    monkeypatch,
):
    protocol_path, paths = _make_protocol_tree(tmp_path, monkeypatch)
    project_root = protocol_path.parent
    base_report = (
        project_root
        / "reports"
        / "qwen-material"
        / "full_cross_modal_quality"
    )
    base_val = _write_evaluation(
        base_report / "val" / "evaluation.json",
        seed=42,
        split="val",
        best_epoch=92,
    )
    base_test = _write_evaluation(
        base_report / "test" / "evaluation.json",
        seed=42,
        split="test",
        best_epoch=92,
    )
    paths["base_val_evaluation"] = base_report / "val" / "evaluation.json"
    paths["base_test_evaluation"] = base_report / "test" / "evaluation.json"
    report_root = protocol_path.parent
    checkpoint_root = (
        project_root / "checkpoints" / "ablations" / "v3_seed_replicates"
    )
    val_runs = [("seed42", base_val)]
    test_runs = [("seed42", base_test)]
    for seed, best_epoch in ((43, 91), (44, 93)):
        name = f"full_cross_modal_quality_seed{seed}"
        checkpoint_dir = checkpoint_root / name
        checkpoint_dir.mkdir(parents=True)
        best = checkpoint_dir / "best.pt"
        last = checkpoint_dir / "last.pt"
        best.write_bytes(f"best-{seed}".encode())
        last.write_bytes(f"last-{seed}".encode())
        _write_json(
            checkpoint_dir / "training_summary.json",
            {
                "completed_epochs": 100,
                "resumed_from": None,
                "checkpoints_saved": True,
            },
        )
        _write_json(
            checkpoint_dir / "history.json",
            [{"epoch": epoch} for epoch in range(1, 101)],
        )
        seed_report = report_root / f"seed{seed}"
        _write_json(
            seed_report / "CHECKPOINT_AUDIT.json",
            {
                "valid": True,
                "training_summary_completed_epochs": 100,
                "history_json_rows": 100,
                "best": {
                    "epoch": best_epoch,
                    "sha256": hashlib.sha256(best.read_bytes()).hexdigest(),
                },
                "last": {
                    "epoch": 100,
                    "sha256": hashlib.sha256(last.read_bytes()).hexdigest(),
                },
            },
        )
        _write_json(
            seed_report / "TRAINING_PROCESS_EXIT.json",
            {
                "seed": seed,
                "exit_code": 0,
                "zero_exit": True,
                "complete_training_bundle": True,
            },
        )
        val = _write_evaluation(
            seed_report / "val" / "evaluation.json",
            seed=seed,
            split="val",
            best_epoch=best_epoch,
        )
        test = _write_evaluation(
            seed_report / "test" / "evaluation.json",
            seed=seed,
            split="test",
            best_epoch=best_epoch,
        )
        val_runs.append((f"seed{seed}", val))
        test_runs.append((f"seed{seed}", test))
    for filename, runs in (
        ("VALIDATION_SEED_SUMMARY.json", val_runs),
        ("TEST_SEED_SUMMARY.json", test_runs),
    ):
        path = report_root / filename
        _write_json(path, summarize_runs(runs))
        path.with_suffix(".csv").write_text("summary\n", encoding="utf-8")

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    for name in ("base_val_evaluation", "base_test_evaluation"):
        protocol["source_bindings"][name] = _binding(
            project_root,
            paths[name],
        )
    _write_json(protocol_path, protocol)
    output = report_root / "OUTCOME.json"

    result = audit.audit_completed_replication(
        protocol_path=protocol_path,
        output_path=output,
    )

    assert result["valid"] is True
    assert result["training"]["seeds"] == [42, 43, 44]
    assert result["test_summary"]["run_count"] == 3
    assert "seed43_best_checkpoint" in result["source_bindings"]
    assert output.is_file()
    assert output.with_suffix(".md").is_file()


def test_paper_addendum_renders_three_seed_statistics(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    report_root = (
        project_root
        / "reports"
        / "qwen-material"
        / "full_cross_modal_quality"
        / "seed-replication"
    )
    report_root.mkdir(parents=True)
    monkeypatch.setattr(addendum_builder, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(addendum_builder, "REPORT_ROOT", report_root)
    protocol = report_root / "PROTOCOL.json"
    protocol.write_text("{}", encoding="utf-8")
    stub = project_root / "source.txt"
    stub.write_text("bound", encoding="utf-8")
    monkeypatch.setattr(addendum_builder, "BUILDER_SOURCE", stub)
    validation_runs = [
        (f"seed{seed}", _evaluation(seed, "val", 90 + seed % 3))
        for seed in (42, 43, 44)
    ]
    test_runs = [
        (f"seed{seed}", _evaluation(seed, "test", 90 + seed % 3))
        for seed in (42, 43, 44)
    ]
    validation_summary = summarize_runs(validation_runs)
    test_summary = summarize_runs(test_runs)
    _write_json(
        report_root / "VALIDATION_SEED_SUMMARY.json",
        validation_summary,
    )
    _write_json(report_root / "TEST_SEED_SUMMARY.json", test_summary)
    outcome_path = report_root / "OUTCOME.json"
    _write_json(
        outcome_path,
        {
            "state": "complete",
            "valid": True,
            "errors": [],
            "training": {
                "seeds": [42, 43, 44],
                "new_replicate_seeds": [43, 44],
                "epochs_per_run": 100,
                "resume_used": False,
                "best_seed_replacement_performed": False,
            },
            "validation_summary": validation_summary,
            "test_summary": test_summary,
            "source_bindings": {
                "stub": addendum_builder._binding(stub),
            },
        },
    )
    output = report_root / "PAPER_ADDENDUM.json"

    result = addendum_builder.build_addendum(
        outcome_path=outcome_path,
        output_path=output,
    )

    assert result["valid"] is True
    assert result["statistics"]["test"]["accuracy"]["sample_std"] > 0
    assert output.with_suffix(".md").is_file()
    assert output.with_name("training_seed_replication.csv").is_file()
