import hashlib
import json

from audit_paper_readiness import (
    EXPECTED_BIBLIOGRAPHY_COVERAGE,
    EXPECTED_BIBLIOGRAPHY_KEYS,
    EXPECTED_LATENT_UNCERTAINTY_SOURCES,
    EXPECTED_METHOD_SOURCES,
    EXPECTED_PAPER_FIGURES,
    EXPECTED_PAPER_SOURCE_REPORTS,
    EXPECTED_PAPER_TABLES,
    EXPECTED_UNCERTAINTY_COMPARISON_SOURCES,
    EXPECTED_UNCERTAINTY_SOURCES,
    _select_evidence_audit_path,
    _sequence_sha256,
    audit_final_claim_bundle,
    audit_bibliography_bundle,
    audit_latent_uncertainty_bundle,
    audit_method_contract_bundle,
    audit_paper_readiness,
    audit_paper_table_bundle,
    audit_scene_protocol_addenda_bundle,
    audit_scene_strata_reporting_recovery_bundle,
    audit_sensor_retraining_exit_bundle,
    audit_yolo_seg_dependency_incident_bundle,
    audit_yolo_seg_dependency_outcome_bundle,
    audit_uncertainty_floor_bundle,
    audit_uncertainty_comparison_bundle,
    audit_variant_lineage_bundle,
    patent_capabilities,
    _training_config_fingerprint,
)
from compare_uncertainty_diagnostics import build_uncertainty_comparison
from audit_uncertainty_floor import (
    _config_fingerprint,
    audit_uncertainty_floor,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pending_readiness_ignores_incomplete_strict_snapshot(tmp_path):
    final_path = tmp_path / "FINAL_EVIDENCE_AUDIT.json"
    pending_path = tmp_path / "FINAL_EVIDENCE_AUDIT.pending.json"
    final_path.write_text(
        json.dumps({"state": "pending", "valid": False}),
        encoding="utf-8",
    )
    pending_path.write_text(
        json.dumps({"state": "pending", "valid": False}),
        encoding="utf-8",
    )

    assert _select_evidence_audit_path(
        tmp_path,
        allow_pending=True,
    ) == pending_path
    assert _select_evidence_audit_path(
        tmp_path,
        allow_pending=False,
    ) == final_path


def test_patent_capabilities_keep_joint_head_hardware_and_rgbd_open():
    capabilities = patent_capabilities(
        {"yolo_seg": {}, "scene_pipeline": {}}
    )

    assert capabilities["target_domain_yolo_seg"]["status"] == "validated"
    assert capabilities["two_stage_scene_pipeline"]["status"] == (
        "validated"
    )
    assert capabilities["joint_multimodal_detector_head"]["status"] == (
        "not_implemented"
    )
    assert capabilities["rgbd_input"]["status"] == "not_implemented"
    assert capabilities[
        "synchronized_hardware_sensor_array"
    ]["status"] == "not_implemented"


def test_paper_table_bundle_checks_source_and_every_artifact_hash(
    tmp_path,
):
    final_audit = tmp_path / "FINAL_EVIDENCE_AUDIT.json"
    final_audit.write_text('{"valid":true}\n', encoding="utf-8")
    table_root = tmp_path / "paper-tables"
    table_root.mkdir()
    artifacts = (
        *EXPECTED_PAPER_TABLES,
        "TABLES.md",
        *EXPECTED_PAPER_FIGURES,
    )
    hashes = {}
    for name in artifacts:
        path = table_root / name
        path.write_text(name, encoding="utf-8")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 2,
        "source_audit_valid": True,
        "source_audit_sha256": hashlib.sha256(
            final_audit.read_bytes()
        ).hexdigest(),
        "artifact_sha256": hashes,
        "formal_ablation_scope": (
            "legacy_full_reference_not_v3_component_ablation"
        ),
        "main_paired_evidence_count": 2,
        "tables": {name: 1 for name in EXPECTED_PAPER_TABLES},
        "figures": list(EXPECTED_PAPER_FIGURES),
        "source_reports": {},
    }
    for name, relative in EXPECTED_PAPER_SOURCE_REPORTS.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path != final_audit:
            path.write_text(name, encoding="utf-8")
        manifest["source_reports"][name] = {
            "path": path.as_posix(),
            "sha256": _sha256(path),
        }
    (table_root / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    result = audit_paper_table_bundle(
        table_root,
        final_audit_path=final_audit,
    )

    assert result["valid"] is True
    assert result["artifact_count"] == 16
    assert result["source_report_count"] == 15
    (table_root / EXPECTED_PAPER_TABLES[0]).write_text(
        "changed",
        encoding="utf-8",
    )
    result = audit_paper_table_bundle(
        table_root,
        final_audit_path=final_audit,
    )
    assert result["valid"] is False
    assert any("hash mismatch" in error for error in result["errors"])
    source_path = tmp_path / EXPECTED_PAPER_SOURCE_REPORTS["v3_gate_paired"]
    source_path.write_text("source changed", encoding="utf-8")
    result = audit_paper_table_bundle(
        table_root,
        final_audit_path=final_audit,
    )
    assert result["valid"] is False
    assert any(
        "source report hash mismatch: v3_gate_paired" in error
        for error in result["errors"]
    )


def test_paper_table_bundle_reports_malformed_manifest(tmp_path):
    final_audit = tmp_path / "FINAL_EVIDENCE_AUDIT.json"
    final_audit.write_text("{}", encoding="utf-8")
    table_root = tmp_path / "paper-tables"
    table_root.mkdir()
    (table_root / "manifest.json").write_text("{", encoding="utf-8")

    result = audit_paper_table_bundle(
        table_root,
        final_audit_path=final_audit,
    )

    assert result["valid"] is False
    assert any("cannot be parsed" in error for error in result["errors"])


def test_final_claim_bundle_checks_both_upstream_hashes(tmp_path):
    final_audit = tmp_path / "FINAL_EVIDENCE_AUDIT.json"
    table_manifest = tmp_path / "paper-tables" / "manifest.json"
    table_manifest.parent.mkdir()
    final_audit.write_text("audit", encoding="utf-8")
    table_manifest.write_text("tables", encoding="utf-8")
    claims = {
        name: {"status": "evidence_complete"}
        for name in (
            "formal_ablations",
            "C13_same_checkpoint_sensor_transfer",
            "C14_scene_strata",
            "C15_independent_sensor_retraining",
            "target_domain_system",
        )
    }
    report = {
        "state": "complete",
        "source_final_audit_sha256": hashlib.sha256(
            final_audit.read_bytes()
        ).hexdigest(),
        "source_paper_table_manifest_sha256": hashlib.sha256(
            table_manifest.read_bytes()
        ).hexdigest(),
        "claims": claims,
    }
    (tmp_path / "FINAL_CLAIM_EVIDENCE.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    (tmp_path / "FINAL_CLAIM_EVIDENCE.md").write_text(
        "claims",
        encoding="utf-8",
    )

    result = audit_final_claim_bundle(
        tmp_path,
        final_audit_path=final_audit,
        paper_manifest_path=table_manifest,
    )

    assert result["valid"] is True
    table_manifest.write_text("changed", encoding="utf-8")
    result = audit_final_claim_bundle(
        tmp_path,
        final_audit_path=final_audit,
        paper_manifest_path=table_manifest,
    )
    assert result["valid"] is False


def test_final_claim_bundle_reports_malformed_json(tmp_path):
    (tmp_path / "FINAL_CLAIM_EVIDENCE.json").write_text(
        "{",
        encoding="utf-8",
    )
    (tmp_path / "FINAL_CLAIM_EVIDENCE.md").write_text(
        "claims",
        encoding="utf-8",
    )

    result = audit_final_claim_bundle(
        tmp_path,
        final_audit_path=tmp_path / "FINAL_EVIDENCE_AUDIT.json",
        paper_manifest_path=tmp_path / "paper-tables" / "manifest.json",
    )

    assert result["valid"] is False
    assert any("cannot be parsed" in error for error in result["errors"])


def test_readiness_enters_table_branch_with_strict_final_path(tmp_path):
    reports = tmp_path / "reports" / "qwen-material"
    reports.mkdir(parents=True)
    (reports / "FINAL_EVIDENCE_AUDIT.json").write_text(
        json.dumps(
            {
                "state": "complete",
                "valid": True,
                "error_count": 0,
                "sections": {},
            }
        ),
        encoding="utf-8",
    )
    (reports / "paper-tables").mkdir()

    result = audit_paper_readiness(
        reports_root=reports,
        allow_pending=False,
    )

    assert result["state"] == "invalid"
    assert any(
        "paper table manifest is missing" in error
        for error in result["errors"]
    )


def test_readiness_reports_missing_evidence_without_hashing_it(tmp_path):
    reports = tmp_path / "reports" / "qwen-material"
    reports.mkdir(parents=True)

    result = audit_paper_readiness(
        reports_root=reports,
        allow_pending=False,
    )

    assert result["state"] == "invalid"
    assert result["source_evidence_audit_sha256"] is None
    assert any("evidence audit is missing" in error for error in result["errors"])


def test_yolo_dependency_incident_bundle_validates_mirrored_sources(
    tmp_path,
):
    project = tmp_path / "project"
    reports = project / "reports" / "qwen-material"
    reports.mkdir(parents=True)
    mirror = (
        reports
        / "training"
        / "interrupted"
        / "yolo_seg_missing_polars_epoch1_20260728"
        / "sources"
    )
    original_names = (
        "formal_status",
        "formal_reporting_status",
        "yolo_status",
        "continuation_status",
        "yolo_stdout",
        "yolo_stderr",
        "continuation_yolo_stdout",
        "continuation_yolo_stderr",
        "run_manifest",
        "integrated_protocol",
        "stratified_protocol",
        "export_report",
        "requirements",
        "requirements_repro",
        "environment",
    )
    source_bindings = {}
    mirror_bindings = {}
    for name in original_names:
        path = mirror / f"{name}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        content = (
            "read_results_csv\n"
            "ModuleNotFoundError: No module named 'polars'\n"
            if name == "yolo_stderr"
            else name
        )
        path.write_text(content, encoding="utf-8")
        relative = path.relative_to(project).as_posix()
        binding = {
            "path": f"original/{name}.txt",
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        source_bindings[name] = binding
        mirror_bindings[name] = {
            "original_path": binding["path"],
            "path": relative,
            "size_bytes": binding["size_bytes"],
            "sha256": binding["sha256"],
        }
    archive = (
        project
        / "checkpoints"
        / "interrupted"
        / "yolo_seg_missing_polars_epoch1_20260728"
    )
    archive.mkdir(parents=True)
    archived_bindings = {}
    for name, content in {
        "args.yaml": "epochs: 100\n",
        "results.csv": "epoch,time\n1,10\n",
        "train_batch0.jpg": "0",
        "train_batch1.jpg": "1",
        "train_batch2.jpg": "2",
    }.items():
        path = archive / name
        path.write_text(content, encoding="utf-8")
        archived_bindings[name] = {
            "path": path.relative_to(project).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    incident_path = reports / "YOLO_SEG_DEPENDENCY_INCIDENT.json"
    incident_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state": "complete",
                "valid": True,
                "failure": {
                    "completed_epochs": 1,
                    "last_completed_epoch": 1,
                    "checkpoint_saved": False,
                    "missing_module": "polars",
                    "failure_point": (
                        "Ultralytics BaseTrainer.save_model/read_results_csv"
                    ),
                },
                "remediation": {
                    "decision": (
                        "clean_restart_same_frozen_protocol_from_epoch_1"
                    ),
                    "archived_file_count": 5,
                },
                "source_bindings": source_bindings,
                "archive_bindings": archived_bindings,
            }
        ),
        encoding="utf-8",
    )
    addendum_path = (
        reports / "YOLO_SEG_DEPENDENCY_INCIDENT_ADDENDUM.json"
    )
    addendum_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state": "complete",
                "valid": True,
                "incident": {
                    "size_bytes": incident_path.stat().st_size,
                    "sha256": _sha256(incident_path),
                },
                "source_mirror": {
                    "file_count": len(mirror_bindings),
                    "bindings": mirror_bindings,
                },
                "checkpoint_archive": {
                    "path": archive.relative_to(project).as_posix(),
                    "file_count": 5,
                    "bindings": archived_bindings,
                },
            }
        ),
        encoding="utf-8",
    )
    incident_path.with_suffix(".md").write_text("incident", encoding="utf-8")
    addendum_path.with_suffix(".md").write_text(
        "addendum", encoding="utf-8"
    )

    result = audit_yolo_seg_dependency_incident_bundle(
        incident_path,
        addendum_path,
        project_root=project,
    )

    assert result["valid"] is True
    assert result["source_mirror_count"] == 15
    assert result["checkpoint_archive_count"] == 5


def test_yolo_dependency_outcome_bundle_checks_bound_sources(tmp_path):
    project = tmp_path / "project"
    reports = project / "reports" / "qwen-material"
    reports.mkdir(parents=True)
    bindings = {}
    for index in range(23):
        source = project / f"source-{index}.txt"
        source.write_text(str(index), encoding="utf-8")
        bindings[f"source_{index}"] = {
            "path": source.relative_to(project).as_posix(),
            "size_bytes": source.stat().st_size,
            "sha256": _sha256(source),
        }
    report_path = reports / "YOLO_SEG_DEPENDENCY_OUTCOME.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state": "complete",
                "valid": True,
                "scope": (
                    "yolo_seg_clean_restart_after_dependency_interruption"
                ),
                "restart": {
                    "failed_run_completed_epochs": 1,
                    "failed_run_checkpoint_saved": False,
                    "formal_completed_epochs": 100,
                    "formal_epoch_sequence": [1, 100],
                    "resume": False,
                    "same_frozen_protocol": True,
                },
                "detector_test": {
                    "source_image_count": 570,
                    "object_count": 825,
                },
                "integrated_scene": {
                    "source_image_count": 570,
                    "ground_truth_object_count": 825,
                    "scenario_count": 2,
                },
                "source_bindings": bindings,
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    report_path.with_suffix(".md").write_text("outcome", encoding="utf-8")

    result = audit_yolo_seg_dependency_outcome_bundle(
        report_path,
        project_root=project,
    )

    assert result["valid"] is True
    assert result["source_binding_count"] == 23


def test_scene_protocol_addenda_bundle_checks_dependency_and_sources(
    tmp_path,
):
    project = tmp_path / "project"
    reports = project / "reports" / "qwen-material" / "yolo-seg"
    reports.mkdir(parents=True)

    def bindings(count):
        values = {}
        for index in range(count):
            source = project / f"source-{count}-{index}.txt"
            source.write_text(f"{count}-{index}", encoding="utf-8")
            values[f"source_{index}"] = {
                "path": source.relative_to(project).as_posix(),
                "size_bytes": source.stat().st_size,
                "sha256": _sha256(source),
            }
        return values

    integrated_path = (
        reports / "INTEGRATED_EVALUATION_PROTOCOL_ADDENDUM.json"
    )
    integrated_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": (
                    "frozen_after_path_identity_failure_before_recovery_rerun"
                ),
                "valid": True,
                "source_bindings": bindings(13),
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    stratified_path = (
        reports / "STRATIFIED_EVALUATION_PROTOCOL_ADDENDUM.json"
    )
    stratified_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": (
                    "frozen_after_integrated_path_addendum_before_"
                    "stratified_run"
                ),
                "valid": True,
                "dependency_protocol_addendum": {
                    "sha256": _sha256(integrated_path)
                },
                "invariants": {
                    "confidence_thresholds": [0.25, 0.8],
                    "bootstrap_repeats": 2000,
                    "bootstrap_seed": 20260727,
                    "expected_row_count": 42,
                    "metric_definitions_unchanged": True,
                },
                "source_bindings": bindings(12),
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    integrated_path.with_suffix(".md").write_text(
        "integrated", encoding="utf-8"
    )
    stratified_path.with_suffix(".md").write_text(
        "stratified", encoding="utf-8"
    )

    result = audit_scene_protocol_addenda_bundle(
        integrated_path,
        stratified_path,
        project_root=project,
    )

    assert result["valid"] is True
    assert result["source_binding_count"] == 25


def test_scene_strata_reporting_recovery_bundle_checks_unchanged_metrics(
    tmp_path,
):
    project = tmp_path / "project"
    reports = project / "reports" / "qwen-material" / "yolo-seg"
    reports.mkdir(parents=True)

    def bindings(prefix, count):
        values = {}
        for index in range(count):
            source = project / f"{prefix}-{index}.txt"
            source.write_text(f"{prefix}-{index}", encoding="utf-8")
            values[f"source_{index}"] = {
                "path": source.relative_to(project).as_posix(),
                "size_bytes": source.stat().st_size,
                "sha256": _sha256(source),
            }
        return values

    incident_path = reports / "SCENE_STRATA_REPORTING_INCIDENT.json"
    incident_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state": "complete",
                "valid": True,
                "failure": {"exception_type": "KeyError"},
                "partial_outputs": {"row_count": 42},
                "scope": {
                    "metric_computation_completed": True,
                    "metric_values_invalidated": False,
                },
                "archive": {"bindings": bindings("archive", 12)},
            }
        ),
        encoding="utf-8",
    )
    addendum_path = (
        reports / "SCENE_STRATA_REPORTING_PROTOCOL_ADDENDUM.json"
    )
    addendum_bindings = bindings("addendum", 17)
    addendum_bindings.pop("source_0")
    addendum_bindings.pop("source_1")
    archive_builder_path = project / "build_evidence_archive.py"
    archive_builder_path.write_text(
        "historical archive builder",
        encoding="utf-8",
    )
    addendum_bindings["evidence_archive_builder"] = {
        "path": archive_builder_path.relative_to(project).as_posix(),
        "size_bytes": archive_builder_path.stat().st_size,
        "sha256": _sha256(archive_builder_path),
    }
    audit_source = project / "audit_paper_readiness.py"
    audit_source.write_text("historical audit source", encoding="utf-8")
    addendum_bindings["paper_readiness_audit"] = {
        "path": audit_source.relative_to(project).as_posix(),
        "size_bytes": audit_source.stat().st_size,
        "sha256": _sha256(audit_source),
    }
    addendum_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": (
                    "frozen_after_metrics_complete_before_report_recovery"
                ),
                "valid": True,
                "incident": {"sha256": _sha256(incident_path)},
                "invariants": {
                    "metric_recomputation": False,
                    "training_rerun": False,
                    "detector_rerun": False,
                    "matching_rerun": False,
                    "bootstrap_rerun": False,
                    "expected_row_count": 42,
                    "confidence_thresholds": [0.25, 0.8],
                    "bootstrap_repeats": 2000,
                    "bootstrap_seed": 20260727,
                },
                "source_bindings": addendum_bindings,
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    hashes = {"evaluation_json": "e", "metrics_csv": "m"}
    outcome_path = (
        reports / "SCENE_STRATA_REPORTING_RECOVERY_OUTCOME.json"
    )
    outcome_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state": "complete",
                "valid": True,
                "incident": {"sha256": _sha256(incident_path)},
                "addendum": {"sha256": _sha256(addendum_path)},
                "recovery": {
                    "scope": (
                        "generate_missing_markdown_from_persisted_metrics"
                    ),
                    "metric_recomputation": False,
                    "training_rerun": False,
                    "detector_rerun": False,
                    "row_count": 42,
                    "metric_hashes_unchanged": True,
                    "before_hashes": hashes,
                    "after_hashes": hashes,
                },
                "source_bindings": bindings("outcome", 6),
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    for path in (incident_path, addendum_path, outcome_path):
        path.with_suffix(".md").write_text(path.stem, encoding="utf-8")

    archive_builder_path.write_text(
        "current non-algorithmic archive builder",
        encoding="utf-8",
    )
    audit_source.write_text("audit source", encoding="utf-8")
    audit_test = project / "tests" / "test_paper_readiness_audit.py"
    audit_test.parent.mkdir(parents=True)
    audit_test.write_text("audit test", encoding="utf-8")
    regression_bindings = {}
    for index, source in enumerate(
        (archive_builder_path, audit_source, audit_test)
    ):
        regression_bindings[f"source_{index}"] = {
            "path": source.relative_to(project).as_posix(),
            "size_bytes": source.stat().st_size,
            "sha256": _sha256(source),
        }
    regression_path = (
        project
        / "reports"
        / "qwen-material"
        / "FULL_REGRESSION_AUDIT.json"
    )
    regression_path.parent.mkdir(parents=True, exist_ok=True)
    regression_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state": "complete",
                "valid": True,
                "combined": {
                    "all_collected_tests_passed": True,
                    "unresolved_skips": 0,
                    "unresolved_failures": 0,
                },
                "source_bindings": regression_bindings,
                "errors": [],
            }
        ),
        encoding="utf-8",
    )

    result = audit_scene_strata_reporting_recovery_bundle(
        incident_path,
        addendum_path,
        outcome_path,
        project_root=project,
    )

    assert result["errors"] == []
    assert result["valid"] is True
    assert result["metric_hashes_unchanged"] is True
    assert result["source_binding_count"] == 35
    transitions = result["historical_source_transitions"]
    assert {item["name"] for item in transitions} == {
        "evidence_archive_builder",
        "paper_readiness_audit",
    }
    assert all(
        item["full_regression_audit"]["sha256"]
        == _sha256(regression_path)
        for item in transitions
    )

    regression_path.unlink()
    unbound = audit_scene_strata_reporting_recovery_bundle(
        incident_path,
        addendum_path,
        outcome_path,
        project_root=project,
    )
    assert unbound["valid"] is False
    assert any(
        "source transition is not regression-bound" in error
        for error in unbound["errors"]
    )


def test_sensor_retraining_exit_bundle_checks_skip_recovery(tmp_path):
    project = tmp_path / "project"
    reports = project / "reports" / "qwen-material"
    reports.mkdir(parents=True)

    def bindings(prefix, count):
        values = {}
        for index in range(count):
            source = project / f"{prefix}-{index}.txt"
            source.write_text(f"{prefix}-{index}", encoding="utf-8")
            values[f"source_{index}"] = {
                "path": source.relative_to(project).as_posix(),
                "size_bytes": source.stat().st_size,
                "sha256": _sha256(source),
            }
        return values

    incident_path = reports / "SENSOR_RETRAINING_EXIT_INCIDENT.json"
    incident_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state": "complete",
                "valid": True,
                "failure": {
                    "exit_code": -1073741819,
                    "stderr_empty": True,
                },
                "completed_training": {
                    "completed_epochs": 100,
                    "best_epoch": 91,
                    "last_epoch": 100,
                },
                "recovery": {
                    "decision": (
                        "reuse_valid_complete_training_bundle_and_"
                        "skip_retraining"
                    )
                },
                "source_bindings": bindings("incident", 16),
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    addendum_path = (
        reports / "SENSOR_RETRAINING_EXIT_INCIDENT_ADDENDUM.json"
    )
    addendum_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": (
                    "frozen_after_recovery_queue_path_failure_before_recovery"
                ),
                "valid": True,
                "incident": {"sha256": _sha256(incident_path)},
                "invariants": {"training_rerun": False},
                "source_bindings": bindings("addendum", 10),
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    original_log = project / "original.log"
    repaired_log = project / "repaired.log"
    original_log.write_text("log", encoding="utf-8")
    repaired_log.write_text("log", encoding="utf-8")
    encoding_repair_path = (
        reports / "SENSOR_RETRAINING_LOG_ENCODING_REPAIR.json"
    )
    encoding_repair_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state": "complete",
                "valid": True,
                "decoded_text_equal": True,
                "original": {
                    "path": original_log.relative_to(project).as_posix(),
                    "size_bytes": original_log.stat().st_size,
                    "sha256": _sha256(original_log),
                    "encoding": "utf-16-with-bom",
                },
                "repaired": {
                    "path": repaired_log.relative_to(project).as_posix(),
                    "size_bytes": repaired_log.stat().st_size,
                    "sha256": _sha256(repaired_log),
                    "encoding": "utf-8",
                },
            }
        ),
        encoding="utf-8",
    )
    encoding_addendum_path = (
        reports / "SENSOR_RETRAINING_EXIT_INCIDENT_ADDENDUM_2.json"
    )
    encoding_addendum_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": (
                    "frozen_after_outcome_log_encoding_failure_before_retry"
                ),
                "valid": True,
                "incident": {"sha256": _sha256(incident_path)},
                "first_addendum": {"sha256": _sha256(addendum_path)},
                "invariants": {
                    "training_rerun": False,
                    "checkpoint_audit_rerun": False,
                    "fixed_test_rerun": False,
                    "paired_comparison_rerun": False,
                },
                "source_bindings": bindings("encoding-addendum", 18),
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    outcome_path = reports / "SENSOR_RETRAINING_EXIT_OUTCOME.json"
    outcome_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state": "complete",
                "valid": True,
                "incident": {"sha256": _sha256(incident_path)},
                "recovery": {
                    "training_rerun": False,
                    "training_skip_verified": True,
                    "completed_epochs": 100,
                    "best_epoch": 91,
                    "last_epoch": 100,
                    "test_sample_count": 826,
                    "paired_bootstrap_repeats": 5000,
                },
                "source_bindings": bindings("outcome", 17),
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    for path in (
        incident_path,
        addendum_path,
        encoding_addendum_path,
        encoding_repair_path,
        outcome_path,
    ):
        path.with_suffix(".md").write_text(path.stem, encoding="utf-8")

    result = audit_sensor_retraining_exit_bundle(
        incident_path,
        addendum_path,
        encoding_addendum_path,
        encoding_repair_path,
        outcome_path,
        project_root=project,
    )

    assert result["valid"] is True
    assert result["training_rerun"] is False
    assert result["source_binding_count"] == 61


def test_method_contract_bundle_detects_source_hash_drift(tmp_path):
    project_root = tmp_path / "project"
    report_path = (
        project_root
        / "reports"
        / "qwen-material"
        / "METHOD_CONTRACT_AUDIT.json"
    )
    report_path.parent.mkdir(parents=True)
    sources = []
    for relative in EXPECTED_METHOD_SOURCES:
        source = project_root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(relative, encoding="utf-8")
        sources.append(
            {
                "path": relative,
                "size_bytes": source.stat().st_size,
                "sha256": _sha256(source),
            }
        )
    fingerprint = hashlib.sha256(
        json.dumps(
            sources,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    report_path.write_text(
        json.dumps(
            {
                "state": "complete",
                "valid": True,
                "config": EXPECTED_METHOD_SOURCES[-1],
                "source_fingerprint_sha256": fingerprint,
                "sources": sources,
                "checks": [{"name": "example", "passed": True}],
                "tensor_contract": {
                    "logits_shape": [2, 19],
                    "raw_grid_quality_shape": [2, 7],
                    "cross_modal_grid_quality_shape": [2, 22],
                    "predicted_grid_shape": [2, 1, 4, 4],
                    "sensor_grid_quality_dim": 7,
                    "sensor_cross_modal_quality_dim": 22,
                    "film_gamma_shape": [3, 512],
                    "film_beta_shape": [3, 512],
                    "film_affine_parameter_count": 3072,
                    "scale_prior_is_trainable": True,
                    "scale_prior_initial": [0.3, 0.5, 0.2],
                    "modality_weight_row_sums": [1.0, 1.0],
                    "effective_sensor_formula_max_abs_error": 0.0,
                    "visual_query_context_max_abs_error": 0.0,
                    "fusion_query_weights_max_abs_error": 0.0,
                    "modality_weight_formula_max_abs_error": 0.0,
                    "missing_reliability": 0.0,
                    "missing_fallback_max_abs_error": 0.0,
                },
                "errors": [],
            }
        ),
        encoding="utf-8",
    )

    result = audit_method_contract_bundle(
        report_path,
        project_root=project_root,
    )
    assert result["valid"] is True

    (project_root / EXPECTED_METHOD_SOURCES[1]).write_text(
        "changed",
        encoding="utf-8",
    )
    result = audit_method_contract_bundle(
        report_path,
        project_root=project_root,
    )
    assert result["valid"] is False
    assert any("source size mismatch" in error for error in result["errors"])
    assert any("source hash mismatch" in error for error in result["errors"])


def test_uncertainty_floor_bundle_recomputes_sources_and_metrics(tmp_path):
    project_root = tmp_path / "project"
    for name, paths in EXPECTED_UNCERTAINTY_SOURCES.items():
        config = {
            "experiment": {"name": name},
            "model": {
                "sensor_log_variance_min": -3.0,
                "sensor_log_variance_max": 2.0,
            },
            "loss": {
                "variance_prior_weight": 0.1,
                "variance_prior_target": -2.0,
            },
            "training": {"resume": False},
        }
        config_path = project_root / paths["config"]
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config), encoding="utf-8")
        history_path = project_root / paths["history"]
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(
            json.dumps(
                {
                    "experiment": {"name": name},
                    "config_fingerprint": _config_fingerprint(config),
                    "history": [
                        {
                            "epoch": 100,
                            "val_sensor_reconstruction": -1.11,
                            "val_sensor_variance_prior": 0.06,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
    report = audit_uncertainty_floor(
        project_root=project_root,
        variants=EXPECTED_UNCERTAINTY_SOURCES,
    )
    report_path = (
        project_root
        / "reports"
        / "qwen-material"
        / "UNCERTAINTY_FLOOR_AUDIT.json"
    )
    report_path.parent.mkdir(parents=True)
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = audit_uncertainty_floor_bundle(
        report_path,
        project_root=project_root,
    )
    assert result["valid"] is True
    assert result["source_count"] == 4

    drift_path = (
        project_root
        / EXPECTED_UNCERTAINTY_SOURCES[
            "full_cross_modal_quality"
        ]["history"]
    )
    drift_path.write_text("{}", encoding="utf-8")
    result = audit_uncertainty_floor_bundle(
        report_path,
        project_root=project_root,
    )
    assert result["valid"] is False
    assert any(
        "source hash mismatch" in error for error in result["errors"]
    )


def test_latent_uncertainty_bundle_binds_sources_and_recomputes_metrics(
    tmp_path,
):
    project_root = tmp_path / "project"
    config = {
        "model": {
            "sensor_log_variance_min": -3.0,
            "sensor_log_variance_max": 2.0,
        },
        "training": {"resume": False},
    }
    source_paths = {}
    for name, relative in EXPECTED_LATENT_UNCERTAINTY_SOURCES.items():
        path = project_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if name == "config":
            path.write_text(json.dumps(config), encoding="utf-8")
        elif name == "manifest":
            path.write_text('{"id":1}\n{"id":2}\n', encoding="utf-8")
        else:
            path.write_bytes(name.encode("utf-8"))
        source_paths[name] = path

    report_path = (
        project_root
        / "reports"
        / "qwen-material"
        / "full_cross_modal_quality"
        / "uncertainty"
        / "uncertainty_diagnostics.json"
    )
    report_path.parent.mkdir(parents=True)
    figure_path = report_path.with_suffix(".png")
    figure_path.write_bytes(b"figure")
    report_path.with_suffix(".md").write_text(
        "diagnostics",
        encoding="utf-8",
    )
    report = {
        "schema_version": 1,
        "valid": True,
        "diagnostic_scope": "sensor_encoder_latent_space",
        "claim_boundary": (
            "This is not physical sensor calibration or external validation."
        ),
        "dataset": {
            "split": "test",
            "evaluated_samples": 2,
            "full_split_samples": 2,
            "max_samples": None,
        },
        "checkpoint": {
            "config_fingerprint": _training_config_fingerprint(config),
        },
        "source_bindings": {
            name: {
                "path": relative,
                "size_bytes": source_paths[name].stat().st_size,
                "sha256": _sha256(source_paths[name]),
            }
            for name, relative in EXPECTED_LATENT_UNCERTAINTY_SOURCES.items()
        },
        "diagnostics": {
            "sample_count": 2,
            "latent_dimension": 3,
            "element_count": 6,
            "configured_bounds": {
                "log_variance_min": -3.0,
                "log_variance_max": 2.0,
            },
            "log_variance": {"minimum": -2.2},
            "standardized_residual": {
                "coverage_absolute_le_1": 0.8,
                "coverage_absolute_le_1_96": 0.97,
            },
            "variance_error_correlation": {
                "sample_mean": {"spearman": 0.8},
            },
            "variance_calibration": {
                "global_mean_predicted_variance": 0.1,
                "global_mean_squared_error": 0.01,
                "global_mse_to_variance_ratio": 0.1,
            },
            "derived_interpretation_values": {
                "predicted_variance_to_mse_ratio": 10.0,
                "minimum_log_variance_distance_above_lower_bound": 0.8,
                "coverage_gap_absolute_le_1_vs_standard_normal": (
                    0.8 - 0.6826894921370859
                ),
                "coverage_gap_absolute_le_1_96_vs_standard_normal": (
                    0.97 - 0.950004209703559
                ),
            },
        },
        "bootstrap": {
            "method": "source_image_cluster_percentile_bootstrap",
            "repeats": 2000,
            "confidence": 0.95,
            "seed": 20260727,
            "sample_count": 2,
            "cluster_count": 2,
            "source_group_assignment_sha256": _sequence_sha256(
                ["1", "2"]
            ),
            "metrics": {
                "sample_mean_spearman": {
                    "estimate": 0.8,
                    "lower": 0.8,
                    "upper": 0.8,
                    "probability_greater_than_zero": 1.0,
                    "replicates": [0.8] * 2000,
                },
                "predicted_variance_to_mse_ratio": {
                    "estimate": 10.0,
                    "lower": 10.0,
                    "upper": 10.0,
                    "probability_greater_than_zero": 1.0,
                    "replicates": [10.0] * 2000,
                },
            },
        },
        "figure": {
            "path": (
                "reports/qwen-material/full_cross_modal_quality/"
                "uncertainty/uncertainty_diagnostics.png"
            ),
            "size_bytes": figure_path.stat().st_size,
            "sha256": _sha256(figure_path),
        },
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = audit_latent_uncertainty_bundle(
        report_path,
        project_root=project_root,
    )

    assert result["valid"] is True
    assert result["source_count"] == 4
    source_paths["checkpoint"].write_bytes(b"changed")
    result = audit_latent_uncertainty_bundle(
        report_path,
        project_root=project_root,
    )
    assert result["valid"] is False
    assert any(
        "source size mismatch: checkpoint" in error
        for error in result["errors"]
    )
    assert any(
        "source hash mismatch: checkpoint" in error
        for error in result["errors"]
    )


def test_uncertainty_comparison_bundle_recomputes_bound_sources(tmp_path):
    reports_root = tmp_path / "reports" / "qwen-material"

    def make_source(ratio, spearman):
        return {
            "schema_version": 1,
            "valid": True,
            "diagnostic_scope": "sensor_encoder_latent_space",
            "dataset": {"split": "test", "max_samples": None},
            "source_bindings": {
                "manifest": {"sha256": "manifest", "size_bytes": 10},
                "sensor_manifest": {"sha256": "sensor", "size_bytes": 20},
            },
            "bootstrap": {
                "method": "source_image_cluster_percentile_bootstrap",
                "repeats": 3,
                "confidence": 0.95,
                "seed": 7,
                "sample_count": 2,
                "cluster_count": 2,
                "source_group_assignment_sha256": "groups",
                "metrics": {
                    "sample_mean_spearman": {
                        "estimate": spearman,
                        "replicates": [
                            spearman - 0.02,
                            spearman,
                            spearman + 0.02,
                        ],
                    },
                    "predicted_variance_to_mse_ratio": {
                        "estimate": ratio,
                        "replicates": [
                            ratio - 1.0,
                            ratio,
                            ratio + 1.0,
                        ],
                    },
                },
            },
            "diagnostics": {
                "sample_count": 2,
                "latent_dimension": 3,
                "element_count": 6,
                "derived_interpretation_values": {
                    "predicted_variance_to_mse_ratio": ratio,
                    "minimum_log_variance_distance_above_lower_bound": 0.7,
                },
                "variance_error_correlation": {
                    "elementwise": {"spearman": spearman - 0.1},
                    "sample_mean": {"spearman": spearman},
                    "latent_dimension_mean": {
                        "spearman": spearman + 0.05
                    },
                },
                "standardized_residual": {
                    "standard_deviation": 0.2,
                    "coverage_absolute_le_1": 0.99,
                    "coverage_absolute_le_1_96": 1.0,
                },
                "bound_proximity": {
                    "lower": {"within_0.1": 0.0},
                },
            },
        }

    sources = {
        "v2b": make_source(50.0, 0.80),
        "v3": make_source(40.0, 0.88),
    }
    source_paths = {}
    for name, relative in EXPECTED_UNCERTAINTY_COMPARISON_SOURCES.items():
        path = reports_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sources[name]), encoding="utf-8")
        source_paths[name] = path
    report_path = reports_root / "UNCERTAINTY_DIAGNOSTICS_COMPARISON.json"
    report = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "source_reports": {
            name: {
                "path": relative,
                "size_bytes": source_paths[name].stat().st_size,
                "sha256": _sha256(source_paths[name]),
            }
            for name, relative in (
                EXPECTED_UNCERTAINTY_COMPARISON_SOURCES.items()
            )
        },
        "comparison": build_uncertainty_comparison(
            sources["v2b"],
            sources["v3"],
        ),
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    report_path.with_suffix(".md").write_text(
        "comparison",
        encoding="utf-8",
    )

    result = audit_uncertainty_comparison_bundle(
        report_path,
        reports_root=reports_root,
    )

    assert result["valid"] is True
    assert result["source_count"] == 2
    source_paths["v3"].write_text("{}", encoding="utf-8")
    result = audit_uncertainty_comparison_bundle(
        report_path,
        reports_root=reports_root,
    )
    assert result["valid"] is False
    assert any(
        "source hash mismatch: v3" in error
        for error in result["errors"]
    )


def test_bibliography_bundle_binds_candidate_bank(tmp_path):
    project_root = tmp_path / "project"
    reports = project_root / "reports" / "qwen-material"
    reports.mkdir(parents=True)
    bibliography = reports / "reference_candidates.bib"
    bibliography.write_text("candidate bank", encoding="utf-8")
    report_path = reports / "BIBLIOGRAPHY_AUDIT.json"
    report = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "error_count": 0,
        "raw_entry_count": 20,
        "parsed_entry_count": 20,
        "citation_keys": sorted(EXPECTED_BIBLIOGRAPHY_KEYS),
        "domain_coverage": {
            name: {"keys": list(keys), "complete": True}
            for name, keys in EXPECTED_BIBLIOGRAPHY_COVERAGE.items()
        },
        "source": {
            "path": "reports/qwen-material/reference_candidates.bib",
            "size_bytes": bibliography.stat().st_size,
            "sha256": _sha256(bibliography),
        },
        "parser": {
            "name": "pandoc",
            "version": "pandoc test",
            "input_format": "biblatex",
            "output_format": "csljson",
        },
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    report_path.with_suffix(".md").write_text("audit", encoding="utf-8")

    result = audit_bibliography_bundle(
        report_path,
        project_root=project_root,
    )

    assert result["valid"] is True
    assert result["entry_count"] == 20
    bibliography.write_text("changed", encoding="utf-8")
    result = audit_bibliography_bundle(
        report_path,
        project_root=project_root,
    )
    assert result["valid"] is False
    assert any(
        "bibliography source hash mismatch" in error
        for error in result["errors"]
    )


def test_variant_lineage_bundle_recomputes_locked_configs(tmp_path):
    from audit_variant_lineage import (
        PROJECT_ROOT as SOURCE_ROOT,
        VARIANT_FILES,
        build_variant_lineage,
    )

    project_root = tmp_path / "project"
    config_root = project_root / "configs" / "ablations"
    config_root.mkdir(parents=True)
    for filename in VARIANT_FILES.values():
        source = SOURCE_ROOT / "configs" / "ablations" / filename
        (config_root / filename).write_bytes(source.read_bytes())
    report = build_variant_lineage(config_root)
    report_path = (
        project_root
        / "reports"
        / "qwen-material"
        / "VARIANT_LINEAGE_AUDIT.json"
    )
    report_path.parent.mkdir(parents=True)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    report_path.with_suffix(".md").write_text(
        "lineage",
        encoding="utf-8",
    )

    result = audit_variant_lineage_bundle(
        report_path,
        project_root=project_root,
    )

    assert result["valid"] is True
    assert result["variant_count"] == 4
    config = json.loads(
        (config_root / VARIANT_FILES["v3"]).read_text(encoding="utf-8")
    )
    config["model"]["dropout"] = 0.4
    (config_root / VARIANT_FILES["v3"]).write_text(
        json.dumps(config),
        encoding="utf-8",
    )
    result = audit_variant_lineage_bundle(
        report_path,
        project_root=project_root,
    )
    assert result["valid"] is False
    assert any(
        "current variant config lineage is invalid" in error
        for error in result["errors"]
    )
