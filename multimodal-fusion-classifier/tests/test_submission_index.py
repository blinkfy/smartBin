import hashlib
import json

import pytest

from build_submission_index import CORE_ARTIFACTS, build_submission_index


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False),
        encoding="utf-8",
    )


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepare_reports(root):
    reports = root / "reports" / "qwen-material"
    final_path = reports / "FINAL_EVIDENCE_AUDIT.json"
    _write_json(
        final_path,
        {
            "state": "complete",
            "valid": True,
            "warning_count": 8,
            "sections": {"dataset": {}, "formal_ablations": {}},
        },
    )
    _write_json(
        reports / "PAPER_READINESS_AUDIT.json",
        {
            "state": "complete",
            "valid": True,
            "internal_paper_evidence_ready": True,
            "full_patent_reproduction_ready": False,
            "external_deployment_claim_ready": False,
            "source_evidence_audit_sha256": _sha256(final_path),
        },
    )
    table_manifest_path = reports / "paper-tables" / "manifest.json"
    _write_json(
        table_manifest_path,
        {
            "source_audit_valid": True,
            "source_audit_sha256": _sha256(final_path),
            "tables": {f"table-{index}.csv": 1 for index in range(10)},
        },
    )
    (reports / "paper-tables" / "TABLES.md").write_text(
        "tables",
        encoding="utf-8",
    )
    _write_json(
        reports / "FINAL_CLAIM_EVIDENCE.json",
        {
            "state": "complete",
            "source_final_audit_sha256": _sha256(final_path),
            "source_paper_table_manifest_sha256": _sha256(
                table_manifest_path
            ),
            "claims": {f"claim-{index}": {} for index in range(5)},
        },
    )
    (reports / "FINAL_CLAIM_EVIDENCE.md").write_text(
        "claims",
        encoding="utf-8",
    )
    for relative in CORE_ARTIFACTS:
        path = reports / relative
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    return reports


def test_build_submission_index_binds_final_artifacts(tmp_path, monkeypatch):
    reports = _prepare_reports(tmp_path)
    monkeypatch.setattr(
        "build_submission_index.PROJECT_ROOT",
        tmp_path.resolve(),
    )

    result = build_submission_index(reports)

    assert result["state"] == "complete"
    assert result["counts"]["paper_tables"] == 10
    assert result["gates"]["internal_paper_evidence_ready"] is True
    assert result["gates"]["full_patent_reproduction_ready"] is False
    assert (
        "UNCERTAINTY_DIAGNOSTICS_COMPARISON.json"
        in result["artifacts"]
    )
    assert (reports / "SUBMISSION_INDEX.json").is_file()
    assert (reports / "SUBMISSION_INDEX.md").is_file()


def test_build_submission_index_rejects_stale_claim_hash(
    tmp_path,
    monkeypatch,
):
    reports = _prepare_reports(tmp_path)
    monkeypatch.setattr(
        "build_submission_index.PROJECT_ROOT",
        tmp_path.resolve(),
    )
    claim_path = reports / "FINAL_CLAIM_EVIDENCE.json"
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    claim["source_final_audit_sha256"] = "0" * 64
    _write_json(claim_path, claim)

    with pytest.raises(ValueError, match="claim evidence"):
        build_submission_index(reports)


def test_build_submission_index_rejects_stale_readiness_hash(
    tmp_path,
    monkeypatch,
):
    reports = _prepare_reports(tmp_path)
    monkeypatch.setattr(
        "build_submission_index.PROJECT_ROOT",
        tmp_path.resolve(),
    )
    readiness_path = reports / "PAPER_READINESS_AUDIT.json"
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    readiness["source_evidence_audit_sha256"] = "0" * 64
    _write_json(readiness_path, readiness)

    with pytest.raises(ValueError, match="paper readiness"):
        build_submission_index(reports)
