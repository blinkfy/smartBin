"""Build an authoritative index for the completed paper evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
CORE_ARTIFACTS = (
    "FINAL_EVIDENCE_AUDIT.json",
    "PAPER_READINESS_AUDIT.json",
    "FINAL_CLAIM_EVIDENCE.json",
    "FINAL_CLAIM_EVIDENCE.md",
    "MANUSCRIPT_STABLE_SECTIONS.md",
    "MANUSCRIPT_STABLE_SECTIONS_AUDIT.json",
    "MANUSCRIPT_STABLE_SECTIONS_AUDIT.md",
    "MANUSCRIPT_DECISION_BRIEF.md",
    "DATASET_CARD.md",
    "EVALUATION_GOVERNANCE.md",
    "STATISTICAL_PROTOCOL.md",
    "METHOD_SPEC.md",
    "PATENT_ALIGNMENT.md",
    "ARTIFACT_RELEASE_CHECKLIST.md",
    "FIGURE_CAPTIONS.md",
    "reference_candidates.bib",
    "THREATS_TO_VALIDITY.md",
    "paper-tables/manifest.json",
    "paper-tables/TABLES.md",
    "UNCERTAINTY_FLOOR_AUDIT.json",
    "UNCERTAINTY_FLOOR_AUDIT.md",
    "BIBLIOGRAPHY_AUDIT.json",
    "BIBLIOGRAPHY_AUDIT.md",
    "VARIANT_LINEAGE_AUDIT.json",
    "VARIANT_LINEAGE_AUDIT.md",
    "ERROR_ANALYSIS.json",
    "ERROR_ANALYSIS.md",
    "GATE_ERROR_ANALYSIS.json",
    "GATE_ERROR_ANALYSIS.md",
    "CLASS_METRIC_INTERVALS.json",
    "CLASS_METRIC_INTERVALS.md",
    "IMAGE_READ_RECOVERY_AUDIT.json",
    "IMAGE_READ_RECOVERY_AUDIT.md",
    "IMAGE_READ_RECOVERY_INCIDENT.json",
    "IMAGE_READ_RECOVERY_INCIDENT.md",
    "IMAGE_READ_RECOVERY_INCIDENT_ADDENDUM.json",
    "IMAGE_READ_RECOVERY_INCIDENT_ADDENDUM.md",
    "IMAGE_READ_RECOVERY_OUTCOME.json",
    "IMAGE_READ_RECOVERY_OUTCOME.md",
    "SOURCE_IMAGE_INTEGRITY_AUDIT.json",
    "SOURCE_IMAGE_INTEGRITY_AUDIT.md",
    "SOURCE_IMAGE_INTEGRITY_CURRENT.json",
    "SOURCE_IMAGE_INTEGRITY_CURRENT.md",
    "SOURCE_IMAGE_REPAIR_TRANSITION.json",
    "SOURCE_IMAGE_REPAIR_TRANSITION.md",
    "FORMAL_QUEUE_INTERRUPTION.json",
    "FORMAL_QUEUE_INTERRUPTION.md",
    "FORMAL_QUEUE_INTERRUPTION_ADDENDUM.json",
    "FORMAL_QUEUE_INTERRUPTION_ADDENDUM.md",
    "FORMAL_QUEUE_INTERRUPTION_OUTCOME.json",
    "FORMAL_QUEUE_INTERRUPTION_OUTCOME.md",
    "YOLO_SEG_DEPENDENCY_INCIDENT.json",
    "YOLO_SEG_DEPENDENCY_INCIDENT.md",
    "YOLO_SEG_DEPENDENCY_INCIDENT_ADDENDUM.json",
    "YOLO_SEG_DEPENDENCY_INCIDENT_ADDENDUM.md",
    "YOLO_SEG_DEPENDENCY_OUTCOME.json",
    "YOLO_SEG_DEPENDENCY_OUTCOME.md",
    "yolo-seg/INTEGRATED_EVALUATION_PROTOCOL_ADDENDUM.json",
    "yolo-seg/INTEGRATED_EVALUATION_PROTOCOL_ADDENDUM.md",
    "yolo-seg/STRATIFIED_EVALUATION_PROTOCOL_ADDENDUM.json",
    "yolo-seg/STRATIFIED_EVALUATION_PROTOCOL_ADDENDUM.md",
    "yolo-seg/SCENE_STRATA_REPORTING_INCIDENT.json",
    "yolo-seg/SCENE_STRATA_REPORTING_INCIDENT.md",
    "yolo-seg/SCENE_STRATA_REPORTING_PROTOCOL_ADDENDUM.json",
    "yolo-seg/SCENE_STRATA_REPORTING_PROTOCOL_ADDENDUM.md",
    "yolo-seg/SCENE_STRATA_REPORTING_RECOVERY_OUTCOME.json",
    "yolo-seg/SCENE_STRATA_REPORTING_RECOVERY_OUTCOME.md",
    "SENSOR_RETRAINING_EXIT_INCIDENT.json",
    "SENSOR_RETRAINING_EXIT_INCIDENT.md",
    "SENSOR_RETRAINING_EXIT_INCIDENT_ADDENDUM.json",
    "SENSOR_RETRAINING_EXIT_INCIDENT_ADDENDUM.md",
    "SENSOR_RETRAINING_EXIT_INCIDENT_ADDENDUM_2.json",
    "SENSOR_RETRAINING_EXIT_INCIDENT_ADDENDUM_2.md",
    "SENSOR_RETRAINING_LOG_ENCODING_REPAIR.json",
    "SENSOR_RETRAINING_LOG_ENCODING_REPAIR.md",
    "SENSOR_RETRAINING_EXIT_OUTCOME.json",
    "SENSOR_RETRAINING_EXIT_OUTCOME.md",
    "UNCERTAINTY_DIAGNOSTICS_COMPARISON.json",
    "UNCERTAINTY_DIAGNOSTICS_COMPARISON.md",
    (
        "full_cross_modal_quality/uncertainty/"
        "uncertainty_diagnostics.json"
    ),
    (
        "full_cross_modal_quality/uncertainty/"
        "uncertainty_diagnostics.md"
    ),
    (
        "full_cross_modal_quality/uncertainty/"
        "uncertainty_diagnostics.png"
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the final paper evidence handoff index"
    )
    parser.add_argument(
        "--reports",
        type=Path,
        default=PROJECT_ROOT / "reports" / "qwen-material",
    )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (
        PROJECT_ROOT / path
    ).resolve()


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


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    temporary.replace(path)


def build_submission_index(reports_root: Path) -> dict[str, Any]:
    reports_root = _resolve(reports_root)
    paths = {
        relative: reports_root / relative
        for relative in CORE_ARTIFACTS
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(
                f"required submission artifact is missing: {path}"
            )

    final_audit = _read_json(paths["FINAL_EVIDENCE_AUDIT.json"])
    readiness = _read_json(paths["PAPER_READINESS_AUDIT.json"])
    claims = _read_json(paths["FINAL_CLAIM_EVIDENCE.json"])
    tables = _read_json(paths["paper-tables/manifest.json"])
    if (
        final_audit.get("state") != "complete"
        or final_audit.get("valid") is not True
    ):
        raise ValueError("strict final evidence audit is not complete")
    if (
        readiness.get("state") != "complete"
        or readiness.get("valid") is not True
        or readiness.get("internal_paper_evidence_ready") is not True
    ):
        raise ValueError("internal paper evidence gate is not complete")
    if claims.get("state") != "complete":
        raise ValueError("dynamic claim evidence is not complete")

    final_hash = _sha256(paths["FINAL_EVIDENCE_AUDIT.json"])
    table_manifest_hash = _sha256(
        paths["paper-tables/manifest.json"]
    )
    if (
        tables.get("source_audit_valid") is not True
        or tables.get("source_audit_sha256") != final_hash
    ):
        raise ValueError("paper tables are not bound to the final audit")
    if claims.get("source_final_audit_sha256") != final_hash:
        raise ValueError("claim evidence is not bound to the final audit")
    if readiness.get("source_evidence_audit_sha256") != final_hash:
        raise ValueError("paper readiness is not bound to the final audit")
    if (
        claims.get("source_paper_table_manifest_sha256")
        != table_manifest_hash
    ):
        raise ValueError("claim evidence is not bound to the paper tables")
    table_count = len(tables.get("tables", {}))
    if table_count != 10:
        raise ValueError(f"expected 10 paper tables, found {table_count}")

    artifacts = {
        relative: {
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for relative, path in paths.items()
    }
    result = {
        "schema_version": 1,
        "state": "complete",
        "scope": "authoritative_internal_paper_evidence_index",
        "gates": {
            "internal_paper_evidence_ready": True,
            "full_patent_reproduction_ready": readiness.get(
                "full_patent_reproduction_ready"
            ),
            "external_deployment_claim_ready": readiness.get(
                "external_deployment_claim_ready"
            ),
        },
        "counts": {
            "paper_tables": table_count,
            "dynamic_claim_groups": len(claims.get("claims", {})),
            "final_audit_sections": len(
                final_audit.get("sections", {})
            ),
            "final_audit_warnings": final_audit.get(
                "warning_count"
            ),
        },
        "artifacts": artifacts,
        "boundaries": [
            "internal fixed benchmark, not external blind validation",
            "Qwen/rule-v1 sidecars are image-conditioned soft sensors",
            "single-seed retraining omits training-seed variance",
            "target-domain detector is separate from the MMFC feature head",
            (
                "latent uncertainty ranking improves, but absolute variance "
                "scale remains over-dispersed and its paired intervals are "
                "post-hoc diagnostics"
            ),
            (
                "one training image has separate damaged and repaired-state "
                "inventories; the write cause is unknown, and one corrupted "
                "partial checkpoint was archived before a full restart"
            ),
            "full patent reproduction and deployment readiness are separate gates",
        ],
    }
    json_path = reports_root / "SUBMISSION_INDEX.json"
    _atomic_write(
        json_path,
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
    )

    gate_rows = [
        (
            "内部论文证据",
            result["gates"]["internal_paper_evidence_ready"],
        ),
        (
            "完整专利复现",
            result["gates"]["full_patent_reproduction_ready"],
        ),
        (
            "外部部署主张",
            result["gates"]["external_deployment_claim_ready"],
        ),
    ]
    lines = [
        "# 最终投稿证据索引",
        "",
        "本文件由严格审计产物自动生成，是证据归档中的权威状态入口。",
        "工作期 `INDEX.md` 和 pending 快照只保留过程历史。",
        "",
        "## 就绪门",
        "",
        "| 门 | 状态 |",
        "|---|---|",
        *[
            f"| {name} | {'true' if value is True else 'false'} |"
            for name, value in gate_rows
        ],
        "",
        "## 核心产物",
        "",
        "| 产物 | SHA-256 |",
        "|---|---|",
        *[
            f"| `{relative}` | `{entry['sha256']}` |"
            for relative, entry in artifacts.items()
        ],
        "",
        "## 证据边界",
        "",
        *[f"- {boundary}；" for boundary in result["boundaries"]],
        "",
    ]
    markdown_path = reports_root / "SUBMISSION_INDEX.md"
    _atomic_write(markdown_path, "\n".join(lines))
    return {
        "state": "complete",
        "json": json_path.as_posix(),
        "markdown": markdown_path.as_posix(),
        **result,
    }


def main() -> None:
    args = parse_args()
    result = build_submission_index(args.reports)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
