from __future__ import annotations

import json

from audit_patent_aligned_joint_seg import (
    _feature_differences,
    _patent_document,
    extract_patent_claims,
)
from build_patent_requirement_matrix import (
    STATUS_LABELS,
    build_matrix,
    requirement_rows,
)
from mmfc.model import PROJECT_ROOT


def test_extracts_all_numbered_patent_claims() -> None:
    claims = extract_patent_claims(_patent_document())

    assert sorted(claims) == list(range(1, 11))
    assert all(anchor in claims[1] for anchor in ("S1", "S2", "S3", "S4"))
    assert "双向注意力" in claims[8]
    assert "多尺度" in claims[10]


def test_requirement_matrix_uses_only_declared_statuses() -> None:
    rows = requirement_rows()

    assert len(rows) >= 10
    assert len({row["id"] for row in rows}) == len(rows)
    assert all(row["status"] in STATUS_LABELS for row in rows)
    assert any(row["status"] == "implemented" for row in rows)
    assert any(row["status"] == "not_implemented" for row in rows)
    physical = next(row for row in rows if row["id"] == "C2-SENSOR")
    assert physical["status"] == "not_implemented"
    assert (
        "mmfc/physical_sensor.py::calibrate_physical_observation"
        in physical["implementation_evidence"]
    )
    assert (
        "mmfc/physical_release.py::release_sidecars"
        in physical["implementation_evidence"]
    )
    assert (
        "schemas/synchronized_hardware_attestation_v1.schema.json"
        in physical["implementation_evidence"]
    )
    assert "实体阵列记录仍为0" in physical["boundary"]
    evidence = [
        item
        for row in rows
        for item in (
            list(row["implementation_evidence"])
            + list(row["experimental_evidence"])
        )
    ]
    assert all(
        (PROJECT_ROOT / item.split("::", maxsplit=1)[0]).is_file()
        for item in evidence
    )


def test_generated_patent_matrix_preserves_utf8_chinese() -> None:
    report_root = PROJECT_ROOT / "reports" / "qwen-material"
    markdown = (report_root / "PATENT_REQUIREMENT_MATRIX.md").read_text(
        encoding="utf-8"
    )
    matrix = json.loads(
        (report_root / "PATENT_REQUIREMENT_MATRIX.json").read_text(
            encoding="utf-8"
        )
    )
    mojibake_markers = ("ä¸", "åˆ", "ï¼", "Â", "\ufffd")

    assert markdown.startswith("# 专利要求证据矩阵")
    assert matrix["requirements"][0]["requirement"].startswith("采集深度图像")
    assert any(
        "真实物理传感器测量" in item
        for item in matrix["paper_claim_policy"]["not_allowed_now"]
    )
    assert not any(marker in markdown for marker in mojibake_markers)
    serialized = json.dumps(matrix, ensure_ascii=False)
    assert not any(marker in serialized for marker in mojibake_markers)


def test_built_matrix_hash_binds_every_declared_evidence_file(tmp_path) -> None:
    matrix = build_matrix(output=tmp_path / "matrix.json")
    declared = {
        item.split("::", maxsplit=1)[0]
        for row in requirement_rows()
        for item in (
            list(row["implementation_evidence"])
            + list(row["experimental_evidence"])
        )
    }

    assert set(matrix["evidence_bindings"]) == declared
    assert matrix["summary"]["hardware_validation_ready"] is False
    assert matrix["summary"]["complete_patent_replication"] is False
    assert (
        matrix["evidence_bindings"][
            "reports/qwen-material/PHYSICAL_SENSOR_INTERFACE_AUDIT.json"
        ]["sha256"]
        == __import__("hashlib")
        .sha256(
            (
                PROJECT_ROOT
                / "reports/qwen-material/PHYSICAL_SENSOR_INTERFACE_AUDIT.json"
            ).read_bytes()
        )
        .hexdigest()
    )


def test_feature_difference_reports_maximum_and_mean() -> None:
    import torch

    left = [torch.tensor([0.0, 1.0]), torch.tensor([2.0])]
    right = [torch.tensor([2.0, 1.0]), torch.tensor([1.0])]

    maximum, mean = _feature_differences(left, right)

    assert maximum == [2.0, 1.0]
    assert mean == [1.0, 1.0]
