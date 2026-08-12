import hashlib
import json
from pathlib import Path

import pytest

from build_scene_strata_protocol_addendum import (
    DEFAULT_FILES,
    build_scene_strata_addendum,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture_files(tmp_path: Path) -> dict[str, Path]:
    files = {name: tmp_path / name for name in DEFAULT_FILES}
    files["original_strata_source"].write_text(
        "old strata source", encoding="utf-8"
    )
    files["revised_strata_source"].write_text(
        "\n".join(
            (
                "STRATIFIED_EVALUATION_PROTOCOL_ADDENDUM.json",
                "_validate_protocol_addenda",
                '"evaluate_scene_pipeline.py": integrated_source_hash',
                '"evaluate_scene_strata.py": stratified_source_hash',
                '"stratified_protocol_addendum"',
            )
        ),
        encoding="utf-8",
    )
    files["original_integrated_source"].write_text(
        "old integrated source", encoding="utf-8"
    )
    files["revised_integrated_source"].write_text(
        "revised integrated source", encoding="utf-8"
    )
    files["final_evidence_queue_source"].write_text(
        "evaluate_scene_strata.py\nscene-strata evaluation failed",
        encoding="utf-8",
    )
    for name in (
        "recovery_queue_source",
        "dataset_export",
        "builder_source",
    ):
        files[name].write_text(name, encoding="utf-8")

    _write_json(
        files["integrated_protocol"],
        {"status": "frozen_before_detector_training"},
    )
    _write_json(
        files["original_protocol"],
        {
            "status": "frozen_before_detector_training",
            "protocol": "preregistered_scene_stratified_analysis",
            "input_hashes": [
                {
                    "path": "evaluate_scene_strata.py",
                    "sha256": _sha256(files["original_strata_source"]),
                },
                {
                    "path": "evaluate_scene_pipeline.py",
                    "sha256": _sha256(files["original_integrated_source"]),
                },
            ],
            "strata": {
                "relative_bbox_area": {
                    "expected_support": {"small": 1, "medium": 1, "large": 1}
                },
                "scene_density": {
                    "expected_support": {"single": 1, "multi": 1}
                },
                "bbox_overlap_proxy": {
                    "expected_support": {"overlap": 1, "isolated": 1}
                },
            },
            "evaluation": {
                "confidence_thresholds": [0.25, 0.8],
                "match_iou": 0.5,
                "prediction_sources": ["detector", "missing", "rule-v1"],
            },
            "bootstrap": {
                "repeats": 2000,
                "confidence": 0.95,
                "seed": 20260727,
            },
        },
    )
    _write_json(
        files["dependency_protocol_addendum"],
        {
            "status": (
                "frozen_after_path_identity_failure_before_recovery_rerun"
            ),
            "valid": True,
            "original_protocol": {
                "sha256": _sha256(files["integrated_protocol"])
            },
            "implementation_fix": {
                "original_evaluation_source": {
                    "sha256": _sha256(files["original_integrated_source"])
                },
                "revised_evaluation_source": {
                    "sha256": _sha256(files["revised_integrated_source"])
                },
            },
        },
    )
    _write_json(
        files["integrated_report"],
        {
            "protocol": "fixed_source_grouped_scene_test",
            "frozen_protocol": {
                "addendum": {
                    "sha256": _sha256(
                        files["dependency_protocol_addendum"]
                    ),
                    "evaluation_source_sha256": _sha256(
                        files["revised_integrated_source"]
                    ),
                }
            },
        },
    )
    return files


def test_scene_strata_addendum_preserves_preregistered_contract(
    tmp_path, monkeypatch
):
    files = _fixture_files(tmp_path)
    monkeypatch.setattr(
        "build_scene_strata_protocol_addendum.PROJECT_ROOT", tmp_path
    )

    report = build_scene_strata_addendum(
        files=files,
        stratified_output=tmp_path / "stratified",
    )

    assert report["valid"] is True
    assert report["invariants"]["expected_row_count"] == 42
    assert report["invariants"]["bootstrap_repeats"] == 2000


def test_scene_strata_addendum_rejects_existing_result(
    tmp_path, monkeypatch
):
    files = _fixture_files(tmp_path)
    output = tmp_path / "stratified"
    output.mkdir()
    (output / "evaluation.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "build_scene_strata_protocol_addendum.PROJECT_ROOT", tmp_path
    )

    report = build_scene_strata_addendum(
        files=files,
        stratified_output=output,
    )

    assert report["valid"] is False
    assert "not empty before run" in report["errors"][-1]


def test_scene_strata_addendum_rejects_incomplete_source_map(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "build_scene_strata_protocol_addendum.PROJECT_ROOT", tmp_path
    )

    with pytest.raises(FileNotFoundError, match="source map is incomplete"):
        build_scene_strata_addendum(
            files={},
            stratified_output=tmp_path / "stratified",
        )


def test_archived_strata_source_matches_frozen_protocol():
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads(
        (
            root
            / "reports"
            / "qwen-material"
            / "yolo-seg"
            / "STRATIFIED_EVALUATION_PROTOCOL.json"
        ).read_text(encoding="utf-8-sig")
    )
    expected = next(
        entry["sha256"]
        for entry in protocol["input_hashes"]
        if entry["path"] == "evaluate_scene_strata.py"
    )
    archived = (
        root
        / "reports"
        / "qwen-material"
        / "training"
        / "interrupted"
        / "scene_path_identity_20260728"
        / "evaluate_scene_strata.pre_addendum.py"
    )

    assert _sha256(archived) == expected
