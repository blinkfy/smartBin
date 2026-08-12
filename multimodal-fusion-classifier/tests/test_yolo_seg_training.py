import json
import csv
import hashlib
from pathlib import Path

import pytest

from train_yolo_seg import (
    _atomic_write_text,
    _json_safe,
    _read_training_summary,
    sha256_tree,
    validate_training_inputs,
    write_or_validate_run_manifest,
)


def test_atomic_write_text_uses_utf8_and_lf(tmp_path: Path):
    path = tmp_path / "nested" / "report.json"
    _atomic_write_text(path, '{"名称": "分割"}\n')

    assert path.read_bytes() == '{"名称": "分割"}\n'.encode("utf-8")
    assert not path.with_suffix(".json.tmp").exists()


def test_run_manifest_is_reused_without_timestamp_drift(tmp_path: Path):
    path = tmp_path / "RUN_MANIFEST.json"
    first = {"created_at_utc": "first", "inputs": {"sha256": "abc"}}
    second = {"created_at_utc": "second", "inputs": {"sha256": "abc"}}

    written, created = write_or_validate_run_manifest(path, first)
    reused, created_again = write_or_validate_run_manifest(path, second)

    assert created is True
    assert created_again is False
    assert written == reused == first
    assert json.loads(path.read_text(encoding="utf-8")) == first

    with pytest.raises(ValueError, match="differs"):
        write_or_validate_run_manifest(
            path,
            {"created_at_utc": "third", "inputs": {"sha256": "changed"}},
        )


def test_sha256_tree_is_relative_name_and_content_stable(tmp_path: Path):
    root = tmp_path / "dataset"
    first = root / "labels" / "train" / "b.txt"
    second = root / "labels" / "val" / "a.txt"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("1 0 0 1 0 1 1\n", encoding="utf-8")
    second.write_text("0 0 0 1 0 1 1\n", encoding="utf-8")

    result = sha256_tree([first, second], root=root)
    reverse = sha256_tree([second, first], root=root)

    assert result == reverse
    assert result["file_count"] == 2
    assert result["size_bytes"] == first.stat().st_size + second.stat().st_size
    assert len(result["sha256"]) == hashlib.sha256().digest_size * 2


def test_read_training_summary_selects_mask_map(tmp_path: Path):
    results_path = tmp_path / "results.csv"
    with results_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "metrics/mAP50-95(B)",
                "metrics/mAP50-95(M)",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "epoch": "1",
                "metrics/mAP50-95(B)": "0.60",
                "metrics/mAP50-95(M)": "0.40",
            }
        )
        writer.writerow(
            {
                "epoch": "2",
                "metrics/mAP50-95(B)": "0.55",
                "metrics/mAP50-95(M)": "0.45",
            }
        )

    summary = _read_training_summary(results_path)

    assert summary["completed_epochs"] == 2
    assert summary["best_recorded_epoch_index"] == 1
    assert summary["best_recorded_epoch"]["epoch"] == pytest.approx(2.0)


def test_json_safe_converts_path_and_scalar(tmp_path: Path):
    class Scalar:
        def item(self):
            return 0.25

    result = _json_safe({"path": tmp_path, "score": Scalar(), "items": (1, 2)})

    assert result["path"] == tmp_path.resolve().as_posix()
    assert result["score"] == pytest.approx(0.25)
    assert result["items"] == [1, 2]


def test_validate_training_inputs_rejects_bad_hyperparameters(tmp_path: Path):
    with pytest.raises(ValueError, match="epochs"):
        validate_training_inputs(
            data_path=tmp_path / "missing.yaml",
            weights_path=tmp_path / "missing.pt",
            checkpoint_project=tmp_path / "runs",
            run_name="run",
            report_dir=tmp_path / "reports",
            epochs=0,
            patience=100,
            batch_size=16,
            image_size=640,
            workers=0,
        )
