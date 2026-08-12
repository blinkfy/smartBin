from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from mmfc.external_blind import (
    difference_hash,
    hash_hamming_distance,
    sha256_file,
    validate_external_blind_bundle,
)


def _json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _jsonl(path: Path, values: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values),
        encoding="utf-8",
    )


def test_difference_hash_is_stable_and_distance_is_symmetric(
    tmp_path: Path,
) -> None:
    image = np.tile(np.arange(18, dtype=np.uint8), (12, 1)) * 10
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.fromarray(image).save(first)
    Image.fromarray(np.fliplr(image)).save(second)
    left = difference_hash(first)
    right = difference_hash(second)
    assert len(left) == 16
    assert left == difference_hash(first)
    assert hash_hamming_distance(left, right) == hash_hamming_distance(
        right, left
    )
    assert hash_hamming_distance(left, left) == 0


def test_hash_distance_rejects_incompatible_hashes() -> None:
    with pytest.raises(ValueError):
        hash_hamming_distance("00", "0000")
    with pytest.raises(ValueError):
        hash_hamming_distance("zz", "00")


def test_incomplete_external_bundle_is_invalid(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.json"
    integrity = tmp_path / "integrity.json"
    dhash = tmp_path / "dhash.jsonl"
    _json(protocol, {})
    _json(integrity, {})
    _jsonl(dhash, [])
    result = validate_external_blind_bundle(
        tmp_path / "missing",
        protocol_path=protocol,
        development_integrity_path=integrity,
        development_dhash_index_path=dhash,
    )
    assert result["valid"] is False
    assert result["state"] == "incomplete"
    assert "release" in " ".join(result["errors"])


def _valid_bundle(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    bundle = tmp_path / "bundle"
    images = bundle / "images"
    images.mkdir(parents=True)
    image_path = images / "external.png"
    pixels = np.zeros((32, 32, 3), dtype=np.uint8)
    pixels[:, :, 0] = np.arange(32, dtype=np.uint8)[None, :] * 7
    pixels[:, :, 1] = 80
    Image.fromarray(pixels).save(image_path)
    image_sha = sha256_file(image_path)
    external_dhash = difference_hash(image_path)
    width = len(external_dhash)
    complement = f"{((1 << (width * 4)) - 1) ^ int(external_dhash, 16):0{width}x}"

    class_names = [f"class_{index}" for index in range(19)]
    protocol = tmp_path / "protocol.json"
    checkpoint_sha = "a" * 64
    _json(
        protocol,
        {
            "schema_version": 1,
            "protocol": "external_blind_confirmation_v1",
            "status": "preregistered_awaiting_external_data",
            "frozen_at": "2026-07-29T00:00:00+08:00",
            "checkpoint": {"sha256": checkpoint_sha},
            "class_names": class_names,
            "intake_criteria": {
                "minimum_total_samples": 1,
                "minimum_source_images": 1,
                "minimum_samples_per_class": 0,
                "minimum_source_images_per_class": 0,
                "minimum_acquisition_batches": 1,
                "minimum_capture_sessions": 1,
                "minimum_label_reviewers": 1,
                "perceptual_hash_error_distance": 2,
                "perceptual_hash_review_distance": 6,
                "allowed_sensor_source_modes": ["qwen-soft-sensor"],
            },
        },
    )
    integrity = tmp_path / "integrity.json"
    _json(
        integrity,
        {
            "schema_version": 1,
            "valid": True,
            "manifest_bindings": [],
            "images": [
                {
                    "source_image_id": "development-source",
                    "effective_sha256": "b" * 64,
                }
            ],
        },
    )
    dhash = tmp_path / "dhash.jsonl"
    _jsonl(
        dhash,
        [
            {
                "source_image_id": "development-source",
                "dhash": complement,
            }
        ],
    )
    _jsonl(
        bundle / "manifest.jsonl",
        [
            {
                "sample_id": "external-sample",
                "source_image_id": "external-source",
                "acquisition_batch_id": "batch-1",
                "capture_session_id": "session-1",
                "image_path": "images/external.png",
                "image_sha256": image_sha,
                "bbox": [0, 0, 32, 32],
                "label": "class_0",
                "label_id": 0,
                "label_review": {
                    "reviewer_count": 1,
                    "consensus": True,
                },
            }
        ],
    )
    _jsonl(
        bundle / "sensors.jsonl",
        [
            {
                "sample_id": "external-sample",
                "normalized_grid": [[0.5] * 4 for _ in range(4)],
                "properties": {
                    "weight_g": 10.0,
                    "temperature_c": 22.0,
                    "moisture_pct": 5.0,
                    "conductivity_us_cm": 2.0,
                    "odor_index": 0.1,
                },
                "source": {
                    "mode": "qwen-soft-sensor",
                    "generator": "qwen3-vl-flash",
                    "annotation_inputs": False,
                    "class_inputs": False,
                },
            }
        ],
    )
    _json(
        bundle / "RELEASE.json",
        {
            "schema_version": 1,
            "dataset_id": "synthetic-external",
            "role": "external_blind_confirmation",
            "custodian_id": "test-custodian",
            "protocol": "external_blind_confirmation_v1",
            "protocol_sha256": sha256_file(protocol),
            "checkpoint_sha256": checkpoint_sha,
            "labels_unsealed_at": "2026-07-30T00:00:00+08:00",
            "prior_model_development_access": False,
            "one_time_evaluation_authorized": True,
            "class_names": class_names,
            "sensor_source_mode": "qwen-soft-sensor",
            "collection_site_ids": ["site-1"],
            "capture_device_ids": ["device-1"],
            "collection_started_at": "2026-07-28T00:00:00+08:00",
            "collection_completed_at": "2026-07-29T12:00:00+08:00",
            "sample_count": 1,
            "source_image_count": 1,
            "acquisition_batch_ids": ["batch-1"],
            "capture_session_ids": ["session-1"],
        },
    )
    return bundle, protocol, integrity, dhash


def test_valid_synthetic_external_bundle_passes(tmp_path: Path) -> None:
    bundle, protocol, integrity, dhash = _valid_bundle(tmp_path)
    result = validate_external_blind_bundle(
        bundle,
        protocol_path=protocol,
        development_integrity_path=integrity,
        development_dhash_index_path=dhash,
    )
    assert result["valid"] is True
    assert result["error_count"] == 0
    assert result["sample_count"] == 1
    assert result["sensor_record_count"] == 1
    assert result["development_overlap"] == {
        "sample_id_count": 0,
        "source_image_id_count": 0,
        "exact_image_hash_count": 0,
        "severe_perceptual_candidate_count": 0,
        "manual_review_candidate_count": 0,
    }


def test_synchronized_hardware_mode_is_bundle_contract_compatible(
    tmp_path: Path,
) -> None:
    bundle, protocol, integrity, dhash = _valid_bundle(tmp_path)
    protocol_value = json.loads(protocol.read_text(encoding="utf-8"))
    protocol_value["intake_criteria"]["allowed_sensor_source_modes"] = [
        "synchronized-hardware"
    ]
    _json(protocol, protocol_value)
    release = json.loads(
        (bundle / "RELEASE.json").read_text(encoding="utf-8")
    )
    release["protocol_sha256"] = sha256_file(protocol)
    release["sensor_source_mode"] = "synchronized-hardware"
    _json(bundle / "RELEASE.json", release)
    sensor = json.loads(
        (bundle / "sensors.jsonl").read_text(encoding="utf-8")
    )
    sensor["source"].update(
        {
            "mode": "synchronized-hardware",
            "generator": "hardware-calibrated-v1",
            "physical_hardware_origin_declared": True,
            "physical_hardware_origin_verified": False,
            "hardware_origin_attested_by_custodian": True,
        }
    )
    _jsonl(bundle / "sensors.jsonl", [sensor])

    result = validate_external_blind_bundle(
        bundle,
        protocol_path=protocol,
        development_integrity_path=integrity,
        development_dhash_index_path=dhash,
    )

    assert result["valid"] is True
    assert result["error_count"] == 0
    assert result["sensor_record_count"] == 1


def test_exact_development_image_overlap_is_rejected(
    tmp_path: Path,
) -> None:
    bundle, protocol, integrity, dhash = _valid_bundle(tmp_path)
    manifest = json.loads(
        (bundle / "manifest.jsonl").read_text(encoding="utf-8")
    )
    integrity_value = json.loads(integrity.read_text(encoding="utf-8"))
    integrity_value["images"][0]["effective_sha256"] = manifest[
        "image_sha256"
    ]
    _json(integrity, integrity_value)
    result = validate_external_blind_bundle(
        bundle,
        protocol_path=protocol,
        development_integrity_path=integrity,
        development_dhash_index_path=dhash,
    )
    assert result["valid"] is False
    assert result["development_overlap"]["exact_image_hash_count"] == 1
    assert "exact image overlaps" in " ".join(result["errors"])
