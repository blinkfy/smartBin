"""Prepare and validate the preregistered rule-v1 retraining config."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "sensor-source-retraining"
    / "PROTOCOL.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "sensor-source-retraining"
    / "resolved_rule_v1_config.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the frozen rule-v1 independent retraining config"
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_text(config: Mapping[str, Any]) -> str:
    return json.dumps(config, ensure_ascii=False, indent=2) + "\n"


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_rule_v1_config(
    base_config: Mapping[str, Any],
) -> dict[str, Any]:
    config = json.loads(json.dumps(base_config))
    config["data"]["sensor_manifest"] = "data/generated/sensors.jsonl"
    config["training"]["checkpoint_dir"] = (
        "checkpoints/source-transfer/full_cross_modal_quality_rule_v1"
    )
    config["experiment"] = {
        "name": "full_cross_modal_quality_rule_v1",
        "group": "sensor_source_retraining",
        "description": (
            "Independent seed-42 retraining of the locked v3 architecture "
            "with rule-v1 sidecars; all non-sensor data and hyperparameters "
            "match the Qwen-trained reference"
        ),
        "formal_retraining_required": True,
        "reference_checkpoint": (
            "checkpoints/ablations/full_cross_modal_quality/best.pt"
        ),
        "reference_sensor_source": "qwen",
        "candidate_sensor_source": "rule-v1",
    }
    return config


def validate_sensor_alignment(
    *,
    qwen_sensor_path: Path,
    rule_sensor_path: Path,
    split_paths: Mapping[str, Path],
) -> dict[str, Any]:
    qwen_records = _read_jsonl(qwen_sensor_path)
    rule_records = _read_jsonl(rule_sensor_path)

    def index(records: list[dict[str, Any]], name: str) -> dict[str, Any]:
        indexed: dict[str, Any] = {}
        for record in records:
            sample_id = str(record.get("sample_id", ""))
            if not sample_id or sample_id in indexed:
                raise ValueError(
                    f"{name} contains a missing or duplicate sample_id"
                )
            indexed[sample_id] = record
        return indexed

    qwen = index(qwen_records, "qwen sensors")
    rule = index(rule_records, "rule-v1 sensors")
    if set(qwen) != set(rule):
        raise ValueError("Qwen and rule-v1 sensor sample IDs differ")
    split_counts: dict[str, int] = {}
    all_split_ids: list[str] = []
    for split, path in split_paths.items():
        records = _read_jsonl(path)
        sample_ids = [str(record["sample_id"]) for record in records]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError(f"{split} manifest contains duplicate sample IDs")
        if not set(sample_ids) <= set(qwen):
            raise ValueError(
                f"{split} manifest is not covered by both sensor sources"
            )
        split_counts[split] = len(sample_ids)
        all_split_ids.extend(sample_ids)
    if len(all_split_ids) != len(set(all_split_ids)):
        raise ValueError("train/val/test sample IDs are not disjoint")
    if set(all_split_ids) != set(qwen):
        raise ValueError("split manifests do not cover all sensor records")
    return {
        "qwen_sensor_count": len(qwen),
        "rule_v1_sensor_count": len(rule),
        "common_sample_count": len(qwen),
        "split_counts": split_counts,
        "sample_id_sets_equal": True,
        "split_ids_disjoint": True,
        "all_records_covered": True,
    }


def _validate_input_hashes(
    protocol: Mapping[str, Any],
) -> list[dict[str, str]]:
    checks = []
    for entry in protocol.get("input_hashes", []):
        path = _resolve(Path(str(entry["path"])))
        expected = str(entry["sha256"]).lower()
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"protocol input hash mismatch for {path}: "
                f"{actual} != {expected}"
            )
        checks.append({"path": path.as_posix(), "sha256": actual})
    if not checks:
        raise ValueError("protocol does not contain input hashes")
    return checks


def prepare_config(
    *,
    protocol_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    protocol_path = _resolve(protocol_path)
    output_path = _resolve(output_path)
    protocol = _read_json(protocol_path)
    if protocol.get("status") != "frozen_before_rule_v1_retraining":
        raise ValueError("sensor-source retraining protocol is not frozen")
    hash_checks = _validate_input_hashes(protocol)
    base_path = _resolve(Path(str(protocol["base_config"]["path"])))
    config = resolve_rule_v1_config(_read_json(base_path))
    text = _config_text(config)
    actual_config_hash = _text_sha256(text)
    expected_config_hash = str(
        protocol["resolved_config_sha256"]
    ).lower()
    if actual_config_hash != expected_config_hash:
        raise ValueError(
            "resolved config hash does not match preregistration: "
            f"{actual_config_hash} != {expected_config_hash}"
        )
    alignment_paths = protocol["sensor_alignment"]["paths"]
    alignment = validate_sensor_alignment(
        qwen_sensor_path=_resolve(Path(alignment_paths["qwen"])),
        rule_sensor_path=_resolve(Path(alignment_paths["rule-v1"])),
        split_paths={
            split: _resolve(Path(path))
            for split, path in alignment_paths["splits"].items()
        },
    )
    expected_alignment = protocol["sensor_alignment"]["expected"]
    if alignment != expected_alignment:
        raise ValueError(
            f"sensor alignment differs from protocol: "
            f"{alignment} != {expected_alignment}"
        )
    if output_path.is_file():
        existing = output_path.read_text(encoding="utf-8-sig")
        if existing != text:
            raise FileExistsError(
                f"resolved config exists with different content: {output_path}"
            )
        if _sha256(output_path) != expected_config_hash:
            temporary = output_path.with_suffix(output_path.suffix + ".tmp")
            temporary.write_bytes(text.encode("utf-8"))
            temporary.replace(output_path)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary.write_bytes(text.encode("utf-8"))
        temporary.replace(output_path)
    return {
        "state": "complete",
        "output": output_path.as_posix(),
        "resolved_config_sha256": actual_config_hash,
        "protocol_sha256": _sha256(protocol_path),
        "input_hash_check_count": len(hash_checks),
        "sensor_alignment": alignment,
    }


def main() -> None:
    args = parse_args()
    result = prepare_config(
        protocol_path=args.protocol,
        output_path=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
