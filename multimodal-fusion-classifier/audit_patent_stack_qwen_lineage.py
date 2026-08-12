"""Audit the immutable v2 protocol and additive v6 Qwen lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from mmfc.model import PROJECT_ROOT
from prepare_patent_stack_qwen_formal_protocol import (
    DEFAULT_OUTPUT as DEFAULT_NEW_PROTOCOL,
    OLD_PROTOCOL,
    PROTOCOL_NAME,
    RUN_PROTOCOL_NAME,
)


DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "joint-multimodal-seg-patent-qwen"
    / "QWEN_FORMAL_EXTENSION_LINEAGE_AUDIT.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit old-v2 versus additive-v6 Qwen lineage"
    )
    parser.add_argument(
        "--legacy-protocol", type=Path, default=OLD_PROTOCOL
    )
    parser.add_argument(
        "--extension-protocol", type=Path, default=DEFAULT_NEW_PROTOCOL
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return (
        path.resolve()
        if path.is_absolute()
        else (PROJECT_ROOT / path).resolve()
    )


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


def _valid_named_bindings(bindings: Mapping[str, Any]) -> bool:
    if not bindings:
        return False
    for name, raw in bindings.items():
        if not isinstance(raw, Mapping):
            return False
        raw_path = raw.get("path", name)
        path = _resolve(Path(str(raw_path)))
        try:
            path.relative_to(PROJECT_ROOT)
        except ValueError:
            return False
        if (
            not path.is_file()
            or path.stat().st_size != raw.get("size_bytes")
            or _sha256(path) != raw.get("sha256")
        ):
            return False
    return True


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def audit(
    legacy_protocol: Path = OLD_PROTOCOL,
    extension_protocol: Path = DEFAULT_NEW_PROTOCOL,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    legacy_path = _resolve(legacy_protocol)
    extension_path = _resolve(extension_protocol)
    output = _resolve(output)
    for path in (legacy_path, extension_path):
        if not path.is_file():
            raise FileNotFoundError(f"lineage input is missing: {path}")
    legacy = _read_json(legacy_path)
    extension = _read_json(extension_path)
    errors: list[str] = []
    legacy_binding = extension.get("source_bindings", {}).get(
        "legacy_v2_protocol", {}
    )
    parent_binding = extension.get("lineage", {}).get(
        "parent_checkpoint", {}
    )
    checks = {
        "legacy_identity_preserved": (
            legacy.get("protocol")
            == "full_image_qwen_external_sensor_joint_seg_v2"
            and legacy.get("status")
            == "preregistered_awaiting_api_credentials_and_data"
        ),
        "legacy_file_hash_bound_by_extension": (
            legacy_binding.get("sha256") == _sha256(legacy_path)
            and legacy_binding.get("size_bytes") == legacy_path.stat().st_size
        ),
        "extension_identity_valid": (
            extension.get("protocol") == PROTOCOL_NAME
            and extension.get("run_protocol") == RUN_PROTOCOL_NAME
            and extension.get("status")
            == "preregistered_awaiting_api_credentials_and_data"
        ),
        "extension_is_additive": (
            extension.get("relationship_to_legacy_v2", {}).get(
                "legacy_protocol_modified"
            )
            is False
            and extension.get("relationship_to_legacy_v2", {}).get(
                "extension_is_additive"
            )
            is True
        ),
        "fixed_parent_disclosed": (
            extension.get("lineage", {}).get(
                "shared_parent_across_seeds"
            )
            is True
            and extension.get("lineage", {}).get(
                "independent_end_to_end_replicates"
            )
            is False
            and _valid_named_bindings({"parent": parent_binding})
        ),
        "separate_training_entrypoints": (
            legacy.get("execution_commands", {})
            .get("train_template", "")
            .startswith("python train_external_sensor_joint_seg.py")
            and extension.get("training", {}).get("architecture")
            == RUN_PROTOCOL_NAME
        ),
        "legacy_source_bindings_valid": _valid_named_bindings(
            legacy.get("source_bindings", {})
        ),
        "extension_source_bindings_valid": _valid_named_bindings(
            extension.get("source_bindings", {})
        ),
        "same_annotation_free_qwen_data_contract": (
            legacy.get("data_generation", {}).get("output")
            == extension.get("data_generation", {}).get("output")
            and legacy.get("data_generation", {}).get("mode")
            == extension.get("data_generation", {}).get("mode")
            == "qwen"
            and legacy.get("data_generation", {}).get(
                "expected_source_image_count"
            )
            == extension.get("data_generation", {}).get(
                "expected_source_image_count"
            )
            == 5990
        ),
    }
    for name, passed in checks.items():
        if not passed:
            errors.append(f"lineage check failed: {name}")
    result = {
        "schema_version": 1,
        "state": "complete",
        "valid": not errors,
        "formal_training_complete": False,
        "legacy_protocol": {
            "protocol": legacy.get("protocol"),
            "binding": _binding(legacy_path),
            "modified_by_extension": False,
        },
        "extension_protocol": {
            "protocol": extension.get("protocol"),
            "run_protocol": extension.get("run_protocol"),
            "binding": _binding(extension_path),
        },
        "lineage": extension.get("lineage"),
        "checks": checks,
        "errors": errors,
        "bindings": {
            "legacy_protocol": _binding(legacy_path),
            "extension_protocol": _binding(extension_path),
            "auditor_source": _binding(Path(__file__)),
        },
    }
    _atomic_text(
        output,
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
    )
    markdown = [
        "# Qwen 正式协议谱系审计",
        "",
        f"- 审计有效：{'是' if result['valid'] else '否'}",
        "- 旧协议：v2 文件未改动，并由新版协议按 SHA-256 绑定。",
        "- 新协议：独立的 v6 Qwen 适配扩展，不覆盖旧训练或报告目录。",
        (
            "- 种子含义：三次运行共享固定 v5 rule-sidecar 父检查点，"
            "只估计适配阶段随机性。"
        ),
        "- 数据约束：两套协议共用整图、无标注输入的 Qwen sidecar 合同。",
        "",
    ]
    _atomic_text(output.with_suffix(".md"), "\n".join(markdown))
    if errors:
        raise ValueError("; ".join(errors))
    return result


def main() -> None:
    args = parse_args()
    result = audit(
        args.legacy_protocol,
        args.extension_protocol,
        args.output,
    )
    print(
        json.dumps(
            {
                "state": result["state"],
                "valid": result["valid"],
                "formal_training_complete": result[
                    "formal_training_complete"
                ],
                "check_count": len(result["checks"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
