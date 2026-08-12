"""Bind the validator-key correction before the final strata rerun."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from mmfc.model import PROJECT_ROOT


ROOT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "standardized-instance-strata"
)
ADJUDICATION = ROOT / "RUNTIME_NUMERICAL_ADJUDICATION.json"
EVALUATOR = PROJECT_ROOT / "evaluate_standardized_instance_strata.py"
DEFAULT_OUTPUT = ROOT / "FINAL_VALIDATION_ADDENDUM.json"
ADJUDICATION_SHA256 = (
    "1a67ef7a4fa191a1ed32b8a194ee6c044682f073832b2e38f7ff661d7bf9f427"
)
PREVIOUS_EVALUATOR_SHA256 = (
    "11d93f29ad8d5135d2081733868da7f4781ddc3ff5fc8d3354d26dcac203ed96"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze final validator-key correction"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


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


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def build_addendum() -> dict[str, Any]:
    if (
        not ADJUDICATION.is_file()
        or _sha256(ADJUDICATION) != ADJUDICATION_SHA256
        or not EVALUATOR.is_file()
        or _sha256(EVALUATOR) == PREVIOUS_EVALUATOR_SHA256
    ):
        raise ValueError("final validator correction evidence is inconsistent")
    adjudication = json.loads(
        ADJUDICATION.read_text(encoding="utf-8-sig")
    )
    if (
        adjudication.get("status")
        != "frozen_after_batch_sensitivity_before_final_rerun"
        or adjudication.get("bindings", {}).get("protocol") is None
        or adjudication.get("bindings", {}).get("original_protocol")
        is not None
    ):
        raise ValueError("adjudication binding-key diagnosis is inconsistent")
    return {
        "schema_version": 1,
        "status": "frozen_after_validator_binding_key_failure_before_final_rerun",
        "valid": True,
        "formal_paper_metric": False,
        "implementation_fix": {
            "failure": (
                "Evaluator requested bindings.original_protocol while the "
                "frozen adjudication correctly stores bindings.protocol."
            ),
            "authorized_change": (
                "Read the existing protocol binding under its actual key."
            ),
            "runtime_or_scientific_change": False,
        },
        "source_revision": {
            "previous_evaluator_sha256": PREVIOUS_EVALUATOR_SHA256,
            "revised_evaluator_sha256": _sha256(EVALUATOR),
        },
        "bindings": {
            "numerical_adjudication": _binding(ADJUDICATION),
            "revised_evaluator": _binding(EVALUATOR),
            "addendum_builder": _binding(Path(__file__)),
        },
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }


def write_addendum(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = output.resolve()
    result = build_addendum()
    if output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8-sig"))
        left = dict(existing)
        right = dict(result)
        left.pop("frozen_at", None)
        right.pop("frozen_at", None)
        if left != right:
            raise FileExistsError(
                "refusing to replace a drifted validation addendum"
            )
        return existing
    _atomic_text(
        output,
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_text(
        output.with_suffix(".md"),
        "\n".join(
            [
                "# 标准化分层最终校验修正附录",
                "",
                "- 修正内容：读取裁决文件中实际存在的`bindings.protocol`。",
                "- 不改变模型、数据、推理、分层、指标或裁决。",
                "- 最终重跑前冻结。",
                "",
            ]
        ),
    )
    return result


def main() -> None:
    result = write_addendum(parse_args().output)
    print(
        json.dumps(
            {
                "state": "complete",
                "valid": result["valid"],
                "status": result["status"],
                "runtime_or_scientific_change": result[
                    "implementation_fix"
                ]["runtime_or_scientific_change"],
                "revised_evaluator_sha256": result["source_revision"][
                    "revised_evaluator_sha256"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
