"""Normalize the completed sensor-retraining pipeline log to UTF-8."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ARCHIVED_LOG = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "training"
    / "interrupted"
    / "sensor_retraining_outcome_log_encoding_20260728"
    / "inner_pipeline.utf16.log"
)
DEFAULT_CURRENT_LOG = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "training"
    / "sensor_source_retraining.log"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "SENSOR_RETRAINING_LOG_ENCODING_REPAIR.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize the completed rule-v1 pipeline log to UTF-8"
    )
    parser.add_argument(
        "--archived-log", type=Path, default=DEFAULT_ARCHIVED_LOG
    )
    parser.add_argument(
        "--current-log", type=Path, default=DEFAULT_CURRENT_LOG
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    return {
        "path": _display_path(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_bytes(path.read_bytes()),
    }


def repair_log_encoding(
    *,
    archived_log: Path,
    current_log: Path,
) -> dict[str, Any]:
    archived = _resolve(archived_log)
    current = _resolve(current_log)
    if not archived.is_file() or not current.is_file():
        raise FileNotFoundError("sensor retraining log input is missing")
    archived_bytes = archived.read_bytes()
    current_bytes = current.read_bytes()
    if archived_bytes != current_bytes:
        raise ValueError("current log differs from the archived pre-repair log")
    if not archived_bytes.startswith((b"\xff\xfe", b"\xfe\xff")):
        raise ValueError("archived log does not have a UTF-16 BOM")
    text = archived_bytes.decode("utf-16")
    required_tokens = (
        "SKIP rule_v1_training (verified complete)",
        "DONE rule_v1_checkpoint_audit",
        "DONE rule_v1_test",
        "DONE rule_v1_vs_qwen_comparison",
    )
    missing_tokens = [token for token in required_tokens if token not in text]
    if missing_tokens:
        raise ValueError(
            "completed pipeline tokens are missing: "
            + ", ".join(missing_tokens)
        )
    repaired_bytes = text.encode("utf-8")
    temporary = current.with_suffix(current.suffix + ".tmp")
    temporary.write_bytes(repaired_bytes)
    temporary.replace(current)
    round_trip = current.read_bytes().decode("utf-8")
    if round_trip != text:
        raise ValueError("UTF-8 log round trip changed decoded text")
    return {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "scope": "sensor_retraining_pipeline_log_encoding_normalization",
        "generated_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "original": {
            **_binding(archived),
            "encoding": "utf-16-with-bom",
        },
        "repaired": {
            **_binding(current),
            "encoding": "utf-8",
        },
        "decoded_text_sha256": _sha256_bytes(text.encode("utf-8")),
        "decoded_text_equal": True,
        "required_tokens": list(required_tokens),
        "boundaries": [
            (
                "Only the completed PowerShell run log encoding changes; "
                "decoded text is identical."
            ),
            (
                "No checkpoint, training metadata, prediction, metric, "
                "configuration, source image, or sensor sidecar is changed."
            ),
        ],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# rule-v1 恢复日志编码规范化",
            "",
            f"生成时间：`{report['generated_at']}`。",
            "",
            (
                "PowerShell 完成日志由带 BOM 的 UTF-16 规范化为 UTF-8；"
                "解码后文本逐字相同，文本 SHA-256 为 "
                f"`{report['decoded_text_sha256']}`。"
            ),
            (
                "该操作只改变日志字节编码，不改变训练、检查点、预测、指标、"
                "配置、图像或传感器 sidecar。"
            ),
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    output = _resolve(args.output)
    markdown = output.with_suffix(".md")
    if output.exists() or markdown.exists():
        raise FileExistsError(
            f"sensor retraining log repair is immutable: {output}"
        )
    report = repair_log_encoding(
        archived_log=args.archived_log,
        current_log=args.current_log,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": output.as_posix(),
                "valid": report["valid"],
                "decoded_text_equal": report["decoded_text_equal"],
                "original_sha256": report["original"]["sha256"],
                "repaired_sha256": report["repaired"]["sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
