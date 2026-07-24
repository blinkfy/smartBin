#!/usr/bin/env python
"""Build object manifests and simulated 4x4 sensor records."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import sys
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PIL import Image

from mmfc.data import (
    DEFAULT_ANNOTATIONS_DIR,
    DEFAULT_IMAGES_DIR,
    DEFAULT_OUTPUT_DIR,
    PROJECT_ROOT,
    SPLITS,
    parse_labelme_annotations,
    read_jsonl,
    resolve_manifest_path,
    split_samples,
    write_json,
    write_jsonl,
)
from mmfc.sensors import (
    PROPERTY_RANGES,
    QwenSensorEnhancer,
    contains_leakage_key,
    qwen_api_key,
    simulate_rule_sensor,
)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.json"


def _bounded_int(
    option: str, *, minimum: int, maximum: int | None = None
) -> Any:
    def parse(value: str) -> int:
        try:
            number = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"{option} must be an integer"
            ) from exc
        if number < minimum or (maximum is not None and number > maximum):
            upper = f"..{maximum}" if maximum is not None else "+"
            raise argparse.ArgumentTypeError(
                f"{option} must be in range {minimum}{upper}"
            )
        return number

    return parse


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从 LabelMe 标注构建对象清单与可复现 4x4 传感器数据。"
    )
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS_DIR)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=(
            "定义 canonical 类别顺序的配置文件。相对路径固定从项目根目录解析，"
            "不依赖当前工作目录。"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VAL", "TEST"),
        default=(0.8, 0.1, 0.1),
    )
    parser.add_argument(
        "--sensor-mode", choices=("rule", "qwen"), default="rule"
    )
    parser.add_argument(
        "--api-key-env",
        default="DASHSCOPE_API_KEY",
        help="Qwen API Key 环境变量名（不会写入产物）。",
    )
    parser.add_argument(
        "--base-url",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    parser.add_argument("--model", default="qwen3-vl-flash")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument(
        "--workers",
        type=_bounded_int("--workers", minimum=1, maximum=32),
        default=8,
        help="Sensor generation concurrency, 1..32.",
    )
    parser.add_argument(
        "--progress-every",
        type=_bounded_int("--progress-every", minimum=1),
        default=10,
        help="Progress line frequency for non-TTY output.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=_bounded_int("--checkpoint-every", minimum=1),
        default=20,
        help="Write output/sensors.jsonl after every N newly completed samples.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress intermediate progress lines; start/end summaries remain.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="复用 output/sensors.jsonl 中已有且属于当前清单的记录。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="仅构建确定性排序后的前 N 个对象，便于冒烟测试。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只解析并打印摘要，不写文件、不开启网络请求。",
    )
    return parser


def _resolve_project_path(
    path: Path | str, *, project_root: Path | str = PROJECT_ROOT
) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    return (Path(project_root).resolve() / path).resolve()


def load_canonical_labels(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    *,
    project_root: Path | str = PROJECT_ROOT,
) -> list[str]:
    """Load the unique class names whose order defines every numeric label."""
    path = _resolve_project_path(config_path, project_root=project_root)
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise ValueError(f"无法读取类别配置 {path}：{exc}") from exc
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"类别配置不是有效 UTF-8 JSON {path}：{exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"类别配置根节点必须是对象：{path}")
    classes = payload.get("classes")
    if not isinstance(classes, list) or not classes:
        raise ValueError(f"类别配置 classes 必须是非空数组：{path}")

    labels: list[str] = []
    for index, item in enumerate(classes):
        if not isinstance(item, Mapping):
            raise ValueError(f"类别配置 classes[{index}] 必须是对象")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"类别配置 classes[{index}].name 必须是非空字符串")
        labels.append(name.strip())
    duplicates = sorted(
        label for label, count in Counter(labels).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"类别配置包含重复名称：{duplicates}")
    return labels


def _write_ordered_json(path: Path, value: object) -> None:
    """Atomically write JSON while preserving canonical mapping insertion order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2))
        handle.write("\n")
    temporary.replace(path)


def _load_resume(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    records: dict[str, dict[str, Any]] = {}
    try:
        for record in read_jsonl(path):
            sample_id = record.get("sample_id")
            if isinstance(sample_id, str) and sample_id:
                records[sample_id] = record
    except (OSError, ValueError) as exc:
        print(f"警告：无法复用旧传感器文件：{exc}", file=sys.stderr)
        return {}
    return records


def _valid_4x4(value: object) -> bool:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(not isinstance(row, list) or len(row) != 4 for row in value)
    ):
        return False
    return all(
        isinstance(item, (int, float))
        and not isinstance(item, bool)
        and math.isfinite(float(item))
        and 0.0 <= float(item) <= 1.0
        for row in value
        for item in row
    )


def resume_record_usable(
    record: Mapping[str, Any],
    *,
    sensor_mode: str,
    qwen_model: str,
) -> bool:
    """Return whether a prior sensor record satisfies the requested mode.

    Qwen resume intentionally rejects rule and fallback records so a later run
    with credentials can upgrade an existing complete rule dataset.
    """
    if contains_leakage_key(record):
        return False
    if record.get("schema_version") != 1:
        return False
    sample_id = record.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        return False
    if record.get("grid_shape") != [4, 4]:
        return False
    if not _valid_4x4(record.get("grid")) or not _valid_4x4(
        record.get("normalized_grid")
    ):
        return False
    properties = record.get("properties")
    if not isinstance(properties, Mapping):
        return False
    for name, (lower, upper) in PROPERTY_RANGES.items():
        value = properties.get(name)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not lower <= float(value) <= upper
        ):
            return False
    source = record.get("source")
    if not isinstance(source, Mapping):
        return False
    generator = source.get("generator")
    if sensor_mode == "rule":
        return generator == "rule-v1"
    if sensor_mode == "qwen":
        return (
            generator == qwen_model
            and source.get("enhancement_status") == "success"
        )
    return False


def _coerce_int_arg(
    args: argparse.Namespace,
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    value = getattr(args, name, default)
    if value is None:
        value = default
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"--{name.replace('_', '-')} must be an integer") from exc
    if number < minimum or (maximum is not None and number > maximum):
        upper = f"..{maximum}" if maximum is not None else "+"
        raise ValueError(
            f"--{name.replace('_', '-')} must be in range {minimum}{upper}"
        )
    return number


def _write_sensor_checkpoint(
    path: Path, sensors_by_id: Mapping[str, Mapping[str, Any]]
) -> None:
    write_jsonl(
        path,
        [sensors_by_id[sample_id] for sample_id in sorted(sensors_by_id)],
    )


def _qwen_status_counts(record: Mapping[str, Any]) -> tuple[int, int]:
    source = record.get("source")
    if not isinstance(source, Mapping):
        return 0, 0
    status = source.get("enhancement_status")
    if status == "success":
        return 1, 0
    if isinstance(status, str) and status.startswith("fallback"):
        return 0, 1
    return 0, 0


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "unknown"
    seconds = max(0, int(round(seconds)))
    minutes, second = divmod(seconds, 60)
    hours, minute = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minute:02d}m{second:02d}s"
    if minutes:
        return f"{minutes}m{second:02d}s"
    return f"{second}s"


class _ProgressReporter:
    def __init__(
        self,
        *,
        total: int,
        progress_every: int,
        no_progress: bool,
        stream: Any,
    ) -> None:
        self.total = total
        self.progress_every = progress_every
        self.no_progress = no_progress
        self.stream = stream
        self.is_tty = bool(getattr(stream, "isatty", lambda: False)())
        self.started = time.monotonic()
        self.last_completed = -1
        self.last_was_tty = False

    def _line(
        self,
        *,
        phase: str,
        completed: int,
        reused: int,
        qwen_success: int,
        fallback: int,
        errors: int,
    ) -> str:
        percent = 100.0 if self.total == 0 else completed / self.total * 100.0
        elapsed = max(time.monotonic() - self.started, 1e-3)
        rate = completed / elapsed if completed else 0.0
        remaining = max(0, self.total - completed)
        eta = remaining / rate if rate > 0 else (0.0 if remaining == 0 else None)
        return (
            f"sensor progress {phase}: completed={completed}/{self.total} "
            f"percent={percent:.1f}% reused={reused} "
            f"qwen_success={qwen_success} fallback={fallback} errors={errors} "
            f"rate={rate:.2f} samples/s ETA={_format_duration(eta)}"
        )

    def emit(
        self,
        *,
        phase: str,
        completed: int,
        reused: int,
        qwen_success: int,
        fallback: int,
        errors: int,
        force: bool = False,
    ) -> None:
        if not force:
            if self.no_progress:
                return
            if not self.is_tty and completed - self.last_completed < self.progress_every:
                return
        line = self._line(
            phase=phase,
            completed=completed,
            reused=reused,
            qwen_success=qwen_success,
            fallback=fallback,
            errors=errors,
        )
        if self.is_tty:
            self.stream.write("\r" + line)
            self.last_was_tty = True
            if phase in {"done", "interrupted"}:
                self.stream.write("\n")
        else:
            if self.last_was_tty:
                self.stream.write("\n")
                self.last_was_tty = False
            self.stream.write(line + "\n")
        self.stream.flush()
        self.last_completed = completed


def _build_sensor_record(
    sample: Mapping[str, Any],
    *,
    sensor_mode: str,
    seed: int,
    enhancer: QwenSensorEnhancer,
) -> dict[str, Any]:
    image_path = resolve_manifest_path(str(sample["image_path"]), PROJECT_ROOT)
    with Image.open(image_path) as image:
        rule_record = simulate_rule_sensor(
            image,
            sample["bbox"],
            str(sample["sample_id"]),
            seed=seed,
            source_image_id=str(sample["source_image_id"]),
        )
        if sensor_mode == "qwen":
            return enhancer.enhance(image, sample["bbox"], rule_record)
        return rule_record


def build(args: argparse.Namespace) -> dict[str, Any]:
    seed = int(getattr(args, "seed", 20260724))
    ratios = tuple(getattr(args, "ratios", (0.8, 0.1, 0.1)))
    sensor_mode = str(getattr(args, "sensor_mode", "rule"))
    if sensor_mode not in {"rule", "qwen"}:
        raise ValueError("--sensor-mode must be rule or qwen")
    limit = getattr(args, "limit", None)
    dry_run = bool(getattr(args, "dry_run", False))
    resume = bool(getattr(args, "resume", False))
    output = Path(getattr(args, "output", DEFAULT_OUTPUT_DIR)).resolve()
    model = str(getattr(args, "model", "qwen3-vl-flash"))
    api_key_env = str(getattr(args, "api_key_env", "DASHSCOPE_API_KEY"))
    workers = _coerce_int_arg(args, "workers", 4, minimum=1, maximum=32)
    progress_every = _coerce_int_arg(args, "progress_every", 10, minimum=1)
    checkpoint_every = _coerce_int_arg(args, "checkpoint_every", 20, minimum=1)

    canonical_labels = load_canonical_labels(
        getattr(args, "config", DEFAULT_CONFIG_PATH)
    )
    label_to_index = {
        label: index for index, label in enumerate(canonical_labels)
    }
    samples, report = parse_labelme_annotations(
        getattr(args, "annotations", DEFAULT_ANNOTATIONS_DIR),
        getattr(args, "images", DEFAULT_IMAGES_DIR),
        project_root=PROJECT_ROOT,
    )
    unknown_labels = sorted(
        {sample["label"] for sample in samples} - set(label_to_index)
    )
    if unknown_labels:
        raise ValueError(
            "annotations contain labels missing from config: "
            f"{unknown_labels}; update the config or fix annotations"
        )
    for sample in samples:
        sample["label_id"] = label_to_index[sample["label"]]
    samples.sort(key=lambda item: item["sample_id"])
    if limit is not None:
        limit = int(limit)
        if limit < 0:
            raise ValueError("--limit must be non-negative")
        samples = samples[:limit]
    split_records = split_samples(samples, seed=seed, ratios=ratios)
    all_samples = [
        sample for split in SPLITS for sample in split_records[split]
    ]
    ordered_samples = sorted(all_samples, key=lambda item: item["sample_id"])
    summary = {
        "samples": len(all_samples),
        "source_images": len({item["source_image_id"] for item in all_samples}),
        "split_counts": {split: len(split_records[split]) for split in SPLITS},
        "class_counts": dict(
            sorted(Counter(item["label"] for item in all_samples).items())
        ),
        "labels": canonical_labels,
        "issue_counts": report["issue_counts"],
        "sensor_mode": sensor_mode,
        "dry_run": dry_run,
    }
    if dry_run:
        return summary

    output.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        write_jsonl(output / f"{split}.jsonl", split_records[split])

    resume_records = _load_resume(output / "sensors.jsonl") if resume else {}
    api_key = qwen_api_key(api_key_env) if sensor_mode == "qwen" else None
    enhancer = QwenSensorEnhancer(
        api_key=api_key,
        base_url=str(
            getattr(
                args,
                "base_url",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        ),
        model=model,
        retries=int(getattr(args, "retries", 2)),
        timeout=float(getattr(args, "timeout", 45.0)),
    )
    sensors_by_id: dict[str, Mapping[str, Any]] = {}
    pending_samples: list[dict[str, Any]] = []
    sensor_issues: list[dict[str, Any]] = []
    reused = 0
    rejected_resume_records = 0
    qwen_success = 0
    fallback = 0

    for sample in ordered_samples:
        sample_id = str(sample["sample_id"])
        old_record = resume_records.get(sample_id)
        if old_record is not None and resume_record_usable(
            old_record,
            sensor_mode=sensor_mode,
            qwen_model=model,
        ):
            sensors_by_id[sample_id] = old_record
            reused += 1
            success_delta, fallback_delta = _qwen_status_counts(old_record)
            qwen_success += success_delta
            fallback += fallback_delta
            continue
        if old_record is not None:
            rejected_resume_records += 1
        pending_samples.append(sample)

    completed = reused
    reporter = _ProgressReporter(
        total=len(ordered_samples),
        progress_every=progress_every,
        no_progress=bool(getattr(args, "no_progress", False)),
        stream=sys.stderr,
    )
    reporter.emit(
        phase="start",
        completed=completed,
        reused=reused,
        qwen_success=qwen_success,
        fallback=fallback,
        errors=len(sensor_issues),
        force=True,
    )

    new_completed = 0
    if pending_samples:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        future_to_sample: dict[
            concurrent.futures.Future[dict[str, Any]], dict[str, Any]
        ] = {}
        try:
            future_to_sample = {
                executor.submit(
                    _build_sensor_record,
                    sample,
                    sensor_mode=sensor_mode,
                    seed=seed,
                    enhancer=enhancer,
                ): sample
                for sample in pending_samples
            }
            for future in concurrent.futures.as_completed(future_to_sample):
                sample = future_to_sample[future]
                sample_id = str(sample["sample_id"])
                try:
                    sensor_record = future.result()
                except Exception as exc:
                    sensor_issues.append(
                        {
                            "code": "sensor_generation_error",
                            "sample_id": sample_id,
                            "detail": f"{type(exc).__name__}: {exc}",
                        }
                    )
                else:
                    sensors_by_id[sample_id] = sensor_record
                    success_delta, fallback_delta = _qwen_status_counts(
                        sensor_record
                    )
                    qwen_success += success_delta
                    fallback += fallback_delta
                completed += 1
                new_completed += 1
                if new_completed % checkpoint_every == 0:
                    _write_sensor_checkpoint(
                        output / "sensors.jsonl", sensors_by_id
                    )
                reporter.emit(
                    phase="progress",
                    completed=completed,
                    reused=reused,
                    qwen_success=qwen_success,
                    fallback=fallback,
                    errors=len(sensor_issues),
                )
        except KeyboardInterrupt:
            for future in future_to_sample:
                future.cancel()
            _write_sensor_checkpoint(output / "sensors.jsonl", sensors_by_id)
            reporter.emit(
                phase="interrupted",
                completed=completed,
                reused=reused,
                qwen_success=qwen_success,
                fallback=fallback,
                errors=len(sensor_issues),
                force=True,
            )
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                executor.shutdown(wait=False)
            raise
        except BaseException:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                executor.shutdown(wait=False)
            raise
        else:
            executor.shutdown(wait=True)

    _write_sensor_checkpoint(output / "sensors.jsonl", sensors_by_id)
    reporter.emit(
        phase="done",
        completed=completed,
        reused=reused,
        qwen_success=qwen_success,
        fallback=fallback,
        errors=len(sensor_issues),
        force=True,
    )

    sensors = [sensors_by_id[sample_id] for sample_id in sorted(sensors_by_id)]
    sensor_position = {
        record["sample_id"]: index for index, record in enumerate(sensors)
    }
    index = {
        "schema_version": 1,
        "seed": seed,
        "split_ratios": list(ratios),
        "labels": canonical_labels,
        "label_to_index": label_to_index,
        "samples": {
            sample["sample_id"]: {
                "split": sample["split"],
                "label_index": label_to_index[sample["label"]],
                "sensor_index": sensor_position.get(sample["sample_id"]),
            }
            for sample in ordered_samples
        },
    }
    _write_ordered_json(output / "index.json", index)

    generator_counts = Counter(
        str(record.get("source", {}).get("generator", "unknown"))
        for record in sensors
    )
    fallback_counts = Counter(
        str(record.get("source", {}).get("enhancement_status", "not_requested"))
        for record in sensors
    )
    stats = {
        "schema_version": 1,
        **summary,
        "sensor_records": len(sensors),
        "sensor_generator_counts": dict(sorted(generator_counts.items())),
        "sensor_enhancement_status_counts": dict(sorted(fallback_counts.items())),
        "resumed_sensor_records": reused,
        "rejected_resume_records": rejected_resume_records,
        "sensor_errors": len(sensor_issues),
        "qwen_success_records": qwen_success,
        "fallback_sensor_records": fallback,
        "qwen_api_key_available": bool(api_key) if sensor_mode == "qwen" else None,
        "workers": workers,
        "checkpoint_every": checkpoint_every,
    }
    write_json(output / "stats.json", stats)
    report["selected_samples"] = len(all_samples)
    report["sensor_issues"] = sensor_issues
    report["sensor_issue_counts"] = dict(
        sorted(Counter(item["code"] for item in sensor_issues).items())
    )
    write_json(output / "build_report.json", report)
    return stats


def main() -> int:
    args = _parser().parse_args()
    try:
        summary = build(args)
    except KeyboardInterrupt:
        print(
            "Build interrupted; completed sensor checkpoint was written when possible.",
            file=sys.stderr,
        )
        return 130
    except (OSError, ValueError) as exc:
        print(f"构建失败：{exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if not getattr(args, "dry_run", False) and summary.get("sensor_errors"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
