"""Build and verify a deterministic evidence archive for the joint-head extension."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import zipfile

from mmfc.model import PROJECT_ROOT
from mmfc.training import resolve_project_path
from prepare_joint_multimodal_seg import sha256_file


DEFAULT_RUN_NAME = "yolo11s-seg-joint-v1-seed42"
DEFAULT_REPORT_RUN = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "joint-multimodal-seg"
    / DEFAULT_RUN_NAME
)
DEFAULT_ADDITIONAL_REPORT_RUNS = tuple(
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "joint-multimodal-seg"
    / f"yolo11s-seg-joint-v1-seed{seed}"
    for seed in (43, 44)
)
RETRAINED_ABLATION_RUN_NAMES = tuple(
    f"yolo11s-seg-joint-v1-{mode}-seed{seed}"
    for mode in ("text-only", "sensor-only")
    for seed in (42, 43, 44)
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "submission-evidence"
    / "joint-head-extension"
    / "joint-head-extension.zip"
)

SOURCE_FILES = (
    "prepare_joint_multimodal_seg.py",
    "train_joint_multimodal_seg.py",
    "evaluate_joint_multimodal_seg.py",
    "audit_joint_multimodal_seg.py",
    "aggregate_joint_multimodal_seg.py",
    "run_joint_seg_retrained_ablations.py",
    "audit_joint_seg_retrained_ablations.py",
    "plot_joint_seg_architecture.py",
    "build_joint_head_deployment_profile.py",
    "build_joint_head_readiness.py",
    "build_joint_head_extension_index.py",
    "build_manuscript_results_sections.py",
    "audit_manuscript_stable_sections.py",
    "audit_paper_readiness.py",
    "build_joint_multimodal_seg_archive.py",
    "README.md",
    "mmfc/joint_seg.py",
    "tests/test_prepare_joint_multimodal_seg.py",
    "tests/test_joint_multimodal_seg.py",
    "tests/test_joint_seg_diagnostics.py",
    "tests/test_joint_seg_audit.py",
    "tests/test_joint_seg_archive.py",
    "tests/test_joint_seg_multiseed.py",
    "tests/test_joint_seg_retrained_queue.py",
    "tests/test_joint_seg_retrained_audit.py",
    "tests/test_joint_seg_architecture_plot.py",
    "tests/test_joint_head_deployment_profile.py",
    "tests/test_joint_head_readiness.py",
    "tests/test_joint_head_extension_index.py",
    "tests/test_paper_readiness_audit.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic joint-head evidence archive."
    )
    parser.add_argument("--report-run", type=Path, default=DEFAULT_REPORT_RUN)
    parser.add_argument(
        "--additional-report-run",
        nargs="*",
        type=Path,
        default=list(DEFAULT_ADDITIONAL_REPORT_RUNS),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _archive_name(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _collect_files(report_runs: list[Path]) -> list[Path]:
    files = [PROJECT_ROOT / relative for relative in SOURCE_FILES]
    prepared = PROJECT_ROOT / "data" / "joint-multimodal-seg"
    report_root = PROJECT_ROOT / "reports" / "qwen-material"
    files.extend(
        [
            prepared / "PREPARATION_REPORT.json",
            prepared / "image_sensor_targets.jsonl",
            prepared / "text_prototypes.npy",
            PROJECT_ROOT
            / "reports"
            / "qwen-material"
            / "yolo-seg"
            / "evaluation.json",
            PROJECT_ROOT
            / "reports"
            / "qwen-material"
            / "PATENT_ALIGNMENT.md",
            report_root
            / "joint-multimodal-seg"
            / "MULTISEED_SUMMARY.json",
            report_root
            / "joint-multimodal-seg"
            / "MULTISEED_SUMMARY.md",
            report_root
            / "joint-multimodal-seg"
            / "MULTISEED_PROTOCOL.json",
            report_root
            / "joint-multimodal-seg"
            / "PERFORMANCE_BENCHMARK.json",
            report_root
            / "joint-multimodal-seg"
            / "PERFORMANCE_BENCHMARK.md",
            report_root
            / "joint-multimodal-seg"
            / "DEPLOYMENT_PROFILE.json",
            report_root
            / "joint-multimodal-seg"
            / "DEPLOYMENT_PROFILE.md",
            report_root
            / "joint-multimodal-seg"
            / "PAPER_READINESS.json",
            report_root
            / "joint-multimodal-seg"
            / "joint_head_architecture.png",
            report_root
            / "joint-multimodal-seg"
            / "joint_head_architecture.pdf",
            report_root
            / "joint-multimodal-seg"
            / "joint_head_architecture.json",
            report_root
            / "joint-multimodal-seg"
            / "JOINT_METHOD_SPEC.md",
            report_root
            / "joint-multimodal-seg"
            / "JOINT_METHOD_SPEC.json",
            report_root / "MANUSCRIPT_RESULTS_SECTIONS.json",
            report_root / "MANUSCRIPT_RESULTS_SECTIONS.md",
            report_root / "MANUSCRIPT_STABLE_SECTIONS.md",
            report_root / "MANUSCRIPT_STABLE_SECTIONS_AUDIT.json",
            report_root / "MANUSCRIPT_STABLE_SECTIONS_AUDIT.md",
            report_root / "MANUSCRIPT_DECISION_BRIEF.md",
            report_root / "FIGURE_CAPTIONS.md",
            report_root / "MOTIVATION_OPTIONS_AFTER_JOINT_EVIDENCE.md",
            report_root / "JOINT_HEAD_EXTENSION_INDEX.json",
            report_root / "JOINT_HEAD_EXTENSION_INDEX.md",
        ]
    )
    for report_run in report_runs:
        files.extend(path for path in report_run.rglob("*") if path.is_file())
        training_run = (
            PROJECT_ROOT
            / "checkpoints"
            / "joint-multimodal-seg"
            / report_run.name
        )
        files.extend(
            path
            for path in training_run.rglob("*")
            if path.is_file()
            and path.suffix.lower() not in {".pt", ".onnx", ".engine"}
        )
    ablation_root = report_root / "joint-multimodal-seg" / "retrained-ablations"
    files.extend(
        path
        for path in ablation_root.rglob("*")
        if path.is_file() and path.suffix.lower() != ".log"
    )
    for run_name in RETRAINED_ABLATION_RUN_NAMES:
        ablation_report_run = (
            report_root / "joint-multimodal-seg" / run_name
        )
        files.extend(
            path for path in ablation_report_run.rglob("*") if path.is_file()
        )
        ablation_training_run = (
            PROJECT_ROOT
            / "checkpoints"
            / "joint-multimodal-seg"
            / run_name
        )
        files.extend(
            path
            for path in ablation_training_run.rglob("*")
            if path.is_file()
            and path.suffix.lower() not in {".pt", ".onnx", ".engine"}
        )
    unique = sorted(set(path.resolve() for path in files), key=_archive_name)
    missing = [path for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing archive inputs: {missing}")
    missing_outcomes = [
        report_run.name
        for report_run in report_runs
        if not (report_run / "OUTCOME.json").is_file()
    ]
    if missing_outcomes:
        raise FileNotFoundError(
            "formal OUTCOME.json must exist before archiving: "
            + ", ".join(missing_outcomes)
        )
    if not (ablation_root / "OUTCOME.json").is_file():
        raise FileNotFoundError(
            "retrained-ablation OUTCOME.json must exist before archiving"
        )
    return unique


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def build_archive(
    report_run: Path,
    output: Path,
    additional_report_runs: list[Path] | None = None,
) -> dict:
    report_runs = [
        resolve_project_path(report_run),
        *[
            resolve_project_path(path)
            for path in (additional_report_runs or [])
        ],
    ]
    if len(report_runs) != len(set(report_runs)):
        raise ValueError("report runs must be unique")
    output = resolve_project_path(output)
    files = _collect_files(report_runs)
    bindings = {
        _archive_name(path): {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    }
    manifest = {
        "schema_version": 2,
        "archive": "joint-head-extension",
        "run_names": [path.name for path in report_runs],
        "retrained_ablation_run_names": list(
            RETRAINED_ABLATION_RUN_NAMES
        ),
        "file_count_excluding_manifest": len(files),
        "checkpoint_included": False,
        "source_images_included": False,
        "hardware_sensor_claim": False,
        "rgb_d_claim": False,
        "bindings": bindings,
    }
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        archive.writestr(_zip_info("ARCHIVE_MANIFEST.json"), manifest_bytes)
        for path in files:
            archive.writestr(_zip_info(_archive_name(path)), path.read_bytes())
    temporary.replace(output)

    errors = []
    with zipfile.ZipFile(output, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            errors.append("archive contains duplicate names")
        if names != ["ARCHIVE_MANIFEST.json", *bindings]:
            errors.append("archive member order or inventory differs from manifest")
        loaded = json.loads(archive.read("ARCHIVE_MANIFEST.json"))
        if loaded != manifest:
            errors.append("embedded archive manifest differs")
        for name, expected in bindings.items():
            payload = archive.read(name)
            if len(payload) != expected["size_bytes"]:
                errors.append(f"size mismatch: {name}")
            if hashlib.sha256(payload).hexdigest() != expected["sha256"]:
                errors.append(f"hash mismatch: {name}")
        forbidden = [
            name
            for name in names
            if Path(name).suffix.lower() in {".pt", ".onnx", ".engine"}
        ]
        if forbidden:
            errors.append(f"checkpoint-like files included: {forbidden[:5]}")

    result = {
        "schema_version": 1,
        "valid": not errors,
        "errors": errors,
        "archive": output.resolve().as_posix(),
        "size_bytes": output.stat().st_size,
        "sha256": sha256_file(output),
        "member_count": 1 + len(files),
        "verified_at": datetime.now().astimezone().isoformat(),
    }
    verification_path = output.parent / "ARCHIVE_VERIFICATION.json"
    verification_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if errors:
        raise ValueError("; ".join(errors))
    return result


def main() -> None:
    args = parse_args()
    result = build_archive(
        args.report_run,
        args.output,
        list(args.additional_report_run),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
