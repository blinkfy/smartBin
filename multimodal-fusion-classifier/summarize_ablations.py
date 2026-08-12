"""Aggregate evaluated ablation reports into paper-ready artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmfc.ablations import (
    ablation_names,
    load_ablation_reports,
    write_ablation_artifacts,
)
from mmfc.training import PROJECT_ROOT, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize ablation reports")
    parser.add_argument(
        "--reports",
        type=Path,
        default=PROJECT_ROOT / "reports" / "qwen-material" / "ablations",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=ablation_names(),
        default=None,
    )
    parser.add_argument(
        "--protocol",
        choices=("formal_retrained", "posthoc_intervention"),
        required=True,
    )
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports_dir = resolve_project_path(args.reports)
    output_dir = (
        resolve_project_path(args.output)
        if args.output is not None
        else reports_dir
    )
    reports = load_ablation_reports(
        reports_dir,
        args.experiments,
        allow_missing=args.allow_missing,
    )
    artifacts = write_ablation_artifacts(
        reports,
        output_dir,
        protocol=args.protocol,
    )
    print(json.dumps(artifacts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
