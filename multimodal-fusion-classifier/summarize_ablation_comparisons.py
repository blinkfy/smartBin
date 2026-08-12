"""Aggregate paired full-vs-ablation comparisons with Holm correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mmfc.ablations import (
    ablation_names,
    write_ablation_comparison_artifacts,
)
from mmfc.training import PROJECT_ROOT, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize paired ablation comparisons"
    )
    parser.add_argument(
        "--reports",
        type=Path,
        default=PROJECT_ROOT / "reports" / "qwen-material" / "ablations",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=tuple(name for name in ablation_names() if name != "full"),
        required=True,
    )
    return parser.parse_args()


def _load_comparisons(
    reports_dir: Path,
    experiments: list[str],
) -> dict[str, dict[str, Any]]:
    comparisons: dict[str, dict[str, Any]] = {}
    for name in experiments:
        path = reports_dir / name / "PAIRED_COMPARISON.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing paired comparison: {path}")
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError(f"comparison must be a JSON object: {path}")
        comparisons[name] = document
    return comparisons


def main() -> None:
    args = parse_args()
    reports_dir = resolve_project_path(args.reports)
    output_dir = (
        resolve_project_path(args.output)
        if args.output is not None
        else reports_dir
    )
    comparisons = _load_comparisons(reports_dir, args.experiments)
    artifacts = write_ablation_comparison_artifacts(comparisons, output_dir)
    print(json.dumps(artifacts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
