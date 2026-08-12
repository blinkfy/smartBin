"""Train one isolated same-input multimodal fusion comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmfc.model import PROJECT_ROOT
from mmfc.recent_baseline_training import train_recent_baseline_from_config
from mmfc.training import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an isolated same-input fusion baseline"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume and args.resume_checkpoint is not None:
        raise ValueError(
            "--resume and --resume-checkpoint are mutually exclusive"
        )
    config = load_config(args.config)
    if args.resume:
        config.setdefault("training", {})["resume"] = True
    if args.resume_checkpoint is not None:
        config.setdefault("training", {})["resume_checkpoint"] = str(
            args.resume_checkpoint
        )
    result = train_recent_baseline_from_config(config)
    print(
        json.dumps(
            {
                "valid": result["valid"],
                "baseline_method": result["baseline_method"],
                "completed_epochs": result["completed_epochs"],
                "checkpoint_dir": result["checkpoint_dir"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
