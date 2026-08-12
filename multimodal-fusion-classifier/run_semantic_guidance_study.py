"""Run semantic-guidance training and matched evaluation sequentially."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIGS = tuple(
    PROJECT_ROOT
    / "configs"
    / "semantic-guidance-v5"
    / f"semantic_fallback_seed{seed}.json"
    for seed in (42, 43, 44)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    run([sys.executable, "prepare_semantic_guidance_ablations.py"])
    for seed, config in zip((42, 43, 44), CONFIGS):
        checkpoint = (
            PROJECT_ROOT
            / "checkpoints"
            / "semantic-guidance-v5"
            / f"seed{seed}"
            / "best.pt"
        )
        if args.force_train or not checkpoint.is_file():
            run([sys.executable, "train.py", "--config", str(config)])
        else:
            print(f"SKIP existing checkpoint {checkpoint}", flush=True)
    run(
        [
            sys.executable,
            "evaluate_semantic_guidance.py",
            "--configs",
            *(str(config) for config in CONFIGS),
            "--split",
            args.split,
        ]
    )


if __name__ == "__main__":
    main()
