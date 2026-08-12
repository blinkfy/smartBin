"""Generate standalone UTF-8 configs for the paper ablation suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmfc.ablations import ablation_names, write_ablation_configs
from mmfc.training import PROJECT_ROOT, load_config, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ablation configs")
    parser.add_argument(
        "--base-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "default.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "configs" / "ablations",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=ablation_names(),
        default=None,
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("checkpoints/ablations"),
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="Smoke mode: run training without writing best.pt/last.pt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = load_config(args.base_config)
    artifacts = write_ablation_configs(
        base_config,
        resolve_project_path(args.output),
        experiments=args.experiments,
        checkpoint_root=args.checkpoint_root,
        epochs=args.epochs,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        save_checkpoints=False if args.no_checkpoints else None,
    )
    print(json.dumps(artifacts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
