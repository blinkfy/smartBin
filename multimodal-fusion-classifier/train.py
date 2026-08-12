"""训练命令行入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmfc.training import PROJECT_ROOT, load_config, train_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练多模态智能垃圾分类器")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "default.json",
        help="UTF-8 JSON 配置文件；相对路径按子项目目录解析",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume exactly from <checkpoint_dir>/last.pt",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Explicit last_resume checkpoint; mutually exclusive with --resume",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume and args.resume_checkpoint is not None:
        raise ValueError(
            "--resume and --resume-checkpoint cannot be used together"
        )
    config = load_config(args.config)
    if args.resume:
        config.setdefault("training", {})["resume"] = True
    if args.resume_checkpoint is not None:
        config.setdefault("training", {})["resume_checkpoint"] = str(
            args.resume_checkpoint
        )
    result = train_from_config(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
