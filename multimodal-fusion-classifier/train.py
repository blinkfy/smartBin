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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    result = train_from_config(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
