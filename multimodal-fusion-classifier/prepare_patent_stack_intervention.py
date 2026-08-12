"""Freeze the same-checkpoint v6 head-conditioning intervention protocol."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from mmfc.model import PROJECT_ROOT
from train_patent_aligned_external_sensor_joint_seg import DEFAULT_NAME


REPORTS = PROJECT_ROOT / "reports" / "qwen-material"
RUN_ROOT = (
    REPORTS
    / "joint-multimodal-seg-patent"
    / DEFAULT_NAME
)
DEFAULT_OUTPUT = RUN_ROOT / "PATENT_STACK_INTERVENTION_PROTOCOL.json"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "joint-multimodal-seg-patent"
    / DEFAULT_NAME
    / "weights"
    / "best.pt"
)
DEFAULT_OBSERVATIONS = (
    PROJECT_ROOT / "data" / "scene-sensor-sidecars" / "observations.jsonl"
)
DEFAULT_DATA = (
    PROJECT_ROOT / "data" / "yolo-seg-unique-linked" / "dataset.yaml"
)
PROTOCOL_NAME = "patent_stack_same_checkpoint_head_intervention_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the v6 same-checkpoint intervention protocol"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--sensor-observations",
        type=Path,
        default=DEFAULT_OBSERVATIONS,
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def freeze_protocol(args: argparse.Namespace) -> dict[str, Any]:
    output = _resolve(args.output)
    checkpoint = _resolve(args.checkpoint)
    observations = _resolve(args.sensor_observations)
    data = _resolve(args.data)
    paths = {
        "checkpoint": checkpoint,
        "observations": observations,
        "data": data,
        "v6_protocol": RUN_ROOT / "PROTOCOL.json",
        "v6_evaluation": RUN_ROOT / "evaluation.json",
        "v6_conditions": RUN_ROOT / "RELIABILITY_CONDITIONS.json",
        "v6_outcome": RUN_ROOT / "OUTCOME.json",
        "joint_model_source": PROJECT_ROOT / "mmfc" / "joint_seg_patent.py",
        "preparer_source": Path(__file__),
        "evaluator_source": (
            PROJECT_ROOT / "evaluate_patent_stack_intervention.py"
        ),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "intervention protocol inputs missing: " + ", ".join(missing)
        )
    if (
        args.batch_size <= 0
        or args.workers < 0
        or args.image_size <= 0
    ):
        raise ValueError("invalid intervention runtime arguments")
    evaluation = json.loads(
        paths["v6_evaluation"].read_text(encoding="utf-8-sig")
    )
    conditions = json.loads(
        paths["v6_conditions"].read_text(encoding="utf-8-sig")
    )
    if (
        evaluation.get("valid") is not True
        or conditions.get("valid") is not True
        or evaluation.get("checkpoint", {}).get("sha256")
        != _sha256(checkpoint)
        or evaluation.get("sensor_observations", {}).get("sha256")
        != _sha256(observations)
    ):
        raise ValueError("v6 evidence bindings are inconsistent")
    protocol = {
        "schema_version": 1,
        "protocol": PROTOCOL_NAME,
        "status": "frozen_before_intervention_evaluation",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "formal_paper_metric": False,
        "evidence_role": "same_checkpoint_posthoc_causal_intervention",
        "question": (
            "Does the trained v6 P3/P4/P5 patent-stack conditioning "
            "causally alter aggregate task metrics?"
        ),
        "conditions": {
            "full_clean": {
                "source": (
                    "reuse clean_matched from the bound v6 condition report"
                ),
                "checkpoint_modified": False,
            },
            "head_conditioning_zeroed": {
                "source": "new fixed-test evaluation",
                "checkpoint_modified": False,
                "in_memory_intervention": (
                    "set weight and bias of all three "
                    "patent_scale_modulators to zero"
                ),
                "expected_changed_state_tensors": 6,
                "all_other_model_state_tensors_unchanged": True,
                "sidecar_condition": "clean_matched",
            },
        },
        "runtime": {
            "image_size": int(args.image_size),
            "batch_size": int(args.batch_size),
            "workers": int(args.workers),
            "device": str(args.device),
            "seed": int(args.seed),
        },
        "analysis": {
            "primary_metrics": [
                "metrics/mAP50-95(B)",
                "metrics/mAP50-95(M)",
            ],
            "comparison_direction": (
                "full_clean_minus_head_conditioning_zeroed"
            ),
            "descriptive_only": True,
            "retrained_ablation": False,
            "formal_model_selection": False,
        },
        "claim_boundaries": [
            "This is a same-checkpoint posthoc intervention, not an independently retrained ablation.",
            "The full condition is reused from the hash-bound v6 condition report.",
            "Only the three final P3/P4/P5 modulator weights and biases are zeroed in memory.",
            "No modified checkpoint is saved.",
            "The sidecar is scene-rule-v1 rather than Qwen or hardware.",
            "The result remains single-seed development evidence.",
        ],
        "source_bindings": {
            path.resolve().as_posix(): {
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in paths.values()
        },
    }
    _atomic_text(
        output,
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
    )
    markdown = [
        "# v6 同检查点头前调制干预协议",
        "",
        "- 状态：评估前冻结；正式论文指标：否。",
        "- 完整条件：复用已绑定的v6 clean_matched结果。",
        "- 干预条件：同一检查点仅在内存中将三个P3/P4/P5最终调制器的weight与bias归零。",
        "- 预期变化状态张量：6；其余模型状态必须逐张量不变。",
        "- 数据：570张固定测试源图；clean scene-rule-v1 sidecar。",
        f"- 运行：imgsz={args.image_size}，batch={args.batch_size}，"
        f"workers={args.workers}，seed={args.seed}。",
        "- 该实验是事后因果干预，不是独立重训练消融，不参与正式模型选择。",
        "",
    ]
    _atomic_text(output.with_suffix(".md"), "\n".join(markdown))
    return protocol


def main() -> None:
    result = freeze_protocol(parse_args())
    print(
        json.dumps(
            {
                "state": result["status"],
                "formal_paper_metric": result["formal_paper_metric"],
                "protocol": result["protocol"],
                "workers": result["runtime"]["workers"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
