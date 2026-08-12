"""Measure per-image separation between matched and shuffled sensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from evaluate_reliable_external_conditions import (
    DEFAULT_DATA,
    DEFAULT_OBSERVATIONS,
    DEFAULT_SHUFFLED,
    _atomic_json,
    _binding,
    _gate_collector,
    _resolve,
)
from evaluate_spatial_external_conditions import (
    DEFAULT_CHECKPOINT,
    DEFAULT_OUTPUT,
)
from prepare_joint_multimodal_seg import sha256_file
from train_spatial_contrastive_external_sensor_joint_seg import PROTOCOL_NAME


DEFAULT_DIAGNOSTIC = (
    DEFAULT_OUTPUT.parent / "SENSOR_SEPARATION.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose matched versus shuffled sensor separation"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--sensor-observations", type=Path, default=DEFAULT_OBSERVATIONS
    )
    parser.add_argument(
        "--shuffled-observations", type=Path, default=DEFAULT_SHUFFLED
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_DIAGNOSTIC)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def binary_auc(
    negative_scores: torch.Tensor,
    positive_scores: torch.Tensor,
) -> float:
    """Return tie-aware ROC AUC using rank sums."""

    negative = negative_scores.detach().double().reshape(-1)
    positive = positive_scores.detach().double().reshape(-1)
    if negative.numel() == 0 or positive.numel() == 0:
        raise ValueError("both AUC populations must be non-empty")
    scores = torch.cat([negative, positive])
    order = torch.argsort(scores, stable=True)
    sorted_scores = scores[order]
    ranks = torch.empty_like(scores)
    start = 0
    while start < scores.numel():
        end = start + 1
        while (
            end < scores.numel()
            and sorted_scores[end] == sorted_scores[start]
        ):
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    positive_rank_sum = ranks[negative.numel() :].sum()
    n_negative = float(negative.numel())
    n_positive = float(positive.numel())
    auc = (
        positive_rank_sum - n_positive * (n_positive + 1.0) / 2.0
    ) / (n_negative * n_positive)
    return float(auc.item())


def _values(captured: dict[str, list[torch.Tensor]], name: str) -> torch.Tensor:
    values = torch.cat(captured[name], dim=0).float().reshape(-1)
    if values.numel() < 570:
        raise ValueError(f"diagnostic population is short: {name}")
    return values[-570:]


def _summary(values: torch.Tensor) -> dict[str, float]:
    quantiles = torch.quantile(
        values.float(),
        torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95]),
    )
    return {
        "mean": float(values.mean().item()),
        "population_standard_deviation": float(
            values.std(unbiased=False).item()
        ),
        "p05": float(quantiles[0].item()),
        "p25": float(quantiles[1].item()),
        "median": float(quantiles[2].item()),
        "p75": float(quantiles[3].item()),
        "p95": float(quantiles[4].item()),
    }


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    from ultralytics import YOLO

    from mmfc.joint_seg_external import load_scene_sensor_index
    from mmfc.joint_seg_reliable import (
        ReliableExternalSensorSegmentationValidator,
    )

    checkpoint = _resolve(args.checkpoint)
    observations = _resolve(args.sensor_observations)
    shuffled = _resolve(args.shuffled_observations)
    data = _resolve(args.data)
    output = _resolve(args.output)
    protocol_path = output.parent / "PROTOCOL.json"
    evaluation_path = output.parent / "evaluation.json"
    for path in (
        checkpoint,
        observations,
        shuffled,
        data,
        protocol_path,
        evaluation_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"separation input is missing: {path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8-sig"))
    evaluation = json.loads(
        evaluation_path.read_text(encoding="utf-8-sig")
    )
    if (
        protocol.get("protocol") != PROTOCOL_NAME
        or evaluation.get("checkpoint", {}).get("sha256")
        != sha256_file(checkpoint)
        or evaluation.get("sensor_observations", {}).get("sha256")
        != sha256_file(observations)
    ):
        raise ValueError("spatial diagnostic bindings are inconsistent")

    indexes = {
        "matched": load_scene_sensor_index(observations),
        "shuffled": load_scene_sensor_index(shuffled),
    }
    populations: dict[str, dict[str, torch.Tensor]] = {}
    for name, index in indexes.items():
        validators = []

        def validator_factory(*factory_args: Any, **factory_kwargs: Any):
            validator = ReliableExternalSensorSegmentationValidator(
                *factory_args,
                **factory_kwargs,
                sensor_observations=index,
                degradation={"enabled": False},
                degradation_seed=42,
            )
            validators.append(validator)
            return validator

        model = YOLO(str(checkpoint))
        captured, hook = _gate_collector(model)
        try:
            model.val(
                validator=validator_factory,
                data=str(data),
                split="test",
                imgsz=640,
                batch=int(args.batch_size),
                device=str(args.device),
                workers=int(args.workers),
                project=str(output.parent / "separation-runs"),
                name=name,
                exist_ok=False,
                plots=False,
                save_json=False,
                verbose=False,
            )
        finally:
            hook.remove()
        if validators[0].external_sensor_sample_count != 570:
            raise ValueError(f"{name} separation run did not consume 570 images")
        populations[name] = {
            key: _values(captured, key)
            for key in (
                "sensor_raw_reliability",
                "sensor_consistency",
                "sensor_discrepancy",
            )
        }

    matched = populations["matched"]
    shuffled_values = populations["shuffled"]
    metrics = {
        "discrepancy_auc_shuffled_positive": binary_auc(
            matched["sensor_discrepancy"],
            shuffled_values["sensor_discrepancy"],
        ),
        "consistency_auc_shuffled_positive": binary_auc(
            -matched["sensor_consistency"],
            -shuffled_values["sensor_consistency"],
        ),
        "raw_reliability_auc_shuffled_positive": binary_auc(
            -matched["sensor_raw_reliability"],
            -shuffled_values["sensor_raw_reliability"],
        ),
        "paired_shuffled_discrepancy_higher_fraction": float(
            (
                shuffled_values["sensor_discrepancy"]
                > matched["sensor_discrepancy"]
            )
            .float()
            .mean()
            .item()
        ),
        "paired_shuffled_reliability_lower_fraction": float(
            (
                shuffled_values["sensor_raw_reliability"]
                < matched["sensor_raw_reliability"]
            )
            .float()
            .mean()
            .item()
        ),
    }
    result = {
        "schema_version": 1,
        "valid": True,
        "formal_paper_metric": False,
        "sample_count_per_population": 570,
        "metrics": metrics,
        "populations": {
            condition: {
                name: _summary(values)
                for name, values in condition_values.items()
            }
            for condition, condition_values in populations.items()
        },
        "interpretation": (
            "AUC=0.5 is chance and AUC=1.0 is perfect separation. "
            "This diagnostic does not establish physical-sensor robustness."
        ),
        "bindings": {
            "checkpoint": _binding(checkpoint),
            "observations": _binding(observations),
            "shuffled_observations": _binding(shuffled),
            "protocol": _binding(protocol_path),
            "evaluation": _binding(evaluation_path),
            "diagnostic_source": _binding(Path(__file__)),
        },
    }
    _atomic_json(output, result)
    return result


def main() -> None:
    result = diagnose(parse_args())
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
