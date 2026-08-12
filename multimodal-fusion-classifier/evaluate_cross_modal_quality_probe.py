"""Evaluate whether visual-text sensor latents can recover raw-grid quality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mmfc.checkpoints import (
    load_checkpoint,
    load_model_state_compatible,
    validated_checkpoint_class_names,
)
from mmfc.cross_modal_quality_probe import (
    aggregate_cross_modal_quality_runs,
    evaluate_cross_modal_quality_probe,
    fit_cross_modal_quality_probe,
    write_cross_modal_quality_multiseed_artifacts,
    write_cross_modal_quality_probe_artifacts,
)
from mmfc.model import build_production_model
from mmfc.training import (
    ManifestDataset,
    PROJECT_ROOT,
    load_config,
    resolve_project_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe visual-text latent reconstruction of clean sensor grids"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs"
        / "ablations"
        / "full_quality_aware_prior.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "checkpoints"
        / "ablations"
        / "full_quality_aware_prior"
        / "best.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "qwen-material"
        / "cross-modal-quality-probe",
    )
    parser.add_argument("--decoder-epochs", type=int, default=50)
    parser.add_argument("--quality-epochs", type=int, default=50)
    parser.add_argument("--decoder-hidden-dim", type=int, default=64)
    parser.add_argument("--quality-hidden-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--ranking-weight", type=float, default=0.2)
    parser.add_argument("--ranking-margin", type=float, default=0.05)
    parser.add_argument("--clean-weight", type=float, default=3.0)
    parser.add_argument(
        "--latent-source",
        choices=("predicted_sensor", "visual_context", "combined"),
        default="combined",
        help="Feature source used to reconstruct the clean 4x4 sensor grid",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--degradation-seed", type=int, default=2026)
    parser.add_argument(
        "--degradation-seeds",
        nargs="+",
        type=int,
        default=None,
        help="Optional multiple degradation seeds for mean/std reporting",
    )
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def _extract_latents(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    latent_source: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    latents: list[torch.Tensor] = []
    grids: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            sensors = batch["sensor"].to(device, non_blocking=True)
            availability = torch.ones(
                sensors.shape[0],
                1,
                device=device,
                dtype=sensors.dtype,
            )
            _, aux = model(
                images,
                sensors,
                sensor_available=availability,
                return_aux=True,
            )
            if latent_source == "predicted_sensor":
                latent = aux["predicted_sensor"]
            elif latent_source == "visual_context":
                latent = torch.cat(
                    (
                        aux["visual_base"],
                        aux["visual_text_context"],
                    ),
                    dim=1,
                )
            elif latent_source == "combined":
                latent = torch.cat(
                    (
                        aux["visual_base"],
                        aux["visual_text_context"],
                        aux["predicted_sensor"],
                    ),
                    dim=1,
                )
            else:
                raise ValueError(f"unsupported latent source: {latent_source}")
            latents.append(latent.detach().cpu())
            grids.append(batch["sensor"].detach().cpu())
    return torch.cat(latents, dim=0), torch.cat(grids, dim=0)


def _loader(
    config: dict,
    split: str,
    class_names: tuple[str, ...],
    *,
    batch_size: int,
    device: torch.device,
) -> DataLoader:
    data = config["data"]
    dataset = ManifestDataset(
        data[f"{split}_manifest"],
        class_names=class_names,
        image_size=int(data.get("image_size", 640)),
        sensor_manifest=data["sensor_manifest"],
        preprocess=config.get("preprocess", {}),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    config = load_config(args.config)
    checkpoint_path = resolve_project_path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, device)
    class_names = validated_checkpoint_class_names(checkpoint, config)
    model = build_production_model(config).to(device)
    load_model_state_compatible(model, checkpoint["model_state"])

    train_latents, train_grids = _extract_latents(
        model,
        _loader(
            config,
            "train",
            class_names,
            batch_size=args.feature_batch_size,
            device=device,
        ),
        device,
        latent_source=args.latent_source,
    )
    test_latents, test_grids = _extract_latents(
        model,
        _loader(
            config,
            "test",
            class_names,
            batch_size=args.feature_batch_size,
            device=device,
        ),
        device,
        latent_source=args.latent_source,
    )
    degradation = config.get("robustness", {}).get(
        "sensor_degradation",
        {},
    )
    decoder, quality_probe, history = fit_cross_modal_quality_probe(
        train_latents,
        train_grids,
        degradation,
        device=device,
        seed=args.seed,
        decoder_epochs=args.decoder_epochs,
        quality_epochs=args.quality_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        decoder_hidden_dim=args.decoder_hidden_dim,
        quality_hidden_dim=args.quality_hidden_dim,
        ranking_weight=args.ranking_weight,
        ranking_margin=args.ranking_margin,
        clean_weight=args.clean_weight,
    )
    degradation_seeds = list(
        dict.fromkeys(
            args.degradation_seeds
            if args.degradation_seeds is not None
            else [args.degradation_seed]
        )
    )
    runs = {}
    decoder_metrics = None
    for degradation_seed in degradation_seeds:
        rows, current_decoder_metrics = evaluate_cross_modal_quality_probe(
            decoder,
            quality_probe,
            test_latents,
            test_grids,
            device=device,
            seed=degradation_seed,
            batch_size=args.batch_size,
        )
        runs[degradation_seed] = rows
        decoder_metrics = current_decoder_metrics
    assert decoder_metrics is not None
    metadata = {
        "config_path": config["_config_path"],
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "train_samples": int(train_latents.shape[0]),
        "test_samples": int(test_latents.shape[0]),
        "latent_dim": int(train_latents.shape[1]),
        "latent_source": args.latent_source,
        "decoder_epochs": args.decoder_epochs,
        "quality_epochs": args.quality_epochs,
        "decoder_hidden_dim": args.decoder_hidden_dim,
        "quality_hidden_dim": args.quality_hidden_dim,
        "batch_size": args.batch_size,
        "feature_batch_size": args.feature_batch_size,
        "learning_rate": args.learning_rate,
        "ranking_weight": args.ranking_weight,
        "ranking_margin": args.ranking_margin,
        "clean_weight": args.clean_weight,
        "training_seed": args.seed,
        "degradation_seeds": degradation_seeds,
        "device": str(device),
        "probe_scope": "visual_text_latent_to_raw_grid_quality",
    }
    output_dir = resolve_project_path(args.output)
    if len(degradation_seeds) == 1:
        summary = runs[degradation_seeds[0]]
        artifacts = write_cross_modal_quality_probe_artifacts(
            decoder,
            quality_probe,
            history,
            summary,
            output_dir,
            metadata=metadata,
            decoder_metrics=decoder_metrics,
        )
    else:
        for degradation_seed, rows in runs.items():
            write_cross_modal_quality_probe_artifacts(
                decoder,
                quality_probe,
                history,
                rows,
                output_dir / "seeds" / f"seed-{degradation_seed}",
                metadata={
                    **metadata,
                    "degradation_seeds": [degradation_seed],
                },
                decoder_metrics=decoder_metrics,
            )
        summary = aggregate_cross_modal_quality_runs(runs)
        artifacts = write_cross_modal_quality_multiseed_artifacts(
            runs,
            output_dir,
            metadata=metadata,
            decoder_metrics=decoder_metrics,
        )
    print(
        json.dumps(
            {
                "decoder_metrics": decoder_metrics,
                "summary": summary,
                "artifacts": artifacts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
