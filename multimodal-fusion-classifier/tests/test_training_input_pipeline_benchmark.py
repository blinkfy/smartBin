from __future__ import annotations

import torch
from torch.utils.data import Dataset

from benchmark_training_input_pipeline import benchmark_options


class TinyDataset(Dataset):
    def __len__(self) -> int:
        return 16

    def __getitem__(self, index: int) -> dict:
        return {
            "sample_id": f"sample-{index}",
            "source_image_id": f"source-{index // 2}",
            "image": torch.full((3, 4, 4), float(index), dtype=torch.float32),
            "sensor": torch.full((1, 4, 4), index / 16.0),
            "label": torch.tensor(index % 3, dtype=torch.long),
        }


def test_input_pipeline_benchmark_preserves_tensor_order():
    report = benchmark_options(
        dataset=TinyDataset(),
        workers=[0],
        batch_size=4,
        epochs=2,
        equivalence_samples=8,
        seed=42,
    )

    assert report["valid"] is True
    assert report["recommended_num_workers"] == 0
    assert report["options"][0]["samples_per_epoch"] == [16, 16]
    assert report["options"][0]["steady_state_samples_per_second"] > 0


def test_input_pipeline_benchmark_rejects_duplicate_workers():
    try:
        benchmark_options(
            dataset=TinyDataset(),
            workers=[0, 0],
            batch_size=4,
            epochs=1,
            equivalence_samples=4,
            seed=42,
        )
    except ValueError as error:
        assert "duplicates" in str(error)
    else:
        raise AssertionError("duplicate worker counts should be rejected")
