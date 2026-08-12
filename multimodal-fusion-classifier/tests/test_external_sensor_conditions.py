from __future__ import annotations

from evaluate_external_sensor_conditions import derange_sensor_records


def test_sensor_derangement_preserves_split_and_has_no_self_pair() -> None:
    records = []
    for split in ("train", "val", "test"):
        for index in range(3):
            source_id = f"{split}-{index}"
            records.append(
                {
                    "source_image_id": source_id,
                    "split": split,
                    "grid": [[index / 3] * 4] * 4,
                    "normalized_grid": [[index / 3] * 4] * 4,
                    "properties": {"weight_g": index + 1},
                    "source": {
                        "observation_scope": "full_image",
                        "full_image_only": True,
                        "annotation_inputs": [],
                    },
                }
            )

    shuffled = derange_sensor_records(records)

    assert len(shuffled) == len(records)
    assert {
        (record["source_image_id"], record["split"])
        for record in shuffled
    } == {
        (record["source_image_id"], record["split"])
        for record in records
    }
    assert all(
        record["source_image_id"]
        != record["source"]["diagnostic_donor_source_image_id"]
        for record in shuffled
    )
