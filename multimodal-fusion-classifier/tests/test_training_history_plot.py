from __future__ import annotations

import json

import pytest

from plot_training_history import load_history, resolve_best_epoch


def test_load_training_history_document(tmp_path):
    path = tmp_path / "history.json"
    path.write_text(
        json.dumps(
            {
                "best_checkpoint_epoch": 2,
                "history": [
                    {"epoch": 1, "val_accuracy": 0.5},
                    {"epoch": 2, "val_accuracy": 0.6},
                ],
            }
        ),
        encoding="utf-8",
    )

    document, rows = load_history(path)

    assert document["best_checkpoint_epoch"] == 2
    assert [row["epoch"] for row in rows] == [1, 2]


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{"epoch": 2}, {"epoch": 1}],
        [{"epoch": 1}, {"epoch": 1}],
    ],
)
def test_load_training_history_rejects_invalid_rows(tmp_path, rows):
    path = tmp_path / "history.json"
    path.write_text(json.dumps({"history": rows}), encoding="utf-8")

    with pytest.raises(ValueError):
        load_history(path)


def test_old_history_infers_first_validation_accuracy_argmax():
    document = {"best_val_accuracy": 0.7}
    rows = [
        {"epoch": 1, "val_accuracy": 0.6},
        {"epoch": 2, "val_accuracy": 0.7},
        {"epoch": 3, "val_accuracy": 0.7},
    ]

    epoch, metric, inferred = resolve_best_epoch(document, rows)

    assert epoch == 2
    assert metric == "accuracy"
    assert inferred is True
