import json
from pathlib import Path

import pytest

from audit_v3_reliability_ablation import _history_rows
from build_v3_reliability_ablation_protocol import (
    BASE_CONFIG,
    COMPARISON_BOOTSTRAP_SEEDS,
    CONFIGS,
    PROJECT_ROOT,
    validate_ablation_config,
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_generated_ablation_configs_are_exact_matched_interventions(seed):
    check = validate_ablation_config(
        _read_json(BASE_CONFIG),
        _read_json(CONFIGS[seed]),
        seed=seed,
    )
    assert check["valid"], check["errors"]


def test_comparison_bootstrap_seeds_are_unique_and_fixed():
    assert COMPARISON_BOOTSTRAP_SEEDS == {
        "42": 20260842,
        "43": 20260843,
        "44": 20260844,
    }
    assert len(set(COMPARISON_BOOTSTRAP_SEEDS.values())) == 3


def test_history_reader_requires_object_with_history_list(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(
        json.dumps({"history": [{"epoch": 1}, {"epoch": 2}]}),
        encoding="utf-8",
    )
    assert [row["epoch"] for row in _history_rows(good)] == [1, 2]

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"epoch": 1}]), encoding="utf-8")
    with pytest.raises(ValueError, match="root must be an object"):
        _history_rows(bad)


def test_queue_uses_stage_only_monitoring_and_safe_powershell_names():
    queue = (
        PROJECT_ROOT
        / "reports"
        / "qwen-material"
        / "training"
        / "run_v3_reliability_ablation.ps1"
    ).read_text(encoding="utf-8-sig")
    assert 'monitoring = "stage_only"' in queue
    assert "$historyRows = @($historyDocument.history)" in queue
    assert "$ReportBase" in queue
    assert "$SeedReportDir" in queue
    assert "$reportRoot" not in queue
    assert "$reportBase" not in queue
    assert "--resume" not in queue
    assert '"--num-workers", "4"' in queue


def test_queue_names_full_model_as_candidate_for_paired_effects():
    queue = (
        PROJECT_ROOT
        / "reports"
        / "qwen-material"
        / "training"
        / "run_v3_reliability_ablation.ps1"
    ).read_text(encoding="utf-8-sig")
    assert '"--reference-name", "v3_no_reliability_seed${Seed}"' in queue
    assert '"--candidate-name", "full_cross_modal_quality_seed${Seed}"' in queue
    assert '"--repeats", "5000"' in queue
