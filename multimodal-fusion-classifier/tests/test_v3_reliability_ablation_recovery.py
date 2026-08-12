import json
from pathlib import Path

import pytest

from audit_v3_reliability_ablation_recovery import CONFIGS, PROJECT_ROOT
from prepare_v3_reliability_ablation_recovery import (
    BASE_CONFIG,
    validate_recovery_config,
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_workers0_recovery_configs_preserve_scientific_intervention(seed):
    check = validate_recovery_config(
        _read_json(BASE_CONFIG),
        _read_json(CONFIGS[seed]),
        seed=seed,
    )
    assert check["valid"], check["errors"]
    config = _read_json(CONFIGS[seed])
    assert config["training"]["num_workers"] == 0
    assert config["training"]["seed"] == seed
    assert config["training"]["checkpoint_dir"].endswith(f"/seed{seed}")


def test_workers4_stall_incident_is_closed_and_not_reused():
    incident = _read_json(
        PROJECT_ROOT
        / "reports"
        / "qwen-material"
        / "confirmatory-ablations"
        / "v3-no-reliability"
        / "WORKERS4_STALL_INCIDENT.json"
    )
    assert incident["state"] == "closed_incomplete_and_archived"
    assert incident["valid"] is True
    assert incident["facts"]["last_completed_epoch"] == 69
    assert incident["recovery_decision"]["reuse_or_resume_partial_run"] is False
    assert incident["recovery_decision"]["fresh_training_from_epoch_1"] is True
    assert incident["recovery_decision"]["replacement_num_workers"] == 0


def test_workers0_recovery_queue_has_fixed_history_and_no_resume():
    queue = (
        PROJECT_ROOT
        / "reports"
        / "qwen-material"
        / "training"
        / "run_v3_reliability_ablation_workers0_recovery.ps1"
    ).read_text(encoding="utf-8-sig")
    assert "$historyRows = @($historyDocument.history)" in queue
    assert 'monitoring = "stage_only"' in queue
    assert "workers4_partial_run_reused = $false" in queue
    assert '"--num-workers", "0"' in queue
    assert "--resume" not in queue
    assert "$ReportBase" in queue
    assert "$SeedReportDir" in queue
    assert "$reportRoot" not in queue
    assert "$reportBase" not in queue


def test_workers0_recovery_keeps_full_as_comparison_candidate():
    queue = (
        PROJECT_ROOT
        / "reports"
        / "qwen-material"
        / "training"
        / "run_v3_reliability_ablation_workers0_recovery.ps1"
    ).read_text(encoding="utf-8-sig")
    assert (
        '"--reference-name",\n'
        '                "v3_no_reliability_workers0_seed${Seed}"'
    ) in queue
    assert '"--candidate-name", "full_cross_modal_quality_seed${Seed}"' in queue
    assert '"--repeats", "5000"' in queue
