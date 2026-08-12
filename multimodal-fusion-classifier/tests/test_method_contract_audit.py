from __future__ import annotations

import json
from pathlib import Path

from audit_method_contract import audit_method_contract


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORMAL_CONFIG = PROJECT_ROOT / "configs/ablations/full_cross_modal_quality.json"


def test_formal_method_contract_is_valid():
    report = audit_method_contract(PROJECT_ROOT, FORMAL_CONFIG)

    assert report["valid"] is True
    assert report["state"] == "complete"
    assert report["tensor_contract"]["raw_grid_quality_shape"] == [2, 7]
    assert report["tensor_contract"]["cross_modal_grid_quality_shape"] == [2, 22]
    assert report["tensor_contract"]["film_gamma_shape"] == [3, 512]
    assert report["tensor_contract"]["film_affine_parameter_count"] == 3072
    assert report["tensor_contract"]["scale_prior_is_trainable"] is True
    assert report["tensor_contract"]["scale_prior_initial"] == [0.3, 0.5, 0.2]
    assert (
        report["tensor_contract"]["modality_weight_formula_max_abs_error"]
        <= 1e-7
    )
    assert report["tensor_contract"]["modality_weight_row_sums"] == [1.0, 1.0]
    assert report["tensor_contract"]["missing_reliability"] == 0.0
    assert report["tensor_contract"]["missing_fallback_max_abs_error"] <= 1e-7
    assert all(item["passed"] for item in report["checks"])


def test_method_contract_rejects_configuration_drift(tmp_path):
    config = json.loads(FORMAL_CONFIG.read_text(encoding="utf-8"))
    config["model"]["scale_prior"] = [0.2, 0.6, 0.2]
    drifted = tmp_path / "drifted.json"
    drifted.write_text(
        json.dumps(config, ensure_ascii=False),
        encoding="utf-8",
    )

    report = audit_method_contract(PROJECT_ROOT, drifted)

    assert report["valid"] is False
    assert "configuration contract mismatch: model.scale_prior" in report["errors"]
