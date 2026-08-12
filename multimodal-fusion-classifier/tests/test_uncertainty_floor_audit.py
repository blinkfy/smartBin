import hashlib
import json
from pathlib import Path

import pytest

from audit_uncertainty_floor import (
    _config_fingerprint,
    audit_uncertainty_floor,
)


def _write_variant(
    root: Path,
    *,
    name: str,
    reconstruction: float = -1.11,
    prior_mse: float = 0.06,
) -> dict[str, str]:
    config = {
        "experiment": {"name": name},
        "model": {
            "sensor_log_variance_min": -3.0,
            "sensor_log_variance_max": 2.0,
        },
        "loss": {
            "variance_prior_weight": 0.1,
            "variance_prior_target": -2.0,
        },
        "training": {"resume": False},
    }
    config_path = root / f"{name}.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    history = {
        "experiment": {"name": name},
        "config_fingerprint": _config_fingerprint(config),
        "history": [
            {
                "epoch": 100,
                "val_sensor_reconstruction": reconstruction,
                "val_sensor_variance_prior": prior_mse,
            }
        ],
    }
    history_path = root / f"{name}.history.json"
    history_path.write_text(json.dumps(history), encoding="utf-8")
    return {
        "config": config_path.as_posix(),
        "history": history_path.as_posix(),
    }


def test_uncertainty_floor_audit_derives_conservative_saturation_bound(
    tmp_path: Path,
):
    paths = _write_variant(tmp_path, name="candidate")

    report = audit_uncertainty_floor(
        project_root=tmp_path,
        variants={"candidate": paths},
    )

    assert report["valid"] is True
    row = report["variants"]["candidate"]
    assert row["theoretical_nll_floor"] == -1.5
    assert row["nll_floor_gap"] == pytest.approx(0.39)
    assert row["near_theoretical_floor"] is False
    assert row["exact_lower_clamp_fraction_upper_bound"] == pytest.approx(
        0.06
    )
    assert row["source_config_fingerprint"] == hashlib.sha256(
        json.dumps(
            {
                "experiment": {"name": "candidate"},
                "loss": {
                    "variance_prior_target": -2.0,
                    "variance_prior_weight": 0.1,
                },
                "model": {
                    "sensor_log_variance_max": 2.0,
                    "sensor_log_variance_min": -3.0,
                },
                "training": {},
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def test_uncertainty_floor_audit_rejects_history_fingerprint_drift(
    tmp_path: Path,
):
    paths = _write_variant(tmp_path, name="candidate")
    history_path = Path(paths["history"])
    history = json.loads(history_path.read_text(encoding="utf-8"))
    history["config_fingerprint"] = "0" * 64
    history_path.write_text(json.dumps(history), encoding="utf-8")

    report = audit_uncertainty_floor(
        project_root=tmp_path,
        variants={"candidate": paths},
    )

    assert report["valid"] is False
    assert "configuration fingerprint mismatch" in report["errors"][0]
