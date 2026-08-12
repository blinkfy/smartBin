from __future__ import annotations

import json
from pathlib import Path
import shutil

from audit_variant_lineage import (
    EXPECTED_SCIENTIFIC_CHANGES,
    PROJECT_ROOT,
    VARIANT_FILES,
    build_variant_lineage,
)


def test_variant_lineage_accepts_locked_project_configs():
    report = build_variant_lineage(PROJECT_ROOT / "configs" / "ablations")

    assert report["valid"] is True
    assert report["state"] == "complete"
    assert report["lineage"] == ["full", "v2b", "v3", "balanced"]
    assert len(report["variants"]) == 4
    assert len(report["comparisons"]) == 3
    for comparison in report["comparisons"]:
        key = (comparison["reference"], comparison["candidate"])
        assert comparison["scientific_change_count"] == len(
            EXPECTED_SCIENTIFIC_CHANGES[key]
        )
        assert comparison["valid"] is True
    assert report["shared_invariants"]["leaf_count"] > 50


def test_variant_lineage_rejects_unregistered_scientific_change(
    tmp_path: Path,
):
    config_root = tmp_path / "configs"
    config_root.mkdir()
    source_root = PROJECT_ROOT / "configs" / "ablations"
    for filename in VARIANT_FILES.values():
        shutil.copy2(source_root / filename, config_root / filename)

    candidate = config_root / VARIANT_FILES["v3"]
    config = json.loads(candidate.read_text(encoding="utf-8"))
    config["model"]["dropout"] = 0.4
    candidate.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report = build_variant_lineage(config_root)

    assert report["valid"] is False
    assert any(
        "unexpected changes: model.dropout" in error
        for error in report["errors"]
    )
