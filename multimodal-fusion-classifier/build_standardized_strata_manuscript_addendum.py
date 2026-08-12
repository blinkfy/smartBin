"""Build paper-ready text for the audited standardized strata diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from mmfc.model import PROJECT_ROOT


ROOT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "standardized-instance-strata"
)
DEFAULT_AUDIT = ROOT / "AUDIT.json"
DEFAULT_OUTPUT = ROOT / "MANUSCRIPT_ADDENDUM.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the standardized-strata manuscript addendum"
    )
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return (
        path.resolve()
        if path.is_absolute()
        else (PROJECT_ROOT / path).resolve()
    )


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.relative_to(PROJECT_ROOT).as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _percent(value: float) -> float:
    return 100.0 * float(value)


def build_addendum(
    audit_path: Path = DEFAULT_AUDIT,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    audit_path = _resolve(audit_path)
    output = _resolve(output)
    if not audit_path.is_file():
        raise FileNotFoundError(f"strata audit is missing: {audit_path}")
    audit = _read(audit_path)
    contract = audit.get("target_contract", {})
    selected = audit.get("selected_paired_results", {})
    interpretation = audit.get("interpretation", {})
    if (
        audit.get("valid") is not True
        or audit.get("formal_paper_metric") is not False
        or contract.get("source_image_count") != 570
        or contract.get("target_count") != 825
        or contract.get("scale_support")
        != {"small": 11, "medium": 486, "large": 328}
        or contract.get("overlap_support")
        != {"isolated": 717, "overlap_low": 88, "overlap_high": 20}
        or not all(interpretation.values())
    ):
        raise ValueError("standardized-strata audit contract is invalid")

    v5 = selected["v5_isolated_mask_mean_recall"]
    v6 = selected["v6_isolated_mask_mean_recall"]
    small = selected["v6_small_mask_mean_recall"]
    overlap_v5 = selected["v5_overlap_high_mask_mean_recall"]
    overlap_v6 = selected["v6_overlap_high_mask_mean_recall"]
    result = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "formal_paper_metric": False,
        "evidence_role": "paper_ready_posthoc_diagnostic_addendum",
        "methods": {
            "dataset": (
                "The frozen test split contained 570 source images and "
                "825 annotated instances."
            ),
            "scale_definition": (
                "Instances were grouped by rasterized visible-mask area "
                "using COCO thresholds: small < 32^2, medium from 32^2 "
                "to < 96^2, and large >= 96^2 pixels."
            ),
            "overlap_definition": (
                "Crowding was approximated by the maximum intersection "
                "area divided by the current target box area: isolated "
                "= 0, low overlap in (0, 0.25), and high overlap >= 0.25."
            ),
            "metric": (
                "Class-aware greedy one-to-one target recall was measured "
                "at mask IoU 0.50, 0.75, and as mean recall over "
                "0.50:0.05:0.95; 95% confidence intervals used 2,000 "
                "source-image bootstrap replicates."
            ),
            "inference": (
                "Inference used imgsz=640, conf=0.25, NMS IoU=0.5, "
                "FP32, retina masks, batch=8, and 64-image chunks."
            ),
        },
        "support": {
            "scale": contract["scale_support"],
            "overlap_proxy": contract["overlap_support"],
        },
        "selected_results_percentage_points": {
            "v5_isolated_mask_mean_recall_difference": {
                "estimate": _percent(v5["difference"]),
                "ci_lower": _percent(v5["ci_lower"]),
                "ci_upper": _percent(v5["ci_upper"]),
            },
            "v6_isolated_mask_mean_recall_difference": {
                "estimate": _percent(v6["difference"]),
                "ci_lower": _percent(v6["ci_lower"]),
                "ci_upper": _percent(v6["ci_upper"]),
            },
            "v6_small_mask_mean_recall_difference": {
                "estimate": _percent(small["difference"]),
                "ci_lower": _percent(small["ci_lower"]),
                "ci_upper": _percent(small["ci_upper"]),
            },
            "v5_high_overlap_mask_mean_recall_difference": {
                "estimate": _percent(overlap_v5["difference"]),
                "ci_lower": _percent(overlap_v5["ci_lower"]),
                "ci_upper": _percent(overlap_v5["ci_upper"]),
            },
            "v6_high_overlap_mask_mean_recall_difference": {
                "estimate": _percent(overlap_v6["difference"]),
                "ci_lower": _percent(overlap_v6["ci_lower"]),
                "ci_upper": _percent(overlap_v6["ci_upper"]),
            },
        },
        "numerical_sensitivity_percentage_points": {
            "large_mask_mean_recall": _percent(
                abs(
                    audit["batch_sensitivity"][
                        "large_mask_mean_recall_difference"
                    ]
                )
            ),
            "high_overlap_mask_mean_recall": _percent(
                abs(
                    audit["batch_sensitivity"][
                        "overlap_high_mask_mean_recall_difference"
                    ]
                )
            ),
            "exact_batch_invariance": False,
        },
        "claim_boundary": {
            "interpretation": (
                "The isolated-instance gains are descriptive internal "
                "development evidence. Small-object and high-overlap "
                "results are sample-limited and cannot establish robust "
                "small-object or occlusion performance."
            ),
            "overlap_is_human_occlusion_label": False,
            "formal_blind_metric": False,
            "supports_original_patent_percentages": False,
        },
        "bindings": {
            "audit": _binding(audit_path),
            "builder_source": _binding(Path(__file__)),
        },
    }
    _atomic_text(
        output,
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
    )

    v5_pp = result["selected_results_percentage_points"][
        "v5_isolated_mask_mean_recall_difference"
    ]
    v6_pp = result["selected_results_percentage_points"][
        "v6_isolated_mask_mean_recall_difference"
    ]
    markdown = [
        "# 标准化实例分层分析附录",
        "",
        "## 方法",
        "",
        (
            "固定测试集包含570张源图像和825个目标。按栅格化可见掩膜面积采用"
            "COCO阈值划分小（<32²）、中（32²至<96²）和大目标（>=96²）；"
            "对应样本量为11、486和328。"
        ),
        (
            "拥挤程度使用“与其他目标框的最大交集面积/当前目标框面积”代理："
            "无重叠、低重叠(0, 0.25)和高重叠>=0.25，对应717、88和20例。"
            "该变量不是人工遮挡标签。"
        ),
        (
            "采用类别一致、贪心一对一匹配的目标级Recall@0.50、Recall@0.75"
            "和mR@0.50:0.95；95%置信区间由源图像级2,000次bootstrap得到。"
            "它是固定阈值诊断，不等同于COCOeval AP。"
        ),
        "",
        "## 结果",
        "",
        (
            "相对RGB基线，v5和v6的无框重叠子集掩膜mR@0.50:0.95分别提高"
            f"{v5_pp['estimate']:.3f}个百分点"
            f"（95% CI {v5_pp['ci_lower']:.3f}至{v5_pp['ci_upper']:.3f}）和"
            f"{v6_pp['estimate']:.3f}个百分点"
            f"（95% CI {v6_pp['ci_lower']:.3f}至{v6_pp['ci_upper']:.3f}）。"
        ),
        (
            "小目标仅11例，高重叠代理仅20例；两组结果只作探索性报告，"
            "不足以形成稳健的小目标或遮挡性能结论。"
        ),
        "",
        "## 数值边界",
        "",
        (
            "为规避显存不足，最终推理采用batch=8、每块64张图像。与未分块"
            "RGB运行相比，R@0.50和R@0.75完全一致，但一个高IoU掩膜匹配发生"
            "变化；大目标与高重叠掩膜mR数值敏感性分别为"
            f"{result['numerical_sensitivity_percentage_points']['large_mask_mean_recall']:.4f}"
            "和"
            f"{result['numerical_sensitivity_percentage_points']['high_overlap_mask_mean_recall']:.3f}"
            "个百分点。"
        ),
        "",
        (
            "本附录属于事后内部诊断，不能用于宣称已复现原专利93.3%小目标"
            "或95.7%遮挡性能。"
        ),
        "",
    ]
    _atomic_text(output.with_suffix(".md"), "\n".join(markdown))
    return result


def main() -> None:
    args = parse_args()
    result = build_addendum(args.audit, args.output)
    print(
        json.dumps(
            {
                "state": result["state"],
                "valid": result["valid"],
                "formal_paper_metric": result["formal_paper_metric"],
                "support": result["support"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
