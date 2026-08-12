"""Build the additive index for the v6 patent-aligned fusion stack."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from mmfc.model import PROJECT_ROOT
from train_patent_aligned_external_sensor_joint_seg import DEFAULT_NAME


REPORTS = PROJECT_ROOT / "reports" / "qwen-material"
RUN_ROOT = (
    REPORTS
    / "joint-multimodal-seg-patent"
    / DEFAULT_NAME
)
DEFAULT_OUTPUT = REPORTS / "PATENT_STACK_EXTENSION_INDEX.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the patent-stack extension index"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
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


def _patent_document() -> Path:
    matches = sorted(PROJECT_ROOT.glob("2026.4.3-1*.docx"))
    if len(matches) != 1:
        raise FileNotFoundError("expected exactly one 2026.4.3-1 patent DOCX")
    return matches[0]


def build_index(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = _resolve(output)
    paths = {
        "builder_source": Path(__file__),
        "patent_document": _patent_document(),
        "requirement_matrix": REPORTS / "PATENT_REQUIREMENT_MATRIX.json",
        "requirement_matrix_md": REPORTS / "PATENT_REQUIREMENT_MATRIX.md",
        "manuscript_addendum": (
            REPORTS / "PATENT_STACK_MANUSCRIPT_ADDENDUM.json"
        ),
        "manuscript_addendum_md": (
            REPORTS / "PATENT_STACK_MANUSCRIPT_ADDENDUM.md"
        ),
        "complexity_benchmark": REPORTS / "PATENT_STACK_COMPLEXITY.json",
        "complexity_benchmark_md": REPORTS / "PATENT_STACK_COMPLEXITY.md",
        "intervention_protocol": (
            RUN_ROOT / "PATENT_STACK_INTERVENTION_PROTOCOL.json"
        ),
        "intervention_protocol_md": (
            RUN_ROOT / "PATENT_STACK_INTERVENTION_PROTOCOL.md"
        ),
        "intervention_result": (
            RUN_ROOT / "PATENT_STACK_INTERVENTION.json"
        ),
        "intervention_result_md": (
            RUN_ROOT / "PATENT_STACK_INTERVENTION.md"
        ),
        "intervention_audit": (
            REPORTS / "PATENT_STACK_INTERVENTION_AUDIT.json"
        ),
        "intervention_audit_md": (
            REPORTS / "PATENT_STACK_INTERVENTION_AUDIT.md"
        ),
        "architecture_figure_metadata": (
            REPORTS
            / "joint-multimodal-seg-patent"
            / "patent_stack_architecture.json"
        ),
        "architecture_figure_png": (
            REPORTS
            / "joint-multimodal-seg-patent"
            / "patent_stack_architecture.png"
        ),
        "architecture_figure_pdf": (
            REPORTS
            / "joint-multimodal-seg-patent"
            / "patent_stack_architecture.pdf"
        ),
        "architecture_figure_caption": (
            REPORTS / "PATENT_STACK_FIGURE_CAPTION.md"
        ),
        "v6_protocol": RUN_ROOT / "PROTOCOL.json",
        "v6_evaluation": RUN_ROOT / "evaluation.json",
        "v6_conditions": RUN_ROOT / "RELIABILITY_CONDITIONS.json",
        "v6_outcome": RUN_ROOT / "OUTCOME.json",
        "v6_outcome_md": RUN_ROOT / "OUTCOME.md",
        "v5_audit": REPORTS / "SPATIAL_RELIABILITY_AUDIT.json",
        "qwen_formal_protocol": (
            REPORTS
            / "joint-multimodal-seg-external"
            / "QWEN_FORMAL_PROTOCOL.json"
        ),
        "qwen_v6_extension_protocol": (
            REPORTS
            / "joint-multimodal-seg-patent-qwen"
            / "QWEN_FORMAL_EXTENSION_PROTOCOL.json"
        ),
        "qwen_v6_extension_protocol_md": (
            REPORTS
            / "joint-multimodal-seg-patent-qwen"
            / "QWEN_FORMAL_EXTENSION_PROTOCOL.md"
        ),
        "qwen_v6_lineage_audit": (
            REPORTS
            / "joint-multimodal-seg-patent-qwen"
            / "QWEN_FORMAL_EXTENSION_LINEAGE_AUDIT.json"
        ),
        "qwen_v6_lineage_audit_md": (
            REPORTS
            / "joint-multimodal-seg-patent-qwen"
            / "QWEN_FORMAL_EXTENSION_LINEAGE_AUDIT.md"
        ),
        "qwen_v6_preflight": (
            REPORTS
            / "joint-multimodal-seg-patent-qwen"
            / "QWEN_FORMAL_EXTENSION_PREFLIGHT.json"
        ),
        "qwen_v6_preflight_md": (
            REPORTS
            / "joint-multimodal-seg-patent-qwen"
            / "QWEN_FORMAL_EXTENSION_PREFLIGHT.md"
        ),
        "motivation_decision_packet": (
            REPORTS / "MOTIVATION_DECISION_PACKET.json"
        ),
        "motivation_decision_packet_md": (
            REPORTS / "MOTIVATION_DECISION_PACKET.md"
        ),
        "paper_readiness_v6_addendum": (
            REPORTS / "PAPER_READINESS_V6_ADDENDUM.json"
        ),
        "paper_readiness_v6_addendum_md": (
            REPORTS / "PAPER_READINESS_V6_ADDENDUM.md"
        ),
        "worker_scaling_4_to_8": (
            REPORTS
            / "joint-multimodal-seg-external"
            / "WORKER_SCALING_4_TO_8.json"
        ),
        "worker_scaling_4_to_8_md": (
            REPORTS
            / "joint-multimodal-seg-external"
            / "WORKER_SCALING_4_TO_8.md"
        ),
        "standardized_strata_protocol": (
            REPORTS / "standardized-instance-strata" / "PROTOCOL.json"
        ),
        "standardized_strata_protocol_md": (
            REPORTS / "standardized-instance-strata" / "PROTOCOL.md"
        ),
        "standardized_strata_recovery": (
            REPORTS
            / "standardized-instance-strata"
            / "RUNTIME_RECOVERY_ADDENDUM.json"
        ),
        "standardized_strata_recovery_md": (
            REPORTS
            / "standardized-instance-strata"
            / "RUNTIME_RECOVERY_ADDENDUM.md"
        ),
        "standardized_strata_chunking": (
            REPORTS
            / "standardized-instance-strata"
            / "RUNTIME_CHUNKING_ADDENDUM.json"
        ),
        "standardized_strata_chunking_md": (
            REPORTS
            / "standardized-instance-strata"
            / "RUNTIME_CHUNKING_ADDENDUM.md"
        ),
        "standardized_strata_adjudication": (
            REPORTS
            / "standardized-instance-strata"
            / "RUNTIME_NUMERICAL_ADJUDICATION.json"
        ),
        "standardized_strata_adjudication_md": (
            REPORTS
            / "standardized-instance-strata"
            / "RUNTIME_NUMERICAL_ADJUDICATION.md"
        ),
        "standardized_strata_final_validation": (
            REPORTS
            / "standardized-instance-strata"
            / "FINAL_VALIDATION_ADDENDUM.json"
        ),
        "standardized_strata_final_validation_md": (
            REPORTS
            / "standardized-instance-strata"
            / "FINAL_VALIDATION_ADDENDUM.md"
        ),
        "standardized_strata_evaluation": (
            REPORTS
            / "standardized-instance-strata"
            / "evaluation-final-chunk64"
            / "evaluation.json"
        ),
        "standardized_strata_report": (
            REPORTS
            / "standardized-instance-strata"
            / "evaluation-final-chunk64"
            / "REPORT.md"
        ),
        "standardized_strata_metrics": (
            REPORTS
            / "standardized-instance-strata"
            / "evaluation-final-chunk64"
            / "metrics.csv"
        ),
        "standardized_strata_paired": (
            REPORTS
            / "standardized-instance-strata"
            / "evaluation-final-chunk64"
            / "paired_differences.csv"
        ),
        "standardized_strata_audit": (
            REPORTS / "standardized-instance-strata" / "AUDIT.json"
        ),
        "standardized_strata_audit_md": (
            REPORTS / "standardized-instance-strata" / "AUDIT.md"
        ),
        "standardized_strata_manuscript": (
            REPORTS
            / "standardized-instance-strata"
            / "MANUSCRIPT_ADDENDUM.json"
        ),
        "standardized_strata_manuscript_md": (
            REPORTS
            / "standardized-instance-strata"
            / "MANUSCRIPT_ADDENDUM.md"
        ),
        "standardized_strata_figure_metadata": (
            REPORTS
            / "standardized-instance-strata"
            / "standardized_instance_strata.json"
        ),
        "standardized_strata_figure_png": (
            REPORTS
            / "standardized-instance-strata"
            / "standardized_instance_strata.png"
        ),
        "standardized_strata_figure_pdf": (
            REPORTS
            / "standardized-instance-strata"
            / "standardized_instance_strata.pdf"
        ),
        "standardized_strata_figure_caption": (
            REPORTS
            / "standardized-instance-strata"
            / "FIGURE_CAPTION.md"
        ),
        "v6_model_source": PROJECT_ROOT / "mmfc" / "joint_seg_patent.py",
        "v6_trainer": (
            PROJECT_ROOT / "train_patent_aligned_external_sensor_joint_seg.py"
        ),
        "v6_evaluator": (
            PROJECT_ROOT / "evaluate_patent_aligned_conditions.py"
        ),
        "v6_auditor": PROJECT_ROOT / "audit_patent_aligned_joint_seg.py",
        "matrix_builder": PROJECT_ROOT / "build_patent_requirement_matrix.py",
        "manuscript_builder": (
            PROJECT_ROOT / "build_patent_stack_manuscript_addendum.py"
        ),
        "runtime_benchmark": (
            PROJECT_ROOT / "benchmark_patent_stack_runtime.py"
        ),
        "intervention_preparer": (
            PROJECT_ROOT / "prepare_patent_stack_intervention.py"
        ),
        "intervention_evaluator": (
            PROJECT_ROOT / "evaluate_patent_stack_intervention.py"
        ),
        "intervention_auditor": (
            PROJECT_ROOT / "audit_patent_stack_intervention.py"
        ),
        "architecture_plotter": (
            PROJECT_ROOT / "plot_patent_stack_architecture.py"
        ),
        "qwen_v6_protocol_builder": (
            PROJECT_ROOT / "prepare_patent_stack_qwen_formal_protocol.py"
        ),
        "qwen_derangement_builder": (
            PROJECT_ROOT / "prepare_qwen_sensor_derangement.py"
        ),
        "qwen_v6_formal_trainer": (
            PROJECT_ROOT / "train_patent_aligned_qwen_formal.py"
        ),
        "qwen_v6_formal_evaluator": (
            PROJECT_ROOT
            / "evaluate_patent_aligned_qwen_formal_conditions.py"
        ),
        "qwen_v6_formal_aggregator": (
            PROJECT_ROOT / "aggregate_patent_stack_qwen_formal.py"
        ),
        "qwen_v6_formal_runner": (
            PROJECT_ROOT / "run_patent_stack_qwen_formal.py"
        ),
        "qwen_v6_lineage_auditor": (
            PROJECT_ROOT / "audit_patent_stack_qwen_lineage.py"
        ),
        "motivation_packet_builder": (
            PROJECT_ROOT / "build_paper_motivation_decision_packet.py"
        ),
        "paper_readiness_v6_addendum_builder": (
            PROJECT_ROOT / "build_paper_readiness_v6_addendum.py"
        ),
        "worker_scaling_auditor": (
            PROJECT_ROOT
            / "audit_external_sensor_worker_scaling_4_to_8.py"
        ),
        "standardized_strata_protocol_builder": (
            PROJECT_ROOT
            / "prepare_standardized_instance_strata_protocol.py"
        ),
        "standardized_strata_evaluator": (
            PROJECT_ROOT / "evaluate_standardized_instance_strata.py"
        ),
        "standardized_strata_recovery_builder": (
            PROJECT_ROOT
            / "prepare_standardized_strata_runtime_addendum.py"
        ),
        "standardized_strata_chunking_builder": (
            PROJECT_ROOT
            / "prepare_standardized_strata_chunking_addendum.py"
        ),
        "standardized_strata_adjudicator": (
            PROJECT_ROOT
            / "prepare_standardized_strata_numerical_adjudication.py"
        ),
        "standardized_strata_final_validator": (
            PROJECT_ROOT
            / "prepare_standardized_strata_final_validation_addendum.py"
        ),
        "standardized_strata_auditor": (
            PROJECT_ROOT / "audit_standardized_instance_strata.py"
        ),
        "standardized_strata_manuscript_builder": (
            PROJECT_ROOT
            / "build_standardized_strata_manuscript_addendum.py"
        ),
        "standardized_strata_plotter": (
            PROJECT_ROOT / "plot_standardized_instance_strata.py"
        ),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "patent-stack index inputs missing: " + ", ".join(missing)
        )

    matrix = _read_json(paths["requirement_matrix"])
    outcome = _read_json(paths["v6_outcome"])
    evaluation = _read_json(paths["v6_evaluation"])
    conditions = _read_json(paths["v6_conditions"])
    addendum = _read_json(paths["manuscript_addendum"])
    complexity = _read_json(paths["complexity_benchmark"])
    intervention = _read_json(paths["intervention_audit"])
    architecture_figure = _read_json(paths["architecture_figure_metadata"])
    v5 = _read_json(paths["v5_audit"])
    qwen = _read_json(paths["qwen_formal_protocol"])
    qwen_v6 = _read_json(paths["qwen_v6_extension_protocol"])
    qwen_lineage = _read_json(paths["qwen_v6_lineage_audit"])
    qwen_preflight = _read_json(paths["qwen_v6_preflight"])
    motivation = _read_json(paths["motivation_decision_packet"])
    readiness_v6 = _read_json(paths["paper_readiness_v6_addendum"])
    worker_scaling = _read_json(paths["worker_scaling_4_to_8"])
    standardized_strata = _read_json(paths["standardized_strata_audit"])
    standardized_strata_manuscript = _read_json(
        paths["standardized_strata_manuscript"]
    )
    standardized_strata_figure = _read_json(
        paths["standardized_strata_figure_metadata"]
    )
    if (
        matrix.get("valid") is not True
        or matrix.get("summary", {}).get("complete_patent_replication")
        is not False
        or outcome.get("valid") is not True
        or outcome.get("formal_paper_metric") is not False
        or evaluation.get("valid") is not True
        or conditions.get("valid") is not True
        or addendum.get("valid") is not True
        or addendum.get("formal_paper_metric") is not False
        or complexity.get("valid") is not True
        or complexity.get("formal_paper_metric") is not False
        or intervention.get("valid") is not True
        or intervention.get("formal_paper_metric") is not False
        or architecture_figure.get("valid") is not True
        or standardized_strata.get("valid") is not True
        or standardized_strata.get("formal_paper_metric") is not False
        or standardized_strata_manuscript.get("valid") is not True
        or standardized_strata_manuscript.get("formal_paper_metric")
        is not False
        or standardized_strata_figure.get("valid") is not True
        or standardized_strata_figure.get("formal_paper_metric")
        is not False
        or standardized_strata_figure.get("render_checks", {})
        .get("manual_visual_inspection", {})
        .get("result")
        != "pass"
        or readiness_v6.get("valid") is not True
        or readiness_v6.get("historical_audits_modified") is not False
    ):
        raise ValueError("v6 patent-stack evidence is not valid")
    if (
        v5.get("valid") is not True
        or v5.get("selected_development_model", {}).get("selected")
        != "spatial_v5_primary"
    ):
        raise ValueError("selected v5 reliability evidence is inconsistent")
    if (
        qwen.get("status")
        != "preregistered_awaiting_api_credentials_and_data"
        or qwen.get("current_gate", {}).get("formal_training_started")
        is not False
    ):
        raise ValueError("formal Qwen gate is inconsistent")
    if (
        qwen_v6.get("protocol")
        != "full_image_qwen_patent_stack_adaptation_v6"
        or qwen_v6.get("status")
        != "preregistered_awaiting_api_credentials_and_data"
        or qwen_v6.get("lineage", {}).get(
            "shared_parent_across_seeds"
        )
        is not True
        or qwen_v6.get("lineage", {}).get(
            "independent_end_to_end_replicates"
        )
        is not False
        or qwen_lineage.get("valid") is not True
        or qwen_lineage.get("formal_training_complete") is not False
        or qwen_preflight.get("valid") is not True
        or qwen_preflight.get("state") != "awaiting_api_credentials"
        or qwen_preflight.get("training_workers") != 4
        or qwen_preflight.get("training_batch_size") != 32
        or motivation.get("valid") is not True
        or motivation.get("state") != "awaiting_user_confirmation"
        or motivation.get("confirmed_option") is not None
        or worker_scaling.get("valid") is not True
        or worker_scaling.get("recommendation", {}).get(
            "formal_training_workers"
        )
        != 4
        or worker_scaling.get("equivalence", {})
        .get("model_state", {})
        .get("exact")
        is not True
    ):
        raise ValueError(
            "v6 Qwen extension or paper-motivation gate is inconsistent"
        )

    test = outcome["test"]
    index = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "scope": "patent_aligned_native_segment_fusion_stack_v6",
        "formal_paper_metric": False,
        "capability_status": {
            "three_independent_512_adapters": "implemented_and_tested",
            "film_gamma_1_beta_0_1": "implemented_and_tested",
            "pairwise_cosine_modality_weighting": "implemented_and_tested",
            "down_512_to_128": "implemented_and_tested",
            "eight_head_three_token_interaction": "implemented_and_tested",
            "up_128_to_512_and_residual": "implemented_and_tested",
            "dynamic_p3_p4_p5_prior_0_3_0_5_0_2": (
                "implemented_and_tested"
            ),
            "native_segment_head_conditioning": "implemented_and_tested",
            "same_gpu_raw_forward_complexity": "measured_descriptively",
            "paper_architecture_png_pdf_and_caption": (
                "rendered_and_visually_checked"
            ),
            "observed_sensor_head_feature_causality": (
                "implemented_and_verified"
            ),
            "availability_zero_fallback_invariance": (
                "implemented_and_verified"
            ),
            "same_checkpoint_final_conditioning_intervention": (
                "implemented_and_audited"
            ),
            "exact_explicit_pairwise_bidirectional_concatenation": (
                "partially_aligned"
            ),
            "literal_8_16_32_spatial_maps": "partially_aligned_to_p3_p4_p5",
            "full_image_qwen_sidecars": "awaiting_api_credentials",
            "v6_qwen_adaptation_protocol": (
                "preregistered_awaiting_api_credentials"
            ),
            "v6_qwen_lineage_audit": "complete_valid",
            "paper_motivation": "awaiting_user_confirmation",
            "paper_readiness_v6_addendum": (
                "complete_valid_additive_update"
            ),
            "workers_4_to_8_scaling": (
                "exact_equivalence_workers4_faster"
            ),
            "standardized_instance_scale_overlap_diagnostic": (
                "audited_posthoc_with_runtime_sensitivity_disclosed"
            ),
            "paper_standardized_strata_png_pdf_and_caption": (
                "rendered_and_visually_checked"
            ),
            "hardware_sensor_array": "not_implemented",
            "rgb_d_input": "not_implemented",
        },
        "v6_development_result": {
            "seed_count": 1,
            "training_epochs": int(
                evaluation.get("training", {}).get("completed_epochs", 0)
            ),
            "sensor_source": "scene-rule-v1",
            "source_image_count": test["source_image_count"],
            "checkpoint_sha256": evaluation["checkpoint"]["sha256"],
            "clean_metrics": test["clean_metrics"],
            "clean_minus_rgb_fallback": test["clean_minus_rgb_fallback"],
            "clean_minus_shuffled": test["clean_minus_shuffled"],
            "maximum_absolute_box_or_mask_map50_95_effect": test[
                "maximum_absolute_box_or_mask_map50_95_effect"
            ],
            "effect_interpretation": test["effect_interpretation"],
            "observed_grid_head_feature_max_abs_difference": outcome[
                "structural_audit"
            ]["observed_grid_head_feature_max_abs_difference"],
            "fallback_head_feature_max_abs_difference": outcome[
                "structural_audit"
            ]["missing_grid_head_feature_max_abs_difference"],
            "formal_performance_evidence": False,
            "complexity": {
                "v5_gflops_per_image": complexity["models"][
                    "v5_spatial_reliability"
                ]["gflops_per_image"],
                "v6_gflops_per_image": complexity["models"][
                    "v6_patent_stack"
                ]["gflops_per_image"],
                "v6_latency_over_v5_fraction": complexity["v6_over_v5"][
                    "median_batch_latency_fraction"
                ],
                "timing_scope": complexity["scope"],
            },
            "conditioning_intervention": intervention[
                "contribution_decomposition"
            ],
        },
        "model_selection": {
            "selected_reliability_development_model": "spatial_v5_primary",
            "selected_patent_structure_model": "patent_stack_v6",
            "v6_promoted_for_task_performance": False,
            "reason": (
                "v6 closes structural S3-to-S4 gaps, while aggregate "
                "sensor-conditioned task effects are negligible"
            ),
        },
        "patent_coverage": matrix["summary"],
        "governance": {
            "rule_sidecar_result_promoted_to_formal": False,
            "original_patent_performance_numbers_inherited": False,
            "checkpoint_in_extension_archive": False,
            "source_images_in_extension_archive": False,
            "annotation_files_in_extension_archive": False,
            "patent_document_in_extension_archive": False,
            "architecture_png_pdf_in_extension_archive": False,
            "standardized_strata_png_pdf_in_extension_archive": False,
            "patent_document_bound_externally_by_sha256": True,
            "hardware_sensor_claim": False,
            "rgb_d_claim": False,
            "legacy_qwen_v2_protocol_modified": False,
            "qwen_v6_extension_is_additive": True,
            "qwen_v6_shared_parent_disclosed": True,
            "standardized_strata_formal_metric": False,
            "bbox_overlap_claimed_as_human_occlusion": False,
            "object_checkpoint_bound_externally_not_archived": True,
        },
        "standardized_strata_diagnostic": {
            "formal_paper_metric": False,
            "source_image_count": standardized_strata["target_contract"][
                "source_image_count"
            ],
            "target_count": standardized_strata["target_contract"][
                "target_count"
            ],
            "scale_support": standardized_strata["target_contract"][
                "scale_support"
            ],
            "overlap_support": standardized_strata["target_contract"][
                "overlap_support"
            ],
            "selected_paired_results": standardized_strata[
                "selected_paired_results"
            ],
            "batch_sensitivity": standardized_strata[
                "batch_sensitivity"
            ],
            "interpretation": standardized_strata["interpretation"],
            "claim_boundary": standardized_strata_manuscript[
                "claim_boundary"
            ],
        },
        "qwen_formal_extension": {
            "protocol": qwen_v6["protocol"],
            "state": qwen_preflight["state"],
            "api_credentials_available": qwen_preflight[
                "api_credentials_available"
            ],
            "qwen_data_ready": qwen_preflight["qwen_data"]["ready"],
            "derangement_ready": qwen_preflight["derangement"]["ready"],
            "planned_seeds": qwen_preflight["planned_seeds"],
            "training_batch_size": qwen_preflight[
                "training_batch_size"
            ],
            "training_workers": qwen_preflight["training_workers"],
            "workers8_slowdown_fraction": worker_scaling["timing"][
                "workers8_slowdown_fraction"
            ],
            "shared_parent_across_seeds": qwen_v6["lineage"][
                "shared_parent_across_seeds"
            ],
            "independent_end_to_end_replicates": qwen_v6["lineage"][
                "independent_end_to_end_replicates"
            ],
            "formal_results_available": False,
        },
        "paper_motivation_gate": {
            "state": motivation["state"],
            "recommended_option": motivation["recommended_option"],
            "confirmed_option": motivation["confirmed_option"],
            "substantive_full_drafting_authorized": motivation[
                "paper_spine_gate"
            ]["substantive_full_drafting_authorized"],
        },
        "paper_readiness_update": {
            "object_checkpoint_found": readiness_v6[
                "object_checkpoint_recovery"
            ]["found"],
            "object_checkpoint_selected_epoch": readiness_v6[
                "object_checkpoint_recovery"
            ]["selected_epoch"],
            "readiness": readiness_v6["readiness"],
            "uncertainty_status": readiness_v6["uncertainty_status"],
        },
        "next_gate": {
            "required_environment_variable": "DASHSCOPE_API_KEY",
            "planned_training_seeds": [42, 43, 44],
            "training_workers": 4,
            "training_batch_size": 32,
            "workers4_retained_after_workers8_benchmark": True,
            "legacy_formal_protocol_modified": False,
            "v6_formal_extension_protocol_frozen": True,
        },
        "claim_boundaries": outcome["claim_boundaries"],
        "bindings": {name: _binding(path) for name, path in paths.items()},
    }
    _atomic_text(
        output,
        json.dumps(index, ensure_ascii=False, indent=2) + "\n",
    )
    coverage = matrix["summary"]["status_counts"]
    markdown = [
        "# v6 专利融合栈扩展索引",
        "",
        "- 状态：结构与机制开发证据完整；正式论文指标：否。",
        "- v6：三个512维Adapter、FiLM、余弦权重、512→128八头交互、128→512残差、"
        "P3/P4/P5动态融合和原生Segment头前调制。",
        "- 结构干预：观测网格能改变三个尺度头前特征；availability=0时三个尺度严格不变。",
        "- 任务结果：干净、回退和乱序的box/mask mAP50-95差异低于0.01个百分点分辨率。",
        "- 同检查点最终调制器干预：完整相对归零的box/mask mAP50-95为"
        f"{100.0 * intervention['contribution_decomposition']['full_stack_conditioning_effect']['box_map50_95']:+.3f}/"
        f"{100.0 * intervention['contribution_decomposition']['full_stack_conditioning_effect']['mask_map50_95']:+.3f}"
        "个百分点；证明整体调制作用，但不能归因于sidecar。",
        "- 专利矩阵：已实现 "
        f"{coverage['implemented']}，部分实现 "
        f"{coverage['partially_implemented']}，未实现 "
        f"{coverage['not_implemented']}，有意变更 "
        f"{coverage['intentional_scope_change']}。",
        "- 可靠性开发模型仍选择v5；v6只作为更接近专利S3/S4的结构模型。",
        (
            "- v6 Qwen适配扩展已独立预注册并通过谱系审计；"
            "仍等待环境变量，训练batch/workers保持为32/4。"
        ),
        (
            "- 三种子共享固定v5父检查点，只度量Qwen适配阶段随机性，"
            "不表述为独立端到端复现实验。"
        ),
        (
            "- workers=8与workers=4逐张量等价，但训练慢"
            f"{worker_scaling['timing']['workers8_slowdown_fraction'] * 100:.2f}%；"
            "因此保留workers=4。"
        ),
        (
            "- 标准化事后分层覆盖570张图、825个目标；v5/v6在无框重叠"
            "子集的掩膜mR50:95相对RGB提高"
            f"{100.0 * standardized_strata['selected_paired_results']['v5_isolated_mask_mean_recall']['difference']:.3f}/"
            f"{100.0 * standardized_strata['selected_paired_results']['v6_isolated_mask_mean_recall']['difference']:.3f}"
            "个百分点。"
        ),
        (
            "- 小目标仅11例、高重叠代理仅20例；结果为事后内部诊断，"
            "且已披露分块推理的高IoU掩膜匹配数值敏感性。"
        ),
        "- 论文动机建议选择A，当前仍等待用户明确确认。",
        (
            "- 对象分类best.pt已恢复并作外部哈希绑定，选择于第"
            f"{readiness_v6['object_checkpoint_recovery']['selected_epoch']}轮；"
            "内部对象证据完整，但权重不进入本扩展归档。"
        ),
        "",
        "本索引不把rule/Qwen图像条件sidecar表述为实体传感器，也不把原专利性能数字"
        "或五轮单种子结果写入正式性能表。",
        "",
    ]
    _atomic_text(output.with_suffix(".md"), "\n".join(markdown))
    return index


def main() -> None:
    result = build_index(parse_args().output)
    print(
        json.dumps(
            {
                "state": result["state"],
                "valid": result["valid"],
                "formal_paper_metric": result["formal_paper_metric"],
                "task_effect": result["v6_development_result"][
                    "effect_interpretation"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
