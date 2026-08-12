"""Render the v6 patent-aligned reliable native Segment-head architecture."""

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
DEFAULT_OUTPUT = (
    REPORTS
    / "joint-multimodal-seg-patent"
    / "patent_stack_architecture.png"
)
DEFAULT_CAPTION = REPORTS / "PATENT_STACK_FIGURE_CAPTION.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the v6 patent-stack architecture"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--caption", type=Path, default=DEFAULT_CAPTION)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


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


def architecture_spec() -> dict[str, Any]:
    return {
        "inputs": [
            "RGB image",
            "19 frozen CLIP class prototypes",
            "4x4 image-conditioned sidecar",
            "sensor availability",
        ],
        "visual_path": [
            "YOLO11s-seg backbone and neck",
            "P3/P4/P5",
            "global pooling and independent 512 projections",
        ],
        "sensor_path": [
            "spatial RGB-to-4x4 prediction and uncertainty",
            "cross-modal reliability gate",
            "effective 4x4 grid",
            "16x16 upsampling",
            "1x1 convolution to 512 and global pooling",
        ],
        "fusion_path": [
            "three independent 512 adapters",
            "FiLM gamma=1 beta=0.1",
            "pairwise cosine modality softmax",
            "512 to 128 projection",
            "8-head 3-token attention",
            "128 to 512 projection and residual",
            "dynamic P3/P4/P5 weights with 0.3/0.5/0.2 prior",
        ],
        "head": "native YOLO11s-seg Segment head",
        "outputs": ["box", "class", "confidence", "instance mask"],
        "boundaries": {
            "rgb_d_input": False,
            "hardware_sensor": False,
            "ground_truth_class_in_text_query": False,
            "formal_qwen_three_seed_result": False,
        },
    }


def draw_architecture(
    output: Path = DEFAULT_OUTPUT,
    caption_path: Path = DEFAULT_CAPTION,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    output = _resolve(output)
    caption_path = _resolve(caption_path)
    sources = {
        "v6_protocol": RUN_ROOT / "PROTOCOL.json",
        "v6_outcome": RUN_ROOT / "OUTCOME.json",
        "intervention_audit": (
            REPORTS / "PATENT_STACK_INTERVENTION_AUDIT.json"
        ),
        "plot_source": Path(__file__),
    }
    missing = [name for name, path in sources.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "architecture figure inputs missing: " + ", ".join(missing)
        )

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    colors = {
        "rgb": "#D9EAF7",
        "yolo": "#DDE9D7",
        "text": "#F5E5A6",
        "sensor": "#F3D1C4",
        "reliability": "#D7ECE6",
        "fusion": "#E2DBEF",
        "scale": "#CFE0F0",
        "head": "#D2E7E2",
        "aux": "#ECEFF1",
        "border": "#263238",
        "line": "#455A64",
        "training": "#A43D4E",
        "muted": "#607D8B",
    }
    figure, axis = plt.subplots(figsize=(20.4, 10.2))
    axis.set_xlim(0.0, 24.0)
    axis.set_ylim(0.0, 10.6)
    axis.axis("off")

    def box(
        x: float,
        y: float,
        width: float,
        height: float,
        text: str,
        color: str,
        *,
        fontsize: float = 8.2,
        linewidth: float = 1.15,
    ) -> None:
        axis.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.018,rounding_size=0.045",
                facecolor=color,
                edgecolor=colors["border"],
                linewidth=linewidth,
                zorder=3,
            )
        )
        axis.text(
            x + width / 2,
            y + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            linespacing=1.22,
            zorder=4,
        )

    def arrow(
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        color: str | None = None,
        dashed: bool = False,
        width: float = 1.15,
    ) -> None:
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=10.5,
                linewidth=width,
                linestyle="--" if dashed else "-",
                color=color or colors["line"],
                connectionstyle="arc3,rad=0.0",
                zorder=2,
            )
        )

    def route(
        points: list[tuple[float, float]],
        *,
        color: str | None = None,
        dashed: bool = False,
        width: float = 1.15,
    ) -> None:
        route_color = color or colors["line"]
        style = "--" if dashed else "-"
        for start, end in zip(points[:-2], points[1:-1]):
            axis.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color=route_color,
                linestyle=style,
                linewidth=width,
                zorder=2,
            )
        arrow(
            points[-2],
            points[-1],
            color=route_color,
            dashed=dashed,
            width=width,
        )

    axis.text(
        12.0,
        10.22,
        "专利对齐的可靠性多模态 YOLO11s-seg 联合头（v6）",
        ha="center",
        va="center",
        fontsize=16.2,
        weight="bold",
    )
    axis.text(
        12.0,
        9.80,
        "RGB 主干 + 冻结 CLIP 类别原型 + 图像条件 4×4 sidecar；融合特征直接调制原生 Segment 头",
        ha="center",
        va="center",
        fontsize=9.2,
        color=colors["muted"],
    )

    box(0.35, 6.55, 1.65, 1.05, "RGB 图像\n640×640", colors["rgb"], fontsize=9.0)
    box(
        2.45,
        6.25,
        2.35,
        1.65,
        "YOLO11s-seg\n主干 + Neck\nRGB 仅 3 通道",
        colors["yolo"],
        fontsize=8.8,
    )
    box(
        5.25,
        6.20,
        1.75,
        1.75,
        "P3 / P4 / P5\n空间特征\n(3 scales)",
        colors["yolo"],
        fontsize=8.7,
    )
    box(
        7.50,
        6.65,
        2.35,
        1.05,
        "GAP + 3 个独立投影\n→ 512 维视觉 token",
        colors["rgb"],
        fontsize=8.2,
    )

    box(
        0.35,
        8.40,
        2.20,
        0.95,
        "19 类文本描述\n冻结 CLIP ViT-B/16",
        colors["text"],
        fontsize=8.5,
    )
    box(
        3.15,
        8.40,
        2.05,
        0.95,
        "类别原型库\n19×512",
        colors["text"],
        fontsize=8.5,
    )
    box(
        7.50,
        8.25,
        2.35,
        1.20,
        "视觉 + 传感器查询\n全部候选原型\n无真实类别输入",
        colors["text"],
        fontsize=8.1,
    )

    box(
        0.35,
        2.55,
        2.25,
        1.15,
        "整图 4×4 sidecar\nrule / Qwen\n图像条件观测",
        colors["sensor"],
        fontsize=8.3,
    )
    box(
        0.35,
        1.05,
        2.25,
        0.95,
        "availability\nmissing / noise /\nbias-drift / stuck",
        colors["aux"],
        fontsize=7.9,
    )
    box(
        4.00,
        3.25,
        2.60,
        1.45,
        "P3/P4/P5 空间汇聚\nRGB→4×4 预测\n均值 + 不确定性",
        colors["sensor"],
        fontsize=8.2,
    )
    box(
        7.50,
        2.85,
        2.65,
        1.85,
        "跨模态可靠性门控\n观测-预测差异 + 一致性\n原始网格质量 + availability\n→ effective 4×4 grid",
        colors["reliability"],
        fontsize=7.9,
    )
    box(
        10.70,
        3.05,
        2.50,
        1.45,
        "4×4 → 16×16\n1×1 Conv: 1→512\nLeakyReLU + GAP",
        colors["sensor"],
        fontsize=8.3,
    )

    box(
        10.65,
        6.30,
        2.55,
        2.25,
        "三模态 512 token\n\nV：三尺度视觉\nT：CLIP 文本上下文\nS：有效传感器",
        colors["scale"],
        fontsize=8.5,
    )
    box(
        13.70,
        6.00,
        3.00,
        2.85,
        "完整专利融合栈\n\n3× 独立 Adapter\nLeakyReLU + dropout 0.3\nFiLM: γ=1.0, β=0.1\n两两余弦 + softmax",
        colors["fusion"],
        fontsize=8.4,
        linewidth=1.45,
    )
    box(
        17.20,
        6.00,
        2.85,
        2.85,
        "深层模态交互\n\n512 → 128\n8 头 / 3 token 注意力\n128 → 512\n残差连接",
        colors["fusion"],
        fontsize=8.5,
        linewidth=1.45,
    )
    box(
        20.55,
        6.25,
        2.85,
        2.30,
        "最终 512 维融合特征\n\n动态 P3/P4/P5 权重\n初始先验 0.3 / 0.5 / 0.2",
        colors["scale"],
        fontsize=8.3,
        linewidth=1.45,
    )
    box(
        18.05,
        3.05,
        2.70,
        1.45,
        "三尺度头前 FiLM\n生成 γ_i、β_i\n调制原始 P3/P4/P5",
        colors["scale"],
        fontsize=8.4,
        linewidth=1.45,
    )
    box(
        21.15,
        3.05,
        2.15,
        1.45,
        "原生 Segment 头\n同一头联合预测",
        colors["head"],
        fontsize=8.6,
        linewidth=1.45,
    )
    box(
        21.15,
        0.75,
        2.15,
        1.35,
        "bbox + class\nconfidence + mask",
        colors["head"],
        fontsize=8.7,
    )
    box(
        7.45,
        0.55,
        5.80,
        1.05,
        "训练期辅助监督：4×4 重建 + 不确定性校准 + 可靠性校准\n（不改变测试标注隔离；测试前向不读取 bbox / mask / class）",
        colors["aux"],
        fontsize=8.1,
    )

    arrow((2.00, 7.08), (2.45, 7.08))
    arrow((4.80, 7.08), (5.25, 7.08))
    arrow((7.00, 7.08), (7.50, 7.08))
    arrow((2.55, 8.88), (3.15, 8.88))
    route([(5.20, 8.88), (6.30, 8.88), (6.30, 8.85), (7.50, 8.85)])

    route([(6.13, 6.20), (6.13, 5.35), (5.30, 5.35), (5.30, 4.70)])
    route([(2.60, 3.12), (3.15, 3.12), (3.15, 3.80), (7.50, 3.80)])
    route([(2.60, 1.52), (3.35, 1.52), (3.35, 3.20), (7.50, 3.20)])
    arrow((6.60, 3.98), (7.50, 3.98))
    arrow((10.15, 3.78), (10.70, 3.78))

    route([(9.85, 7.18), (10.20, 7.18), (10.20, 7.75), (10.65, 7.75)])
    route([(9.85, 8.85), (10.20, 8.85), (10.20, 7.42), (10.65, 7.42)])
    route([(13.20, 3.78), (13.45, 3.78), (13.45, 6.72), (13.20, 6.72)])
    route([(13.20, 3.78), (10.35, 3.78), (10.35, 8.25), (9.85, 8.25)])

    arrow((13.20, 7.42), (13.70, 7.42))
    arrow((16.70, 7.42), (17.20, 7.42))
    arrow((20.05, 7.42), (20.55, 7.42))
    route([(21.98, 6.25), (21.98, 5.15), (19.40, 5.15), (19.40, 4.50)])
    route(
        [
            (7.00, 6.48),
            (7.20, 6.48),
            (7.20, 5.35),
            (16.95, 5.35),
            (16.95, 3.78),
            (18.05, 3.78),
        ]
    )
    arrow((20.75, 3.78), (21.15, 3.78))
    arrow((22.22, 3.05), (22.22, 2.10))

    route(
        [(5.30, 3.25), (5.30, 2.15), (8.50, 2.15), (8.50, 1.60)],
        color=colors["training"],
        dashed=True,
    )
    route(
        [(8.82, 2.85), (8.82, 2.10), (10.70, 2.10), (10.70, 1.60)],
        color=colors["training"],
        dashed=True,
    )

    axis.plot(
        [0.55, 1.15],
        [5.35, 5.35],
        color=colors["line"],
        linewidth=1.5,
    )
    axis.text(
        1.28,
        5.35,
        "推理 / 指标前向",
        va="center",
        fontsize=8.2,
    )
    axis.plot(
        [0.55, 1.15],
        [4.93, 4.93],
        color=colors["training"],
        linestyle="--",
        linewidth=1.5,
    )
    axis.text(
        1.28,
        4.93,
        "训练期辅助监督",
        va="center",
        fontsize=8.2,
    )
    axis.text(
        0.55,
        0.35,
        "边界：当前无 RGB-D、无同步实体传感器；sidecar 为图像条件软观测；正式 Qwen 三种子结果尚未生成。",
        fontsize=7.8,
        color=colors["muted"],
        va="center",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        dpi=240,
        bbox_inches="tight",
        facecolor="white",
    )
    pdf_path = output.with_suffix(".pdf")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    caption = (
        "图X 专利对齐的可靠性多模态YOLO11s-seg联合头。RGB主干输出P3、P4和P5；"
        "冻结CLIP类别原型、视觉特征与经跨模态可靠性门控的4×4图像条件sidecar"
        "分别形成512维视觉、文本和传感器token。三模态依次经过独立Adapter、"
        "FiLM、两两余弦加权、512→128的8头交互、128→512残差及动态尺度融合，"
        "最终生成P3/P4/P5头前FiLM参数并进入同一原生Segment头。实线为指标前向，"
        "红色虚线为训练期辅助监督。当前输入为RGB，sidecar不是同步实体传感器，"
        "正式Qwen三种子实验尚未完成。"
    )
    caption_lines = [
        "# v6 专利融合栈图注",
        "",
        f"**建议图注：** {caption}",
        "",
        "**文件：** "
        "`joint-multimodal-seg-patent/patent_stack_architecture.pdf`、"
        "`joint-multimodal-seg-patent/patent_stack_architecture.png`",
        "",
        "**解释边界：** 图示证明的是已实现结构与数据流，不等价于实体传感器增益；"
        "v6五轮单种子rule结果不进入正式Qwen主表。三个模态token的8头自注意力"
        "是对原专利双向交互的工程实现，不应描述成六个显式有向输出完全复刻。",
        "",
    ]
    _atomic_text(caption_path, "\n".join(caption_lines))
    report = {
        "schema_version": 1,
        "state": "complete",
        "valid": output.is_file() and pdf_path.is_file(),
        "formal_paper_metric": False,
        "specification": architecture_spec(),
        "caption": caption,
        "artifacts": {
            "png": _binding(output),
            "pdf": _binding(pdf_path),
            "caption": _binding(caption_path),
        },
        "source_bindings": {
            name: _binding(path) for name, path in sources.items()
        },
        "errors": [],
    }
    _atomic_text(
        output.with_suffix(".json"),
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    return report


def main() -> None:
    args = parse_args()
    result = draw_architecture(args.output, args.caption)
    print(
        json.dumps(
            {
                "state": result["state"],
                "valid": result["valid"],
                "formal_paper_metric": result["formal_paper_metric"],
                "artifacts": result["artifacts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
