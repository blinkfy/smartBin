# Figure Asset Map

| Figure ID | Role | Asset | Manuscript Placement | Caption Boundary | Status |
|---|---|---|---|---|---|
| F01 | 主方法框架 | `final_paper/figures/method_architecture_ai_no_icons_word.jpg`（论文嵌入适配版）；`reports/qwen-material/method_architecture_ai_no_icons.png`（无图标AI辅助重绘母版）；`reports/qwen-material/method_architecture.pdf`（可复现结构底稿） | 方法章节开头 | 仅保留张量、4×4 网格、标量、公式和数据流，不使用装饰性图标；明确RGB与对象记录来自真实采集图像，网格为逐条绑定真实样本的 Qwen/rule 图像条件 sidecar，分类头不等于 YOLO Segment head | ready |
| F02 | 退化鲁棒性与 gate 干预 | `reports/qwen-material/full_cross_modal_quality/robustness_comparison.png` | 核心结果 | 软件退化；五个扰动种子不是独立训练种子；同检查点 no-gate 为机制干预 | ready |
| F03 | 完整模型退化曲线 | `reports/qwen-material/full_cross_modal_quality/robustness/robustness_multiseed_curves.png` | 核心结果或补充材料 | 只描述 seed42 检查点在五个退化随机种子上的趋势 | ready |
| F04 | 匹配无可靠性变体退化曲线 | `reports/qwen-material/confirmatory-ablations/v3-no-reliability-workers0/seed42/robustness/robustness_multiseed_curves.png` | 与 F03 并列或补充材料 | 不把扰动种子写成训练复现 | ready |
| F05 | 潜表示不确定性诊断 | `reports/qwen-material/full_cross_modal_quality/uncertainty/uncertainty_diagnostics.png` | 结果/讨论 | 512 维潜空间诊断，不是物理传感器单位校准 | ready |
| F06 | 逐类 F1 与训练支持量 | `reports/qwen-material/class_performance_v2b_v3.pdf` | 长尾结果或补充材料 | 固定内部 test；单种子逐类差异不含训练种子方差 | ready |
| F07 | 类别平衡负结果 | `reports/qwen-material/class_performance_v2b_v3_balanced.pdf` | 补充材料 | 不能从局部少数类改善推导整体策略有效 | ready |
| F08 | 历史正式消融差值 | `reports/qwen-material/paper-tables/formal_ablation_effects.png` | 补充材料 | 单训练种子，Holm 校正后无显著组件优势 | ready |
| F09 | 软传感器来源比较 | `reports/qwen-material/paper-tables/sensor_source_comparison.png` | 补充材料 | 同检查点迁移与独立重训必须分开；Qwen/rule 均为图像条件 | ready |
| F10 | 独立 YOLO-seg 结果 | `reports/qwen-material/paper-tables/yolo_segmentation_metrics.png` | 系统扩展附录 | 仅 RGB 目标域分割，不是 MMFC 分类结果 | ready |
| F11 | 联合分割头结构 | `reports/qwen-material/joint-multimodal-seg/joint_head_architecture.pdf` | 系统扩展附录 | 正式 test 只读取 RGB，Qwen/rule 仅提供训练软目标 | ready |
| F12 | 两阶段系统比较 | `reports/qwen-material/paper-tables/scene_pipeline_comparison.png` | 系统扩展附录 | detector、missing、rule-v1 类别来源分别解释 | ready |
| F13 | 分层场景诊断 | `reports/qwen-material/paper-tables/scene_strata_diagnostics.png` | 补充材料 | relative area、密度与 bbox overlap 是代理变量 | ready |

## Main-Text Selection

主文默认只使用 F01、F02、F05，并视版面加入 F06。F03/F04 用于补充支持，
F07-F13 进入附录或补充材料。这个选择让图件顺序直接服务方案 A：
方法如何形成可靠性，然后看退化时是否抑制，再看不确定性诊断是否支持门控解释。

## Required Tables

| Table ID | Content | Machine Source | Placement |
|---|---|---|---|
| T01 | 数据划分、源图数、Qwen/rule 数 | `DATASET_CARD.md` and build JSON | 数据章节 |
| T02 | v3 三训练种子主结果 | `full_cross_modal_quality/seed-replication/OUTCOME.json` | 核心结果 |
| T03 | 完整 v3 与无可靠性变体匹配种子差值及区间 | `MATCHED_TEST_SEED_SUMMARY.json` | 核心结果 |
| T04 | 软件退化场景效应 | `full_cross_modal_quality/ROBUSTNESS_COMPARISON.json` | 核心结果 |
| T05 | 潜空间不确定性诊断 | `UNCERTAINTY_DIAGNOSTICS_COMPARISON.json` | 结果/讨论 |
| T06 | 主要训练与退化超参数 | `configs/ablations/full_cross_modal_quality.json` | 实验设置 |
| T07 | 历史消融、来源与系统扩展 | `paper-tables/*.csv` | 附录 |
