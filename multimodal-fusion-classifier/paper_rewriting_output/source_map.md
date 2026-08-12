# 论文来源映射

## 证据优先级

| 层级 | 来源 | 论文用途 | 使用规则 |
|---|---|---|---|
| E0 | 冻结协议、逐样本预测、JSON/CSV、检查点审计与 SHA-256 绑定 | 实验数字、统计区间、模型身份 | 最高优先级；正文数字必须能够回溯到该层 |
| E1 | `mmfc/`、训练/评估脚本、`configs/` | 方法、超参数、数据流和实现边界 | 与 E0 不一致时停止写作并审计，不以描述性报告覆盖代码事实 |
| E2 | `reports/qwen-material/*.md` 与生成脚本 | 已审计解释、表格、图注和治理规则 | 只复用已通过审计的结论，不从历史 pending 文件取最终状态 |
| E3 | 原专利 DOCX 与 `PATENT_REQUIREMENT_MATRIX.json` | 结构来源、专利逐项对齐及差异 | 原专利性能数字没有可复核清单和权重，不进入本文结果 |
| L0 | 同行评议论文官方页面或公开正文 | 研究背景、相关工作、方法来源 | 不用于证明本项目实验效果 |
| L1 | 官方技术报告或产品文档 | Qwen3-VL、YOLO11 实现背景 | 明确其非同行评议或具体托管修订版未冻结 |

## 主文证据链

| 主张单元 | 权威材料 | 允许写法 | 禁止外推 |
|---|---|---|---|
| 数据组成与切分 | `DATASET_CARD.md`、`DATA_LEAKAGE_AUDIT.json`、`SOURCE_IMAGE_INTEGRITY_CURRENT.json` | 5,990 张真实采集源图、8,654 个对象记录，按源图分组切分 | 独立采集外部队列 |
| Qwen/rule-v1 来源 | `data/qwen-material/sensors.jsonl`、构建报告、`SENSOR_SOURCE_AUDIT.json` | 逐条绑定真实对象图像的图像条件软传感器或 sidecar | 经硬件标定的物理重量、温湿度或电导率实测 |
| v3 方法 | `mmfc/model.py`、`mmfc/losses.py`、`configs/ablations/full_cross_modal_quality.json`、`METHOD_CONTRACT_AUDIT.json` | 潜表示预测、缺失回退、网格质量和跨模态一致性门控 | 对任意传感器或任意缺失机制的理论保证 |
| 对象级主结果 | `full_cross_modal_quality/seed-replication/OUTCOME.json`、`confirmatory-ablations/v3-no-reliability-workers0/OUTCOME.json` | 三训练种子描述、匹配种子可靠性消融和源图簇区间 | 由单个最佳种子代表总体 |
| 软件退化 | `full_cross_modal_quality/ROBUSTNESS_COMPARISON.json` | missing/noise/bias/stuck/mixed 的固定软件干预 | 硬件故障试验 |
| 分类校准 | `paper-tables/main_classification.csv`、`LATENT_UNCERTAINTY_TEMPERATURE_ADDENDUM.json` | ECE/NLL/Brier 与潜空间尺度诊断分开报告 | 潜空间方差等于物理传感器不确定性 |
| 目标域分割 | `yolo-seg/evaluation.json` | 独立 RGB YOLO11s-seg 的 box/mask 指标 | 多模态融合带来的分割提升 |
| 联合头扩展 | `joint-multimodal-seg/MULTISEED_SUMMARY.json`、`retrained-ablations/OUTCOME.json` | 结构接入和三种子小幅变化 | sidecar 产生显著任务增益 |
| Qwen 蒸馏 | `OBJECT_QWEN_DISTILLED_JOINT_ADDENDUM.json` | RGB 学生网格优于均值基线但任务指标无增益 | 正式全图 Qwen 三种子结论 |
| 外部验证 | `external-blind-v1/PROTOCOL.json`、`EXTERNAL_BLIND_READINESS_V1.json`、`SYNCHRONIZED_HARDWARE_RELEASE.md` | 协议、验收和保管人证明发布链已就绪，等待独立数据 | 已完成外部泛化或硬件性能验证 |

## 图表来源

- 主分类、消融、鲁棒性、来源分析和系统结果优先取自
  `reports/qwen-material/paper-tables/`。
- 论文架构图使用无装饰性图标的AI辅助重绘
  `method_architecture_ai_no_icons_word.jpg` 作为排版适配版，PNG母版保留为
  `method_architecture_ai_no_icons.png`，结构由
  `method_architecture.pdf` 和实现代码交叉校对；联合头与专利栈仅在扩展章节使用对应
  PDF/PNG，并采用 `FIGURE_CAPTIONS.md` 中的边界化图注。
- 不使用原专利训练曲线作为本文实验图，也不把原专利图中的指标数字转录为结果。

## 文献来源

`reference_materials/source_index.md` 先列出 500 项本地材料，再列出 20 项经官方
页面核验的论文或文档。`reports/qwen-material/reference_candidates.bib` 是当前
20 条候选书目底稿；进入最终参考文献前还需通过引用支持库和 BibTeX 审计。
