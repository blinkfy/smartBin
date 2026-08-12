# Evidence Bank

本表只登记可以进入论文的本地证据。E0 为机器可读结果和冻结协议，E1 为代码与配置，
E2 为已审计解释。正文数字优先由 E0 产生；E2 只能解释，不能覆盖 E0。

| ID | Evidence / Claim Unit | Primary Source | Level | Status | Allowed Manuscript Use | Boundary |
|---|---|---|---|---|---|---|
| E01 | 冻结对象数据含 8,654 条记录、5,990 张源图；train/val/test 为 7,002/826/826 | `reports/qwen-material/DATASET_CARD.md`; `data/qwen-material/*.jsonl` | E0/E2 | verified | 数据规模、划分和任务定义 | 是内部对象级数据，不是外部队列 |
| E02 | 同一源图不跨集合，test 含 570 张源图；存在 24 对高置信跨集合近重复候选，敏感性结论不变 | `reports/qwen-material/DATA_LEAKAGE_AUDIT.json`; `DATASET_CARD.md` | E0/E2 | verified | 说明源图分组切分、近重复审计和限制 | 无拍摄批次或物体实例 ID，不能声称完全独立 |
| E03 | 8 条重复对象记录；主协议保留 826 条 test，去重敏感性为 825 条 | `reports/qwen-material/DATASET_CARD.md`; `reports/qwen-material/DATA_LEAKAGE_AUDIT.json` | E0/E2 | verified | 数据质量披露与敏感性分析 | 不得事后改写主测试集规模 |
| E04 | 8,639 条 Qwen 辅助记录、15 条 rule-v1 回退；有效率 100%，Qwen 覆盖率 99.83% | `data/qwen-material/build_report.json`; `SENSOR_SOURCE_AUDIT.json`; `DATASET_CARD.md` | E0/E2 | verified | 软传感器来源说明 | 不是物理重量、温湿度、电导率或气味实测 |
| E05 | 当前模型只读取 4×4 `normalized_grid`，sidecar 五个标量不参与分类 | `mmfc/dataset.py`; `DATASET_CARD.md`; `METHOD_CONTRACT_AUDIT.json` | E1/E2 | verified | 输入张量和方法边界 | 禁止把五个标量写成已融合特征 |
| E06 | v3 由 YOLO 多尺度视觉、CLIP 文本原型、4×4 网格编码、潜表示预测、回退、可靠性门控、Adapter/FiLM/注意力组成 | `mmfc/model.py`; `configs/ablations/full_cross_modal_quality.json`; `METHOD_CONTRACT_AUDIT.json` | E1/E0 | verified | 方法结构和公式 | FiLM 是全局逐模态仿射，不是文本动态生成器 |
| E07 | 可靠性证据联合 availability、潜表示差异、不确定性、原始网格质量和跨模态网格差异 | `mmfc/model.py`; `mmfc/losses.py`; `METHOD_CONTRACT_AUDIT.json` | E1/E0 | verified | 方案 A 的核心机制 | 只表示已定义软传感器条件下的相对可用性 |
| E08 | missing 时可靠性置零，有效传感器表示完全回退至视觉文本预测潜表示 | `mmfc/model.py`; `METHOD_CONTRACT_AUDIT.json`; `MANUSCRIPT_STABLE_SECTIONS.md` | E1/E2 | verified | 回退公式和机制解释 | 不构成对任意缺失机制的理论保证 |
| E09 | 训练退化含 missing、noise、bias/drift、stuck，可组合形成 mixed；参数由正式配置冻结 | `configs/ablations/full_cross_modal_quality.json`; `ABLATION_PROTOCOL.md`; `MANUSCRIPT_STABLE_SECTIONS.md` | E0/E1 | verified | 退化定义和训练设置 | 均为软件退化，不是硬件故障试验 |
| E10 | v3 seed42 固定内部测试 Accuracy 88.14%、Macro F1 82.44%、ECE 2.46% | `full_cross_modal_quality/test/evaluation.json`; `paper-tables/main_classification.csv` | E0 | verified | 单种子主检查点结果 | 不按 test 选 seed，不称外部盲测 |
| E11 | v3 seeds 42/43/44：Accuracy 88.26% ± 0.21 pp，Macro F1 82.26% ± 0.31 pp | `full_cross_modal_quality/seed-replication/OUTCOME.json`; `MANUSCRIPT_RESULTS_SECTIONS.md` | E0/E2 | verified | 训练种子稳定性 | 三个种子仅支持描述性均值与样本标准差 |
| E12 | 完整 v3 相对匹配无可靠性重训：Accuracy +0.81 pp、Macro F1 +0.51 pp，方向均为 3/3 | `confirmatory-ablations/v3-no-reliability-workers0/OUTCOME.json`; `PAPER_ADDENDUM.json` | E0 | verified | 核心独立重训消融 | 效应温和，不能写成大幅或普遍显著 |
| E13 | seed42/43/44 的 Accuracy 差分别为 +0.36、+0.73、+1.33 pp；仅 seed44 区间下界高于零 | `MATCHED_TEST_SEED_SUMMARY.json`; `MANUSCRIPT_RESULTS_SECTIONS.md` | E0/E2 | verified | 效应量、源图簇 bootstrap 区间 | 单种子区间多跨零，必须如实呈现 |
| E14 | 同检查点开关 gate：Accuracy +2.78 pp [1.23,4.33]，Macro F1 +3.50 pp [1.46,5.89] | `reports/qwen-material/full_cross_modal_quality/GATE_PAIRED_COMPARISON.json`; `reports/qwen-material/paper-tables/TABLES.md` | E0 | verified | 机制干预支持证据 | 属于事后同检查点干预，不等价于独立重训 |
| E15 | seed42 支持性退化比较中，完整 v3 在 missing/noise/bias/stuck/mixed 的 Accuracy 与 Macro F1 效应均为正 | `full_cross_modal_quality/ROBUSTNESS_COMPARISON.json`; `MANUSCRIPT_RESULTS_SECTIONS.md` | E0/E2 | verified | 软件退化鲁棒性表和图 | 退化随机种子不是训练随机种子 |
| E16 | v3 潜空间样本均值方差—误差 Spearman 0.892，预测方差/MSE 44.3；相对排序较好但尺度保守 | `full_cross_modal_quality/uncertainty/uncertainty_diagnostics.json`; `UNCERTAINTY_DIAGNOSTICS_COMPARISON.json` | E0 | verified | 不确定性诊断与限制 | 是 512 维潜空间，不是物理量标定 |
| E17 | v3 相对 v2b 的样本均值 Spearman 增量区间 [0.0477,0.0954]，过分散比下降区间 [11.08,14.02] | `UNCERTAINTY_DIAGNOSTICS_COMPARISON.json` | E0 | verified | 事后补充诊断 | 非预注册主要终点，不支持外部校准主张 |
| E18 | 旧谱系八项基础消融为单训练种子，Holm 校正最小 p=0.4767 | `ablations/ablation_pairwise_statistics.json`; `paper-tables/formal_ablation_statistics.csv` | E0 | verified | 补充材料中的历史结构对照 | 不用来声称组件显著优越 |
| E19 | 类别平衡候选未提高验证或测试 Macro F1 | `CLASS_BALANCE.md`; `class_performance_v2b_v3_balanced.json` | E0/E2 | verified negative | 报告负结果、说明未继续多种子 | 禁止只挑少数类局部提升 |
| E20 | Qwen 与 rule-v1 单种子独立来源重训为 88.14/82.44 与 87.29/81.87（Accuracy/Macro F1） | `reports/qwen-material/sensor-source-retraining/RULE_V1_VS_QWEN_RETRAINED_COMPARISON.json`; `reports/qwen-material/paper-tables/sensor_source_retraining.csv` | E0 | verified | 来源敏感性补充 | 两者都由图像条件生成，且仅一个训练种子 |
| E21 | 19 类标签高度不均衡，训练最大/最小类约 29.75 倍，test 最小 support 为 5 | `DATASET_CARD.md`; `CLASS_METRIC_INTERVALS.json` | E0/E2 | verified | 选择 Macro F1 和逐类指标的理由 | 少数类结论需谨慎 |
| E22 | 570 张测试源图、825 个唯一对象上的 YOLO11s-seg：box mAP50-95 96.55%，mask mAP50-95 95.13% | `yolo-seg/evaluation.json`; `paper-tables/yolo_segmentation.csv` | E0 | verified extension | 目标域定位/分割扩展 | 不是对象分类增益，也不是融合头结果 |
| E23 | 联合头三种子 box/mask mAP50-95 为 97.09% ±0.02 / 95.34% ±0.02 pp；变化小 | `joint-multimodal-seg/MULTISEED_SUMMARY.json`; `retrained-ablations/OUTCOME.json` | E0 | verified extension | 扩展章节说明结构可接入 Segment head | 正式前向只读 RGB，不是硬件多模态 |
| E24 | 两阶段整图系统在 conf=0.25 时定位 P/R 97.98/99.88%，端到端 Macro F1 96.80% | `yolo-seg/integrated/evaluation.json`; `paper-tables/scene_pipeline.csv` | E0 | verified extension | 系统范围演示 | detector 类别路径与 MMFC 复分类路径必须分开 |
| E25 | 外部盲测协议已冻结且 tooling ready，但外部数据和结果不存在 | `external-blind-v1/PROTOCOL.json`; `EXTERNAL_BLIND_READINESS_V1.json` | E0 | verified missing | 讨论未来验证与当前边界 | 严禁写成外部验证完成 |
| E26 | 原专利 10 项权利要求已有逐项矩阵；当前实现仍缺 RGB-D、同步硬件阵列等 | `PATENT_REQUIREMENT_MATRIX.json`; 原专利 DOCX | E0/E3 | verified extension | 专利对齐附录/讨论 | 不能声称完整复现专利 |
| E27 | 正式环境为 RTX 4090、Python 3.9.23、PyTorch 2.6.0+cu126 等 | `requirements-repro.txt`; `REPRODUCIBILITY.md`; `MANUSCRIPT_STABLE_SECTIONS.md` | E0/E2 | verified | 实现细节 | 运行环境事实不等于跨设备性能 |
| E28 | 一张 train 图像的只读镜像恢复、一次检查点损坏后从头重训等事件均有冻结审计 | `reports/qwen-material/IMAGE_READ_RECOVERY_INCIDENT.json`; `reports/qwen-material/IMAGE_READ_RECOVERY_INCIDENT_ADDENDUM.json`; `reports/qwen-material/ablations/full_no_reconstruction/CHECKPOINT_AUDIT.json`; `reports/qwen-material/MANUSCRIPT_STABLE_SECTIONS.md` | E0/E2 | verified incident | 可复现性与偏差披露 | 不用于解释模型间性能差异 |
| E29 | 实体4×4记录的软件校准、同步和保管人证明发布链已就绪，并与冻结盲测的`synchronized-hardware`模式兼容 | `PHYSICAL_SENSOR_INTERFACE_AUDIT.json`; `SYNCHRONIZED_HARDWARE_RELEASE.md`; `EXTERNAL_BLIND_READINESS_V1.json` | E0/E1/E2 | verified software interface | 实施路径、未来实验和当前边界 | 实体记录仍为0，文件哈希与声明核验不证明硬件来源或性能 |

## Evidence Gaps

| Gap ID | Missing Evidence | Consequence |
|---|---|---|
| G01 | 独立外部采集垃圾数据的一次性盲测结果 | 不能声称外部泛化或真实部署有效 |
| G02 | 与 RGB 同步的真实 4×4 物理传感器阵列 | 不能把 sidecar 解释为硬件测量或独立模态互补性 |
| G03 | 采集批次、物体实例和时间序列标识 | 不能证明跨集合完全消除了同批次弱相关 |
| G04 | 直接全图 Qwen 软传感器的三训练种子复现 | 不能把当前来源比较外推为稳定的 Qwen 优势 |
| G05 | 真实 RGB-D 输入和硬件端延迟/功耗测试 | 不能声称完整专利栈或智能垃圾桶部署性能 |
