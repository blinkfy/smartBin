# Rewrite Matrix

## Decision Rule

本矩阵以 `original_logic_map.md` 的源单元为起点，以 `confirmed_motivation.md` 的方案A为控制线。所有数值只允许来自 `evidence_bank.md`、冻结清单、检查点审计或已核验文献；专利中的无归档性能点估计不迁移。最终论文按研究问题重新组织，而不是保持专利段落顺序。

| Section | Unit ID | Current Function | Motivation Link | Operation | Intended Move | Evidence Source | Model Pattern | Target Length | Logic Change | Decision |
|---|---|---|---|---|---|---|---|---|---|---|
| 摘要 | O01 | 模块清单与强性能宣称 | 软传感器退化会反向伤害分类 | REWRITE | 问题—方法—三种子证据—适用边界 | `evidence_bank.md` E01-E08；`results_validation.md` | motivation-gap-method-evidence-limit | 500-700字 | 从发明概述改为受证据约束的研究摘要 | 完成 |
| 引言1 | O09 | 医疗废物和纯视觉缺陷 | 一般垃圾分类中的对象级任务边界 | REWRITE | 区分分类、检测、分割及工业多传感器场景 | `citation_support_bank.md` C01-C07 | field-context-scope | 2段 | 取消医疗强限定并避免跨任务指标比较 | 完成 |
| 引言2 | O02,O09 | 假定同步物理传感器 | 图像条件sidecar与物理模态不同源假设不成立 | REWRITE | 正式定义图像条件软传感器及共享来源风险 | 数据清单；Qwen/rule-v1构建报告；C08-C12 | boundary-gap | 1段 | 从硬件互补信息改为同源辅助表示治理 | 完成 |
| 引言3 | O03-O08 | 多模块融合不足 | 观测质量未显式参与融合 | REWRITE | 提出缺失回退、连续质量门控和多层评价三个缺口 | `sota_gap_map.md`；C13-C20 | gap-response | 2段 | 从“融合更深”改为“坏模态如何不伤害” | 完成 |
| 引言4 | O01,O10 | 高精度、高鲁棒性总括 | 内部、有限、可检验的贡献 | DELETE+ADD | 删除98.8%等无证据数值，列出三项受限贡献 | `confirmed_contribution.md`；E01-E08 | contribution-promise | 1段 | 贡献与后续结果小节逐项对应 | 完成 |
| 相关工作1 | O09 | 无引用的背景概述 | 垃圾视觉与工业分选的任务谱系 | ADD | 组织视觉分类、检测分割和物理传感分选文献 | `research_dossier.md`；C01-C10 | taxonomy-contrast | 2段 | 为任务边界提供外部证据 | 完成 |
| 相关工作2 | O04,O08 | 余弦/双向注意力模块陈述 | 缺失、低质量和动态融合最近邻 | ADD | 比较缺失模态恢复、质量感知门控与不确定性建模 | `sota_gap_map.md`；C11-C20 | nearest-work-gap | 2段 | 避免以组件命名代替新颖性论证 | 完成 |
| 方法总览 | O11 | 四模块设备流程 | 可靠性失败模式驱动的方法链 | MOVE+REWRITE | 在方法开头给出端到端流程图和变量关系 | 当前代码；`figure_asset_map.md` F01 | overview-rationale | 1段+图 | 总览图从模块细节之后前移到方法开头 | 完成 |
| 数据与任务 | O02,O05 | 深度图、物理阵列、检测分割 | 19类对象分类与源图分组 | REWRITE | 定义对象样本、图像路径、文本描述、4x4 sidecar和分组划分 | manifests；`validate_dataset.py`报告；E01-E03 | governance-definition | 2-3段 | 用可审计数据单元替换假定硬件流程 | 完成 |
| 软传感器生成 | O02,O06 | 传感器采集、上采样与归一化 | 图像条件属性估计及确定性回退 | REWRITE | 说明固定提示、strict JSON、4x4范围、rule-v1和冻结工件复现 | `build_dataset.py`；`sensors.py`；构建报告 | provenance-boundary | 3段 | 明确不是独立物理测量，生成过程与工件复现分开 | 完成 |
| 表示编码 | O03,O06,O07 | YOLO/CLIP/1x1卷积和适配 | 保留可验证结构连续性 | REWRITE | 按实际权重、维度、标准化、激活和dropout定义编码 | `model.py`；`configs/default.json` | implementation-rationale | 3段 | 删除与代码不符的冻结层和RGB-D预处理参数 | 完成 |
| 可靠性机制 | 无对应源单元 | 原文未处理观测质量 | 缺失时回退、退化时连续抑制 | ADD | 定义视觉—文本预测潜表示、异方差、质量头、gate与fallback | `model.py`；`losses.py`；E04-E08 | failure-mechanism | 4段+公式 | 新建全文核心机制，替代简单三模态加权假设 | 完成 |
| 训练目标 | O03,O12 | CE+Focal参数列表 | 退化感知监督和可靠性校准 | REWRITE | 展开CE、Focal、NLL、校准、方差先验、网格及排序损失 | `losses.py`；`training.py`；配置审计 | objective-definition | 4段+公式 | 从两项分类损失扩展为可复现的联合目标 | 完成 |
| 实验设置 | O12,O14 | 100轮训练和三组点估计 | 公平比较与统计协议 | REWRITE | 固定划分、种子、最佳检查点、ECE和源图簇bootstrap | 配置；检查点；`results_validation.md` | protocol-before-results | 4段+表 | 删除无来源对比，建立可重复评估协议 | 完成 |
| 模型比较 | O04,O14 | “本发明/传统模型”粗粒度对照 | 匹配重训与同检查点干预回答不同问题 | ADD | 统一v3、无可靠性、no-gate和v2b定义 | 实验汇总；审稿修订记录 | comparison-map | 1段+表 | 防止将复合子系统和后验干预混为一谈 | 完成 |
| 结果1 | O10,O14 | 无区间的高性能宣称 | 先报告内部固定测试的真实性能 | DELETE+ADD | 报告seed42、三种子均值、Macro F1、ECE和长尾边界 | E01-E05；`results_validation.md` | answer-evidence-limit | 3段+表图 | 以88.14%/88.26%等冻结结果替代98.8% | 完成 |
| 结果2 | O04,O08 | 结构被默认有效 | 可靠性复合子系统是否带来增益 | ADD | 报告三种子匹配差值、簇bootstrap及谨慎解释 | E06-E08 | paired-effect-qualification | 3段+表 | 将“模块存在”变成匹配证据问题 | 完成 |
| 结果3 | 无对应源单元 | 原文没有缺失/退化检验 | 门控是否按设计响应故障 | ADD | 报告missing/noise/bias/stuck和同检查点no-gate行为 | 退化评估产物；Figure F03 | mechanism-stress-test | 3段+图 | 新增可靠性行为证据，并限定为分布内退化 | 完成 |
| 结果4 | 无对应源单元 | 原文没有不确定性校准 | 方差是否既排序有效又绝对校准 | ADD | 报告Spearman 0.892、variance/MSE 44.3和过分散结论 | 不确定性诊断；Figure F04 | diagnosis-boundary | 2段+图 | 保留负面校准发现，避免把潜方差解释为物理置信度 | 完成 |
| 讨论与局限 | O15 | 专利保护范围式收束 | 负结果、来源敏感性和外部有效性 | REWRITE+SPLIT | 分成负结果、系统扩展、局限性和复现声明 | 审稿意见；检查点审计；源清单 | implication-limit | 5-7段 | 将过程细节与边界从结果尾部拆开，明确待补实验 | 完成 |
| 结论 | O01,O10,O15 | 重复高精度和广泛适用 | 回答控制性问题并封闭主张范围 | REWRITE | 总结内部证据、机制价值及外部盲测/硬件缺口 | 全文已核验证据 | answer-boundary-next-step | 2段 | 由发明效果宣告改为有限结论和下一步验证 | 完成 |

## Coverage Summary

- Source units mapped: O01-O15, 15/15.
- Primary operations: REWRITE 13 rows, ADD 8 rows, DELETE-related 2 rows, MOVE/SPLIT related 2 rows; no paragraph以原样KEEP。
- Unsupported source claims removed: medical-only scope, synchronized hardware measurement, RGB-D acquisition, 98.8% Accuracy, small-object/occlusion rates, 15.2 ms, detection/segmentation superiority.
- Preserved structural lineage: YOLO visual features, CLIP class text, 4x4 sidecar, Adapter, FiLM, attention, CE and Focal.
- New paper spine: degraded side information -> latent prediction and fallback -> reliability evidence and gate -> matched evaluation -> uncertainty/negative findings -> external-validity boundary.
