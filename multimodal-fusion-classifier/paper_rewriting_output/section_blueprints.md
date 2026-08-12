# Section Blueprints

## Whole-Work Throughline

论文围绕一个问题展开：图像条件软传感器并非始终可信，模型应如何在利用其信息的
同时识别缺失或退化、抑制错误观测并回退到可用表示。证据顺序固定为：

1. 说明垃圾分类中的多模态价值与低质量模态风险。
2. 明确本文 sidecar 的图像条件来源及其与硬件观测的区别。
3. 给出潜表示预测、可靠性证据与连续回退机制。
4. 先用同检查点干预验证机制会改变行为，再用匹配三训练种子验证收益方向。
5. 用注册软件退化、不确定性诊断和负结果解释方法适用范围。
6. 把外部盲测、真实硬件和专利完整复现留作未完成边界。

## Title

**面向图像条件软传感器退化的可靠多模态垃圾分类方法**

标题不出现“医疗废物”“真实传感器”“外部验证”或“智能垃圾桶部署”，因为现有
核心证据是 19 类一般垃圾对象、图像条件 sidecar 和内部固定基准。

## Abstract

| Unit | Planned Content | Evidence / Citation | Claim Limit |
|---|---|---|---|
| A1 Background | 多模态垃圾分类可利用附加属性，但图像条件软传感器在缺失、噪声、偏移或卡死时可能注入低质量证据 | C019-C025 | 不写成真实硬件故障 |
| A2 Method | 一句话概括视觉—文本预测潜表示、availability/质量/一致性/不确定性门控和连续回退 | E06-E09 | 不逐个包装通用模块为首创 |
| A3 Data | 8,654 个对象、19 类、源图分组内部划分；4×4 网格来自 Qwen/rule 图像条件估计 | E01-E05 | 明确内部固定基准 |
| A4 Results | 三种子均值；相对匹配无可靠性重训的平均差值和 3/3 方向；退化结果只作概括 | E10-E15 | 报告效应温和，不写 SOTA |
| A5 Conclusion | 方法在已定义软件退化下提供可审计的利用、抑制和回退行为 | E12-E17 | 外部和硬件有效性仍待验证 |

建议关键词：垃圾分类；多模态融合；软传感器；模态退化；可靠性门控；不确定性

## 1 Introduction

### Paragraph 1: Application Problem

从自然/工业垃圾识别由封闭集分类走向检测、分割和多模态材料识别展开，说明仅依赖
RGB 容易受外观相似、遮挡和长尾类别影响。引用 WasteNet、TACO、ZeroWaste、
SortWaste、ScatSpotter，不与其跨任务比较数值。

### Paragraph 2: Multimodal Opportunity and Risk

引用多传感器 bulky waste 与 SpectralWaste 说明同步物理模态可提供材料响应；
随后转折到本文图像条件软传感器：它可作为结构化先验和接口测试载体，却与 RGB
共享来源，退化时不能被无条件接纳。

### Paragraph 3: Research Gap

将垃圾多模态研究与通用缺失/低质量多模态方法连接起来。缺口不是缺一个新的注意力
模块，而是缺少围绕图像条件 4×4 软传感器来源的潜表示回退、跨模态质量判别和
机制级评估闭环。

### Paragraph 4: Proposed Response

直接给出核心机制与实验问题：能否在观测可用时利用、退化时抑制、缺失时回退，并
通过同检查点干预、匹配训练种子和注册退化进行检验。

### Paragraph 5: Contributions

贡献严格三点：

1. 面向图像条件 4×4 软传感器退化的可靠融合结构。
2. 结合潜表示预测、网格质量、一致性和不确定性的样本级门控及回退目标。
3. 由同检查点干预、三匹配训练种子、注册退化与源图簇区间组成的审计式评价。

本段同步说明 sidecar 非硬件、测试为内部固定基准。

## 2 Related Work

### 2.1 Waste Recognition from Classification to Scene Understanding

按任务演化组织，不按论文逐篇罗列。区分对象裁剪分类、整图检测和实例分割，
解释为什么本文主指标是 Accuracy/Macro F1，而 mAP 只在扩展章节出现。

### 2.2 Multimodal Waste Sorting

讨论 RGB 与热、近红外、高光谱、太赫兹等同步硬件观测。用独立物理模态的价值
反衬本文软传感器设定的研究边界，不暗示结果可替代硬件验证。

### 2.3 Low-Quality and Missing-Modality Learning

围绕动态质量融合、模态恢复、偶然不确定性和概率映射归纳已有路线。收束到本文的
区别：来源特定的 4×4 网格质量、视觉文本条件潜表示和软件退化机制审计。

## 3 Task and Data

### 3.1 Problem Formulation

定义输入 RGB 裁剪 `x`、19 类文本原型 `E`、可选网格 `g`、availability `a` 和
类别标签 `y`。明确对象分类器不预测 bbox/mask。

### 3.2 Dataset and Split

表 T01 报告对象数、源图数和来源覆盖。描述 source-image grouped split、
类别长尾、8 条重复对象及近重复敏感性。

### 3.3 Image-Conditioned Soft Sensor

说明 Qwen 提示不含类别字段、结构化校验、15 条 rule-v1 回退和仅网格入模。
采用“图像条件软传感器”作为全篇唯一主术语。

### 3.4 Governance Boundary

内部固定 test 已用于路线判断；外部盲测协议已冻结但没有数据。说明原图授权、
隐私和硬件校准尚未完成，避免正文末尾才补救性披露。

## 4 Method

### 4.1 Visual and Text Representation

给出 YOLO11s-seg 三尺度 token、可学习尺度权重和 CLIP 19 类原型查询。
视觉查询所有类别原型，不输入真实标签。

### 4.2 Sensor Encoding and Latent Prediction

说明 4×4 网格标准化、上采样和传感器编码；由视觉文本上下文预测 `mu_s` 与
有界 `log_var_s`，定义差异 `D`、不确定性 `U` 和一致性 `C`。

### 4.3 Cross-Modal Quality and Reliability Gate

这是方法章节中心。先解释原始网格统计，再解释预测 clean 网格和 22 维跨模态差异，
最后给出三个 logit 残差相加的可靠性。强调 `stopgrad` 的作用和 reliability 的
相对含义。

### 4.4 Missing Fallback and Multimodal Fusion

给出 `s_eff = mu_s + r(s-mu_s)`。从连续插值解释“用、抑制、回退”三个状态，
再描述 Adapter、FiLM-style 仿射、三 token 自注意力和动态尺度残差。

### 4.5 Objectives and Degradation Training

分类损失为 CE+Focal；辅助损失为异方差重建、可靠性校准、方差先验、clean 网格
重建和排序。表 T06 给出权重及退化概率，正文不堆叠所有实现常数。

## 5 Experimental Design

### 5.1 Implementation

报告 RTX 4090 环境、输入尺寸、优化器、batch、100 epoch、warmup 和按验证指标
选 best。训练速度优化不改变正式冻结结果，不在本稿重新训练。

### 5.2 Compared Variants

主要比较只含完整 v3 与匹配无可靠性重训。旧谱系结构消融、类别平衡、来源重训
作为补充，不与核心三种子证据混在同一主表。

### 5.3 Evaluation Protocol

报告 Accuracy、Macro Precision/Recall/F1、Weighted F1、ECE、NLL、Brier；
570 个源图簇 bootstrap，匹配比较 5,000 次。解释 McNemar 只作对象级敏感性。

### 5.4 Registered Software Degradation

定义 clean、missing_100、noise_015、bias_008、stuck、mixed。区分训练随机种子
42/43/44 与固定 seed42 模型上的退化随机种子 2026-2030。

## 6 Results

### 6.1 Main Internal Performance and Seed Stability

表 T02 先报告 seed42，再报告三训练种子均值。点明 test 不参与选 seed。

### 6.2 Reliability Mechanism

表 T03 先放匹配三训练种子独立重训，再用同检查点 gate 干预作为机制支持。
三种子平均差值和每种子区间同时呈现，避免只写 3/3 方向。

### 6.3 Robustness under Software Degradation

表 T04 和图 F02。逐场景解释完整 v3 的 Accuracy/Macro F1 方向，并如实指出
noise/bias 的 ECE 未改善。结论限定为 seed42 支持性证据。

### 6.4 Uncertainty and Reliability Diagnostics

图 F05、表 T05。先报告高相对排序，再报告 44.3 倍过分散，形成“能排序但尺度
保守”的双面结论。

### 6.5 Long-Tail and Negative Results

简要报告逐类 F1、历史消融和类别平衡负结果。若版面紧张移至补充材料，但正文
保留一句说明 Macro F1 低于 Accuracy 的原因。

## 7 Discussion

### 7.1 What the Evidence Supports

核心价值是可靠性行为和审计闭环，而不是干净集大幅增益。结合 E12-E17 解释
训练种子收益温和、同检查点机制效应较大以及退化场景支持之间的关系。

### 7.2 Why the Soft-Sensor Boundary Matters

讨论图像条件 sidecar 可用于接口开发、回退学习和软件故障注入，但不能证明独立
物理模态互补性。说明未来硬件替换需同步采集、schema 对齐和重新校准。

### 7.3 Limitations and External Validation

集中列出内部 test 多次访问、近重复风险、少数类 support、外部盲测缺失、
无真实硬件及托管 Qwen 版本未冻结。外部协议 readiness 只作为准备状态。

### 7.4 System and Patent Extensions

一段概述 YOLO-seg、联合头、两阶段链与专利矩阵，所有 mAP 进入附录。明确
当前系统不是 RGB-D 或完整专利复现。

## 8 Conclusion

回到“利用、抑制、回退”问题；复述三种子方向一致但温和的结果，不引入新数字。
最后一句给出下一步：在冻结协议下接入同步物理阵列并执行一次性外部盲测。

## Appendices / Supplement

- A：完整类别定义、支持量和逐类指标。
- B：旧谱系结构消融及 Holm 校正。
- C：Qwen/rule-v1 来源敏感性与类别平衡负结果。
- D：YOLO-seg、联合头和两阶段整图系统。
- E：专利权利要求对齐矩阵。
- F：训练完整性事件、哈希和复现依赖。

