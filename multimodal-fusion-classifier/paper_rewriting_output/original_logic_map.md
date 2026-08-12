# Original Logic Map

## Source Scope

- Primary source: `2026.4.3-1基于多模态识别的医疗废品识别方法.docx`
- Extracted audit copy: `qa/patent_source.txt`
- Source genre: patent-style technical disclosure, not an evaluated research paper.
- Source argument: medical-waste application need -> visual/text/sensor feature extraction -> Adapter/FiLM/attention fusion -> YOLO detection and segmentation outputs -> asserted performance benefits.
- Current evidence mismatch: the implemented project is a 19-class object classifier using image-conditioned soft-sensor sidecars. It does not provide synchronized physical measurements, an RGB-D acquisition protocol, a medical-only benchmark, or evidence for the patent's 98.8% accuracy and detection/segmentation claims.

## Ordered Source Units

| Unit ID | Source location | Reader question answered | Dominant move | Evidence used | Logic issue for a paper rewrite |
|---|---|---|---|---|---|
| O01 | 摘要 | 该发明做什么？ | 概括S1-S4模块并宣称精度、遮挡和实时性收益。 | 无可追溯实验表或数据来源。 | 先列模块、后给强结果，缺少可检验问题和证据边界。 |
| O02 | 权利要求1，S1 | 输入是什么？ | 假定深度图像、类别文本和传感器阵列同步获得并预处理。 | 权利要求式描述。 | 与当前RGB对象裁剪及图像条件sidecar来源不一致。 |
| O03 | 权利要求1，S2 | 如何提取三模态特征？ | 以YOLO11s-seg、CLIP和1x1卷积形成视觉、文本和传感器表示。 | 模块名称和CE/Focal。 | 可保留技术谱系，但没有说明软传感器退化或信息同源性。 |
| O04 | 权利要求1，S3 | 如何融合？ | Adapter、FiLM、余弦注意力、双向交互和多尺度融合串联。 | 模块公式。 | 组件较多但缺少设计需求、消融和可靠性机制。 |
| O05 | 权利要求1，S4 | 输出是什么？ | 进入检测头和分割头，经NMS与阈值输出类别、位置和置信度。 | 无当前项目对应评估。 | 当前核心任务是对象级分类，不能沿用检测/分割主张。 |
| O06 | 权利要求2-3 | 预处理和骨干怎样配置？ | 给出双边滤波、CLAHE、锐化、仿射变换、冻结层和ViT-B/16。 | 固定参数叙述。 | 部分参数与当前代码不一致，不能作为复现配置保留。 |
| O07 | 权利要求4-5 | 如何对齐与调制？ | 线性Adapter后使用FiLM缩放和偏移。 | 两个局部公式。 | 是可迁移结构元素，但原文没有把它们与退化观测联系起来。 |
| O08 | 权利要求6-10 | 如何进行深层交互？ | 余弦权重、softmax融合、下投影、双向注意力、残差和多尺度融合。 | 多个结构公式。 | 叙述属于实现流程，缺少问题驱动的论证与统计验证。 |
| O09 | 技术领域与背景技术 | 为什么需要该方法？ | 从纯视觉在光照、材质、小目标和遮挡方面的不足，引出物理多传感器。 | 概括性陈述，无文献引用。 | 场景被限定为医疗废物，且把当前未采集的物理传感器当作既有输入。 |
| O10 | 发明内容 | 方法解决什么问题？ | 宣称深度融合和分层微调可同时提高精度、鲁棒性和实时性。 | 98.8%、93.3%、95.7%、15.2 ms。 | 数值没有可审计产物支持，必须删除，不能迁移到论文。 |
| O11 | 具体实施方式：系统流程 | 系统如何运行？ | 按预处理、特征提取与对齐、融合、识别/分割头展开。 | 图1及流程叙述。 | 可以保留端到端组织意识，但需改为分类任务和可靠性路径。 |
| O12 | 具体实施方式：训练配置 | 如何训练？ | 给出100 epoch、batch 16、AdamW、warmup和余弦退火。 | 参数列表。 | 部分配置与现项目接近，但需要以实际配置和检查点审计为准。 |
| O13 | 具体实施方式：网络细节 | 各模块内部怎样实现？ | 说明512维适配、dropout、FiLM、8头注意力和固定尺度权重。 | 手工设定与示例。 | 存在YOLO11s-seg/YOLOv8 Head混用等内部不一致，不能直接照录。 |
| O14 | 具体实施方式：比较结果 | 方法是否优于基线？ | 列出视觉模型、传统多模态模型和本发明结果。 | 三组未经归档的点估计。 | 没有清单、检查点、随机种子和统计区间，证据不可复核。 |
| O15 | 附图说明与结尾 | 有哪些验证材料？ | 罗列17张训练/验证曲线并以专利保护范围收束。 | 图号占位。 | 当前文件不含可验证图像内容，且专利式结尾不适合研究论文。 |

## Original Move Sequence

1. 以医疗废物和纯视觉局限建立应用必要性。
2. 假定视觉、文本和同步物理传感器三类输入已经存在。
3. 依次列出预处理、编码、Adapter、FiLM、注意力、残差和多尺度融合组件。
4. 将融合表示连接到检测与实例分割头。
5. 直接给出高精度、小目标、遮挡和时延主张。
6. 用训练曲线图号和实施例说明结束，而不形成研究问题、对照协议、统计检验和局限性讨论。

## Transfer Decisions

| Source element | Decision | Reason |
|---|---|---|
| YOLO视觉特征、CLIP类别文本、4x4网格和Adapter/FiLM/注意力 | REFRAME | 与实现存在结构连续性，但必须由当前代码、配置和检查点重新定义。 |
| 一般多模态垃圾分类方向 | RETAIN | 与当前数据和19类分类任务一致，取消医疗废物强限定。 |
| 同步深度图像与物理传感器采集 | DELETE | 当前数据没有这类采集证据。 |
| 检测、位置坐标、NMS和实例分割作为主任务 | DELETE | 当前主实验是对象级分类；分割只能作为非可比扩展说明。 |
| 98.8%准确率、93.3%小目标、95.7%遮挡、15.2 ms | DELETE | 缺乏可审计实验产物，不能进入论文证据链。 |
| 组件串联式方法描述 | REWRITE | 改成潜表示预测、可靠性证据、门控回退和退化监督的因果设计链。 |
| CE与Focal处理类别不平衡 | RETAIN-WITH-AUDIT | 由现有训练实现和负结果审计支持，但不夸大为已解决长尾问题。 |
| “高精度、高鲁棒性”总括性结论 | NARROW | 仅陈述内部固定测试和预定义软件退化下的温和、方向一致收益。 |

## Required New Argument

原专利没有回答当前论文的控制性问题：当辅助网格由图像条件生成且可能缺失、加噪、偏置或卡死时，如何防止它向分类器注入错误证据？因此，论文不能只对原文做语言润色，而必须建立新的论证顺序：

1. 界定图像条件sidecar与物理传感器的差异。
2. 把“低质量辅助模态伤害分类”定义为可操作的失败模式。
3. 设计可用性回退、潜表示预测、跨模态质量证据和连续可靠性门控。
4. 用匹配重训与同检查点干预区分性能效应和行为依赖。
5. 用源图簇bootstrap、校准、退化场景和负结果限制主张。
6. 在结论中明确外部盲测、未知退化和硬件验证仍未完成。
