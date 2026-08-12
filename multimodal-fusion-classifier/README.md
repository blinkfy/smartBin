# 多模态融合分类器

这是一个与原 `smartBin` 系统完全隔离的一般智能垃圾对象级分类子项目。它参考历史专利
《基于多模态识别的医疗废品识别方法》实现视觉、候选类别文本和 4×4
软传感器阵列的融合分类，但默认论文定位不再限定为医疗废品；同时直接使用原数据路径，不复制原图，也不修改
`../ultralytics/`、`../myprogram/`、`../device-simulator/` 或原有权重。

> 当前原始数据实际是 19 类 LabelMe 标注，类别同时包含电池、药品包装、
> 瓶罐、纸张、厨余和建筑碎料，并非专利所述的完整医疗废品 5 大类/20
> 小类。本项目忠实使用现有标签，不伪造专利的 98.8% 指标。

## 已实现内容

- 从 `../data/Annotations/*.json` 读取 LabelMe polygon，在内存中计算对象框。
- 按源图分组做稳定的 train/val/test 切分，防止同一原图泄漏到不同集合。
- 清单仅保存 `image_path + bbox`；训练时才读取并裁剪原图。
- 默认用图像裁剪统计、目标框几何和稳定随机种子生成不读取真实标签的
  4×4 传感器数据。
- 可选 Qwen3-VL-Flash：把目标框裁剪编码为 Base64 图像，通过
  OpenAI-compatible `chat/completions` 请求 JSON 传感器数据；支持校验、
  钳位、重试、断点续建和逐样本规则回退。
- 当前分类网络只读取 sidecar 的 `normalized_grid`；五项属性标量保留作审计和
  未来硬件对接，不参与现有模型前向。
- YOLO11s-seg 三尺度视觉特征、CLIP ViT-B/16 全候选类别文本原型、
  4×4→16×16 传感器编码。
- 三路 Adapter、FiLM-style 全局逐模态仿射、余弦模态权重、512→128 下投影、
  8 头三 token 自注意力、128→512 上投影、残差和 0.3/0.5/0.2 初始化的动态
  多尺度融合。
- Cross Entropy + Focal Loss、AdamW、5 epoch warmup 和余弦退火。
- 专利 S1 的双边滤波、暗图 CLAHE、Laplacian 锐化、可配置仿射变换和
  480×480 缩放。

文本分支不会把样本真实标签作为输入。它始终编码全部候选类别描述，再由
视觉和传感器观测查询文本原型库；传感器模拟器的 API 同样没有 label 参数。

## 目录

当前实验结论和论文证据入口见 `reports/qwen-material/INDEX.md`。
论文动机已由用户确认采用方案 A，锁定题目为《面向图像条件软传感器退化的可靠
多模态垃圾分类方法》。当前审计母稿位于
`paper_rewriting_output/final_paper/main.tex`、`paper.zh.docx` 和
`paper.pdf`；确认后的结果治理状态见
`reports/qwen-material/MANUSCRIPT_RESULT_GOVERNANCE_V14.md`。V12/V13作为历史
治理链保留；新增确认性实验
完成后，由低频接力脚本自动生成论文增量并重建 Word/PDF；机器构建状态见
`reports/qwen-material/training/plan_a_manuscript_refresh_status.json`，最终
仍保留逐页视觉复核和 PaperSpine 总关卡，不由无人值守脚本冒充完成。

```text
multimodal-fusion-classifier/
├─ configs/                  # 默认参数与 13 组消融/增强配置
├─ data/generated/           # 可复现的 rule-v1 对照数据（无图片）
├─ data/qwen-material/       # 当前 Qwen 辅助传感器数据（无图片）
├─ mmfc/                     # 数据、模拟器、模型、损失与训练代码
├─ reports/                  # 独立评估指标、表格和混淆矩阵
├─ review-stage/             # 计划和各轮验证证据
├─ tests/
├─ build_dataset.py
├─ validate_dataset.py
├─ export_yolo_seg_dataset.py
├─ validate_yolo_seg_export.py
├─ audit_dataset_leakage.py
├─ audit_bibliography.py
├─ audit_checkpoints.py
├─ audit_method_contract.py
├─ audit_uncertainty_floor.py
├─ audit_variant_lineage.py
├─ audit_final_evidence.py
├─ audit_paper_readiness.py
├─ build_claim_evidence.py
├─ build_evidence_archive.py
├─ build_submission_index.py
├─ verify_evidence_archive.py
├─ build_paper_tables.py
├─ plot_paper_tables.py
├─ analyze_error_patterns.py
├─ train.py
├─ train_yolo_seg.py
├─ evaluate.py
├─ evaluate_uncertainty.py
├─ evaluate_scene_pipeline.py
├─ evaluate_scene_strata.py
├─ evaluate_sensor_sources.py
├─ evaluate_robustness.py
├─ evaluate_quality_probe.py
├─ evaluate_cross_modal_quality_probe.py
├─ evaluate_sensor_probe.py
├─ compare_predictions.py
├─ compare_uncertainty_diagnostics.py
├─ prepare_ablations.py
├─ prepare_sensor_source_retraining.py
├─ summarize_ablations.py
├─ summarize_ablation_comparisons.py
├─ predict.py
└─ predict_scene.py
```

## 环境安装

建议在本子目录创建独立虚拟环境，避免影响原版系统：

```powershell
cd E:\program\smtbin\multimodal-fusion-classifier
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
```

复现正式实验时优先使用精确主依赖快照：

```powershell
python -m pip install -r requirements-repro.txt
```

该文件锁定当前 `torch/torchvision + cu126`、Ultralytics、Transformers、NumPy、
OpenCV、scikit-learn、Matplotlib、Pillow 和 pytest 版本；宽松
`requirements.txt` 继续用于兼容开发，不代表正式结果的逐版本环境。

如需特定 CUDA 版本，请先按 PyTorch 官方方式安装匹配的 `torch/torchvision`，
再安装其余依赖。生产视觉分支使用隔离环境中的 `ultralytics==8.4.70`，
只读取 `../yolo11s-seg.pt`；不会导入仓库内已有的旧 `ultralytics` 源码。

## 构建与校验数据

当前论文与正式训练绑定冻结版本 `data/qwen-material/`。先直接校验现有清单：

```powershell
python validate_dataset.py --output data/qwen-material
```

`data/generated/` 是可复现的 rule-v1 对照，也是构建 CLI 为向后兼容保留的默认
输出目录。独立重建时建议写入新目录，避免覆盖既有对照：

```powershell
python build_dataset.py --sensor-mode rule --output data/rule-v1-rebuild
python validate_dataset.py --output data/rule-v1-rebuild
```

只读审计 train/val/test 的精确重复和高置信近重复，并可复算剔除敏感性：

```powershell
python audit_dataset_leakage.py `
  --prediction baseline=reports/qwen-material/test/predictions.jsonl `
  --prediction v2b=reports/qwen-material/full_quality_aware_prior/test/predictions.jsonl
```

当前划分保证同一源图的对象不跨集合，但不保证连续拍摄序列隔离；审计边界和结果
见 `reports/qwen-material/DATASET_CARD.md`。

固定线性探针可分别量化 4x4 网格和五项属性的独立可分性：

```powershell
python evaluate_sensor_probe.py --split val `
  --output reports/qwen-material/sensor-linear-probe/val
python evaluate_sensor_probe.py --split test `
  --output reports/qwen-material/sensor-linear-probe/test
```

结果解释和图像条件软传感器的表述限制见
`reports/qwen-material/SOFT_SENSOR_PROBE.md`。

当前产物统计：

- 5,990 张有效源图；
- 8,654 个对象样本和 8,654 条传感器记录；
- train/val/test 为 7,002/826/826；
- 19 类；
- 182 个缺图标注和 1 个非 polygon shape 被明确记录在
  `data/qwen-material/build_report.json`；
- `data/qwen-material/` 中没有图片副本。

正式训练配置使用 `data/qwen-material/`，其样本数和切分与规则数据一致；
其中 8,639 条为 Qwen3-VL-Flash 辅助估计，15 条因结构化输出不合法而使用
完整 rule-v1 回退，Qwen 覆盖率为 99.83%。

这里的 8,654 是冻结清单记录数。对象身份审计发现其中 8 条为同源图、同类别、
同 bbox 的完全重复记录，唯一对象裁剪数为 8,646；既有分类结果保留原 826 条
test 协议并附 825 条唯一对象敏感性，YOLO-seg 训练出口则在训练前确定性去重。

### 使用 Qwen3-VL-Flash 增强

先用单独输出目录做少量冒烟，避免覆盖完整规则清单：

```powershell
$env:DASHSCOPE_API_KEY = "你的百炼 API Key"
python build_dataset.py `
  --sensor-mode qwen `
  --limit 20 `
  --workers 4 `
  --progress-every 1 `
  --checkpoint-every 5 `
  --output data/qwen-smoke
python validate_dataset.py --output data/qwen-smoke
```

改进后的构建阶段会在请求前去除 8 条完全重复对象，全量新建最多产生 8,646 次
模型请求。必须输出到新的数据版本，不能原地覆盖当前冻结清单：

```powershell
python build_dataset.py `
  --sensor-mode qwen `
  --output data/qwen-material-unique `
  --workers 4 `
  --progress-every 10 `
  --checkpoint-every 20
```

`--resume` 只复用同一数据版本内已经成功的当前 Qwen 模型记录；规则记录或失败回退记录会
继续尝试增强。没有 `DASHSCOPE_API_KEY` 时不会发起网络请求，每个样本自动
回退到规则模拟。业务空间专属地址可用 `--base-url` 指定，模型名可用
`--model` 指定。

Qwen 构建默认使用 `--workers 4` 并发请求，主线程统一输出进度，包含完成数、
复用数、Qwen 成功数、回退数、错误数、速率和 ETA。若遇到服务端限流或账号
配额压力，先把 `--workers` 降到 `1` 或 `2`。`--checkpoint-every 20` 会周期性
原子写入 `output/sensors.jsonl`，中断后再次加 `--resume` 可复用已成功的 Qwen
记录。

必须只通过环境变量显式提供密钥，例如 `$env:DASHSCOPE_API_KEY = "..."`
后再运行构建命令；不要把密钥写进命令参数、日志或源码。早期源码曾误暴露过
一个密钥，若你曾复制或使用过它，请在百炼控制台轮换/吊销对应凭据。完整数据集
当前冻结版本有 8,654 条记录；新建唯一对象版本最多会产生 8,646 次模型请求。
不要对 `data/qwen-material` 原地 `--resume` 迁移，正式跑全量前先用上面的
`--limit 20` smoke 命令确认费用、限流和输出 schema。

## 训练

当前论文主模型是 v3，训练配置与输出目录固定为
`configs/ablations/full_cross_modal_quality.json` 和
`checkpoints/ablations/full_cross_modal_quality/`：

```powershell
python train.py --config configs/ablations/full_cross_modal_quality.json
```

训练中断后使用同一配置精确恢复：

```powershell
python train.py --config configs/ablations/full_cross_modal_quality.json --resume
```

`configs/default.json` 与 `checkpoints/qwen-material/` 保留为早期 `full`
基线，供历史结果复核和正式基础消融使用，不是当前推荐主模型入口。

`--resume` 只读取配置中 `checkpoint_dir/last.pt`；也可用
`--resume-checkpoint <path>` 显式指定。恢复检查点包含模型、优化器、调度器、
DataLoader/CPU/CUDA 随机状态和完整历史，并校验训练关键配置 SHA-256 指纹。
`last.pt` 用于续训，`best.pt` 是不含优化器的轻量评估检查点，不能用于恢复训练。
每轮历史原子写入 `history.json`，正常结束后另写 `training_summary.json`。

训练完成后可一次核验轮次、最佳点、角色、配置指纹、类别顺序和状态 schema：

```powershell
python audit_checkpoints.py `
  --config configs/ablations/full_quality_aware_prior.json `
  --output reports/qwen-material/full_quality_aware_prior/CHECKPOINT_AUDIT.json

python audit_uncertainty_floor.py

python audit_variant_lineage.py

python evaluate_uncertainty.py `
  --config configs/ablations/full_cross_modal_quality.json `
  --checkpoint checkpoints/ablations/full_cross_modal_quality/best.pt `
  --split test

python compare_uncertainty_diagnostics.py

python analyze_error_patterns.py

python analyze_error_patterns.py `
  --reference reports/qwen-material/full_cross_modal_quality/posthoc_no_reliability/test/predictions.jsonl `
  --candidate reports/qwen-material/full_cross_modal_quality/test/predictions.jsonl `
  --reference-name gate_disabled --candidate-name gate_enabled `
  --output reports/qwen-material/GATE_ERROR_ANALYSIS.json

python analyze_class_metric_intervals.py

python audit_bibliography.py

python audit_manuscript_stable_sections.py

pandoc reports/qwen-material/MANUSCRIPT_STABLE_SECTIONS.md `
  --citeproc `
  --bibliography reports/qwen-material/reference_candidates.bib `
  --to html5 `
  --output "$env:TEMP\mmfc-manuscript-stable.html"
```

`audit_uncertainty_floor.py` 把 v2b/v3 的末轮异方差重建项与配置理论下界重新
计算并绑定到配置、`history.json` 的 SHA-256。它只支持“聚合日志未出现整体
贴底”，不把潜空间方差解释为逐维或物理传感器不确定性校准。
`audit_variant_lineage.py` 则锁定 `full→v2b→v3→balanced` 的配置哈希和逐键
差异，防止把旧 `full` 的正式消融误写成 v3 单组件因果证据。

`evaluate_uncertainty.py` 在 826 条固定测试样本、512 维传感器编码空间中直接比较
预测方差和实际重建误差，并绑定配置、best 检查点、测试清单和传感器清单。当前
v3 没有元素接近方差下界，逐样本
Spearman 为 0.892，但预测方差约为潜空间 MSE 的 44.3 倍，因此只能支持
相对风险排序，不能写成绝对量值或物理传感器校准良好。
`compare_uncertainty_diagnostics.py` 绑定
v2b/v3 两份完整诊断的 SHA-256：v3 将逐样本 Spearman 从 0.822 提高到
0.892，并将方差/MSE 比从 56.9 降至 44.3。按 570 个 `source_image_id`
聚类的 2,000 次成对 bootstrap 给出 Spearman 增量 `[0.0477, 0.0954]` 和
过分散比下降 `[11.08, 14.02]` 的 95% 区间；二者均明确标记为事后诊断，
不是预注册主终点。

`audit_bibliography.py` 通过 Pandoc 的 `biblatex -> csljson` 解析锁定的 20 条
候选文献，检查键集合、标题、作者、年份、URL、重复 DOI/URL 和五类主题覆盖，并
把源 BibTeX SHA-256 写入 `BIBLIOGRAPHY_AUDIT.json/.md`。该结构审计不替代
全文阅读、撤稿核查或目标期刊适配。
`audit_manuscript_stable_sections.py` 则复核结果无关正文的章节、引用键、数据
治理、训练完整性、专利边界和八项正式方法契约，并绑定其 15 项来源。随后用
Pandoc/citeproc 实际渲染到系统临时目录，可提前发现 BibTeX 或引用语法错误；
标题、摘要、结果和结论仍不在该稳定稿范围内。

`analyze_error_patterns.py` 不重新推理或读取原图，只对齐 v2b/v3 的 826 条已保存
预测，报告主要定向混淆、正确/错误置信度、14 条纠正与 15 条回退，并保存稳定
sample ID 供后续人工误差查看。它属于事后描述性分析，不是预注册显著性终点。
同一程序另生成 `GATE_ERROR_ANALYSIS.json/.md`，比较一个 v3 检查点关闭/启用
门控后的预测：门控净增加 23 条正确，主要来自 `china`、`stone` 和 `drug`，
同时高置信错误减少。该同检查点干预可解释门控的前向贡献，但不能替代分别重训
有/无门控模型。
`analyze_class_metric_intervals.py` 对 v3 的逐类 Precision/Recall/F1 按
570 个源图簇做 2,000 次百分位 bootstrap，并记录类别缺席导致的未定义重复数。
例如 support 仅 5 的 `drugbox` F1 为 0.714，95% 描述性区间为
`[0.333, 1.000]`，因此少数类点估计不能被写成稳定外部性能。

2026-07-27 的正式消融中，`s_cabblestone_6.jpg` 在
`full_no_degradation` 第 6 轮触发 Pillow 完整解码失败。该路径只对应 1 条
train 样本，val/test 为 0；原数据树内同名镜像可解码、尺寸和长度一致，两文件
仅 16 字节不同。恢复链不改写原图或清单，而是在显式 `PYTHONPATH` 和四个
哈希锁定环境变量下只读重定向该路径，并从第 5 轮 `last.pt` 续训。故障前状态、
部分检查点、正式运行清单、损坏源、镜像和恢复脚本哈希见
`IMAGE_READ_RECOVERY_AUDIT.json/.md` 与
`IMAGE_READ_RECOVERY_INCIDENT.json/.md`。由于事前没有逐图哈希，不能声称镜像与
较早变体看到的字节逐位相同；正式消融需披露这一条训练输入恢复偏差。
`audit_source_image_integrity.py` 另对全部 5,990 张源图完整解码并计算
filename-free 内容清单：当前只有上述 1 张失败且已恢复；原始文件清单指纹为
`fc6fd136...da23`，应用镜像后的有效输入指纹为 `0d2dba53...ebf1`。完整哈希见
`SOURCE_IMAGE_INTEGRITY_AUDIT.json/.md`。
首次失败版、中间审计版和功能修订版 shim 的哈希时间线见
`IMAGE_READ_RECOVERY_INCIDENT_ADDENDUM.json/.md`。首次尝试因误处理训练实际
传入的 `pathlib.Path` 在 `5→5` 轮退出；机器审计可从当前文件删除审计 opener
行并逆向替换 `PathLike` 分支，重建前两版哈希。
恢复完成后，`audit_image_recovery_outcome.py` 将第 5 轮 resume 事件、连续
`1..100` 轮 history、best/last 检查点审计和 826 条固定 test 结果绑定到
`IMAGE_READ_RECOVERY_OUTCOME.json/.md`。该结案审计证明恢复运行完成且测试使用
最佳检查点，但仍不把镜像训练输入宣称为与故障前字节一致。
随后损坏路径被观测到已与锁定镜像逐字节相同；写入来源和精确时刻无法归因。
`SOURCE_IMAGE_REPAIR_TRANSITION.json/.md` 同时绑定损坏态审计和修复态
`SOURCE_IMAGE_INTEGRITY_CURRENT.json/.md`。修复后 5,990 图原始/有效指纹均为
先前的有效输入指纹 `0d2dba53...ebf1`，后续运行不再启用 shim。
旧恢复链退出时，`full_no_reconstruction` 已记录 68 轮，但第 68 轮 `last.pt`
缺少 ZIP 中央目录；第 67 轮 `best.pt` 又不含优化器、调度器和 RNG 状态，因此
拒绝不完整 warm start。部分目录被保留到 `checkpoints/interrupted/`，正式变体以
原冻结配置从第 1 轮完整重训；快照和决策见
`FORMAL_QUEUE_INTERRUPTION.json/.md` 及其 `ADDENDUM`。清洁重训已连续完成
100 轮，best/last 检查点审计和 826 条固定 test 结果由不可变
`FORMAL_QUEUE_INTERRUPTION_OUTCOME.json/.md` 结案。
独立 YOLO-seg 首次启动完成第 1 轮训练与验证后，在 Ultralytics 保存
`results.csv` 时因环境缺少 Polars 退出，且未生成任何模型权重。五个部分文件
移入 `checkpoints/interrupted/yolo_seg_missing_polars_epoch1_20260728/`；
失败状态、traceback、冻结协议和修复后的依赖版本由
`YOLO_SEG_DEPENDENCY_INCIDENT.json/.md` 及其 `ADDENDUM` 逐哈希绑定。补齐
Polars、nvidia-ml-py 和 ultralytics-thop 后，训练只按原冻结协议从第 1 轮
干净重启；正式 detector 以 `resume=false` 连续完成 100 轮，best/last、固定
test 和恢复后的整图结果由 `YOLO_SEG_DEPENDENCY_OUTCOME.json/.md` 结案，失败
运行不进入最终指标。

首次整图评估随后因 Ultralytics 路径列表转 PIL 时返回 `image{index}.jpg` 合成名
而在路径一致性校验处退出，且未创建结果目录。
`yolo-seg/INTEGRATED_EVALUATION_PROTOCOL_ADDENDUM.json/.md` 只授权按合成索引
校验原列表顺序，检查点、预测、阈值、匹配、指标和 bootstrap 保持冻结；恢复运行
已完成。分层结果生成前另冻结
`yolo-seg/STRATIFIED_EVALUATION_PROTOCOL_ADDENDUM.json/.md`，用于绑定整图修订
和分层评估器的新旧源码哈希。
正式分层运行随后完整写出 42 行 JSON/CSV，但 Markdown 渲染因分类准确率区间
字段前缀不一致而退出。`SCENE_STRATA_REPORTING_INCIDENT.json/.md` 与恢复前
补充协议绑定该现场；恢复只从已持久化 JSON 生成缺失报告，并要求 JSON/CSV 哈希
保持不变，不重算训练、检测、匹配、bootstrap 或指标。

默认参数来自专利实施例：

- batch size 16；
- 初始学习率 `1e-4`；
- AdamW，weight decay `1e-5`；
- 100 epochs；
- 5 epochs warmup；
- CE + Focal；
- YOLO 前 5 层冻结，CLIP 文本 Transformer 前 3 层冻结。

检查点写入本子目录的 `checkpoints/`，不会进入原版 `runs/`。

## 固定留出测试集评估

当前比较结果使用按验证指标选出的 `best.pt` 在固定留出测试集上评估：

```powershell
python evaluate.py `
  --config configs/ablations/full_cross_modal_quality.json `
  --checkpoint checkpoints/ablations/full_cross_modal_quality/best.pt `
  --split test `
  --bootstrap-repeats 2000 `
  --output reports/qwen-material/full_cross_modal_quality/test
```

历史基线结果仍位于 `reports/qwen-material/test/`；不要把它与上述 v3
结果目录混用。

该集合不参与梯度训练或类别权重统计，但已用于多轮预注册候选比较，因此不再称为
“从未查看的最终盲测集”。测试访问历史、允许的论文表述和投稿前新批次确认要求见
`reports/qwen-material/EVALUATION_GOVERNANCE.md`。

评估过程不启用软件传感器退化，输出 Accuracy、Macro Precision/Recall/F1、
Weighted F1、Top-3 Accuracy、NLL、Brier Score、ECE、逐类别指标、逐样本预测、
混淆矩阵及模型前向耗时和参数量。`reports/qwen-material/test/` 中生成：

- `evaluation.json`
- `classification_report.csv`
- `confusion_matrix.csv`
- `confusion_matrix_normalized.csv`
- `confusion_matrix.png`
- `predictions.jsonl`

推理耗时只统计模型前向传播，不包含图片读取和预处理，并按评估 batch
摊销到每个样本，不等同于 batch=1 的端到端延迟；正式论文评估不要使用
`--max-samples`，如需报告实时单目标延迟应额外使用 `--batch-size 1` 复测。
`--bootstrap-repeats` 按 `source_image_id` 聚类重采样，避免同一原图内多个对象
被当作独立样本；默认关闭，正式 clean 测试建议使用 2,000 次和固定种子。
两个模型已有逐样本结果时，可直接做配对源图 bootstrap，无需重新前向：

```powershell
python compare_predictions.py `
  --reference reports/qwen-material/test/predictions.jsonl `
  --candidate reports/qwen-material/full_quality_aware_prior/test/predictions.jsonl `
  --reference-name baseline `
  --candidate-name full_quality_aware_prior `
  --repeats 5000 `
  --output reports/qwen-material/full_quality_aware_prior/PAIRED_COMPARISON.json
```

## 传感器退化鲁棒性评估

固定同一个 `best.pt`，在 clean、不同缺失率、不同噪声强度、bias/drift、stuck
和 mixed 场景下评估：

```powershell
python evaluate_robustness.py `
  --config configs/ablations/full_cross_modal_quality.json `
  --checkpoint checkpoints/ablations/full_cross_modal_quality/best.pt `
  --split test `
  --seeds 2026 2027 2028 2029 2030 `
  --output reports/qwen-material/full_cross_modal_quality/robustness
```

每个场景会输出完整分类指标、混淆矩阵和逐样本预测；逐样本记录包含传感器质量
目标、预测可靠性、一致性、差异和不确定性，连续可靠性同时报告
MAE、RMSE、Pearson、Spearman 和成对排序准确率。多种子原始结果位于
`seeds/seed-<编号>/`，根目录额外生成 `robustness_multiseed_summary.csv`、
`robustness_multiseed_summary.json` 和 `robustness_multiseed_curves.png`，
统计量采用 mean +/- sample std。单次探索仍可使用 `--seed 2026`，其目录结构和
原有 `robustness_summary.*` 产物保持不变。当前详细解读见
`reports/qwen-material/ROBUSTNESS.md`。

## 多模态消融实验

下列命令生成 12 组基础独立消融/增强配置；另有单独锁定的
`full_cross_modal_quality_balanced`，因此目录当前共 13 组：

```powershell
python prepare_ablations.py `
  --base-config configs/default.json `
  --output configs/ablations
```

配置覆盖视觉-only、视觉+文本、视觉+传感器、三模态等权融合、保留
FiLM/注意力但关闭可靠性门控、完整模型，以及分别关闭训练退化、重建损失和
校准损失；`full_quality_aware` 额外启用原始网格质量校正分支和更宽的退化强度
分布，`full_quality_aware_prior` 再加入方差先验与 clean 样本校准锚点，避免
异方差重建项贴近方差下界；`full_cross_modal_quality` 进一步由视觉/文本潜特征
预测 clean 4×4 网格，并用跨模态网格差异和成对排序损失校正可靠性；
`full_cross_modal_quality_balanced` 在 v3 上加入 sqrt-inverse CE 类别权重，
已按预注册门槛保留为负结果。每个变体应独立训练，例如：

```powershell
python train.py --config configs/ablations/vision_only.json

python evaluate.py `
  --config configs/ablations/vision_only.json `
  --checkpoint checkpoints/ablations/vision_only/best.pt `
  --split test `
  --output reports/qwen-material/ablations/vision_only
```

所有变体评估完成后生成论文表格和图：

```powershell
python summarize_ablations.py `
  --reports reports/qwen-material/ablations `
  --protocol formal_retrained
```

训练配置支持 `max_train_samples` 和 `max_val_samples`，仅用于管线冒烟，正式结果
不得设置样本上限。当前同一 epoch 97 检查点的前向干预诊断位于
`reports/qwen-material/posthoc-ablations/`，不能替代独立重训；正式协议、当前
负面结果及判定门槛见 `reports/qwen-material/ABLATION_PROTOCOL.md`。

`full_quality_aware_prior` 已完成 100 轮正式训练和五种子退化评估。其测试
Accuracy 为 88.26%，同检查点门控在全部场景均优于无门控干预，但连续可靠性
排序校准仍有不足；完整结果和表述边界见
`reports/qwen-material/full_quality_aware_prior/REPORT.md`。推理期门控对照可用：

```powershell
python evaluate_robustness.py `
  --config configs/ablations/full_quality_aware_prior.json `
  --checkpoint checkpoints/ablations/full_quality_aware_prior/best.pt `
  --disable-sensor-reliability `
  --seeds 2026 2027 2028 2029 2030 `
  --output reports/qwen-material/full_quality_aware_prior/posthoc_no_reliability/robustness
```

正式训练可靠性 v2 前，可单独验证 7 个原始网格质量特征是否携带退化信号：

```powershell
python evaluate_quality_probe.py `
  --config configs/ablations/full_quality_aware.json `
  --output reports/qwen-material/quality-probe
```

该探针只读取 train/test 的 4×4 sidecar，不读取图片或类别标签，也不代表完整模型
性能。当前结果显示它能显著降低强噪声的质量 MAE，但 clean 偏差和样本级相关性
仍不足，详细边界见 `reports/qwen-material/QUALITY_PROBE.md`。

跨模态 v3 在正式训练前还使用冻结的 v2b 检查点验证视觉/文本潜特征能否预测
clean 网格并识别退化：

```powershell
python evaluate_cross_modal_quality_probe.py `
  --config configs/ablations/full_quality_aware_prior.json `
  --checkpoint checkpoints/ablations/full_quality_aware_prior/best.pt `
  --latent-source combined `
  --degradation-seeds 2026 2027 2028 2029 2030 `
  --output reports/qwen-material/cross-modal-quality-probe-combined
```

轻量组合探针对中强噪声、stuck 和 mixed 有信号，扩大解码器容量后反而过拟合；
轻噪声与 bias 的连续排序仍未解决。v3 的完整探针、负面容量对照和 5 轮子集诊断见
`reports/qwen-material/CROSS_MODAL_QUALITY_PROBE.md`。

v3 正式训练和预注册评估已经完成。独立测试 Accuracy/Macro F1 为
88.14%/82.44%，与 v2b 的 clean 差异不显著；但主要非缺失退化的连续排序均
改善，同检查点门控相对无门控的 clean Accuracy/Macro F1 提高
2.78/3.50 pp，且十个场景的门控 Macro F1 均更高。完整边界和统计见
`reports/qwen-material/full_cross_modal_quality/REPORT.md`。

以 v3 为底座的 sqrt-inverse CE 类别平衡候选也已完成 100 轮独立训练。种子 42
测试 Accuracy/Macro F1 为 87.65%/82.25%，相对 v3 为 -0.48/-0.19 pp；
Macro Recall 提高 1.34 pp，但 Macro Precision 下降 1.59 pp，验证 Macro F1
下降 1.64 pp。该候选未通过预注册继续门槛，已停止种子 43/44，且不再根据逐类
测试结果调权。完整负结果见
`reports/qwen-material/full_cross_modal_quality_balanced/REPORT.md`。

### v3 新增监督项的确认性消融

基础八项正式消融不能逐项归因 v3 新增的网格重建与可靠性排序监督。方案 A 确认后，
已另行冻结两类组件消融：`v3_no_grid_reconstruction` 和 `v3_no_ranking`，
各使用训练种子 42/43/44、100 轮独立重训，并与同种子完整 v3 做源图簇配对统计。
协议位于
`reports/qwen-material/confirmatory-ablations/v3-loss-components/PROTOCOL.json`。

```powershell
python prepare_v3_loss_component_ablations.py
python build_v3_loss_component_protocol.py
powershell -ExecutionPolicy Bypass -File `
  reports/qwen-material/training/run_v3_loss_component_ablations.ps1
```

正式队列使用 `num_workers=0`。短输入基准虽证明 workers 0/4 的样本顺序和张量
字节一致，但既往 workers=4 长周期确认性训练在第 69 轮失去活性，因此其正式
提速建议已撤回；不以短测吞吐替代 100 轮稳定性证据。队列只按阶段写状态，不逐轮
轮询，正、负和零效应都必须在 `OUTCOME.json` 审计通过后报告。

### 未见传感器条件评估

为检验可靠性门控是否只适应训练时预设退化，另行冻结了不再训练模型的三种子评估
协议。协议覆盖干净输入、越界噪声/偏置、脉冲噪声、非线性响应、空间错位、跨样本
传感器替换、卡死中值，以及三种传感器可用性真值/上报组合，共 11 个条件；所有
扰动按样本标识确定性生成，并与同种子干净条件做源图簇配对 bootstrap。

协议位于
`reports/qwen-material/unseen-sensor-conditions-v1/PROTOCOL.json`。低频接力
脚本 `reports/qwen-material/training/run_unseen_sensor_conditions_after_components.ps1`
每 300 秒只检查一次上述组件消融的阶段状态；上游全部审计通过后才运行评估，上游
失败则关闭后续评估。该协议当前仅表示预注册和排队完成，不代表已有结果。

### 近期鲁棒融合直接基线

QMF（ICML 2023）与 TMC（ICLR 2021）的同输入适配已经实现。两者共享当前
YOLO 多尺度视觉编码器、CLIP 候选类别原型、4×4 sidecar、数据划分、预处理、
训练退化和优化日程，只替换为各自的决策融合及训练目标。原冻结协议位于
`reports/qwen-material/recent-robust-baselines/PROTOCOL.json`；按用户对计算
成本的后续决定，正式范围修订位于同目录 `PROTOCOL_AMENDMENT.json`：QMF保留
种子42/43/44三个完整100轮运行，TMC仅保留完整seed42作为单种子参考。TMC
seed43在34轮终止并隔离，seed44未启动，二者均不进入统计；不能把seed42写成
多次运行，且TMC样本标准差记为不适用。

文本侧严格称为 `visual_text` 分支：全部候选类别原型由目标视觉特征查询，不宣称
存在独立采集的逐样本文本。QMF 使用能量置信度动态融合和累计损失排序，TMC 使用
Dirichlet 证据与 Dempster–Shafer 组合。两者均为公式级同输入适配，不冒充原作者
数据集和编码器的逐字节复现。

原低频接力脚本
`reports/qwen-material/training/run_recent_robust_baselines_after_unseen.ps1`
在 v3 的 11 条件评估审计完成后启动原计划训练。最终 `OUTCOME.json` 仅汇总
范围修订允许的4个完整运行，并对TMC单种子结果保留源图簇配对区间。

### 方案 A 自动论文结案

组件消融、11 条件评估和范围修订后的 QMF/TMC 直接基线审计完成后，
`run_plan_a_manuscript_refresh_after_baselines.ps1` 会按以下顺序接力：

1. `build_plan_a_confirmatory_addendum.py` 读取三份最终 `OUTCOME.json`，生成
   组件表、未见条件表、直接基线表、预选条件配对表及 LaTeX 论文增量。
2. `apply_plan_a_confirmatory_manuscript_update.py` 先快照现有交付件，再把审计
   增量插入规范 `main.tex`，并替换已经过时的“尚未验证”限制文本。
3. 固定使用 Anaconda 文档环境重建 `paper.zh.docx`，执行 LaTeX/Word guard，
   再由 Microsoft Word 导出字节一致的 `paper.pdf` 与 `main.pdf`。
4. `audit_plan_a_automated_paper_build.py` 检查引用、关键段落、表格数、PDF页数、
   旧占位表述和 PDF 别名哈希；状态随后停在 `awaiting_visual_qa`。
5. 在基础 Anaconda 文档环境运行 `prepare_plan_a_visual_qa.py`，用 PyMuPDF 把
   正式 `paper.pdf` 全部页面渲染为哈希锁定 PNG；生成的决定模板默认所有页面
   `reviewed=false`、`overall_pass=false`。当前由
   `run_plan_a_visual_qa_prepare_after_refresh.ps1` 在论文状态就绪后自动执行该
   准备步骤，但不会代替页面检查。
6. 逐页实际查看后单独形成 `REVIEW_DECISION.json`，再运行
   `complete_plan_a_visual_qa.py`；只有页数、PDF哈希、每页PNG哈希全部一致、
   每页已复核且问题列表为空时，才生成 `plan_a_visual_qa.json/.md`。
7. `run_plan_a_post_visual_finalize.ps1` 只在上述视觉审计存在后接力：刷新
   `final_artifact_manifest.md` 和交付报告，重跑 PaperSpine 最终硬关卡及论文
   就绪审计；方案A使用独立的 `PLAN_A_MANUSCRIPT_STABLE_SECTIONS_AUDIT.*`
   和 `PLAN_A_PAPER_READINESS_AUDIT.json`，不覆盖旧论文门。随后生成不可变的
   `PLAN_A_SUBMISSION_INDEX.json/.md`，再构建并校验独立的
   `submission-evidence-plan-a/`。旧审计、旧索引和旧归档继续作为历史版本保留。

该链只每 300 秒读取一次上游阶段状态，不监控训练 epoch，也不占用 GPU。自动
内容审计通过后仍须逐页查看最终渲染，再刷新 PaperSpine 最终审计、交付清单和
证据包；外部盲测、实体传感器和 RGB-D 仍是独立的外部证据门。

## 单目标预测

预测需要原图、目标框和真实或模拟的 4×4 传感器数据，不需要类别文本或真实标签：

```powershell
python predict.py `
  --image ..\data\images\brick_38.jpg `
  --bbox 109.5 279.5 219.5 393.5 `
  --sensor-json .\example-sensor.json
```

`sensor-json` 可直接使用一条 `data/qwen-material/sensors.jsonl` 记录拆出的 JSON
对象；训练和预测都会优先读取 `normalized_grid`。默认加载已锁定的 v3 配置与
最佳检查点；旧基线需显式传入 `configs/default.json` 和
`checkpoints/qwen-material/best.pt`。

## 整图检测适配

`predict_scene.py` 在本项目内独立加载外部 Ultralytics 检测/分割权重，使用其
NMS 后的 bbox 和可选 mask 批量裁剪目标，再交给 v3 多模态分类器复分类。它不修改
原版 YOLO 源码或图片。默认 `confidence=0.8`、`NMS IoU=0.5`，与专利实施例一致：

```powershell
python predict_scene.py `
  --image ..\data\images\scene.jpg `
  --detector-checkpoint D:\weights\garbage-seg-best.pt `
  --sensor-source missing `
  --include-masks `
  --output-json reports\scene-result.json `
  --output-image reports\scene-result.jpg
```

`--sensor-source rule-v1` 可显式生成图像条件软传感器；`sidecar` 模式要求
`--sensor-json` 为每个 `detection_index` 提供一个 4×4 阵列，并拒绝
class/target 字段。目标域 YOLO-seg 已完成 100 轮正式训练和固定 test 评估；
`predict_scene.py` 仍是两阶段系统集成入口，不能把其对象复分类结果与联合头
box/mask mAP 混为同一指标。

现有 LabelMe polygon 已按锁定源图划分导出。输入 8,654 条对象记录中检测到
8 条完全重复记录；`data/yolo-seg-unique-metadata` 确定性保留 8,646 个唯一
polygon。可训练的 `data/yolo-seg-unique-linked` 显式创建 5,990 个原图符号
链接，图片复制数为 0，并通过自有校验和 Ultralytics 原生全量扫描：

```powershell
python validate_yolo_seg_export.py --dataset data/yolo-seg-unique-linked

python train_yolo_seg.py --validate-only
```

正式 YOLO11s-seg 已按
`reports/qwen-material/yolo-seg/RUN_MANIFEST.json` 的冻结输入完成 100 轮训练；
独立 test 的 box/mask mAP50-95 分别为 0.9655/0.9513。随后
`evaluate_scene_pipeline.py` 已在 570 张固定 test 源图上比较 YOLO 原类别、
v3 缺失传感器复分类和 v3 rule-v1 复分类，并同时报告
confidence 0.25 与专利阈值 0.8 下的定位、匹配分类和端到端指标。协议已在 detector
训练前冻结到 `yolo-seg/INTEGRATED_EVALUATION_PROTOCOL.json`。详细边界见
`reports/qwen-material/SEGMENTATION_EXPORT.md`。

## 联合多模态分割头

`JointMultimodalSegmentationModel` 在独立 YOLO11s-seg 的 P3/P4/P5 特征进入
原生 `Segment` 头之前，注入冻结 CLIP 类别原型和 RGB 预测的 4×4 软传感器条件。
Qwen/rule 网格只作为训练期辅助蒸馏目标；正式 test 前向只接收 RGB，不读取
标注、目标框或 sidecar。该支线不修改原版 Ultralytics 源码和原始图片：

```powershell
python prepare_joint_multimodal_seg.py

python train_joint_multimodal_seg.py `
  --name yolo11s-seg-joint-v1-seed42 `
  --fusion-mode full `
  --epochs 20 --patience 5 --batch-size 32 `
  --workers 0 --freeze-layers 24 --seed 42

python evaluate_joint_multimodal_seg.py `
  --checkpoint checkpoints/joint-multimodal-seg/yolo11s-seg-joint-v1-seed42/weights/best.pt `
  --output reports/qwen-material/joint-multimodal-seg/yolo11s-seg-joint-v1-seed42/diagnostics

python audit_joint_multimodal_seg.py `
  --report-run reports/qwen-material/joint-multimodal-seg/yolo11s-seg-joint-v1-seed42

python aggregate_joint_multimodal_seg.py

# 按冻结协议串行完成 text_only/sensor_only 的 3 种子独立重训
python run_joint_seg_retrained_ablations.py
python audit_joint_seg_retrained_ablations.py
```

种子 42/43/44 的固定 test box/mask mAP50-95 为
`0.9709 ± 0.0002`/`0.9534 ± 0.0002`（均值 ± 样本标准差），相对独立 RGB
基线的均值变化为 +0.54/+0.21 个百分点。同检查点 `no_text`、`no_sensor` 和
`vision_only` 仅是分支屏蔽诊断，不是独立重训消融。正式分支消融使用
`--fusion-mode text_only` 和 `--fusion-mode sensor_only` 分别独立训练：

| 模型 | Box mAP50-95 | Mask mAP50-95 |
|---|---:|---:|
| 独立 RGB | 0.9655 | 0.9513 |
| 仅文本 adapter（3 种子） | 0.9705±0.0000 | 0.9525±0.0000 |
| 仅预测软传感器 adapter（3 种子） | 0.9707±0.0001 | 0.9533±0.0001 |
| 完整联合头（3 种子） | 0.9709±0.0002 | 0.9534±0.0002 |

两个单分支均值都高于 RGB 基线；完整头相对仅文本的 box/mask 均值变化为
+0.038/+0.094 个百分点，相对仅预测软传感器为 +0.020/+0.007 个百分点。
最后一项 mask 差异在三个种子间符号不一致，不能据此声称强协同或统计显著性。
权威多种子结果见
`reports/qwen-material/joint-multimodal-seg/MULTISEED_SUMMARY.json`；每个种子的
正式结果、软传感器诊断和主张边界位于对应 run 目录。独立重训消融协议与审计
结果见 `joint-multimodal-seg/retrained-ablations/PROTOCOL.json` 和
`OUTCOME.json`。

冻结骨干的输入等价测速中，`batch_size=32` 约为 39.8--40.0 秒/轮；
`batch_size=64` 反而慢约 6.4%，因此正式联合头和分支消融统一使用 32。该结论
只适用于当前显卡、图像大小和冻结策略，后续更换环境应重新做短基准。

同一 RTX 4090 环境的固定 test 验证记录表明，联合 adapter 增加 483,494 个
参数（约为 RGB 模型的 4.79%）；三种子联合头的验证器推理摊销时间约为
`2.033±0.018 ms/图`，RGB 单次正式验证为 `1.775 ms/图`。该对照用于说明当前
环境的描述性开销，不是受控单样本延迟测试，也不能写成复现专利的 15.2 ms。
权威数据见 `joint-multimodal-seg/DEPLOYMENT_PROFILE.json`。

### 整图外部传感器联合头 v2

旧 `image_sensor_targets.jsonl` 的空间网格由对象 polygon 投影得到，只能作为
训练期蒸馏目标，禁止在 test 时直接输入检测头。独立 v2 改用完整场景图生成的
一图一条 4×4 sidecar，构建过程不读取标签、bbox、polygon、mask 或类别：

```powershell
# 已完成的无标注 rule 集成数据与校验
python build_scene_sensor_sidecars.py `
  --mode rule --output data/scene-sensor-sidecars `
  --workers 8 --checkpoint-every 200 --progress-every 500
python validate_scene_sensor_sidecars.py `
  --output data/scene-sensor-sidecars

# 1 轮接口 smoke；不进入正式论文表格
python train_external_sensor_joint_seg.py `
  --name yolo11s-seg-joint-external-rule-smoke `
  --epochs 1 --patience 1 --batch-size 32 `
  --workers 0 --freeze-layers 24 --seed 42
python audit_external_sensor_joint_seg.py
```

当前 rule 数据共 5,990 条，train/val/test 为 4,823/597/570，校验错误为 0。
v2 用第四个 loader 通道传送量化网格，但模型在 YOLO 前立即拆分；检查点审计确认
主干第一层在外部观测和缺失回退两种前向中都只收到 3 个 RGB 通道。缺少 sidecar
时继续使用 RGB 预测软传感器。

整图 Qwen 正式协议已锁定到
`reports/qwen-material/joint-multimodal-seg-external/QWEN_FORMAL_PROTOCOL.json`，
计划使用种子 42/43/44。当前环境没有 `DASHSCOPE_API_KEY`，所以正式 Qwen 数据和
训练尚未启动；缺少密钥时构建器会在请求网络和写文件之前终止。详细数据边界见
`reports/qwen-material/SCENE_SENSOR_DATA.md`。

为补充专利关注的小目标/遮挡诊断，同时避免虚构当前数据没有的人工属性，
`STRATIFIED_EVALUATION_PROTOCOL.json` 已在 detector 训练前冻结三种辅助分层：
标注框面积四分位、单/多目标场景、标注框正面积重叠代理。整图结果生成后运行：

```powershell
python evaluate_scene_strata.py
```

该分析复用同一批 detector 输出并按源图聚类 bootstrap。`relative_small` 只是
当前固定 test 的相对面积下四分位，`bbox_overlap` 也只是拥挤/潜在遮挡代理，
均不能写成标准小目标或人工遮挡识别率。

整个后台链完成后可执行严格证据审计：

```powershell
python audit_final_evidence.py
```

它统一核对冻结数据数量、9 组正式消融、8 组配对统计、YOLO-seg test、
整图双阈值/三类别来源结果和协议哈希。队列运行期间可加 `--allow-pending`
做非阻断预检，此模式只写 `FINAL_EVIDENCE_AUDIT.pending.json`，不会占用严格
文件名；后台守护会在 YOLO/整图流程完成后自动生成
`reports/qwen-material/FINAL_EVIDENCE_AUDIT.json`。严格审计成功后，同一守护流程
继续执行：

```powershell
python build_paper_tables.py
```

它只接受 `valid=true` 且 `state=complete` 的最终审计结果，并在
`reports/qwen-material/paper-tables/` 生成 10 张 UTF-8-SIG CSV、`TABLES.md`、
5 张论文结果 PNG 和带 15 项来源/全部 16 项产物 SHA-256 的
`manifest.json`；Markdown 另直接带入
v3-v2b 与 v3 gate/no-gate 两条锁定配对证据。历史评估未计算的单模型置信区间
保留为空值/`NA`，不会用事后估计补齐。多文件表组先在同目录临时工作区完整生成，
再一次性提交到正式目录；中断不会留下被误判为正式结果的半成品。来源路径相对
`reports/qwen-material` 记录，跨机器解压后不会绑定原工作区盘符。

表组生成后，后台继续执行 `audit_paper_readiness.py`。它分别输出：

- `internal_paper_evidence_ready`：内部固定基准上的论文证据是否齐全；
- `full_patent_reproduction_ready`：RGB-D、实体传感器和联合检测头是否全部实现；
- `external_deployment_claim_ready`：是否具备外部盲测和真实部署证据。

现有最终就绪审计已完成；第一项已具备内部论文证据，后两项在缺少真实硬件、
RGB-D 和外部盲测时必须保持 `false`，防止把“可以写内部实验论文”误写成
“完整复现专利或可部署”。联合头增量补齐了融合特征进入检测/分割头的结构路径，
但不会改变这两个边界。
不依赖动态结果的研究背景、相关工作、数据治理、方法、实验设置、训练完整性、
统计和专利边界正文已经整理到
`MANUSCRIPT_STABLE_SECTIONS.md`。用户随后确认方案 A，标题、摘要、结果段和结论
已形成通过 PaperSpine 最终门禁的完整母稿；v7 保留确认前历史状态，v8 增量治理
文件记录已确认主线和仍在运行的组件消融，v9/v10 依次冻结未见条件与
QMF/TMC直接基线，v11 锁定实验完成后的自动论文结案顺序及人工视觉复核边界。

在表格和就绪审计之间，`build_claim_evidence.py` 会生成
`FINAL_CLAIM_EVIDENCE.json/.md`。它保留训练前 `PAPER_CLAIM_REGISTER.md` 的历史
状态，并从最终结果自动填充正式消融、C13-C15 和目标域系统证据；配对区间按指标
方向标记为 `candidate_better`、`candidate_worse` 或
`interval_inconclusive`，不由人工挑选正结果。构建时还逐项核对主张所读取的
9 份指标来源与论文表格 manifest 中的路径和 SHA-256。

最终就绪门通过后，`build_submission_index.py` 先生成
`SUBMISSION_INDEX.json/.md`，把最终审计、论文门、动态主张、10 张表、文献
结构审计、方差下界审计和 v2b-v3 潜表示不确定性诊断绑定到文件哈希；它是
交付后的权威状态入口，工作期 pending 索引仅保留过程历史。
随后 `build_evidence_archive.py` 生成
`submission-evidence/paper-evidence.zip`、文件级 `evidence-manifest.json` 和
外层 `archive-manifest.json`。ZIP 使用固定时间戳和排序，重复构建哈希一致；
整个三文件归档先在同目录临时工作区生成，全部成功后再一次性提交；
只包含审计、表格、动态主张、核心报告/图、源码、正式配置、测试和队列脚本，
另保留 `history.json`/`training_summary.json` 训练元数据以复算曲线和异方差审计；
不包含原专利 DOCX、原图、标注、sidecar JSONL、`.pt` 模型权重、训练日志、PID
或逐样本预测；文本白名单中如出现疑似 `sk-...` 令牌，构建器只报告文件路径并
拒绝归档，不回显令牌内容。归档生成后由独立校验器逐项核对外层哈希、内外清单字节、成员
白名单、路径安全性、文件大小和 SHA-256：

潜表示诊断在本地就绪审计中绑定 v2b/v3 `best.pt` 哈希，但权重本体不进入
证据 ZIP。接收方可验证报告字节和记录的检查点身份；若要重跑推理，仍需另行取得
相同哈希且授权允许使用的检查点。
两份错误分析绑定 v2b/v3 及 v3 无门控 `predictions.jsonl` 哈希，但逐样本 JSONL
同样不进入证据 ZIP；归档接收方可核验聚合报告及其输入身份，重算样例排序时需
另行取得允许共享的同哈希预测文件。

```powershell
python verify_evidence_archive.py
```

该 ZIP 是内部论文证据包，不是自动获得再分发许可的开源发布包。源码许可、原始
图片/标注来源、第三方权重、Qwen 输出条款和隐私审查的待确认项见
`reports/qwen-material/ARTIFACT_RELEASE_CHECKLIST.md`。

同一 v3 检查点的 Qwen/rule-v1/缺失来源迁移测试也已在运行前冻结：

```powershell
python evaluate_sensor_sources.py
```

协议位于 `reports/qwen-material/sensor-source-transfer/PROTOCOL.json`。该测试只衡量
运行时 sidecar 来源敏感性，不等同于分别使用 Qwen 与 rule-v1 数据独立重训。

两套 sidecar 的 8,654 个 sample ID 已验证一一对应。另一个独立重训协议冻结在
`sensor-source-retraining/PROTOCOL.json`：保持 v3 架构、对象清单、种子 42、
损失和 100 轮训练不变，只把训练/验证/测试传感器来源替换为 rule-v1。它在现有
YOLO 与来源迁移队列释放 GPU 后自动执行，再与 Qwen 训练的 v3 做 5,000 次配对
源图 bootstrap 和精确 McNemar。该对照仍只有一个训练种子，且两种来源都是
图像条件软传感器。

rule-v1 训练子进程在完整持久化 100 轮 history、training summary、best.pt 和
第 100 轮 last.pt 后，以 Windows `0xC0000005` 退出，训练 stderr 为空。独立
检查点审计已成功加载两份权重并核对轮次、配置指纹、状态结构和 SHA-256。
`SENSOR_RETRAINING_EXIT_INCIDENT.json/.md` 锁定失败现场；恢复不重训，而是重跑
原冻结队列并触发其 `Test-TrainingComplete` 跳过分支，再继续固定 test 与配对
统计。首份恢复脚本的路径参数绑定问题在运行前被发现，原稿保留，
`SENSOR_RETRAINING_EXIT_INCIDENT_ADDENDUM.json/.md` 只授权 v2 的路径组合修订。
v2 完成全部下游分析后，结案审计因 PowerShell UTF-16 日志按 UTF-8 读取而退出；
`SENSOR_RETRAINING_LOG_ENCODING_REPAIR.json/.md` 证明编码规范化前后解码文本一致，
`SENSOR_RETRAINING_EXIT_INCIDENT_ADDENDUM_2.json/.md` 将 v3 限制为只重试结案
审计及后续最终证据链。

## 测试

```powershell
python -m pytest -q
python -m compileall -q .
```

测试不下载 YOLO 或 CLIP 权重。生产冒烟已经验证本地
`yolo11s-seg.pt` 可通过 `ultralytics 8.4.70` 捕获三个 512 维尺度。
2026-07-30 在正式训练环境完成全量回归：537 项通过；因该环境未安装
PyMuPDF 和 `python-docx`，2 个文档模块级测试由基础 Anaconda 文档环境补跑，
对应 7 项全部通过。`FULL_REGRESSION_AUDIT.json/.md` 合并后记录 544 项唯一
测试通过、0 个未解决跳过和 0 个失败，并绑定当前测试、源码、配置和队列脚本。
此前全仓库 148 个 Python 文件已通过字节码编译检查；实际数据校验为 8,654 个样本、
8,654 条传感器记录、0 个错误和 8 个已解释的完全重复对象告警。

## 重要限制

模拟传感器数据只是用于完成数据管线、模型开发和接口联调。它与输入图像
相关，仍不是独立的真实物理观测；基于它得到的验证准确率不能作为真实部署
精度或专利指标的证据。上线前必须用同步采集的真实压力、光谱、湿度、电导率
等传感器数据替换或校准模拟数据，并重新划分独立测试集。

### 实体4×4传感器替换接口

`mmfc/physical_sensor.py` 和 `prepare_physical_sensor_sidecars.py` 已固定实体阵列
替换Qwen sidecar时的输入契约。原始记录必须提供设备编号、测量单位、带时区的
图像/传感器时间戳、严格4×4读数、五项属性，以及逐格零点/满量程校准。转换器
默认拒绝超过100 ms的同步误差、超过365天的校准、重复样本和任何类别/标注字段，
并审计超量程裁剪、原始网格哈希和校准哈希。

```powershell
python prepare_physical_sensor_sidecars.py `
  --input data/physical-capture/raw-observations.jsonl `
  --output data/physical-capture/sensors.jsonl `
  --report data/physical-capture/BUILD_REPORT.json
```

输出继续使用现有 `normalized_grid` schema，无需修改模型张量接口。详细字段、公式、
采集验收清单和表述边界见
`reports/qwen-material/PHYSICAL_SENSOR_INTERFACE.md`；机器审计见
`PHYSICAL_SENSOR_INTERFACE_AUDIT.json/.md`。当前审计明确记录实体数据为0条、
硬件验证为 `false`，因此这里只能称“软件接入就绪”，不能称硬件已经实现。

物理转换输出保留 `source.mode=physical-array-v1`，不能直接改名冒充冻结盲测
协议中的实体来源。独立数据保管人完成来源核验后，可用下列命令把 sidecar、
物理构建报告、对象清单和保管人证明做四方哈希绑定，再发布为
`synchronized-hardware`：

```powershell
python prepare_synchronized_hardware_release.py `
  --sidecars data/physical-capture/sensors.jsonl `
  --build-report data/physical-capture/BUILD_REPORT.json `
  --manifest data/external-blind/manifest.jsonl `
  --attestation data/physical-capture/ATTESTATION.json `
  --output data/external-blind/sensors.jsonl
```

证明格式见 `schemas/synchronized_hardware_attestation_v1.schema.json`，完整发布边界
见 `reports/qwen-material/SYNCHRONIZED_HARDWARE_RELEASE.md`。该步骤只验证哈希和
保管人声明，软件仍明确记录未验证硬件来源与性能；外部盲测的 `RELEASE.json`、
图像/对象清单、双人标签共识和一次性评估仍不可省略。

当前项目总状态以
`reports/qwen-material/PROJECT_COMPLETION_AUDIT_V1.json/.md` 为工作期入口。
该审计把 Qwen 方案 A 的十一个论文门与完整专利复现、实体硬件、RGB-D 和外部部署门分开，
并动态检查确认性队列、自动论文构建、视觉复核、提交索引及证据归档是否为当前
版本。旧 `PAPER_READINESS_AUDIT`、稳定正文审计和 `SUBMISSION_INDEX` 保留为
不可变历史证据，但不再代表方案A最终交付状态。方案A使用独立的
`PLAN_A_PAPER_READINESS_AUDIT.json` 和
`PLAN_A_MANUSCRIPT_STABLE_SECTIONS_AUDIT.json/.md`，最终入口固定为
`PLAN_A_SUBMISSION_INDEX.json/.md` 和 `submission-evidence-plan-a/`；总状态
审计核对其中的文件级哈希，而不只比较修改时间。

## 论文创新：跨模态一致性感知的传感器可靠性融合

本轮创新把任务定位为一般智能垃圾对象分类，不强行限定为医疗废物。现有
`data/qwen-material/sensors.jsonl` 中的 4×4 软传感器在当前阶段按已经观测到的
sidecar 传感器记录使用；后续只要真实传感器保持相同 `normalized_grid` schema，
训练、预测和缺失处理接口可以无缝替换。

训练时会在软件层对传感器网格做可配置退化：missing、noise、bias/drift 和 stuck。
退化函数返回 `degraded_grid`、`availability ∈ [0,1]` 和 `quality_target ∈ [0,1]`。
验证和普通预测默认不退化。

模型先由视觉多尺度融合特征 `v` 和视觉查询得到的类别文本上下文 `t_v` 预测传感器
潜表示分布：

```text
mu_s, log_var_s = g([v, t_v])
```

其中 `log_var_s` 被限制在配置范围内。再用实际传感器潜表示 `s` 与预测潜表示
`mu_s` 的不确定性加权差异、余弦一致性、预测不确定性和 availability 估计可靠性：

```text
r = availability * sigmoid(h(discrepancy(s, mu_s), cosine(s, mu_s), uncertainty, availability))
effective_sensor = mu_s + r * (s - mu_s)
```

因此传感器可用且一致时主要使用实际传感器；传感器缺失时 `availability=0`，
`effective_sensor` 完全退回 `mu_s`。后续 Adapter、FiLM、注意力和分类头保持原有融合结构。

联合损失保留 CE + Focal，并提供潜表示重建、可靠性校准、方差先验、原始网格
重建和可靠性排序等可配置辅助项：

```text
L_rec = mean(0.5 * (exp(-log_var_s) * (mu_s - stopgrad(s_clean))^2 + log_var_s))
L_cal = BCE(sensor_reliability, quality_target)
L_grid = SmoothL1(predicted_clean_grid, clean_grid)
L_rank = pairwise_BCE(logit(r_i) - logit(r_j), quality_i > quality_j)
L = w_ce L_ce + w_focal L_focal + w_rec L_rec + w_cal L_cal
    + w_var L_var + w_grid L_grid + w_rank L_rank
```

训练使用 clean sensor encoder feature 作为 detached 重建目标，不使用退化后的错误
传感器表示作为监督。epoch JSON 会额外输出 `sensor_reconstruction`、
`reliability_calibration`、`sensor_reliability` 和 `sensor_consistency` 分项。

论文主模型训练命令为：

```powershell
python train.py --config configs/ablations/full_cross_modal_quality.json
```

缺失传感器预测可显式传入：

```powershell
python predict.py `
  --image ..\data\images\brick_38.jpg `
  --bbox 109.5 279.5 219.5 393.5 `
  --sensor-missing
```

旧 checkpoint 缺少本轮新增的传感器预测和可靠性头时，只支持 `availability=1` 的
实际传感器直通，以保持旧分类路径行为。缺失传感器推理（`--sensor-missing` 或
`sensor_available<1`）必须使用本轮重新训练后保存的新 checkpoint。

相关开关集中在配置的 `model`、`loss` 与 `robustness` 段；当前主模型实例见
`configs/ablations/full_cross_modal_quality.json`：

- 关闭训练退化：`robustness.sensor_degradation.enabled=false`
- 关闭缺失/噪声/偏移/卡死：对应概率设为 `0`
- 关闭潜表示重建：`loss.reconstruction_weight=0`
- 关闭可靠性校准：`loss.calibration_weight=0`
- 关闭网格重建：`loss.sensor_grid_reconstruction_weight=0`
- 关闭可靠性排序：`loss.calibration_ranking_weight=0`
- 关闭跨模态网格质量分支：`model.sensor_cross_modal_quality_enabled=false`
- 启用温和类别平衡：`loss.class_weighting.method=sqrt_inverse`
- 平衡实验按 Macro F1 选检查点：`training.selection_metric=macro_f1`
- 保留网格绝对响应水平：`model.sensor_standardize=false`
- 调整不确定性范围：`model.sensor_log_variance_min/max`

限制：当前传感器仍是模拟观测，不代表真实物理传感器精度。论文实验可以把它作为
软件观测 sidecar 研究鲁棒融合，但若要声明真实部署效果，仍必须用同步采集的真实
压力、光谱、湿度、电导率等数据重新训练、校准和评估。

原始专利与当前对象级分类实现的逐项对应、未实现的检测/实例分割边界，以及两条
后续研究路线见 `reports/qwen-material/PATENT_ALIGNMENT.md`；20项拆分要求、
实现证据和禁止外推见 `reports/qwen-material/PATENT_REQUIREMENT_MATRIX.md`。
可直接用于论文限制章节的构念、内部、数据、统计、外部、专利与复现有效性边界见
`reports/qwen-material/THREATS_TO_VALIDITY.md`。
19 类训练分布、逐类测试误差和默认关闭的类别权重方案见
`reports/qwen-material/CLASS_BALANCE.md`。
正式运行的硬件、CUDA/PyTorch/Ultralytics 版本和随机性边界见
`reports/qwen-material/ENVIRONMENT.md`。
与当前代码逐项一致的张量路径、可靠性公式、停止梯度位置和联合损失定义见
`reports/qwen-material/METHOD_SPEC.md`。
逐划分源图数、19 类分布、Qwen/rule 回退和数据使用边界见
`reports/qwen-material/DATASET_CARD.md`。
