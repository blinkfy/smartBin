# 多模态融合分类器

这是一个与原 `smartBin` 系统完全隔离的一般智能垃圾对象级分类子项目。它参考历史专利
《基于多模态识别的医疗废品识别方法》实现视觉、候选类别文本和 4×4
传感器阵列的融合分类，但默认论文定位不再限定为医疗废品；同时直接使用原数据路径，不复制原图，也不修改
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
- YOLO11s-seg 三尺度视觉特征、CLIP ViT-B/16 全候选类别文本原型、
  4×4→16×16 传感器编码。
- 三路 Adapter、FiLM、余弦模态权重、512→128 下投影、8 头模态双向注意力、
  128→512 上投影、残差和 0.3/0.5/0.2 初始化的动态多尺度融合。
- Cross Entropy + Focal Loss、AdamW、5 epoch warmup 和余弦退火。
- 专利 S1 的双边滤波、暗图 CLAHE、Laplacian 锐化、可配置仿射变换和
  480×480 缩放。

文本分支不会把样本真实标签作为输入。它始终编码全部候选类别描述，再由
视觉和传感器观测查询文本原型库；传感器模拟器的 API 同样没有 label 参数。

## 目录

```text
multimodal-fusion-classifier/
├─ configs/                  # 类别描述与专利默认参数
├─ data/generated/           # JSONL 清单与模拟传感器（无图片）
├─ mmfc/                     # 数据、模拟器、模型、损失与训练代码
├─ review-stage/             # 计划和各轮验证证据
├─ tests/
├─ build_dataset.py
├─ validate_dataset.py
├─ train.py
└─ predict.py
```

## 环境安装

建议在本子目录创建独立虚拟环境，避免影响原版系统：

```powershell
cd F:\projects\fentouxia\smartBin\multimodal-fusion-classifier
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
```

如需特定 CUDA 版本，请先按 PyTorch 官方方式安装匹配的 `torch/torchvision`，
再安装其余依赖。生产视觉分支使用隔离环境中的 `ultralytics==8.4.70`，
只读取 `../yolo11s-seg.pt`；不会导入仓库内已有的旧 `ultralytics` 源码。

## 构建与校验数据

现有完整规则数据已经生成。如需可复现地重建：

```powershell
python build_dataset.py
python validate_dataset.py
```

当前产物统计：

- 5,990 张有效源图；
- 8,654 个对象样本和 8,654 条传感器记录；
- train/val/test 为 7,002/826/826；
- 19 类；
- 182 个缺图标注和 1 个非 polygon shape 被明确记录在
  `data/generated/build_report.json`；
- `data/generated/` 中没有图片副本。

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

全量增强会按对象产生最多 8,654 次模型请求，请先确认费用和限流。正式执行：

```powershell
python build_dataset.py `
  --sensor-mode qwen `
  --resume `
  --workers 4 `
  --progress-every 10 `
  --checkpoint-every 20
```

`--resume` 只复用已经成功的当前 Qwen 模型记录；规则记录或失败回退记录会
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
当前有 8,654 个对象，全量 Qwen 增强最多会产生 8,654 次模型请求；正式跑全量前
先用上面的 `--limit 20` smoke 命令确认费用、限流和输出 schema。

## 训练

```powershell
python train.py --config configs/default.json
```

默认参数来自专利实施例：

- batch size 16；
- 初始学习率 `1e-4`；
- AdamW，weight decay `1e-5`；
- 100 epochs；
- 5 epochs warmup；
- CE + Focal；
- YOLO 前 5 层冻结，CLIP 文本 Transformer 前 3 层冻结。

检查点写入本子目录的 `checkpoints/`，不会进入原版 `runs/`。

## 单目标预测

预测需要原图、目标框和真实或模拟的 4×4 传感器数据，不需要类别文本或真实标签：

```powershell
python predict.py `
  --image ..\data\images\brick_38.jpg `
  --bbox 109.5 279.5 219.5 393.5 `
  --sensor-json .\example-sensor.json `
  --checkpoint .\checkpoints\best.pt
```

`sensor-json` 可直接使用一条 `data/generated/sensors.jsonl` 记录拆出的 JSON
对象；训练和预测都会优先读取 `normalized_grid`。

## 测试

```powershell
python -m pytest -q
python -m compileall -q mmfc build_dataset.py validate_dataset.py train.py predict.py
```

测试不下载 YOLO 或 CLIP 权重。生产冒烟已经验证本地
`yolo11s-seg.pt` 可通过 `ultralytics 8.4.70` 捕获三个 512 维尺度。
最终回归结果为 72 项测试通过；实际数据校验为 8,654 个样本、
8,654 条传感器记录、0 个错误和 0 个警告。

## 重要限制

模拟传感器数据只是用于完成数据管线、模型开发和接口联调。它与输入图像
相关，仍不是独立的真实物理观测；基于它得到的验证准确率不能作为真实部署
精度或专利指标的证据。上线前必须用同步采集的真实压力、光谱、湿度、电导率
等传感器数据替换或校准模拟数据，并重新划分独立测试集。

## 论文创新：跨模态一致性感知的传感器可靠性融合

本轮创新把任务定位为一般智能垃圾对象分类，不强行限定为医疗废物。现有
`data/generated/sensors.jsonl` 中的 4×4 模拟传感器在当前阶段按已经观测到的
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

联合损失保留 CE + Focal，并新增两项可配置辅助项：

```text
L_rec = mean(0.5 * (exp(-log_var_s) * (mu_s - stopgrad(s_clean))^2 + log_var_s))
L_cal = BCE(sensor_reliability, quality_target)
L = w_ce L_ce + w_focal L_focal + w_rec L_rec + w_cal L_cal
```

训练使用 clean sensor encoder feature 作为 detached 重建目标，不使用退化后的错误
传感器表示作为监督。epoch JSON 会额外输出 `sensor_reconstruction`、
`reliability_calibration`、`sensor_reliability` 和 `sensor_consistency` 分项。

训练命令不变：

```powershell
python train.py --config configs/default.json
```

缺失传感器预测可显式传入：

```powershell
python predict.py `
  --image ..\data\images\brick_38.jpg `
  --bbox 109.5 279.5 219.5 393.5 `
  --sensor-missing `
  --checkpoint .\checkpoints\best.pt
```

旧 checkpoint 缺少本轮新增的传感器预测和可靠性头时，只支持 `availability=1` 的
实际传感器直通，以保持旧分类路径行为。缺失传感器推理（`--sensor-missing` 或
`sensor_available<1`）必须使用本轮重新训练后保存的新 checkpoint。

消融开关集中在 `configs/default.json`：

- 关闭训练退化：`robustness.sensor_degradation.enabled=false`
- 关闭缺失/噪声/偏移/卡死：对应概率设为 `0`
- 关闭潜表示重建：`loss.reconstruction_weight=0`
- 关闭可靠性校准：`loss.calibration_weight=0`
- 调整不确定性范围：`model.sensor_log_variance_min/max`

限制：当前传感器仍是模拟观测，不代表真实物理传感器精度。论文实验可以把它作为
软件观测 sidecar 研究鲁棒融合，但若要声明真实部署效果，仍必须用同步采集的真实
压力、光谱、湿度、电导率等数据重新训练、校准和评估。
