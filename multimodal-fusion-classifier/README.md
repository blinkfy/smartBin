# 多模态融合分类器

这是一个与原 `smartBin` 系统完全隔离的对象级分类子项目。它按照专利
《基于多模态识别的医疗废品识别方法》实现视觉、候选类别文本和 4×4
传感器阵列的融合分类，同时直接使用原数据路径，不复制原图，也不修改
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
  --output data/qwen-smoke
python validate_dataset.py --output data/qwen-smoke
```

全量增强会按对象产生最多 8,654 次模型请求，请先确认费用和限流。正式执行：

```powershell
python build_dataset.py --sensor-mode qwen --resume
```

`--resume` 只复用已经成功的当前 Qwen 模型记录；规则记录或失败回退记录会
继续尝试增强。没有 `DASHSCOPE_API_KEY` 时不会发起网络请求，每个样本自动
回退到规则模拟。业务空间专属地址可用 `--base-url` 指定，模型名可用
`--model` 指定。

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
最终回归结果为 57 项测试通过；实际数据校验为 8,654 个样本、
8,654 条传感器记录、0 个错误和 0 个警告。

## 重要限制

模拟传感器数据只是用于完成数据管线、模型开发和接口联调。它与输入图像
相关，仍不是独立的真实物理观测；基于它得到的验证准确率不能作为真实部署
精度或专利指标的证据。上线前必须用同步采集的真实压力、光谱、湿度、电导率
等传感器数据替换或校准模拟数据，并重新划分独立测试集。
