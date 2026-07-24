# Round 1：模型、训练与推理实现记录

## 已实现

- `mmfc/model.py`
  - 生产视觉编码器通过隔离环境安装的兼容 `ultralytics` 包加载仓库本地 `../yolo11s-seg.pt`，冻结前 5 层，并通过分割头的 `head.f` 捕获、池化三个主干尺度；不导入或修改仓库既有 Ultralytics 源码。
  - 生产文本编码器使用 CLIP ViT-B/16 的 512 维文本投影，一次编码全部候选类别描述，冻结文本 Transformer 前 3 层。
  - 文本上下文查询只来自视觉与传感器观测，使用候选原型 bank 的 softmax 加权和；`forward` 不含 label/target 参数。
  - 传感器路径接受 `[B,1,4,4]`，逐样本归一化、上采样到 16×16、1×1 卷积到 512 维并池化。
  - 三个独立 Adapter、LeakyReLU、Dropout(0.3)、FiLM 初值 γ=1/β=0.1、两两余弦模态权重、512→128、8 头跨模态双向注意力、128→512、残差、三尺度动态视觉细化（先验 0.3/0.5/0.2）和分类头。
  - Sensor 1×1 Conv 使用 Kaiming normal，Adapter 使用 Xavier normal，512→128/128→512 使用 Xavier normal；以上 bias 均初始化为 0。
  - `forward_from_features` 可绕开全部可选生产依赖。
- `mmfc/losses.py`
  - Cross Entropy + 多分类 Focal 联合损失；支持标量/逐类别 alpha、gamma 和两项损失权重。
- `mmfc/training.py`、`train.py`
  - JSONL 数据读取，兼容内嵌/sidecar/独立 JSON 的常见传感器 grid 字段。
  - 优先读取 `normalized_grid`（兼容原始 `grid`），并统一使用 `sample_id`/`id` 查找 sidecar。
  - 只在内存中按 bbox 裁剪原图；路径相对子项目稳定解析。可配置 S1 包括 d=3、sigmaSpace=1.0、sigmaColor=0.8 的双边滤波，低亮度 CLAHE（clipLimit=2、8×8），3×3 Laplacian 锐化，可选 2×3 affine 以及最终 resize；关闭时不导入 OpenCV。
  - AdamW、5 epoch warmup、余弦退火、验证准确率以及 best/last 检查点。
- `predict.py`
  - 仅接受 image、bbox 和传感器 JSON/grid，不接受目标标签或目标文本。
- `configs/default.json`
  - 记录 19 类候选描述以及专利默认超参数。

## 验证证据

- `tests/test_model.py` 只依赖 torch（间接代码路径兼容 numpy/Pillow），不加载或下载 YOLO/CLIP 权重。
- 覆盖 logits/注意力/模态权重/尺度权重形状与归一化、尺度先验、FiLM 初值、全候选描述编码、前向签名无标签参数、传感器形状检查以及联合损失反向传播。
- 使用 `C:\Users\29513\.conda\envs\LLMs\python.exe`（PyTorch 2.10.0+cpu）执行真实 pytest，最终结果 `7 passed in 3.31s`。由于该环境未单独安装 pytest，命令将 base 环境纯 Python pytest 包追加到 `sys.path`：

  ```powershell
  @'
  import sys
  sys.path.append(r'D:\Anaconda3\Lib\site-packages')
  import pytest
  raise SystemExit(pytest.main(['tests/test_model.py', '-q']))
  '@ | C:\Users\29513\.conda\envs\LLMs\python.exe -
  ```

- `python -m compileall -q mmfc train.py predict.py` 与 `python -m json.tool configs/default.json` 均通过。
- 额外验证 `import mmfc` 不会导入 torch，数据 CLI 不再被模型可选依赖耦合。
- 使用 `C:\Users\29513\.conda\envs\yolov8\python.exe`、已安装 `ultralytics 8.4.70` 和仓库本地权重完成视觉生产冒烟：捕获来源 `(16, 19, 22)`，输出三个 `(1,512)` 特征，前 5 层均冻结。
- 读取本轮实际生成数据首样本成功：train 共 7002 条，图像 `(3,480,480)`，传感器 `(1,4,4)` 且范围 `[0,1]`，证明正式 sidecar 的 `normalized_grid` 已进入训练路径。

## 已知限制

- 生产 YOLO/CLIP 冒烟测试取决于本机可选依赖与 CLIP 权重缓存/网络；缺失时实现会给出明确错误，不影响纯融合单测。
- 仓库既有 `ultralytics/cfg/default.yaml` 无法被 PyYAML 解析，因此生产编码器刻意不把 `smartBin` 注入 `sys.path`，而依赖隔离环境 requirements 安装的兼容包；本地权重路径仍固定在仓库内。
- base Python 3.12 用户目录中的 PyTorch 因 `c10.dll` 报 WinError 1114，故测试改用上面的 LLM conda Python；这是解释器环境问题而非本子项目测试失败。
- Ultralytics 三尺度捕获依赖当前项目模型结构公开的 `head.f`；若未来替换为不兼容的模型头，会快速失败而不是静默选错层。
- 本轮不负责数据构建和模拟器文件；训练读取器已为清单内嵌或 sidecar 传感器数据保留兼容入口。
