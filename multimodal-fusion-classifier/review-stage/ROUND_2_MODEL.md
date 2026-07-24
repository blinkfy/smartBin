# Round 2 模型与集成修复

## 完成项

1. `mmfc/model.py`
   - 新增外部 Ultralytics 隔离导入：导入期间临时排除解析到当前 cwd 和
     `SMARTBIN_ROOT` 的 `sys.path` 项，并在结束后原样恢复。
   - 若 `sys.modules` 中任一 `ultralytics` 包模块已来自
     `SMARTBIN_ROOT/ultralytics`，立即清晰报错，不删除、不替换已加载模块。
   - 校验实际 `ultralytics.__file__` 不属于本地旧源码，并要求版本
     `>=8.3,<9`；生产 smoke 使用已测试版本 `8.4.70`。
   - `freeze_yolo_layers` 与 `freeze_clip_text_layers` 从 JSON 配置透传到两个
     生产编码器，不再硬编码。
2. `mmfc/training.py`
   - 内嵌传感器记录优先使用顶层 `normalized_grid`。
   - 图像在按 bbox 裁剪前调用 `ImageOps.exif_transpose`。
   - 清单同时包含 `label` 和 `label_id` 时，强制二者与配置类别顺序一致；
     未知 label、越界或冲突均拒绝加载。
3. `predict.py`
   - 先加载 checkpoint，再构建生产模型。
   - 要求 checkpoint 包含有效 `class_names`，并与当前配置 `classes` 顺序
     完全相同；不一致时在推理和输出前失败。
   - 校验通过后，输出标签及概率字典使用 checkpoint 中的类别名称。
4. `tests/test_model.py`
   - 新增本地预加载包不变性、cwd/sys.path 隔离、版本范围、冻结层透传、
     内嵌归一化阵列优先级、类别映射冲突、EXIF 裁剪和预测配置重排回归测试。

未修改 `SMARTBIN_ROOT/ultralytics` 原始源码或任何外部安装包。

## 验证证据

### 模型定向测试

环境：`C:\Users\29513\.conda\envs\yolov8\python.exe`（Python 3.9，
通过追加现有纯 Python pytest 路径运行）。

结果：

```text
tests/test_model.py
15 passed in 3.99s
```

并通过：

```text
python -m compileall -q mmfc predict.py tests/test_model.py
```

### 从 smartBin cwd 的真实生产 smoke

工作目录：`F:\projects\fentouxia\smartBin`

使用同一 `yolov8` Python 实际导入环境包、加载 `yolo11s-seg.pt`，并以
`[1,3,160,160]` 张量完成三尺度前向：

```text
ultralytics_version=8.4.70
ultralytics_path=C:\Users\29513\.conda\envs\yolov8\lib\site-packages\ultralytics\__init__.py
source_indices=(16, 19, 22)
frozen_prefix=[True, True, True, True, True]
feature_shapes=[(1, 512), (1, 512), (1, 512)]
```

### 全量测试状态

本次模型改动测试均通过；当时全量结果为 `24 passed, 2 failed`。两项失败
均在并行数据侧文件 `build_dataset.py:145`，原因是 Python 3.9 不支持
`Path.write_text(..., newline=...)`，与本模型改动无关，已通知主代理协调
数据侧修复。
