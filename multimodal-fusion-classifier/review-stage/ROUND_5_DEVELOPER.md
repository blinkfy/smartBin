# Round 5 Developer Report

## Scope

本轮实现“跨模态一致性感知的传感器可靠性融合”，修改范围：

- `mmfc/model.py`
- `mmfc/training.py`
- `mmfc/losses.py`
- `predict.py`
- `configs/default.json`
- `tests/test_model.py`
- `README.md`
- `review-stage/ROUND_5_DEVELOPER.md`

保留并适配既有 `forward_from_features(...)`、`criterion(logits, labels)` 和正常传感器预测调用。

## Implemented

1. 训练期纯软件传感器退化：
   - 新增 `degrade_sensor_grid()`，支持 missing、noise、bias/drift、stuck。
   - 返回 `degraded_grid`、`availability [B,1]`、`quality_target [B,1]`。
   - 固定 `torch.Generator` 可复现；验证阶段默认不退化。
2. 模型可靠性融合：
   - 由视觉融合特征和视觉查询文本上下文预测 `predicted_sensor` 与受限 `sensor_log_variance`。
   - 用实际/预测潜表示差异、余弦一致性、不确定性和 availability 估计 `sensor_reliability`。
   - `effective_sensor = predicted + reliability * (actual - predicted)`；缺失时 reliability 为 0，完全退回预测潜表示。
   - aux 新增 `sensor_reliability`、`predicted_sensor`、`sensor_log_variance`、`sensor_available`、`sensor_consistency` 等。
3. 损失与训练记录：
   - `JointClassificationLoss` 保留 CE + Focal，新增 heteroscedastic reconstruction 与 reliability calibration 可选项。
   - 训练使用 clean sensor encoder feature 的 detached 输出作为重建 target。
   - epoch JSON 新增 reconstruction、calibration、reliability、consistency 指标。
4. 预测缺失传感器：
   - 新增 `--sensor-missing`，使用零网格并传入 `availability=0`。
   - 正常 `--sensor-json` / `--sensor-grid` 路径保持原行为。
   - 预测加载旧 checkpoint 时，只宽容本轮新增传感器可靠性头的缺失参数；其他不兼容 key 仍报错。
5. 配置、测试和说明：
   - `configs/default.json` 新增 model/robustness/loss 配置与消融开关。
   - 测试覆盖退化范围/缺失、模型全模态/缺失前向、可靠性范围、辅助损失反传、旧 API、预测缺失 helper、旧 checkpoint 兼容。
   - README 增加论文定位、方法公式、训练/缺失预测命令、消融和模拟数据限制。

## Verification Evidence

解释器：

```text
C:\Users\29513\.conda\envs\yolov8\python.exe
```

编译检查：

```text
python -m compileall -q mmfc train.py predict.py
exit code 0
```

定向测试：

```text
pytest -q tests/test_model.py --basetemp=.pytest-tmp-innovation
27 passed in 5.42s
```

全量回归：

```text
pytest -q --basetemp=.pytest-tmp-innovation
63 passed in 4.93s
```

`yolov8` 环境未安装 pytest，测试命令按既有方式临时追加
`D:\Anaconda3\Lib\site-packages` 到 `sys.path`。测试后已删除 `.pytest_cache`、
`.pytest-tmp-innovation` 和本轮产生的 `__pycache__`。

## Known Limits

- 未运行真实 YOLO/CLIP 生产训练或预测冒烟，避免下载或实例化外部模型。
- 当前 sidecar 仍是模拟传感器观测；论文实验可用于鲁棒融合验证，但真实部署结论仍需同步采集的物理传感器数据重新训练和评估。
- 工作树中已有的 `build_dataset.py` 修改和 `review-stage/INNOVATION_PLAN.md` 未跟踪项不是本轮改动，未触碰。
