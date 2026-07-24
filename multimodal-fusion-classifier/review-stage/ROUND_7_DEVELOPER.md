# Round 7 Developer Report

## Scope

本轮只修独立审查指出的小问题：

- `mmfc/model.py`
- `mmfc/losses.py`
- `predict.py`
- `tests/test_model.py`
- `README.md`
- `review-stage/ROUND_7_DEVELOPER.md`

未触碰 `build_dataset.py`，未回退他人改动。

## Implemented

1. legacy checkpoint 行为兼容：
   - `_load_model_state_compatible()` 现在区分完整新 checkpoint、完整 legacy checkpoint 和 partial missing。
   - 仅当全部新增 sensor predictor/reliability head key 都缺失时启用 legacy passthrough。
   - 部分新增 head 缺失直接拒绝。
   - legacy passthrough 不写入 checkpoint state，只是模型运行时 bool 标志。
   - legacy 且 `availability=1` 时 `sensor_reliability=1`，`effective_sensor` 精确等于 actual sensor。
   - legacy 且 `sensor_available<1` / `--sensor-missing` 会明确报错，避免使用未训练 predicted sensor。
2. 新 checkpoint 行为保持：
   - 完整新 state 加载后关闭 legacy 标志。
   - 缺失传感器仍走 predicted fallback 路径。
3. loss mask NaN 修复：
   - `_mask_mean()` 改为 bool 索引聚合。
   - reconstruction loss 在计算 heteroscedastic NLL 前先按有效 mask 构造 safe tensors，避免 masked-out NaN 在反传中污染梯度。
4. 测试补强：
   - legacy available exact passthrough。
   - legacy missing 拒绝。
   - partial missing 拒绝。
   - full new state 正常。
   - masked-out NaN 不污染 reconstruction loss，合法样本仍有梯度。
5. README：
   - 说明旧 checkpoint 只支持有传感器直通，缺失推理必须使用本轮重新训练的新 checkpoint。
   - 测试总数更新为 72。

## Verification Evidence

解释器：

```text
C:\Users\29513\.conda\envs\yolov8\python.exe
```

定向测试：

```text
pytest -q tests/test_model.py --basetemp=.pytest-tmp-innovation
36 passed in 4.80s
```

全量回归：

```text
pytest -q --basetemp=.pytest-tmp-innovation
72 passed in 5.42s
```

测试按既有方式临时追加 `D:\Anaconda3\Lib\site-packages` 到 `sys.path` 以加载 pytest。

编译检查：

```text
python -m compileall -q mmfc train.py predict.py
exit code 0
```

JSON 解析：

```text
configs/default.json ok
```

## Known Limits

- 未运行真实 YOLO/CLIP 生产训练或预测冒烟。
- legacy checkpoint 只保证 `availability=1` 的旧传感器直通路径；缺失传感器推理需要重新训练后的新 checkpoint。
