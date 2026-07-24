# Round 4 Developer Report

## Scope

本轮仅修改与验证以下范围：

- `mmfc/sensors.py`
- `predict.py`
- `tests/test_sensors.py`
- `tests/test_model.py`
- `review-stage/ROUND_4_DEVELOPER.md`

## Implemented

1. 扩展共享递归泄漏键集合：
   - 保留既有 `category_id`。
   - 新增 `label_id`、`class_id`、`target_id`。
   - 新增 `label_index`、`class_index`、`target_index`、
     `category_index`。
   - 继续支持不区分大小写和对象/数组任意层级递归检查。
2. 回归测试覆盖数字值泄漏：
   - `contains_leakage_key()` 对所有数字 ID/index 别名均能识别。
   - `build_dataset.resume_record_usable()` 的 rule 和 qwen 路径均拒绝
     污染记录，clean 记录仍可复用。
   - 使用实际最小生成数据调用 `validate_dataset.validate_generated()`，
     验证 validator 经共享检测器拒绝所有数字别名，clean 数据仍通过。
3. 预测 checkpoint 只调用
   `torch.load(..., weights_only=True)`：
   - 删除 `weights_only=False` 及无参数重试。
   - PyTorch 不支持 `weights_only` 时直接抛出清晰 `RuntimeError`，要求升级。
   - 保留 dict/`model_state` 格式校验及既有 `class_names` 顺序校验。
   - monkeypatch 测试确认唯一调用携带 `weights_only=True`，且不支持路径只调用
     一次并拒绝继续加载。
4. 预测 CLI 的传感器网格增加 `[0,1]` 范围校验，负值或大于 1 的值直接拒绝。

## Verification Evidence

解释器：

```text
C:\Users\29513\.conda\envs\yolov8\python.exe
```

测试通过在启动代码中把 `D:\Anaconda3\Lib\site-packages` 追加到 `sys.path`
来加载 pytest，避免覆盖 yolov8 环境自身的 NumPy/Pillow 二进制依赖。临时目录统一为
项目内 `.round4-tmp`。

针对性测试：

```text
pytest -q tests/test_sensors.py tests/test_model.py --basetemp=.round4-tmp
51 passed in 4.56s
```

全量测试：

```text
pytest -q --basetemp=.round4-tmp
57 passed in 4.55s
```

编译检查：

```text
compileall.compile_dir(".", quiet=1)
exit code 0
```

数据校验：

```text
sample_count: 8654
sensor_count: 8654
source_image_count: 5990
error_count: 0
warning_count: 0
valid: true
```

## External-service Isolation

本轮未发起真实 Qwen/API 请求，未下载或实例化外部模型，未写入 API Key。
