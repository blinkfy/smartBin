# Round 3 Developer Report

## Scope

本轮仅修改与验证以下范围：

- `mmfc/sensors.py`
- `build_dataset.py`
- `validate_dataset.py`
- `mmfc/training.py`
- `tests/test_sensors.py`
- `tests/test_model.py`

## Implemented

1. 在 `mmfc/sensors.py` 增加共享的递归泄漏键检测：
   - 覆盖 `label`、`labels`、`class`、`classes`、`target`、`targets`、
     `target_label`、`class_name`、`category`、`category_id`。
   - 键名比较不区分大小写。
   - 递归检查嵌套对象和数组。
2. `build_dataset.resume_record_usable()` 在判断 rule/qwen 模式之前拒绝任意层级
   含泄漏键的旧记录，避免 `--resume` 复用污染数据。
3. `validate_dataset.py` 导入并使用同一检测函数，生成与校验采用统一规则。
4. `ManifestDataset` 加载 sensor sidecar 时统一解析 `sample_id`/`id`，遇到重复
   ID 立即抛出带文件、行号和重复 ID 的 `ValueError`。
5. 增加回归测试：
   - 10 个泄漏键的大小写、递归结构，以及 rule/qwen 的拒绝行为。
   - clean rule/qwen 记录仍可复用。
   - `sample_id` 与 `id` 别名之间的重复 sidecar ID 被拒绝。
   - tiny 注入模型和内存 batch 的单轮 train/eval 冒烟测试；不加载 YOLO/CLIP，
     不调用 Qwen；验证指标有限、训练权重更新、评估阶段不更新权重。

## Verification Evidence

解释器：

```text
C:\Users\29513\.conda\envs\yolov8\python.exe
```

测试启动前将 `D:\Anaconda3\Lib\site-packages` 追加到 `sys.path`。由于沙箱不允许
访问系统 pytest 临时目录，测试使用项目内
`.pytest-tmp-round3` 作为 `TEMP`、`TMP` 和 `--basetemp`。

针对性测试：

```text
pytest -q tests/test_sensors.py tests/test_model.py
32 passed in 4.77s
```

全量测试：

```text
pytest -q
38 passed in 4.08s
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

本轮测试未下载或实例化 YOLO/CLIP 模型，未发起真实 Qwen 请求，也未写入 API Key。
