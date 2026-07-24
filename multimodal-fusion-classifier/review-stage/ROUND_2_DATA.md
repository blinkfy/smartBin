# Round 2 数据映射修复

## 完成项

1. `build_dataset.py`
   - 新增 `--config`，默认使用项目内 `configs/default.json`；相对配置路径固定从
     `PROJECT_ROOT` 解析，不依赖启动目录。
   - `classes` 数组的名称及其顺序成为唯一类别映射来源；空配置、非法名称和重复名称
     都会在写产物前失败。
   - 在应用 `--limit` 前校验全部已解析标注标签；任何配置外标签都会明确失败，
     不再静默生成新类别。
   - 三个 split 清单中的每个样本都写入与配置顺序一致的 `label_id`。
   - `index.json.labels`、`label_to_index` 和每个样本的 `label_index` 均使用配置顺序。
     索引使用保序原子写入，因此 JSON 中 `label_to_index` 的键顺序也不是字母序。
   - 原有 rule/Qwen resume 判定和升级语义保持不变。
2. `validate_dataset.py`
   - 新增同一 canonical 配置入口。
   - 逐样本要求 `label` 已知、`label_id` 为非布尔整数且严格等于配置中的 ID。
   - 严格检查 `index.json.labels` 的列表顺序、`label_to_index` 的键顺序和值、
     每个索引样本的 `label_index`/`split`/`sensor_index`。
   - 配置外标签、清单 ID 冲突、索引重排或错误映射均会令校验失败。
3. `tests/test_data.py`
   - 新增真实临时构建回归，证明 `potato` 按配置映射为 0，并同时写入 manifest
     和 index。
   - 新增配置外标注标签构建失败测试。
   - 新增篡改 `index.json` 类别顺序和映射后校验失败测试。
4. Python 3.9 兼容
   - `build_dataset.py` 的保序 JSON 写入，以及额外授权修改的
     `mmfc/data.py::write_json`，均使用
     `Path.open(..., newline="\n")`，不再调用 Python 3.9 不支持的
     `Path.write_text(..., newline=...)`。

## 全量规则数据重建

在项目根目录执行无 resume 全量重建：

```text
python build_dataset.py --sensor-mode rule
```

结果：

| 项目 | 结果 |
| --- | ---: |
| 有效对象样本 | 8,654 |
| 有效源图 | 5,990 |
| train / val / test | 7,002 / 826 / 826 |
| rule 传感器记录 | 8,654 |
| 传感器生成错误 | 0 |
| resume 复用记录 | 0 |
| manifest label/label_id 不一致 | 0 |
| `data/generated` 中图像副本 | 0 |

配置顺序和产物均以 `potato` 开头；抽查及全量扫描结果为：

```text
index.json.labels[0]=potato
list(index.json.label_to_index)[0]=potato
index.json.label_to_index.potato=0
potato manifest label_id=0
manifest_label_id_mismatches=0
```

本轮产物 SHA-256：

```text
train.jsonl   35845E64BA73AD28ACB370D60EA4D24182B91DBA4C1695C8D2F60C127765A353
val.jsonl     2ED293861169D9613E80E995B3B8C14279B52B4D1A2A2089EB3CF0D0AA7E4925
test.jsonl    A01AAEA8AD40A1BCBE454D9C82B76B50433B8E806F63F97C4B14ED5203DFC0FD
sensors.jsonl D2048729FD18D3F9163B5863EC9F094FC07A047CA0C391A75AFC74CD70D8B635
index.json    1A908B557FACE4FC6A6D3E04EA12FE7807F7BBBF15CCE787C1494F117FAA54CA
```

## 校验与测试证据

全量产物校验：

```text
python validate_dataset.py

valid=true
sample_count=8654
sensor_count=8654
source_image_count=5990
error_count=0
warning_count=0
```

在 `yolov8` 的 Python 3.9.19 环境中，追加现有纯 Python pytest 路径后运行：

```text
tests/test_data.py tests/test_sensors.py
11 passed in 0.57s

tests
26 passed in 2.86s
```

同一 Python 3.9.19 解释器还实际完成了 3 样本
`build_dataset.py --limit 3` 到 `validate_dataset.py` 的端到端冒烟：

```text
sample_count=3
sensor_count=3
valid=true
error_count=0
warning_count=0
```
