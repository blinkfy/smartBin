# Round 1：数据管线实现与验证记录

## 实现范围

- `mmfc/data.py`
  - 按 UTF-8/UTF-8 BOM 读取 LabelMe JSON。
  - 将合法 polygon 的所有点转换并裁剪为 `[x1, y1, x2, y2]`。
  - 对缺失原图、损坏 JSON、非法 polygon、空标签和非 polygon shape 逐项报告并跳过。
  - 对象清单只记录原图路径和 bbox，不复制或修改图像。
  - `sample_id` 由源图 ID、标注文件、shape 序号和 bbox 确定，明确不使用标签。
  - 使用源图 ID 的带种子 SHA-256 阈值做稳定 train/val/test 分组切分。
- `mmfc/sensors.py`
  - 从内存 crop 的亮度、饱和度、纹理、4x4 灰度结构和 bbox 几何生成确定性传感器数据。
  - 保存 4x4 原始/归一化阵列、重量、温度、湿度、气味指数、电导率及生成来源元数据。
  - 模拟器 API 不接收标签；传感器记录不保存 label/class/target。
  - 可选 qwen3.5-flash 使用 OpenAI-compatible `chat/completions`、Base64 data URL 和
    `response_format={"type":"json_object"}`；提示只要求基于图像估计物理量，禁止类别识别。
  - 支持 DashScope Key 环境变量、base URL/model、超时、重试、响应校验/钳位、逐样本规则回退。
  - 缺少 Key 时不调用传输层；构建 CLI 支持 `--resume`、`--dry-run` 和 `--limit`。
  - `--resume` 具有模式感知语义：规则模式只复用结构和范围合法的 `rule-v1`；
    Qwen 模式只复用 `source.generator` 等于当前模型且
    `enhancement_status=success` 的记录。已有规则记录、无 Key fallback 和错误 fallback
    都会重新尝试增强，因此可以从完整规则数据安全地断点升级，而不会出现 0 次 Qwen 调用。
  - 训练模型应读取 `normalized_grid`：规则模式下它是规则阵列的 0-1 归一化值，
    Qwen 成功时则替换为经校验和钳位的增强 4x4 阵列；`grid` 始终保留规则模拟的
    原始 4x4 响应，便于审计和回退。
- `build_dataset.py` 生成三个 JSONL 清单、`sensors.jsonl`、`index.json`、
  `stats.json` 和 `build_report.json`。
- `validate_dataset.py` 校验 schema、原图路径、bbox、ID 唯一性、源图跨集合泄漏、
  传感器一一对应、4x4 范围、物理量范围、目标标签字段和索引一致性。

## 真实数据构建证据

在 `multimodal-fusion-classifier` 根目录执行：

```text
python build_dataset.py
```

规则构建成功，结果如下：

| 项目 | 结果 |
| --- | ---: |
| LabelMe JSON 文件 | 6,181 |
| 有效源图 | 5,990 |
| 有效对象样本 | 8,654 |
| train | 7,002 |
| val | 826 |
| test | 826 |
| 规则传感器记录 | 8,654 |
| 传感器生成错误 | 0 |
| 类别数 | 19 |

构建报告记录并跳过：

- `missing_image`: 182 个标注文件。部分文件含多个 shape，因此其缺失对象数大于 182；
- `unsupported_shape_type`: 1 个 `points` shape。

完整问题明细保存在 `data/generated/build_report.json`，没有静默丢弃。

产物 SHA-256：

```text
train.jsonl   A304DC1BB803ECBE410E2C6ED11D46274918B1EDB210E7F92B80A3DB9628B2E7
val.jsonl     55AF77A7F245F50943185FC3B45936C6794F299FF84096E0229F3E373990AB6E
test.jsonl    7FB754C55A8C525D7B8507A53C0AD03F0F8200A10E54301D8CCEF44BCF55A344
sensors.jsonl D2048729FD18D3F9163B5863EC9F094FC07A047CA0C391A75AFC74CD70D8B635
```

`data/generated` 共 7 个文件，图像文件数为 0；清单内示例路径为
`../data/images/brick_38.jpg`，由项目根目录解析，不依赖调用者 CWD。

## 校验与测试

```text
python validate_dataset.py
```

结果：

```text
valid=true
sample_count=8654
sensor_count=8654
source_image_count=5990
split_counts=train:7002,val:826,test:826
error_count=0
warning_count=0
```

这同时证明：

- 8,654 个 `sample_id` 全部唯一，清单/传感器/索引键一致；
- 同一 `source_image_id` 未跨 train/val/test；
- 8,654 个样本和传感器记录严格一一对应；
- 所有原图路径、bbox、4x4 数组及物理量范围合法；
- 传感器记录不含目标标签字段。

```text
python -m pytest -q tests/test_data.py tests/test_sensors.py
8 passed
```

测试覆盖临时 LabelMe fixture、polygon/bbox、缺图和非法 shape 报告、稳定分组切分、
规则模拟确定性与标签无关性、Qwen Base64 图片 JSON 请求、数值钳位、缺 Key 零调用、
重试后逐样本规则回退，以及 rule/Qwen/自定义模型/fallback/非法记录的模式感知 resume。
