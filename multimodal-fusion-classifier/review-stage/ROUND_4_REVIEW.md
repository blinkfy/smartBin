# 第四轮独立终审

## 结论

- 决策：STOP
- 总分：4.5/5
- 关键问题：无

## 评分

- 需求符合度：4.6/5
- 正确性：4.5/5
- 完整性与边界条件：4.4/5
- 可维护性：4.4/5
- 验证证据：4.6/5
- 隔离性与安全性：4.2/5

## 已验证

- `validate_dataset.py`：8,654 个样本、8,654 条传感器记录、0 个错误、
  0 个警告。
- pytest：57 项测试通过。
- `data/generated/` 中无图片；4×4 传感器归一化值无越界；源图在
  train/val/test 间无泄漏。
- resume 与 validator 会递归拒绝 `label_id`、`class_id`、`target_id`、
  `category_id` 及其 `*_index` 变体。
- 预测检查点仅使用 `weights_only=True` 安全加载，旧版 PyTorch 不支持时
  明确拒绝，不进行不安全回退。
- canonical 类别映射、重复 sidecar、`normalized_grid` 优先级、
  Qwen fallback/resume 和外部 Ultralytics 导入隔离均有测试覆盖。

## 发布前处理

终审发现交付目录残留 pytest 临时目录与 Python 缓存。发布前已将这些目录
删除，并补充 `.gitignore` 规则，避免临时图片或 manifest 被误打包。

## 剩余风险

本轮未执行真实 YOLO/CLIP 权重的完整端到端训练或预测；已有验证覆盖数据
路径、模拟器、安全路径、融合核心，以及本地 YOLO11s-seg 的生产编码器冒烟。
