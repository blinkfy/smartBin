# Round 3 独立功能审查

## Score

- Overall: 4.1/5
- Requirement fit: 4.5
- Correctness: 4.1
- Completeness/edge cases: 4.0
- Maintainability: 4.2
- Evidence: 4.7
- Isolation: 4.6

Decision: continue。证据充分，但发现两个新的安全边界缺口。

## Critical Issues

1. 共享泄漏键集合未包含 `label_id`、`class_id`、`target_id` 等数字类别别名，
   因而旧 sidecar 仍可能在 resume 时被复用并通过验证。
2. `predict.py` 对 CLI 指定的 checkpoint 使用
   `torch.load(weights_only=False)`，会启用不安全 pickle 反序列化。

## Required Revisions

1. 扩展数字 ID/index 泄漏键，并为 resume 与 validator 增加回归测试。
2. 预测只使用 `weights_only=True` 安全加载，拒绝旧版不支持或不安全格式；
   增加定向测试。

## Verified Evidence

- 全量测试 38 passed。
- 数据验证 8,654 samples/sensors，0 error/warning。
- 外部 Ultralytics + 本地 YOLO11s-seg 三尺度生产前向成功。
- `data/generated` 无图片副本。

