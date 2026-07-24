# Round 1 独立功能审查

## Score

- Overall: 4.0/5
- Requirement fit: 4.2
- Correctness: 3.8
- Edge cases: 3.7
- Maintainability: 4.0
- Evidence/tests: 4.2
- Isolation: 3.4

Decision: continue。

## Critical Issues

1. `ultralytics` 导入隔离不够稳定。从 `smartBin` 根目录启动时，本地旧
   `smartBin/ultralytics` 可能先于隔离环境包被导入，造成加载失败或行为不一致。
2. 类别映射有两个顺序来源。`index.json` 使用字母序，而训练/预测使用
   `configs/default.json` 顺序；`potato` 分别为 15 和 0，存在静默错标风险。
   预测也尚未校验 checkpoint 中的 `class_names` 与当前配置顺序。

## Important Revisions

1. 生产导入时屏蔽 `smartBin` 根路径并拒绝已加载的本地旧 Ultralytics，校验
   实际包路径和兼容版本；从 `smartBin` cwd 做生产冒烟。
2. 以 `configs/default.json` 为唯一类别顺序，构建时写入同序 `label_id` 和
   `index.json`；训练校验 label/label_id 一致，预测拒绝 checkpoint/config
   类别顺序不一致。
3. 内嵌传感器记录也应优先识别顶层 `normalized_grid`。
4. 将配置里的 YOLO/CLIP 冻结层数真正透传给编码器。
5. 训练裁剪路径与构建/模拟器统一应用 EXIF orientation。

## Evidence Reviewed

- 全量数据校验：8,654 samples/sensors，5,990 source images，0 error/warning。
- 生成目录中无图片副本。
- Qwen 无 Key、JSON 校验、重试、fallback 和 resume 有自动测试。
- 全部测试在可用 PyTorch 环境中为 15 passed。

