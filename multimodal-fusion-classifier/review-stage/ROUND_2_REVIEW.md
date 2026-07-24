# Round 2 独立功能审查

## Score

- Overall: 4.1/5
- Requirement fit: 4.5
- Correctness: 4.1
- Completeness/edge cases: 3.9
- Maintainability: 4.0
- Evidence: 3.7
- Isolation: 4.6

Decision: continue。当前工件没有确认的关键缺陷，但尚未达到 4.3 停止线。

## Important Revisions

1. `resume_record_usable()` 还应拒绝包含 label/class/target 等泄漏字段的旧
   sensor 记录，而不是等完整生成后再由 validator 报错。
2. 增加子项目本地 pytest 配置，防止继承父 `smartBin/pyproject.toml` 的
   `--doctest-modules`。
3. 扩展泄漏键检测，并在 sidecar 加载阶段拒绝重复 sample ID。
4. 使用本机可用的隔离 PyTorch/Ultralytics 解释器再次运行全量测试和轻量
   训练/预测契约测试，形成可复现证据。

## Verified Evidence

- 8,654 samples/sensors、5,990 source images、19 classes。
- 数据校验 0 error/warning，split 泄漏 0，sensor 缺失/孤立 0。
- `data/generated` 无图片副本。
- 类别配置、manifest 与 index 映射一致。
- 数据/Qwen 子集测试 11 passed，compileall 通过。

