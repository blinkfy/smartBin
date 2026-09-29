# 文生图模型选择

在 `textToImage.js` 顶部修改常量，刷新页面即可切换：

```js
export const TEXT_TO_IMAGE_MODEL_ID = 'sdxs-512-dreamshaper';
export const LCM_INFERENCE_STEPS = 4;
```

- `lcm-tiny-sd`：实验选项，LCM 少步推理；从 4 步开始，可对比 2 / 8 步。
- `sdxs-512-dreamshaper`：默认选项，原来的单步模型，权重、预置文本嵌入和推理路径保留。

首次准备 LCM 权重：

```powershell
python scripts/prepare_lcm_tiny_sd.py
```

准备脚本需要 `onnx`，从 `akameswa/lcm-tiny-sd-onnx-fp16` 下载 FP16 UNet，
从 `akameswa/lcm-tiny-sd` 下载匹配的 CLIP-L、分词器和调度器配置。
每次准备记录使用的上游 revision 和文件 SHA256。
首次加入 Python 路由后需要重启模拟器，之后修改前端选择常量只需刷新页面。

LCM 与 SDXS 共用原来的轻量 TAESD 解码器，LCM 不下载标准 VAE。
LCM UNet 约 617 MiB，浏览器生成时仍只加载选中的一套 UNet。
由于保留旧模型，磁盘上会同时存在两套 UNet 和各自的文本编码器。
预置按钮仍使用同一组描述；LCM 通过自己的文本编码器生成嵌入，
SDXS 继续读取原来的预置嵌入。

LCM 使用自己的调度器和标准正态初始噪声；TAESD 使用 scaling_factor=1.0，
直接解码扩散潜变量，不能套用标准 VAE 的 0.18215 缩放。
4 步表示运行 UNet 四次，文本只编码一次、图片只解码一次。
这是社区提供的 Tiny-SD LCM 权重，实际物品生成质量需要对比，不能保证所有描述优于 SDXS。

数值校验（使用已安装 torch、diffusers、numpy 的 Python 环境）：

```powershell
python scripts/verify_lcm_scheduler.py
```

参考：[LCM-Tiny-SD](https://huggingface.co/akameswa/lcm-tiny-sd)、
[FP16 ONNX](https://huggingface.co/akameswa/lcm-tiny-sd-onnx-fp16)。
