#寻找最优超参数
from ultralytics import YOLO

model = YOLO("yolov8s-seg.pt")

# Define search space
search_space = {
    "lr0": (0.007, 0.011),  #0.00716/30epoches
    "lrf": (1e-3,1e-1),
    "momentum": (0.7,0.98),
    "weight_decay": (1e-4,1e-3),
    "warmup_epochs": (1.0,4.0),
    "warmup_momentum": (0.1,0.9),
    "cls": (0.3,2.0),
    "hsv_h": (1e-4,0.1),
    "hsv_s": (1e-2,0.9),
    "hsv_v": (1e-3,0.8),
}

# Tune hyperparameters on COCO8 for 30 epochs
model.tune(
    data="coco.yaml",
    epochs=30,
    batch=108,
    iterations=100,
    imgsz=480,
    workers=18,
    optimizer="AdamW",
    space=search_space,
    plots=False,
    save=False,
    device=[0,1,2]
)
# - 启用cosine学习率：cos_lr=True

# - 降低初始学习率lr0=0.001，或者使用更小的lrf（比如0.1，让最终学习率是初始的0.1倍）

# - 增加mixup=0.1或copy_paste=0.2以增强数据

# - 调整close_mosaic=30，在后期关闭mosaic增强

# - 如果显存足够，增加batch size到64

# - 增加weight_decay到0.001以增强正则化

# - 使用梯度裁剪（如果参数支持）

# 学习率调整：

# 启用余弦退火配合lrf=0.2可产生周期性震荡，帮助跳出局部最优

# 保持较高初始学习率配合AdamW优化器提升收敛速度

# 数据增强平衡：

# 降低颜色增强强度（hsv参数）减少噪声干扰

# 增加几何形变增强（旋转/剪切）提升几何鲁棒性

# 控制mosaic概率避免过度扭曲目标

# 显存优化：

# 通过增大batch size到64可使GPU利用率提升20%+

# 提升分辨率到640能更好捕捉细节特征

# 正则化增强：

# Dropout和权重衰减配合防止过拟合

# 调整损失权重平衡定位与分类任务

