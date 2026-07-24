import os, glob, cv2,json
import numpy as np

# 配置
img_dir   = 'data\\images'     # 原图文件夹
label_dir = 'data\\Annotations'     # 对应 json 标签文件夹
label_dir2= 'data\\val'     # 对应 json 标签文件夹
out_dir   = 'labels_viz'       # 可视化输出
os.makedirs(out_dir, exist_ok=True)
class_names=['potato', 'rabbitcut', 'battery1', 'battery5', 'bottle', 'brick', 'can', 'china', 'stone', 'drug', 'drugbag', 'mooli', 'battery', 'bottle2', 'drugbox', 'papercup','capsule','potatocut','paper']
colors = {}
n = len(class_names)
for i, cls in enumerate(class_names):
    hue = int(180 * i / n)                       # OpenCV H 范围 [0,180)
    hsv = np.uint8([[[hue, 255, 255]]])          # 饱和度、明度都设 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0,0]
    colors[cls] = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
# YOLO txt 每行： class x_center y_center width height  （归一化坐标）
for img_path in glob.glob(os.path.join(img_dir, '*.jpg')):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    basename = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(label_dir, basename + '.json')
    label_path2= os.path.join(label_dir2,basename + '.json')
    if os.path.exists(label_path):
        with open(label_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif os.path.exists(label_path2):
        with open(label_path2,'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        if os.path.exists(os.path.join(label_dir2,basename + '.txt')):
            print(label_path,'only has txt')
        else:
            print(label_path)
        continue

    # shapes 是一个列表，每个元素是一个实例
    for shape in data['shapes']:
        cls = shape['label']
        pts = np.array(shape['points'], dtype=np.int32)  # e.g. [[x1,y1], [x2,y2], ...]

        # 如果 JSON 中存的是归一化坐标，需要先 * w / h
        # pts = (pts * np.array([w, h])).astype(np.int32)

        # 画多边形轮廓
        color = colors.get(cls, (125,125,255))
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)

        # 如果想填充区域：
        # cv2.fillPoly(img, [pts], color=(0,255,0,50))  # 需要 BGRA 或手工混合半透明
        # 在多边形第一个顶点处标类名
        x0, y0 = pts[0]
        cv2.putText(img, cls, (x0, y0-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    # 保存可视化结果
    cv2.imwrite(os.path.join(out_dir, basename + '.jpg'), img)