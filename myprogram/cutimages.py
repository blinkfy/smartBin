import cv2
import os
for file in os.listdir(r"E:\大学日志\2024.07\集训\垃圾分类学长材料\数据集\Images"):
    print(file)
    img=cv2.imread("Images\\"+file)
    img=img[:,80:-80]#640 480 to 480 480
    cv2.imwrite("image\\"+file,img)
    print(img.shape)