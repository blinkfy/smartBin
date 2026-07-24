#yolo train data=wheat.yaml model=yolov8s.pt epochs=300 imgsz=640 batch=9 workers=32 device=cpu

from ultralytics import YOLO
import cv2
import torch
import numpy as np
# Load a model
model = YOLO("../runs/detect/train/weights/best.pt")  # pretrained YOLOv8n model
source=["../data/images/potato.jpg"]
# Run batched inference on a list of images
results = model(source)

img=results[0].orig_img
cnt=0
for box in results[0].boxes.xyxy:
    xyxy=box.numpy().astype(int)
    imgcut=cv2.cvtColor(img[xyxy[1]:xyxy[3],xyxy[0]:xyxy[2]],cv2.COLOR_BGR2GRAY)
    img_2=cv2.threshold(imgcut, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cv2.imshow('img2',img_2)
    conters,herarchy=cv2.findContours(img_2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.rectangle(img,(xyxy[0],xyxy[1]),(xyxy[2],xyxy[3]),(0,255,0))
    cv2.putText(img,results[0].names[int(results[0].boxes.cls[cnt])],(xyxy[0],xyxy[1]),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(0,0,0))
    cv2.putText(img,str(results[0].boxes.numpy().conf[cnt]),(xyxy[0],xyxy[1]+20),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.7,(0,0,0))
    #img=cv2.drawContours(img,conters,-1,(0,0,255),3,16)
    print(conters[0])
    for xy in conters[0][0]:
        cv2.circle(img,(xy[0],xy[1]),3,(0,0,255),-1)
    #print(conters)
    cnt+=1
cv2.imshow('img',img)
print(results[0].keypoints)
cv2.waitKey(0)
quit()

video=cv2.VideoCapture(0)
while video.isOpened():
    flag,frame=video.read()
    if flag:
        results=model(frame)
        frame=results[0].plot()
        cv2.imshow('video',frame)
    if cv2.waitKey(1)==27:
        break
video.release()