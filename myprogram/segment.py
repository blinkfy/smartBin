from ultralytics import YOLO
import cv2
import math
import numpy as np

# Load a model
model = YOLO("runs/segment/best.pt")
# Run batched inference on a list of images
results=model(r"data/combination/group0.jpg")
img=results[0].plot()
xylist=results[0].boxes.xyxy
masks=results[0].masks.xy
for i in range(len(masks)):
    mask=masks[i]
    xyxy=xylist[i]
    rows,cols = img.shape[:2]#直线拟合
    try:
        dx,dy,x,y = cv2.fitLine(mask, cv2.DIST_L2,0,0.01,0.01) #返回直线的点斜
        dx,dy,x,y=*dx,*dy,int(*x),int(*y)
        cv2.circle(img,(x,y),5,(0,0,255),-1)
        lefty = int((-x*dy/dx) + y)
        righty = int(((cols-x)*dy/dx)+y)
        cv2.line(img,(cols-1,righty),(0,lefty),(255,255,0))
        angle=-math.atan(dy/dx)*180/math.pi 
        print(results[0].names[int(results[0].boxes.cls[i])],angle)
        cv2.putText(img,str(angle),(x+20,y),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,0))
    except:
        pass
    for xy in mask:
        cv2.circle(img,(int(xy[0]),int(xy[1])),1,(0,255,0))
cv2.imshow('img',img)
cv2.waitKey(0)

quit()
video=cv2.VideoCapture(0)
while video.isOpened():
    flag,frame=video.read()
    if cv2.waitKey(1)==27:
        break
    if flag:
        results=model(frame)
        frame=results[0].plot()
        xylist=results[0].boxes.xyxy
        if results[0].masks==None:
            cv2.imshow('video',frame)
            continue
        masks=results[0].masks.xy
        for i in range(len(masks)):
            mask=masks[i]
            xyxy=xylist[i]
            rows,cols = frame.shape[:2]#直线拟合
            try:
                [dx,dy,x,y] = cv2.fitLine(mask, cv2.DIST_L2,0,0.01,0.01) #返回直线的点斜
                cv2.circle(frame,(int(x),int(y)),5,(0,0,255),-1)
                lefty = int((-x*dy/dx) + y)
                righty = int(((cols-x)*dy/dx)+y)
                cv2.line(frame,(cols-1,righty),(0,lefty),(255,255,0),2)
            except:
                pass
            for xy in mask:
                cv2.circle(frame,(int(xy[0]),int(xy[1])),1,(0,255,0))
        cv2.imshow('video',frame)
    
video.release()