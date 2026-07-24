from ultralytics import YOLO
import cv2
import numpy as np
import os
import threading
import time
showsize=640
points=[]
px,py=0,0
perspect=False
framesize=480,480
cnt=0
pause=False
rubbishDict={'bottle':'可回收垃圾','bottle2':'可回收垃圾','can':'可回收垃圾','papercup':'可回收垃圾','paper':'可回收垃圾','battery':'有害垃圾','battery1':'有害垃圾','battery5':'有害垃圾','drug':'有害垃圾','drugbag':'有害垃圾','drugbox':'有害垃圾','potato':'厨余垃圾','rabbitcut':'厨余垃圾','mooli':'厨余垃圾','brick':'其他垃圾','stone':'其他垃圾','china':'其他垃圾'}

print("camera init")
video=cv2.VideoCapture(1)
if not video.isOpened():
    video=cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 2688)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1512)
video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
video.set(cv2.CAP_PROP_FPS, 90)

def mouse_callback(event,x,y,flags,userdata:any):
    global px,py,perspect
    px,py=int(x/showsize*framesize[1]),int(y/showsize*framesize[0])
    if event == cv2.EVENT_LBUTTONDOWN:
        print(event,px,py,flags,userdata)
        if len(points)<4:
            points.append((px,py))
            points.sort(key=lambda x:x[1])
            points[:2]=sorted(points[:2],key=lambda x:x[0])
            points[2:]=sorted(points[2:],key=lambda x:x[0],reverse=True)
            print(points)
    elif event==cv2.EVENT_RBUTTONDOWN:
        if len(points)>0 and not perspect:
            points.pop()
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        if len(points)==4:
            perspect=True
    elif event == cv2.EVENT_RBUTTONDBLCLK:
        perspect=False
        points.clear()
cv2.namedWindow('modle compare')
cv2.setMouseCallback('modle compare',mouse_callback,"click")

modlelist=os.listdir('runs/segment')
modlelist=[x for x in modlelist if x[-2:]=='pt' or x[-3:]=='ine']
print(*modlelist)
modles = [YOLO('runs/segment/' + modle) for modle in modlelist]
modlenum=len(modles)
itemscnt=[{} for i in range(modlenum)]
if modlenum<=2:showsize=640
else: showsize=1440//modlenum

def compute(arg:YOLO,result,index):#存在线程安全问题
    modle=arg
    results=modle.predict(frame,verbose=False)
    img=results[0].plot()
    cv2.putText(img,modlelist[index],(20,60),cv2.FONT_HERSHEY_COMPLEX,1,0,3,16)
    result[index]=img,[results[0].names[int(x)] for x in results[0].boxes.cls]
results = [np.ndarray]*modlenum
tm=time.time()
imgcnt=1
namelist=[]

while video.isOpened():
    flag,frame=video.read()
    key=cv2.waitKey(1)
    if flag:
        frame=frame[:,(frame.shape[1]-frame.shape[0])//2:(frame.shape[1]+frame.shape[0])//2]
        if perspect:
            pts=np.float32([(0,0),(frame.shape[1]-1,0),(frame.shape[1]-1,frame.shape[0]-1),(0,frame.shape[0]-1)])
            M=cv2.getPerspectiveTransform(np.float32(points),pts)
            cv2.warpPerspective(frame,M,(frame.shape[1],frame.shape[0]),frame)
        
        if framesize[0]==480:framesize=frame.shape[:2]
        if not perspect:
            cv2.circle(frame,(px,py),3,(0,255,0),-1)
            for point in points:
                cv2.circle(frame,point,3,(0,0,255),-1)
            if len(points)>1:
                cv2.polylines(frame,[np.array(points)],True,(0,255,0),1,16)
                
        try:
            os.system('cls')
            print(f'{imgcnt:<15}',end='')
            for name in namelist:print(f'{name:^{max(len(name)+1,4)}}',end='')
            print('可回收垃圾 有害垃圾  厨余垃圾  其他垃圾 sum')
            for i in range(modlenum):
                for name in results[i][1]:
                    itemscnt[i][name]=itemscnt[i].get(name,0)+1
                for name in itemscnt[i].keys():
                    if name not in namelist:namelist.append(name)
                print(f'{modlelist[i]:<14}:',end='')
                rbclass={'可回收垃圾':0,'有害垃圾':0,'厨余垃圾':0,'其他垃圾':0}
                for name in namelist:
                    print(f'{itemscnt[i].get(name,0):^{max(len(name)+1,4)}}',end='')
                    rbclass[rubbishDict[name]]+=itemscnt[i].get(name,0)
                for rbc in rbclass.values():
                    print(f'{rbc:^10}',end='')
                print(sum(itemscnt[i].values()))
            tt=time.time()
            print(int((tt-tm)*1000),'ms')
            tm=tt
        except:print('print err')
        if not pause:
            threads = []
            for i,modle in enumerate(modles):
                thread = threading.Thread(target=compute, args=(modle, results, i))
                threads.append(thread)
                thread.start()
            # 等待所有线程完成
            for thread in threads:
                thread.join()
            
            ans=np.hstack([x[0] for x in results])
            imgcnt+=1
        else: ans=np.hstack([frame for i in range(modlenum)])
        cv2.imshow('modle compare',cv2.resize(ans,(showsize*modlenum,showsize)))
    if key==27:
        break
    elif key==ord('s'):
        cv2.imwrite('runs/segment/'+str(cnt))
        print('save',cnt)
        cnt=cnt+1
    elif key==ord('c'):
        imgcnt=0
        namelist=[]
        itemscnt=[{} for i in range(modlenum)]
    elif key==ord('p'):
        pause=not pause
video.release()
cv2.destroyAllWindows()

