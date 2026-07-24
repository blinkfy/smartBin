#按P拍照，单击标点，双击确定，右键取消
DEEPCAMERA=False
import cv2
import numpy as np
import time
cnt=80          #注意
size=480
size2=640
showsize=800
name='mycoombination'
points=[]
px,py=0,0
perspect=False
savenow=False
if DEEPCAMERA:
    import pyrealsense2 as rs
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    pipeline.start(config)
    align=rs.align(rs.stream.color)
else:
    video=cv2.VideoCapture(0)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 2688)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1512)
    video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    video.set(cv2.CAP_PROP_FPS, 60)
print("start")

def mouse_callback(event,x,y,flags,userdata:any):
    global px,py,perspect,savenow
    px,py=int(x/showsize*framesize[1]),int(y/showsize*framesize[0])
    if event == cv2.EVENT_LBUTTONDOWN:
        print(event,px,py,flags,userdata)
        if len(points)<4:
            points.append((px,py))
            points.sort(key=lambda x:x[1])
            points[:2]=sorted(points[:2],key=lambda x:x[0])
            points[2:]=sorted(points[2:],key=lambda x:x[0],reverse=True)
            print(points)
        elif perspect:
            if frame.shape[0]//2+frame.shape[0]//25>=px>=frame.shape[0]//2-frame.shape[0]//25 and y<frame.shape[1]//35:
                savenow=True
            elif px>=frame.shape[0]*19//20 and y<frame.shape[1]//35:
                quit()
    elif event==cv2.EVENT_RBUTTONDOWN:
        if len(points)>0 and not perspect:
            points.pop()
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        if len(points)==4:
            perspect=True
    elif event == cv2.EVENT_RBUTTONDBLCLK:
        perspect=False
        points.clear()
cv2.namedWindow('video')
cv2.setMouseCallback('video',mouse_callback,"userdata")
cv2.resizeWindow('video',width=showsize,height=showsize)
try:
    while DEEPCAMERA or video.isOpened():
        if DEEPCAMERA:
            frames = pipeline.wait_for_frames()
            frames=align.process(frames)
            frame = frames.get_color_frame()
            frame = np.asanyarray(frame.get_data())
        else:
            flag,frame=video.read()
        key=cv2.waitKey(1)
        if DEEPCAMERA or flag:
            #print(frame.shape)
            if DEEPCAMERA:frame=frame[:,170:frame.shape[0]+170]
            else:frame=frame[190:frame.shape[0]+190]
            #print(frame.shape)
            if perspect:
                pts2=np.float32([(0,0),(frame.shape[1]-1,0),(frame.shape[1]-1,frame.shape[0]-1),(0,frame.shape[0]-1)])
                M=cv2.getPerspectiveTransform(np.float32(points),pts2)
                cv2.warpPerspective(frame,M,(frame.shape[1],frame.shape[0]),frame)
            
            framesize=frame.shape[:2]
            if key==ord('p') or savenow:
                frame480=cv2.resize(frame,(size,size))
                cv2.imwrite(r'photo/'+name+str(cnt)+".jpg",frame480)
                frame640=cv2.resize(frame,(size2,size2))
                cv2.imwrite(r'photo2/'+name+str(cnt)+".jpg",frame640)
                frame[:]=0
                print(name+str(cnt))
                cnt+=1
                savenow=False
            if not perspect:
                cv2.circle(frame,(px,py),3,(0,255,0),-1)
                for point in points:
                    cv2.circle(frame,point,3,(0,0,255),-1)
                if len(points)>1:
                    cv2.polylines(frame,[np.array(points)],True,(0,255,0),1,16)
            else:
                cv2.rectangle(frame,(frame.shape[0]//2-frame.shape[0]//25,1),(frame.shape[0]//2+frame.shape[0]//25,frame.shape[1]//35),0,2,16)
                cv2.putText(frame,'photo',(frame.shape[0]//2-frame.shape[0]//25+3,frame.shape[1]//55),cv2.FONT_HERSHEY_COMPLEX,1,0,2,16)
                cv2.rectangle(frame,(frame.shape[0]*19//20-2,1),(frame.shape[0]-1,frame.shape[1]//35),0,2,16)
                cv2.putText(frame,'exit',(frame.shape[0]*19//20+2,frame.shape[1]//55),cv2.FONT_HERSHEY_COMPLEX,1,0,2,16)
            cv2.imshow('video',cv2.resize(frame,(showsize,showsize)))
            #print(time.gmtime())
        if key==27:
            break
finally:
    if DEEPCAMERA:
        pipeline.stop()
    else:
        video.release()