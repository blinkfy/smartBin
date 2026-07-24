#拼图（和标注）,在父目录下执行
#yolo segment train data=coco.yaml model=yolo11s-seg.pt batch=32 epochs=100 imgsz=480 device=0 workers=8 patience=30 hsv_h=0.1 hsv_s=0.05 hsv_v=0.1
#yolo segment predict model=runs\segment\train\weights\best.pt source=data\combination\group5.jpg
#4090->batch=36 59s/epoch; 3080->batch=36 66s/epoch; v100->batch=36 65s/epoch
#yolo segment train data=coco.yaml model=best.pt batch=32 epochs=15 freeze=10 imgsz=480 device=0 workers=8 lr0=0.001 patience=10 hsv_h=0.1 hsv_s=0.05 hsv_v=0.1
#yolo segment train data=coco.yaml model=best2.pt batch=32 epochs=30 freeze=0 imgsz=480 device=0 workers=8 lr0=0.004 patience=15 hsv_h=0.1 hsv_s=0.05 hsv_v=0.1
import json
import cv2
import numpy as np
import random
import os
import time
from tqdm import tqdm
IMGSOURSE=r"data/images"
JSONPATH=r"data/Annotations"
SAVEPATH=r"data/masks"
COMBINATIONPATH=r"data/combination"
VALPATH=r'data/val'
VAL_RATIO=0.15
LABELNAMES=['potato','rabbitcut','battery1','battery5','bottle','brick','can','china','stone','drug','drugbag','mooli','battery','bottle2','drugbox','papercup','capsule','potatocut','paper']
THISPATH=os.getcwd()
labelcount=dict.fromkeys(LABELNAMES,0)#计数用
labelmap={}
starttime=time.time()

#traintxt=open('data/train.txt','w')
imglist=os.listdir(IMGSOURSE)
jsonlist=os.listdir(JSONPATH)
morph=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
for name in tqdm(imglist,desc='Generating masks'):                                           #生成蒙版
    imgpath=IMGSOURSE+r'/'+name
    jsonpath=JSONPATH+r'/'+(name[:name.find('.')]+".json")
    if (name[:name.find('.')]+".json") not in jsonlist:
        tqdm.write(f"{name} 未找到标注文件 {name[:name.find('.')]}.json")
    else:
        #print(jsonpath)
        fp=open(jsonpath)
        data=json.load(fp)
        fp.close()
        cnt=1
        for lines in data['shapes']:
            # if lines['label'] not in LABELNAMES:
            #     LABELNAMES.append(lines['label'])
            labelcount[lines['label']]+=1                   #计数
            points=np.array(lines['points']).astype(int)
            img=cv2.imread(imgpath)
            cv2.imwrite(COMBINATIONPATH+'/'+name,img)
            img=np.zeros(img.shape[:2],np.uint8)
            cv2.fillPoly(img,[points],255,16)
            cv2.GaussianBlur(img, (3,3), 0,img)
            #img=cv2.erode(img,morph)
            #print(f'{name} ok {cnt}')
            cv2.imwrite(f"{SAVEPATH}/{name[:name.find('.')]}-{cnt}{name[name.find('.'):]}",img)
            labelmap[f"{name[:name.find('.')]}-{cnt}{name[name.find('.'):]}"]=LABELNAMES.index(lines['label'])
            cnt+=1

        '''原文件也生成txt'''
        ftxt=open(COMBINATIONPATH+r'/'+name[:name.find('.')]+".txt",'w')
        for data2 in data['shapes']:
            ftxt.write(str(LABELNAMES.index(data2['label'])))
            for data3 in data2['points']:
                ftxt.write(f' {data3[0]/img.shape[1]} {data3[1]/img.shape[0]}')
            ftxt.write('\n')
        ftxt.close()

        #源文件放入train.txt  注意把图像手动复制过去
        #traintxt.write("/home/featurize/"+COMBINATIONPATH+'/'+name+'\n')
        
print(labelcount)
#print(labelmap)

np.random.seed(int(time.time()))
masklist=os.listdir(SAVEPATH)
list2=[]
background0=cv2.imread(r"data/void0.jpg")
background1=cv2.imread(r"data/void1.jpg")
background2=cv2.imread(r"data/void2.jpg")
background3=cv2.imread(r"data/void3.jpg")
background4=cv2.imread(r"data/void4.jpg")
background5=cv2.imread(r"data/void5.jpg")
background6=cv2.imread(r"data/void6.jpg")
background7=cv2.imread(r"data/void7.jpg")
background8=cv2.imread(r"data/void8.jpg")
background0=cv2.resize(background0,(480,480))#调整到相应大小
background1=cv2.resize(background1,(480,480))
background2=cv2.resize(background2,(480,480))
background3=cv2.resize(background3,(480,480))
background4=cv2.resize(background4,(480,480))
background5=cv2.resize(background5,(480,480))
background6=cv2.resize(background6,(480,480))
background7=cv2.resize(background7,(480,480))
background8=cv2.resize(background8,(480,480))
for i in tqdm(range(4000),'Generating images'):
    isVal=np.random.random(1)<VAL_RATIO
    if len(list2)<5:
        np.random.shuffle(masklist)
        list2.extend(masklist)
    img=random.choice([background0,background1,background2,background3,background4,background5,background6,background7,background8]).copy()
    ftxt=open((VALPATH if isVal else COMBINATIONPATH)+r"/group"+str(i)+".txt",mode='w')
    hadmask=np.zeros(img.shape[:2],np.uint8)
    
    for j in range(3):
        name=list2.pop(0)
        mask=cv2.imread(SAVEPATH+r"/"+name,cv2.IMREAD_GRAYSCALE)
        #mask=cv2.resize(mask,(480,480))
        org=cv2.imread(IMGSOURSE+r"/"+name[:name.find('-')]+name[name.find('.'):])
        #org=cv2.resize(org,(480,480))
        Mm=cv2.getRotationMatrix2D((img.shape[0]/2,img.shape[1]/2),np.random.randint(0, 360),1)#旋转
        Mm[0][2]=Mm[0][2]+np.random.randint(-80, 80)#平移
        Mm[1][2]=Mm[1][2]+np.random.randint(-80, 80)
        cv2.warpAffine(mask, Mm,mask.shape[:2],mask)
        #防止两个物体重合,通过iou判断
        mask_th=cv2.threshold(mask,128,255,cv2.THRESH_BINARY)[1]
        if j>=1:
            try:
                iou=cv2.bitwise_and(hadmask,mask_th)
                c,_=cv2.findContours(iou,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                area1=cv2.contourArea(c[0])
                c,_=cv2.findContours(mask_th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                area2=cv2.contourArea(c[0])
                c,_=cv2.findContours(hadmask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                area3=cv2.contourArea(c[0])
                if max(area1/area2,(area1/area3)*j)>0.4:
                    continue
            except:pass
        cv2.bitwise_or(hadmask,mask_th,hadmask)

        '''方框
        ret,mask_2=cv2.threshold(cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY),127,255,cv2.THRESH_BINARY)#矩形标注
        contours,hierarchy=cv2.findContours(mask_2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        ct=contours[0]
        left = ct[ct[:, :, 0].argmin()][0]  # 图像轮廓的最左端像素点的坐标
        right = ct[ct[:, :, 0].argmax()][0]
        top = ct[ct[:, :, 1].argmin()][0]
        bottom = ct[ct[:, :, 1].argmax()][0]
        ftxt.write(str(labelmap[name])+' '+str(left[0]/img.shape[1])+' '+str(top[1]/img.shape[0])+' '+str((right[0]-left[0])/img.shape[1])+' '+str((top[0]-bottom[0])/img.shape[1])+'\n')
        cv2.rectangle(img,(left[0],top[1]),(right[0],bottom[1]),(0,0,255))'''
        
        with open(JSONPATH+r"/"+name[:name.find('-')]+'.json','r') as fp:
            data=json.load(fp)
        shape=[]
        cnt_out=0.0
        for tu in data['shapes'][int(name[name.find('-')+1:name.find('.')])-1]['points']:
            xyt=np.array([tu[0],tu[1],1],int)
            xy1=np.dot(Mm,xyt)
            if xy1[0]<0:
                xy1[0]=0
                cnt_out+=1
            elif xy1[0]>=img.shape[1]:
                xy1[0]=img.shape[1]-1
                cnt_out+=1
            if xy1[1]<0:
                xy1[1]=0
                cnt_out+=1
            elif xy1[1]>=img.shape[0]:
                xy1[1]=img.shape[0]-1
                cnt_out+=1
            shape.append(str(xy1[0]/img.shape[1])+' '+str(xy1[1]/img.shape[0]))
        if cnt_out/len(data['shapes'][int(name[name.find('-')+1:name.find('.')])-1]['points'])>=0.5:#一半的点超出
            continue
        ftxt.write(str(labelmap[name])+' '+str(' '.join(shape))+'\n')
        cv2.warpAffine(org, Mm,org.shape[:2],org)
        mask = np.expand_dims(mask, axis=-1)
        alpha = np.repeat(mask, 3, axis=-1) / 255.0
        img = (org * alpha + img * (1 - alpha)).astype(np.uint8)        #边缘渐变式融合
        #img=cv2.bitwise_or(cv2.bitwise_and(img,cv2.bitwise_not(mask)),cv2.bitwise_and(org,mask))
    ra=np.random.random()
    if ra<0.1:#clahe、锐化
        img=cv2.bilateralFilter(img,11,30,5)
        yuv=cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
        yuv[:,:,0]=cv2.createCLAHE(2,(8,8)).apply(yuv[:,:,0])
        cv2.cvtColor(yuv,cv2.COLOR_YUV2RGB,img)
    elif ra<0.2:
        img=cv2.bilateralFilter(img,11,30,5)
        img=cv2.filter2D(img,8,np.array([[-0.1,-0.5,-0.1],[-0.5,3.4,-0.5],[-0.1,-0.5,-0.1]]))
        
    cv2.imwrite(f'{VALPATH if isVal else COMBINATIONPATH}/group{i}.jpg',img)
    #if not isVal: traintxt.write(f'/home/featurize/{COMBINATIONPATH}/group{i}.jpg\n')
    ftxt.close()
    #if i%16==0:print(i)
#traintxt.close()
print(len(labelcount),labelcount)
print("using",time.time()-starttime)