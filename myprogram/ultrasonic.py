import Jetson.GPIO as GPIO
import time
 
TRIG1 = 11  # 超声波模块Tring控制管脚
ECHO1 = 12  # 超声波模块Echo控制管脚
TRIG2 = 11
ECHO2 = 12
TRIG3 = 11
ECHO3 = 12
TRIG4 = 11
ECHO4 = 12
distance1,distance2,distance3,distance4=100,100,100,100
GPIO.setmode(GPIO.BOARD)      # 采用实际的物理管脚给GPIO口
GPIO.setwarnings(False)       # 忽略GPIO操作注意警告
GPIO.setup(TRIG1, GPIO.OUT) # Tring设置为输出模式
GPIO.setup(ECHO1, GPIO.IN)  # Echo设置为输入模式
 
# 超声波计算距离函数
def ur_disMeasure():
    global distance1,distance2,distance3,distance4
    GPIO.output(TRIG1, 0)  # 开始起始
    time.sleep(0.000002)           # 延时2us
    GPIO.output(TRIG1, 1)  # 超声波启动信号，延时10us
    time.sleep(0.00001)            # 发出超声波脉冲
    GPIO.output(TRIG1, 0)           # 设置为低电平
 
    us_time0 = time.time()
    while GPIO.input(ECHO4) == 0:
        if time.time()-us_time0>0.1:
            print('ultrasonic timeout1')
            return distance1,distance2,distance3,distance4
    us_time1 = time.time()
    while GPIO.input(ECHO4) == 1:
        if time.time()-us_time0>0.01:
            print('ultrasonic timeout2')
            return distance1,distance2,distance3,distance4
    us_time2 = time.time()                # 获取当前时间
    distance1=(us_time2 - us_time1)*17000 # t*340/2*100 (cm)
    return distance1,100,100,100
    GPIO.output(TRIG2, 0)
    time.sleep(0.000002)
    GPIO.output(TRIG2, 1)
    time.sleep(0.00001)
    GPIO.output(TRIG2, 0)
 
    us_time0 = time.time()
    while GPIO.input(ECHO4) == 0:
        if time.time()-us_time0>0.1:
            print('ultrasonic timeout1')
            return distance1,distance2,distance3,distance4
    us_time1 = time.time()
    while GPIO.input(ECHO4) == 1:
        if time.time()-us_time0>0.01:
            print('ultrasonic timeout2')
            return distance1,distance2,distance3,distance4
    us_time2 = time.time()
    distance2=(us_time2 - us_time1)*17000

    GPIO.output(TRIG3, 0)
    time.sleep(0.000002)
    GPIO.output(TRIG3, 1)
    time.sleep(0.00001)
    GPIO.output(TRIG3, 0)
 
    us_time0 = time.time()
    while GPIO.input(ECHO4) == 0:
        if time.time()-us_time0>0.1:
            print('ultrasonic timeout1')
            return distance1,distance2,distance3,distance4
    us_time1 = time.time()
    while GPIO.input(ECHO4) == 1:
        if time.time()-us_time0>0.01:
            print('ultrasonic timeout2')
            return distance1,distance2,distance3,distance4
    us_time2 = time.time()
    distance3=(us_time2 - us_time1)*17000

    GPIO.output(TRIG4, 0)
    time.sleep(0.000002)
    GPIO.output(TRIG4, 1)
    time.sleep(0.00001)
    GPIO.output(TRIG4, 0)
 
    us_time0 = time.time()
    while GPIO.input(ECHO4) == 0:
        if time.time()-us_time0>0.1:
            print('ultrasonic timeout1')
            return distance1,distance2,distance3,distance4
    us_time1 = time.time()
    while GPIO.input(ECHO4) == 1:
        if time.time()-us_time0>0.01:
            print('ultrasonic timeout2')
            return distance1,distance2,distance3,distance4
    us_time2 = time.time()
    distance4=(us_time2 - us_time1)*17000
    return distance1,distance2,distance3,distance4

# try:
#     while True:
#         us_dis = ur_disMeasure()   # 获取超声波计算距离
#         print(us_dis)
#         time.sleep(0.2)
# except KeyboardInterrupt: # 当按下Ctrl+C时，将执行destroy()子程序。
#     GPIO.cleanup()