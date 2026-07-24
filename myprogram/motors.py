
import serial
import time
import crcmod
import math

platformR=0          # 目前没用
rotate=0             # 目前没用
axis_x=26000         # y轴量程，目前没用
axis_y=22000         # x轴量程，目前没用

l0 = 9010            # 轴承中心到爪子中心的距离，换算为电机的步数
all_width = 81       # 爪子宽度为 81

# 下列4个参数用于规定平台边界范围（两边各向里20mm）    
x0_min = 795         # 5mm
x0_max = 26235       # 5+160mm
y0_min = 5088        # 32mm
y0_max = 30528       # 32+160mm

# 下列4个参数用于规定爪子中心位于垃圾桶上方时电机的步数  
can_x_min = 5000
can_x_max = 26000 
can_y_min = 8000 
can_y_max = 28000 
can3_x_max = 28000
can3_y_max = 30000

# 下列4个参数用于确定拨边缘垃圾时电机走的步数       当前爪子中心位于平台4个边的稍微靠里一点位置
take_edge_x_min = 8000
take_edge_x_max = 20500
take_edge_y_min = 11000
take_edge_y_max = 24000

# 下列4个参数用于确定拨角落垃圾时电机走的步数       当前爪子中心位于平台的4个顶点
take_corner_x_min = 5000
take_corner_x_max = 25000
take_corner_y_min = 9000
take_corner_y_max = 26000

# 下列两个参数用于标定电机运动的时间系数，即每运动1步所需要的时间
motor_1_time = 1 / 25000      # 运动25000步需要1s
motor_2_time = 1 / 25000      # 运动25000步需要1s

# 下列参数用于表定舵机运动的时间系数，即没运动1步所需要的时间
servo2_time = 1.4 / 360       # 旋转360度需要1.4s 
servo3_time = 1.3 / 4070      # 升降4070步需要1.3s
servo5_time = 0.7 / 2890      # 5号舵机在位置模式下运动2890步需要0.7s



# 1
#该函数可以得到电机16进制指令的校验码
#下面的veritydata表示16进制指令中除去校验码的部分
def crc16(veritydata):
    veritydata_bytes = bytes.fromhex(veritydata.replace(' ', ''))
    crc16 = crcmod.mkCrcFun(0x18005, rev=True, initCrc=0xFFFF, xorOut=0x0000)               #此处为一个计算校验码的函数
    return crc16(veritydata_bytes)
#此后的crc16（'中间为16进制指令代码')表示一个十进制的校验码，用hex（）函数将该数据转化成16进制


# 2
#该函数将输入的十进制数字x转化成16进制的数字表述形式
def transform(x) :
    hex_data_0x = hex(x)
    hex_data = hex_data_0x[2:]
    final_data = hex_data.zfill(8)
    final_data_byte1 = final_data[0:2]
    final_data_byte2 = final_data[2:4]
    final_data_byte3 = final_data[4:6]
    final_data_byte4 = final_data[6:8]
    transform_bytes = [ final_data_byte1 , final_data_byte2 , final_data_byte3 , final_data_byte4 ]
    return transform_bytes


# 3
#下列函数用于控制舵机
#FF FF xx(ID) 0A 03 29 XX(加速度) xx（位置低） xx（位置高） 00 00 xx（速度低） xx（速度高） xx（校验码）
#加速度为0-255，位置为0-4095，速度为0-2000
def servo_sport( ID, position , v , a ) :
    if ID==7:position+=platformR
    instruction = [ 'FF' , 'FF']
    id = format(ID,'02X')
    instruction.append(id)
    instruction.append('0A')
    instruction.append('03')
    instruction.append('29')
    instruction.append(hex(a)[2:].zfill(2))
    instruction.append(hex(position)[2:].zfill(4)[-2:])
    instruction.append(hex(position)[2:].zfill(4)[-4:-2])
    instruction.append('00')
    instruction.append('00')
    instruction.append(hex(v)[2:].zfill(4)[-2:])
    instruction.append(hex(v)[2:].zfill(4)[-4:-2])
    check_code = hex(~( int(ID) + 10 + 3 + 41 + int(a) + int(hex(position)[2:].zfill(4)[-2:],16) + int(hex(position)[2:].zfill(4)[-4:-2],16) + 0 + 0 + int(hex(v)[2:].zfill(4)[-2:],16) + int(hex(v)[2:].zfill(4)[-4:-2],16) )&0xFF)[2:].zfill(2)
    instruction.append(check_code)
    hexadecimal_instruction = ''
    for i in instruction :
        hexadecimal_instruction += str(i) + ' '
    hexadecimal_instruction.strip()
    ser2_servo_data_to_send = bytes.fromhex(hexadecimal_instruction) 
    ser2.write(ser2_servo_data_to_send)
    time.sleep(0.02)


# 4
#下列函数用于修改舵机3、5的工作模式，0为位置模式，2为开环模式
def servo5_change(number) :
    ser2_servo5_unlock = bytes.fromhex('FF FF 05 04 03 37 00 BC')                           #解锁
    ser2.write(ser2_servo5_unlock)
    time.sleep(0.02)
    if number == 0 :
        ser2_servo5_data_to_send = bytes.fromhex('FF FF 05 04 03 21 00 D2')
    if number == 2 :
        ser2_servo5_data_to_send = bytes.fromhex('FF FF 05 04 03 21 02 D0')
    ser2.write(ser2_servo5_data_to_send)
    time.sleep(0.02)
    ser2_servo5_lock = bytes.fromhex('FF FF 05 04 03 37 01 BB')                              #上锁
    ser2.write(ser2_servo5_lock)
    time.sleep(0.02)


def servo3_change(number):
    ser2_servo3_unlock = bytes.fromhex('FF FF 05 04 03 37 00 BC')                           #解锁
    ser2.write(ser2_servo3_unlock)
    time.sleep(0.02)
    if number == 0 :
        ser2_servo3_data_to_send = bytes.fromhex('FF FF 03 04 03 21 00 D4')
    if number == 2 :
        ser2_servo3_data_to_send = bytes.fromhex('FF FF 03 04 03 21 02 D2')
    ser2.write(ser2_servo3_data_to_send)
    time.sleep(0.02)
    ser2_servo3_lock = bytes.fromhex('FF FF 05 04 03 37 01 BB')                              #上锁
    ser2.write(ser2_servo3_lock)
    time.sleep(0.02)


# 下列函数用于控制舵机开环模式下的扭矩
#  FF FF XX(ID) 05 03 2C XX(扭矩低) XX（扭矩高） XX（校验码）
def servo_torque( ID , torque ) :
    if torque < 0 :
        torque = -1 * torque + 1024
    instruction = [ 'FF' , 'FF']
    id = format(ID,'02X')
    instruction.append(id)
    instruction.append('05')
    instruction.append('03')
    instruction.append('2C') 
    instruction.append(hex(torque)[2:].zfill(4)[-2:])
    instruction.append(hex(torque)[2:].zfill(4)[-4:-2])
    check_code = hex(~( int(ID) + 5 + 3 + 44 + int(hex(torque)[2:].zfill(4)[-2:],16) + int(hex(torque)[2:].zfill(4)[-4:-2],16) )&0xFF)[2:].zfill(2)
    instruction.append(check_code)
    hexadecimal_instruction = ''
    for i in instruction :
        hexadecimal_instruction += str(i) + ' '
    hexadecimal_instruction.strip()
    ser2_servo_data_to_send = bytes.fromhex(hexadecimal_instruction) 
    ser2.write(ser2_servo_data_to_send)
    time.sleep(0.02)


# 5
#下列函数用于控制电机
#xx(ID) 10 xx xx(前两个为寄存器地址) 00 02 04 xx xx xx xx(前四个为寄存器值) xx xx(前两个为校验码)
def motor_sport(ID,address,value) :
    if value == 0 :
        value =1
    instruction = []
    instruction.append(format(ID,'02X'))
    instruction.append('10')
    instruction.append(hex(address)[2:].zfill(4)[-4:-2])
    instruction.append(hex(address)[2:].zfill(4)[-2:])
    instruction.append('00')
    instruction.append('02')
    instruction.append('04')
    a = transform(int(value))
    instruction.append(a[0])
    instruction.append(a[1])
    instruction.append(a[2])
    instruction.append(a[3])
    hexadecimal_instruction_string = ''
    for i in instruction :
        hexadecimal_instruction_string += i + ' '
    crc16_value = hex(crc16(hexadecimal_instruction_string))[2:].zfill(4)
    crc16_value_byte1 = crc16_value[-2:]
    crc16_value_byte2 = crc16_value[-4:-2]
    hexadecimal_instruction_string += crc16_value_byte1
    hexadecimal_instruction_string += ' ' + crc16_value_byte2
    if ID == 1:
        ser1_data1_to_send = bytes.fromhex(hexadecimal_instruction_string)
        ser1.write(ser1_data1_to_send)
        time.sleep(0.02)
    if ID == 2:
        ser4_data1_to_send = bytes.fromhex(hexadecimal_instruction_string)
        ser4.write(ser4_data1_to_send)
        time.sleep(0.02)


# 6                                      观察是否有负数情况出现！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
#下列函数表示电机1、2的移动
#下面的x，y表示电机中的步数
position=(0,0)
def goto_garbage_position(x,y) :
    if x > 0 and y > 0 :
        motor_sport(1,2004,x)
        motor_sport(2,2004,y)
    else:
        print("ERRORRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR")
        # 此处延时后面加


def goto_x_position(position_x):                    # 用于页面调试，目前没用
    global position
    if position_x>axis_y:position_x=axis_y
    step_x=int(position_x)-position[0]
    if position_x==-1:
        step_x=1
    position=(position_x,position[1])
    if step_x>0:
        motor_sport(1,2004,step_x)
    elif step_x<0:
        motor_sport(1,2006,-step_x)
    time.sleep(1.5*abs(step_x)/axis_y)
    

def goto_y_position(position_y):                    # 用于页面调试，目前没用
    global position
    if position_y>axis_x:position_y=axis_x
    step_y=int(position_y)-position[1]
    if position_y==-1:
        step_y=1
    position=(position[0],position_y)
    if step_y>0:
        motor_sport(2,2004,step_y)
    elif step_y<0:
        motor_sport(2,2006,-step_y)
    time.sleep(1.5*abs(step_y)/axis_x)


# 7
#下列函数使得2号舵机进行旋转
#下面的angle表示 -180 到 180 度										
def servo_2_rotate(angle):
    position = int( ( angle / 360 ) * 4095 ) + 2048
    servo_sport(2,position,90,250)
    # 旋转延时在后面加


# 8  
def height_grab(height,width) :
    height_limit = 66                                            # 爪子碰到平台时，爪子下降的距离
    two_length = 340                                             # 摄像头到平台的距离的99%，初值为346
    new_height = height_limit - ( two_length - height )          # 爪子抓取该物体时需要下降的高度
    if new_height<1:new_height=1
    all_height = 75                                              # 可以下降的最大高度
    
    if height <= two_length - 3 :
        position = int ( ( ( 4090 - 20 ) / all_height ) * new_height )                          # 高度变化量
        servo3_position = 4090 -  position                                                      # 4090为顶端位置，20为底端位置
        servo_sport(3,servo3_position,90,250)                                                   # 高度下降
        time.sleep( servo3_time * position )

        servo5_change(2)
        servo_torque(5,500)                                                                     # 向里抓，先扭矩为400
        time.sleep(1.2)   #！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！

        servo_torque(5,300)                                                                     # 向里抓，后扭矩为200

        servo_sport(3,4090,90,250)                                                              # 高度上升
        time.sleep( servo3_time * position )
        
    elif height > two_length - 3 :        # 此时说明垃圾高度很小，产生下压再抓起来
        servo_sport(3,500,90,250)                                                                # 先以90的速度高度下降到500
        time.sleep( servo3_time * 3590 )                                                         # 此时高度变化量为3590
        servo_sport(3,20,50,100)                                                                 # 再以50的速度下到20，产生下压
        time.sleep( servo3_time * 480 * 1.3)                                                     # 高度变化量为480，为了延长一点时间所以乘以1.3

        servo5_change(2)
        servo_torque(5,500)                                                                     # 向里抓，先扭矩为400
        time.sleep( 1 * width / all_width + 0.8 )  #！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！

        servo_torque(5,300)                                                                     # 向里抓，后扭矩为200

        servo_sport(3,4090,90,250)                                                              # 高度上升
        time.sleep( servo3_time * 4070 )


# 9       
#下列函数将8、9、10、11号舵机打开 
def open_around() :
    servo_sport(8,2600,90,250)
    servo_sport(9,2600,90,250)                                        
    time.sleep(0.1)
    servo_sport(10,2600,90,250)
    servo_sport(11,2600,90,250)


# 10       
#下列函数将8、9、10、11号舵机闭合 
def close_around() :       
    servo_sport(10,2010,90,250)
    servo_sport(11,2010,90,250)
    time.sleep(0.1)
    servo_sport(8,1980,90,250)
    servo_sport(9,1980,90,250)


# 11
#下列函数用于将电机1、2移动到目标垃圾桶的上方
# x、y为垃圾的位置
def goto_garbage_can_position(x,y,garbage_can_number) :    
    if garbage_can_number == 1 :
        if x > can_x_min and y > can_y_min :
            x = x - can_x_min
            y = y - can_y_min
            motor_sport(1,2006,x)
            motor_sport(2,2006,y)
        elif x < can_x_min and y < can_y_min :
            x = can_x_min - x
            y = can_y_min - y
            motor_sport(1,2004,x)
            motor_sport(2,2004,y)
        elif x > can_x_min and y < can_y_min :
            x = x - can_x_min
            y = can_y_min - y
            motor_sport(1,2006,x)
            motor_sport(2,2004,y)
        else :
            x = can_x_min - x
            y = y - can_y_min
            motor_sport(1,2004,x)
            motor_sport(2,2006,y)

    elif garbage_can_number == 2 :
        if x > can_x_max and y > can_y_min :
            x = x - can_x_max
            y = y - can_y_min
            motor_sport(1,2006,x)
            motor_sport(2,2006,y)
        elif x < can_x_max and y < can_y_min :
            x = can_x_max - x
            y = can_y_min - y
            motor_sport(1,2004,x)
            motor_sport(2,2004,y)
        elif x > can_x_max and y < can_y_min :
            x = x - can_x_max
            y = can_y_min - y
            motor_sport(1,2006,x)
            motor_sport(2,2004,y)
        else :
            x = can_x_max - x
            y = y - can_y_min
            motor_sport(1,2004,x)
            motor_sport(2,2006,y)

    elif garbage_can_number == 3 :          # 模拟赛做了修改，记得改回去
        if x > can3_x_max and y > can3_y_max :    
            x = x - can3_x_max
            y = y - can3_y_max
            motor_sport(1,2006,x)
            motor_sport(2,2006,y)
        elif x < can3_x_max and y < can3_y_max :   
            x = can3_x_max - x                     
            y = can3_y_max - y
            motor_sport(1,2004,x)
            motor_sport(2,2004,y)
        elif x > can3_x_max and y < can3_y_max :
            x = x - can3_x_max
            y = can3_y_max - y
            motor_sport(1,2006,x)
            motor_sport(2,2004,y)
        else :
            x = can3_x_max - x
            y = y - can3_y_max
            motor_sport(1,2004,x)
            motor_sport(2,2006,y)

    elif garbage_can_number == 4 :
        if x > can_x_min and y > can_y_max :
            x = x - can_x_min
            y = y - can_y_max
            motor_sport(1,2006,x)
            motor_sport(2,2006,y)
        elif x < can_x_min and y < can_y_max :
            x = can_x_min - x
            y = can_y_max - y
            motor_sport(1,2004,x)
            motor_sport(2,2004,y)
        elif x > can_x_min and y < can_y_max :
            x = x - can_x_min
            y = can_y_max - y
            motor_sport(1,2006,x)
            motor_sport(2,2004,y)
        else :
            x = can_x_min - x
            y = y - can_y_max
            motor_sport(1,2004,x)
            motor_sport(2,2006,y)

    time.sleep( max ( ( motor_1_time * x ) , ( motor_2_time * y ) ))          


# 12
#该函数使得舵机5向外侧移动，将物体放下
def put_down() :
    servo_sport(7,1536,50,50)                # 平台旋转45度
    time.sleep(0.4)      #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    servo5_change(0)
    servo_sport(5,4090,90,250)               # 将物体放下
    time.sleep(0.8)      #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    servo_sport(7,2048,50,50)                # 平台归位
    # 归为延时在后面加


# 13
def pan_tilt(x,y,garbage_can_number) :          
    if garbage_can_number == 1 :
        if y >= ( ( y0_max - y0_min ) / ( x0_max - x0_min ) ) * ( x - x0_min ) + ( y0_max + y0_min ) / 2  :
            servo_sport(7,1706,50,50)     # 逆时针旋转30度
            time.sleep(0.4)     #！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！!!!!!!!!!!!!!!!!!!!!!!!!!!
            servo_sport(6,1550,90,250)    # 比正常少倾倒100

        elif y <= ( ( y0_max - y0_min ) / ( x0_max - x0_min ) ) * ( x - x0_max ) + ( y0_max + y0_min ) / 2 :
            servo_sport(7,2390,50,50)     # 顺时针旋转30度
            time.sleep(0.4)     #！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            servo_sport(6,1550,90,250)    # 比正常少倾倒100

        else:
            servo_sport(7,2048,50,50)
            #time.sleep(0.4)              # 7号舵机本身就在2048，所以取消延时
            servo_sport(6,1450,90,250)    # 正常值倾倒为1450

    elif garbage_can_number == 2 :
        if y >= ( ( y0_max - y0_min ) / ( x0_max - x0_min ) ) * ( x - x0_min ) + ( y0_max + y0_min ) / 2  :
            servo_sport(7,682,50,50)      # 逆时针旋转120度
            time.sleep(0.8)       #！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！!!!!!!!!!!!!!!!!
            servo_sport(6,1550,90,250)    # 比正常少倾倒100

        elif y <= ( ( y0_max - y0_min ) / ( x0_max - x0_min ) ) * ( x - x0_max ) + ( y0_max + y0_min ) / 2 :
            servo_sport(7,1366,50,50)     # 逆时针旋转60度
            time.sleep(0.8)        #！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            servo_sport(6,1550,90,250)    # 比正常少倾倒100

        else:
            servo_sport(7,1024,50,50)     # 逆时针旋转90度
            time.sleep(0.8)       #！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            servo_sport(6,1450,90,250)    # 正常值倾倒为1450


    elif garbage_can_number == 3:
        if y >= ( ( y0_max - y0_min ) / ( x0_max - x0_min ) ) * ( x - x0_min ) + ( y0_max + y0_min ) / 2  :
            servo_sport(7,2390,50,50)     # 顺时针旋转30度
            time.sleep(0.4)           #！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            servo_sport(6,2550,90,250)    # 比正常少倾倒100

        elif y <= ( ( y0_max - y0_min ) / ( x0_max - x0_min ) ) * ( x - x0_max ) + ( y0_max + y0_min ) / 2 :
            servo_sport(7,1706,50,50)     # 逆时针旋转30度
            time.sleep(0.4)           #！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            servo_sport(6,2550,90,250)    # 比正常少倾倒100

        else:
            servo_sport(7,2048,50,50)
            #time.sleep(0.8)              # 7号舵机本身就在2048，所以取消延时
            servo_sport(6,2650,90,250)    # 正常值倾倒为2650

    elif garbage_can_number ==4:
        if y >= ( ( y0_max - y0_min ) / ( x0_max - x0_min ) ) * ( x - x0_min ) + ( y0_max + y0_min ) / 2  :
            servo_sport(7,1366,50,50)     # 逆时针旋转60度
            time.sleep(0.8)         #！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            servo_sport(6,2550,90,250)    # 比正常少倾倒100

        elif y <= ( ( y0_max - y0_min ) / ( x0_max - x0_min ) ) * ( x - x0_max ) + ( y0_max + y0_min ) / 2 :
            servo_sport(7,682,50,50)      # 逆时针旋转120
            time.sleep(0.8)   #！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            servo_sport(6,2550,90,250)    # 比正常少倾倒100

        else:
            servo_sport(7,1024,50,50)     # 逆时针旋转90度
            time.sleep(0.8)   #！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            servo_sport(6,2650,90,250)    # 正常值倾倒为2650

    time.sleep(1.2)                       # 此处为倾倒时间，需要长一些，防止垃圾掉不下去   ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！

    # 平台归位，速度需要慢一些
    servo_sport(6,2048,50,80)
    servo_sport(7,2048,50,80)
    # 归为延时在后面加


#14                    
#下列函数使得电机和爪子归位
def motor_reset(garbage_can_number) :
    if garbage_can_number == 1 :
        motor_sport(1,2006,can_x_min)
        motor_sport(2,2006,can_y_min)

    if garbage_can_number == 2 :
        motor_sport(1,2006,can_x_max)
        time.sleep( motor_1_time * 18000 )                                       # 让电机1先走18000步，再走电机2，防止磕碰
        motor_sport(2,2006,can_y_min)

    if garbage_can_number == 3 :                   # 模拟赛做了修改，记得改回去
        motor_sport(1,2006,can3_x_max)
        motor_sport(2,2006,can3_y_max)

    if garbage_can_number == 4 :
        motor_sport(2,2006,can_y_max)
        time.sleep( motor_2_time * 15000 )                                       # 让电机2先走20000步，再走电机1，防止磕碰
        motor_sport(1,2006,can_x_min)


# 15 提前抓
def early_contraction(width):        
    if width > all_width :
        width = all_width
    position= 4090 - ( 4090 - 500 ) * ( 1 - width / all_width )            # 4090 为爪子在最外侧的位置， 1400 为爪子闭合时尖端触碰时的位置
    final_position = int(position) + 600
    if final_position > 4090:
        final_position = 4090
    servo5_change(0)
    servo_sport(5,final_position,90,250)
    # 此处延时后面加


# 16 下面三个函数用于重新分配x、y、angel的值
def define_xy_left(x,y,angle):                              # 轴承中心在爪子中心左边时
    angle_rad = math.radians(angle)
    x = int( x -l0 * math.sin(angle_rad) )
    y = int( y +l0 * math.cos(angle_rad) )
    return x,y


def define_xy_right(x,y,angle):                             # 轴承中心在爪子中心右边时
    angle_rad = math.radians(angle)
    x = int( x +l0 * math.sin(angle_rad) )
    y = int( y -l0 * math.cos(angle_rad) )
    return x,y


def assignment(x_0,y_0,angle,width) :                        #  1 - 8 号位置需要修改！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！

    if x_0 <= x0_min and y_0 <= y0_min :                        # 如果垃圾在 1 号位置
        angle = 45 
        x = 6000
        y = 6000

    elif x_0 >= x0_max and y_0 <= y0_min :                       # 如果垃圾在 2 号位置
        angle = 135
        x = 22000
        y = 6000

    elif x_0 >= x0_max and y_0 >= y0_max :                       # 如果垃圾在 3 号位置
        angle = -135 
        x = 22000
        y = 26000

    elif x_0 <= x0_min and y_0 >= y0_max :                       # 如果垃圾在 4 号位置
        angle = -45 
        x = 6000
        y = 26000

    elif x0_min < x_0 < x0_max and y_0 <= y0_min :                #如果垃圾在 5 号位置
        angle = 90
        x = x_0
        y = y_0 + l0

    elif x_0 >= x0_max and  y0_min < y_0 < y0_max :                #如果垃圾在 6 号位置
        angle = -180
        x = x_0 - l0
        y = y_0

    elif x0_min < x_0 < x0_max and  y_0 >= y0_max :                #如果垃圾在 7 号位置
        angle = -90
        x = x_0
        y = y_0 - l0

    elif x_0 <= x0_min and  y0_min < y_0 < y0_max :                #如果垃圾在 8 号位置
        angle = 0
        x = x_0 + l0
        y = y_0

    elif x0_min < x_0 < ( x0_min + x0_max ) / 2 and y0_min < y_0 < ( ( y0_min + y0_max ) / 2 - 1500 ) :           # 如果垃圾在 9 号位置   -1500是为了防止向右旋转磕碰到下侧的舵机连接件
        if -90 <= angle < 0 :
            x,y = define_xy_left(x_0,y_0,angle)
            angle = 90 + angle 
        
        elif 0 <= angle <= 40 :
            x,y = define_xy_left(x_0,y_0,angle)
            angle = 90 + angle
            if x <= 0 :
                x,y = define_xy_left(x_0,y_0,10)
                angle = 90 + 10
                width = width + 20

        elif 40 < angle <=90 :
            x,y = define_xy_right(x_0,y_0,angle)
            angle = angle - 90
            if y <= 0 :
                x,y = define_xy_right(x_0,y_0,80)
                angle = 80 - 90
                width = width + 20

    elif ( x0_min + x0_max ) / 2 < x_0 < x0_max and y0_min < y_0 < ( y0_min + y0_max ) / 2 :           # 如果垃圾在 10 号位置
        if -90 <= angle <= -30 :
            x,y = define_xy_right(x_0,y_0,angle)
            angle = angle - 90 
            if y <= 0 :
                x,y = define_xy_right(x_0,y_0,-80)
                angle = -80 - 90
                width = width + 20
        
        elif -30 < angle < 0 :
            x,y = define_xy_left(x_0,y_0,angle)
            angle = angle + 90

        elif 0 <= angle <=90 :
            x,y = define_xy_left(x_0,y_0,angle)
            angle = 90 + angle

    elif ( x0_min + x0_max ) / 2 < x_0 < x0_max and ( y0_min + y0_max ) / 2 < y_0 < y0_max :           # 如果垃圾在 11 号位置
        if -90 <= angle <= 0 :
            x,y = define_xy_right(x_0,y_0,angle)
            angle = angle - 90 
        
        elif 0 < angle <= 40 :
            x,y = define_xy_right(x_0,y_0,angle)
            angle = angle - 90

        elif 40 < angle <=90 :
            x,y = define_xy_left(x_0,y_0,angle)
            angle = 90 + angle
     
    elif x0_min < x_0 < ( x0_min + x0_max ) / 2 and ( ( y0_min + y0_max ) / 2 - 1500 ) < y_0 < y0_max :           # 如果垃圾在 12 号位置  -1500是为了防止向右旋转磕碰到下侧的舵机连接件
        if -90 <= angle <= -40 :
            x,y = define_xy_left(x_0,y_0,angle)
            angle = angle + 90 
        
        elif -40 < angle < 0 :
            x,y = define_xy_right(x_0,y_0,angle)
            angle = angle - 90
            if x <= 0 :
                x,y = assignment(x_0,y_0,-10,width)
                angle = -10 - 90
                width = width + 20

        elif 0 <= angle <= 90 :
            x,y = define_xy_right(x_0,y_0,angle)
            angle = angle - 90

    return x,y,angle,width


# 17 整个抓取过程               抓取过程和延时优化！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
def whole_process_grab(x,y,angle,garbage_can_number,width,height) :

    x,y,angle,width = assignment(x,y,angle,width)              # 先分配x、y、angle、width的值

    #电机移动、提前抓  2者同时进行
        
    goto_garbage_position(x,y)
    early_contraction(width)
    time.sleep( max ( ( motor_1_time * 5000 ) , ( motor_2_time * 5000 ) ) )         # 等2个电机都运动5000步后再旋转

    servo_2_rotate(angle)                             
    time.sleep( servo2_time * abs ( angle ) )                                       # 旋转所需要的时间

    height_grab(height,width)        

    open_around()

    if garbage_can_number==1 :
        servo_2_rotate(45)
    elif garbage_can_number==2 :
        servo_2_rotate(135)
    elif garbage_can_number==3 :
        servo_2_rotate(-135)
    elif garbage_can_number==4 :
        servo_2_rotate(-45)
            
    goto_garbage_can_position(x,y,garbage_can_number)

    put_down()
    servo_2_rotate(-10)
    time.sleep( servo2_time * 145 )                                                  # 按照最大的145度

    servo5_change(0)
    servo_sport(5,3000,90,250)
    time.sleep(0.1)

    motor_reset(garbage_can_number)
    close_around()
    servo_sport(3,3950,90,250)
    # 此处延时后面加


# 18 整个托盘倾倒过程                
def whole_process_pan(self,garbage_can_number,x,y) :
    open_around()
    time.sleep(0.3)

    pan_tilt(x,y,garbage_can_number)
    time.sleep(0.7)       #！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！

    close_around()        


# 19 将边缘和角落垃圾拨到中间       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def take_garbage_in(x,y,height,width) :
    move_step = 7000                 # 往里拨的步数

    if width > all_width :
        width = all_width
    position= 4090 - ( 4090 - 500 ) * ( 1 - width / all_width )        # 4090 为爪子在最外侧的位置， 500 为爪子闭合时尖端触碰时的位置

    servo5_position = int(position) + 100

    corner_servo5_position = int(position) + 400
    if corner_servo5_position > 4090 :
        corner_servo5_position = 4090
    
    height_limit = 66                                                  # 爪子碰到平台时，爪子下降的距离
    two_length = 342.5                                                 # 摄像头到平台的距离的99%，初值为346
    all_height = 75                                                    # 可以下降的最大高度

    garbage_height = two_length - 3 * ( two_length - height )          # 垃圾的高度

    if garbage_height < 1 :
        garbage_height = 1

    new_height = int( height_limit - ( two_length - garbage_height ) )                      # 爪子抓取该物体时需要下降的高度
    position = int ( ( ( 4090 - 20 ) / all_height ) * new_height )                          # 高度变化量
    servo3_position = 4090 -  position - 1000                                                # 减400为了碰到垃圾
    if servo3_position < 20 :
        servo3_position = 20                                                      

    if x <= x0_min and y <= y0_min :                                   # 如果垃圾在 1 号位置
        motor_sport(1,2004,take_corner_x_min)
        motor_sport(2,2004,take_corner_y_min)
        time.sleep( max ( (motor_1_time * take_corner_x_min ) , ( motor_2_time * take_corner_y_min ) ) )
        
        servo5_change(0)
        servo_sport(5,corner_servo5_position,90,250)
        servo_2_rotate(45)
        time.sleep(0.2)

        servo_sport(3,servo3_position + 300,90,250)                           # 先以90的速度下降到position+500
        time.sleep( servo3_time * ( 4090 - position - 500 ) )

        servo_sport(3,servo3_position,50,100)                                 # 再以50的速度下降到position
        time.sleep( servo3_time * 500 * 1.3)                           # 高度变化量为5000，为了延长一点时间所以乘以1.3

        motor_sport(1,2004,move_step)
        motor_sport(2,2004,move_step)
        time.sleep( max ( (motor_1_time * move_step ) , ( motor_2_time * move_step ) ) )

        servo_sport(3,4090,90,250)   
        time.sleep( servo3_time * ( 4090 - position ) )

        servo_sport(5,3000,90,250)
        servo_2_rotate(-10)
        time.sleep(0.1)                                              # 先旋转0.1s后电机再走，有时会出现电机回来了但旋转没回来的情况
        motor_sport(1,2006, take_corner_x_min + move_step )
        motor_sport(2,2006, take_corner_y_min + move_step )
        time.sleep( max ( (motor_1_time * ( take_corner_x_min + move_step ) ) , ( motor_2_time * ( take_corner_y_min + move_step ) ) ) )

        return True
    
    elif x >= x0_max and y <= y0_min :                        # 如果垃圾在 2 号位置
        motor_sport(1,2004,take_corner_x_max)
        motor_sport(2,2004,take_corner_y_min)
        time.sleep( max ( (motor_1_time * take_corner_x_max ) , ( motor_2_time * take_corner_y_min ) ) )
        
        servo5_change(0)
        servo_sport(5,corner_servo5_position,90,250)
        servo_2_rotate(135)
        time.sleep(0.6)

        servo_sport(3,servo3_position + 300,90,250)                           # 先以90的速度下降到position+500
        time.sleep( servo3_time * ( 4090 - position - 500 ) )

        servo_sport(3,servo3_position,50,100)                                 # 再以50的速度下降到position
        time.sleep( servo3_time * 500 * 1.3)                           # 高度变化量为5000，为了延长一点时间所以乘以1.3

        motor_sport(1,2006,move_step)
        motor_sport(2,2004,move_step)
        time.sleep( max ( (motor_1_time * move_step ) , ( motor_2_time * move_step ) ) )

        servo_sport(3,4090,90,250)   
        time.sleep( servo3_time * ( 4090 - position ) )

        servo_sport(5,3000,90,250)
        servo_2_rotate(-10)
        time.sleep(0.1)                                         # 先旋转0.1s后电机再走，有时会出现电机回来了但旋转没回来的情况
        motor_sport(1,2006, take_corner_x_max - move_step )
        time.sleep( motor_1_time * 15000 )                                       # 让电机1先走15000步，再走电机2，防止磕碰
        motor_sport(2,2006, take_corner_y_min + move_step )
        time.sleep( max ( (motor_1_time * ( take_corner_x_max - move_step ) ) , ( motor_2_time * ( take_corner_y_min + move_step ) ) ) - motor_1_time * 18000 )  # 让电机1先走18000步，再走电机2，防止磕碰)

        return True
    
    elif x >= x0_max and y >= y0_max :                        # 如果垃圾在 3 号位置
        motor_sport(1,2004,take_corner_x_max)
        motor_sport(2,2004,take_corner_y_max)
        time.sleep( max ( (motor_1_time * take_corner_x_max ) , ( motor_2_time * take_corner_y_max ) ) )
        
        servo5_change(0)
        servo_sport(5,corner_servo5_position,90,250)
        servo_2_rotate(-135)
        time.sleep(0.6)

        servo_sport(3,servo3_position + 300,90,250)                           # 先以90的速度下降到position+500
        time.sleep( servo3_time * ( 4090 - position - 500 ) )

        servo_sport(3,servo3_position,50,100)                                 # 再以50的速度下降到position
        time.sleep( servo3_time * 500 * 1.3)                           # 高度变化量为5000，为了延长一点时间所以乘以1.3

        motor_sport(1,2006,move_step)
        motor_sport(2,2006,move_step)
        time.sleep( max ( (motor_1_time * move_step ) , ( motor_2_time * move_step ) ) )

        servo_sport(3,4090,90,250)   
        time.sleep( servo3_time * ( 4090 - position ) )

        servo_sport(5,3000,90,250)
        servo_2_rotate(-10)
        time.sleep(0.1)                                      # 先旋转0.1s后电机再走，有时会出现电机回来了但旋转没回来的情况
        motor_sport(1,2006, take_corner_x_max - move_step )
        motor_sport(2,2006, take_corner_y_max - move_step )
        time.sleep( max ( (motor_1_time * ( take_corner_x_max - move_step ) ) , ( motor_2_time * ( take_corner_y_max - move_step ) ) ) )
    
        return True
    
    elif x <= x0_min and y >= y0_max :                        # 如果垃圾在 4 号位置
        motor_sport(1,2004,take_corner_x_min)
        motor_sport(2,2004,take_corner_y_max)
        time.sleep( max ( (motor_1_time * take_corner_x_min ) , ( motor_2_time * take_corner_y_max ) ) )
        
        servo5_change(0)
        servo_sport(5,corner_servo5_position,90,250)
        servo_2_rotate(-45)
        time.sleep(0.2)

        servo_sport(3,servo3_position + 300,90,250)                           # 先以90的速度下降到position+500
        time.sleep( servo3_time * ( 4090 - position - 500 ) )

        servo_sport(3,servo3_position,50,100)                                 # 再以50的速度下降到position
        time.sleep( servo3_time * 500 * 1.3)                           # 高度变化量为5000，为了延长一点时间所以乘以1.3

        motor_sport(1,2004,move_step)
        motor_sport(2,2006,move_step)
        time.sleep( max ( (motor_1_time * move_step ) , ( motor_2_time * move_step ) ) )

        servo_sport(3,4090,90,250)   
        time.sleep( servo3_time * ( 4090 - position ) )

        servo_sport(5,3000,90,250)
        servo_2_rotate(-10)
        time.sleep(0.1)                                              # 先旋转0.1s后电机再走，有时会出现电机回来了但旋转没回来的情况
        motor_sport(1,2006, take_corner_x_min + move_step )
        motor_sport(2,2006, take_corner_y_max - move_step )
        time.sleep( max ( (motor_1_time * ( take_corner_x_min + move_step ) ) , ( motor_2_time * ( take_corner_y_max - move_step ) ) ) )

        return True
    
    elif x0_min < x < x0_max and y <= y0_min :                # 如果垃圾在 5 号位置
        motor_sport(1,2004,x)
        motor_sport(2,2004,take_edge_y_min)
        time.sleep( max ( (motor_1_time * x ) , ( motor_2_time * take_edge_y_min ) ) )
        
        servo5_change(0)
        servo_sport(5,servo5_position,90,250)
        servo_2_rotate(90)
        time.sleep(0.4)

        servo_sport(3,servo3_position + 300,90,250)                           # 先以90的速度下降到position+500
        time.sleep( servo3_time * ( 4090 - position - 500 ) )

        servo_sport(3,servo3_position,50,100)                                 # 再以50的速度下降到position
        time.sleep( servo3_time * 500 * 1.3)                           # 高度变化量为5000，为了延长一点时间所以乘以1.3

        motor_sport(2,2004,move_step)
        time.sleep( motor_2_time * move_step )

        servo_sport(3,4090,90,250)   
        time.sleep( servo3_time * ( 4090 - position ) )

        servo_sport(5,3000,90,250)
        servo_2_rotate(-10)
        time.sleep(0.1)                                              # 先旋转0.1s后电机再走，有时会出现电机回来了但旋转没回来的情况
        motor_sport(1,2006, x )
        motor_sport(2,2006, take_edge_y_min + move_step )
        time.sleep( max ( (motor_1_time * x ) , ( motor_2_time * ( take_edge_y_min + move_step ) ) ) )

        return True
    
    elif x >= x0_max and  y0_min < y < y0_max :               # 如果垃圾在 6 号位置
        motor_sport(1,2004,take_edge_x_max)
        motor_sport(2,2004,y)
        time.sleep( max ( (motor_2_time * y ) , ( motor_1_time * take_edge_x_max ) ) )
        
        servo5_change(0)
        servo_sport(5,servo5_position,90,250)
        servo_2_rotate(-180)
        time.sleep(0.8)

        servo_sport(3,servo3_position + 300,90,250)                           # 先以90的速度下降到position+500
        time.sleep( servo3_time * ( 4090 - position - 500 ) )

        servo_sport(3,servo3_position,50,100)                                 # 再以50的速度下降到position
        time.sleep( servo3_time * 500 * 1.3)                           # 高度变化量为5000，为了延长一点时间所以乘以1.3

        motor_sport(1,2006,move_step)
        time.sleep( motor_1_time * move_step )

        servo_sport(3,4090,90,250)   
        time.sleep( servo3_time * ( 4090 - position ) )

        servo_sport(5,3000,90,250)
        servo_2_rotate(-10)
        time.sleep(0.1)                                              # 先旋转0.1s后电机再走，有时会出现电机回来了但旋转没回来的情况
        motor_sport(1,2006, take_edge_x_max - move_step )
        motor_sport(2,2006, y )
        time.sleep( max ( (motor_2_time * y ) , ( motor_1_time * ( take_edge_x_max - move_step ) ) ) )

        return True
    
    elif x0_min < x < x0_max and  y >= y0_max :               # 如果垃圾在 7 号位置
        motor_sport(1,2004,x)
        motor_sport(2,2004,take_edge_y_max)
        time.sleep( max ( (motor_1_time * x ) , ( motor_2_time * take_edge_y_max ) ) )
        
        servo5_change(0)
        servo_sport(5,servo5_position,90,250)
        servo_2_rotate(-90)
        time.sleep(0.4)

        servo_sport(3,servo3_position + 300,90,250)                           # 先以90的速度下降到position+500
        time.sleep( servo3_time * ( 4090 - position - 500 ) )

        servo_sport(3,servo3_position,50,100)                                 # 再以50的速度下降到position
        time.sleep( servo3_time * 500 * 1.3)                           # 高度变化量为5000，为了延长一点时间所以乘以1.3

        motor_sport(2,2006,move_step)
        time.sleep( motor_2_time * move_step )

        servo_sport(3,4090,90,250)   
        time.sleep( servo3_time * ( 4090 - position ) )

        servo_sport(5,3000,90,250)
        servo_2_rotate(-10)
        time.sleep(0.1)                                              # 先旋转0.1s后电机再走，有时会出现电机回来了但旋转没回来的情况
        motor_sport(1,2006, x )
        motor_sport(2,2006, take_edge_y_max - move_step )
        time.sleep( max ( (motor_1_time * x ) , ( motor_2_time * ( take_edge_y_max - move_step ) ) ) )
        
        return  True
    
    elif x <= x0_min and  y0_min < y < y0_max :               # 如果垃圾在 8 号位置
        motor_sport(1,2004,take_edge_x_min)
        motor_sport(2,2004,y)
        time.sleep( max ( (motor_2_time * y ) , ( motor_1_time * take_edge_x_min ) ) )
        
        servo5_change(0)
        servo_sport(5,servo5_position,90,250)
        servo_2_rotate(0)
        time.sleep(0.1)

        servo_sport(3,servo3_position + 300,90,250)                           # 先以90的速度下降到position+500
        time.sleep( servo3_time * ( 4090 - position - 500 ) )

        servo_sport(3,servo3_position,50,100)                                 # 再以50的速度下降到position
        time.sleep( servo3_time * 500 * 1.3)                           # 高度变化量为5000，为了延长一点时间所以乘以1.3

        motor_sport(1,2004,move_step)
        time.sleep( motor_2_time * move_step )

        servo_sport(3,4090,90,250)   
        time.sleep( servo3_time * ( 4090 - position ) )

        servo_sport(5,3000,90,250)
        servo_2_rotate(-10)
        time.sleep(0.1)                                              # 先旋转0.1s后电机再走，有时会出现电机回来了但旋转没回来的情况
        motor_sport(1,2006, take_edge_x_min + move_step )
        motor_sport(2,2006, y )
        time.sleep( max ( (motor_2_time * y ) , ( motor_1_time * ( take_edge_x_min + move_step ) ) ) )
        
        return True
    
    return False


# 20  将靠的近的垃圾分开    cx和cy为锁定垃圾的边缘的坐标，cl为该垃圾边缘到另一个距离较近的垃圾边缘的距离
def separate_garbage(height,angle,cx,cy,width) : 
        cx,cy,angle,width = assignment(cx,cy,angle,width)                # 分配数值

        height_limit = 66                                            # 爪子碰到平台时，爪子下降的距离
        two_length = 342.5                                           # 摄像头到平台的距离的99%，初值为346
        new_height = height_limit - ( two_length - height )          # 爪子抓取该物体时需要下降的高度
        all_height = 75                                              # 可以下降的最大高度
        position = int ( ( ( 4090 - 20 ) / all_height ) * new_height )                          # 高度变化量
        servo3_position = 4090 -  position                                                      # 4090为顶端位置，20为底端位置

        motor_sport(1,2004,cx)
        motor_sport(2,2004,cy)
        servo5_change(0)
        servo_sport(5,1200,90,250)                           # 爪子完全闭合
        time.sleep( max ( ( motor_1_time * 5000 ) , ( motor_2_time * 5000 ) ) )         # 等2个电机都运动5000步后再旋转

        servo_2_rotate(angle)
        time.sleep( servo2_time * abs ( angle + 10 ) )       # 初始角度值为-10度，所以角度差值为（angle+10）

        servo_sport(3,servo3_position,90,250)                # 高度下降
        time.sleep( servo3_time * position )

        servo_sport(5,4090,90,250)                           # 爪子完全张开
        time.sleep( servo5_time * 4070)

        servo_sport(3,4090,90,250)                           # 高度上升
        time.sleep( servo3_time * position )

        servo_2_rotate(-10)
        servo_sport(5,3000,90,250)
        time.sleep( max ( ( servo5_time * 590 ) , ( servo2_time * abs ( angle + 10 ) ) ) )
        
        motor_sport(1,2006,cx)
        motor_sport(2,2006,cy)
        time.sleep( max ( ( motor_1_time * cx ) , ( motor_2_time * cy ) ) )

'''
# 21 将3号升降舵机的顶部位置校正为4090，防止脱带
def servo3_correct_top_position() :
    ser2_servo3_data = bytes.fromhex('FF FF 03 04 02 38 02 BC')               # 读取当前位置
    ser2.write(ser2_servo3_data)
    time.sleep(0.02)
    
    data = ser2.read(8)
    position_low = hex(data[5])
    position_high = hex(data[6])
    position = int(position_high[2:].zfill(2) + position_low[2:].zfill(2),16)
    time.sleep(0.02)
    if ( 4090 - position ) > 30 :
        old_position_offset = -1 * ( 1420 ) + 2048                                # 原始位置偏移量为-240
        new_position_offset = ( 4090 - position ) + old_position_offset           # 新位置偏移量
        instruction = [ 'FF' , 'FF']
        id = format(3,'02X')
        instruction.append(id)
        instruction.append('05')
        instruction.append('03')
        instruction.append('1F') 
        instruction.append(hex(new_position_offset)[2:].zfill(4)[-2:])
        instruction.append(hex(new_position_offset)[2:].zfill(4)[-4:-2])
        check_code = hex(~( int(3) + 5 + 3 + 31 + int(hex(new_position_offset)[2:].zfill(4)[-2:],16) + int(hex(new_position_offset)[2:].zfill(4)[-4:-2],16) )&0xFF)[2:].zfill(2)
        instruction.append(check_code)
        hexadecimal_instruction = ''
        for i in instruction :
            hexadecimal_instruction += str(i) + ' '
        hexadecimal_instruction.strip()
        ser2_servo3_data_to_send = bytes.fromhex(hexadecimal_instruction) 
        ser2.write(ser2_servo3_data_to_send)
        time.sleep(0.02)
'''

# 22
def ser_close() :
    if ser1.is_open:
        ser1.close()
    if ser2.is_open:
        ser2.close()
    if ser4.is_open:
        ser4.close()


# 23
def init():
    global ser1,ser2,ser4,ser5,ser6,ser7,ser8

    ser1 = serial.Serial('/dev/ttyCH9344USB0',9600)                                                    #打开电机1的串口，波特率为9600

    try:
        ser2 = serial.Serial('/dev/ttyUSB0',115200)                                                    #打开舵机的串口，波特率为115200
    except  Exception as e:
        print('open USB err: ',e)
        try:
            ser2 = serial.Serial('/dev/ttyUSB1',115200)
        except Exception as e:
            print('open USB err: ',e)
            try:
                ser2=serial.Serial('/dev/ttyUSB2',115200)
            except Exception as e:
                print('open USB err: ',e)
                quit()
        
    
    ser4 = serial.Serial('/dev/ttyCH9344USB3',9600)                                                    #打开电机2的串口，波特率为9600
    
    ser5 = serial.Serial('/dev/ttyCH9344USB7',115200)                                                  #打开超声波1的串口，波特率为115200
    ser6 = serial.Serial('/dev/ttyCH9344USB1',115200)                                                  #打开超声波2的串口，波特率为115200
    ser7 = serial.Serial('/dev/ttyCH9344USB2',115200)                                                  #打开超声波3的串口，波特率为115200
    ser8 = serial.Serial('/dev/ttyCH9344USB4',115200)                                                  #打开超声波4的串口，波特率为115200
    
init()


# 24
def get_distances():
    ser5_data_to_send = bytes.fromhex('01 03 00 10 00 01 85 cf')
    ser6_data_to_send = bytes.fromhex('02 03 00 10 00 01 85 fc')
    ser7_data_to_send = bytes.fromhex('03 03 00 10 00 01 84 2d')
    ser8_data_to_send = bytes.fromhex('04 03 00 10 00 01 85 9a')

    ser5.write(ser5_data_to_send)
    ser6.write(ser6_data_to_send)
    ser7.write(ser7_data_to_send)
    ser8.write(ser8_data_to_send)

    time.sleep(0.12)

    try:
        distance1 = ser5.read(ser5.in_waiting)[4]
    except:
        distance1=131
    try:
        distance2 = ser6.read(ser6.in_waiting)[4]
    except:
        distance2=131
    try:
        distance3 = ser7.read(ser7.in_waiting)[4]
    except:
        distance3=131
    try:
        distance4 = ser8.read(ser8.in_waiting)[4]
    except:
        distance4=131
    return [ distance1 , distance2 , distance3 , distance4 ]



takeintime=0                                       #防止戳起来没完

#  这部分代码可能需要修改！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
def all_grab(self,x,y,angle,k,width,height):
    global takeintime
    servo_sport(3,4090,90,250)
    servo_sport(6,2048,30,50)
    servo_sport(7,2048,30,50)
    close_around()
    if takeintime<3 and take_garbage_in(x,y,height,width) :
        takeintime += 1
        return False
    takeintime=0
    whole_process_grab(x,y,angle,k,width,height)
    time.sleep(0.6 if k==1 else 0.9)
    return True