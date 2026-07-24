import serial
import time
import crcmod
import random


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
#下列函数用于修改舵机5的工作模式，0为位置模式，2为开环模式
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


# 5
#下列函数用于控制电机
#xx(ID) 10 xx xx(前两个为寄存器地址) 00 02 04 xx xx xx xx(前四个为寄存器值) xx xx(前两个为校验码)
def motor_sport(ID,address,value) :
    instruction = []
    instruction.append(format(ID,'02X'))
    instruction.append('10')
    instruction.append(hex(address)[2:].zfill(4)[-4:-2])
    instruction.append(hex(address)[2:].zfill(4)[-2:])
    instruction.append('00')
    instruction.append('02')
    instruction.append('04')
    a = transform(value)
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
    ser1_data1_to_send = bytes.fromhex(hexadecimal_instruction_string)
    ser1.write(ser1_data1_to_send)
    time.sleep(0.05)


# 6
#下列函数表示电机1、2的移动
#下面的x，y表示电机中的步数，未进行换算
def goto_garbage_position(x,y) :
    motor_sport(1,2004,x)
    motor_sport(2,2004,y)


# 7
#下列函数使得2号舵机进行旋转
#下面的angle表示-90到90度										
def servo_2_rotate(angle) :
    position = int((angle/360)*4095)+2048
    servo_sport(2,position,50,50)


# 8
#下列函数用于将垃圾抓起来，电动
def electric_grab() :
    # 3、4舵机：上
    servo_sport(3,2048,30,50)
    servo_sport(4,2048,30,50)
    # 5舵机：外侧，开环模式
    servo5_change(2)
    ser2_servo5_in = bytes.fromhex('FF FF 05 05 03 2C 32 04 90')                            #舵机5向外扩展，运行时间（扭矩）为50（+1024）
    ser2.write(ser2_servo5_in)                   
    time.sleep(0.2)

    # 3、4舵机：下
    servo_sport(3,2700,30,50)
    servo_sport(4,2700,30,50)
    time.sleep(0.8)
    # 5舵机：内侧，开环模式,先500后100
    #servo5_change(2)
    ser2_servo5_in = bytes.fromhex('FF FF 05 05 03 2C F4 01 D1')                            #舵机5向内夹住物体，运行时间（扭矩）为500
    ser2.write(ser2_servo5_in)
    time.sleep(2.5)
    ser2_servo5_in = bytes.fromhex('FF FF 05 05 03 2C 64 00 62')                            #舵机5向内夹住物体，运行时间（扭矩）为100
    ser2.write(ser2_servo5_in)
    time.sleep(4)

    # 3、4舵机：上
    servo_sport(3,2048,30,50)
    servo_sport(4,2048,30,50)


#8
#下列函数用于将垃圾抓起来,气动（无3、4号舵机）
def pneumatic_grab() :
    #气缸缩回
    data2_to_send = bytes.fromhex('01 05 00 A0 00 00 CD E8')          #一路关
    ser3.write(data2_to_send)
    time.sleep(0.2)
    # 5舵机：外侧，开环模式
    servo5_change(2)
    ser2_servo5_in = bytes.fromhex('FF FF 05 05 03 2C 32 04 90')                            #舵机5向外扩展，运行时间（扭矩）为50（+1024）
    ser2.write(ser2_servo5_in)                  
    time.sleep(0.2)
    #气缸伸出
    data1_to_send = bytes.fromhex('01 05 00 A0 FF 00 8C 18')          #一路开
    ser3.write(data1_to_send)
    time.sleep(0.2)

    # 5舵机：内侧，开环模式,先500后100
    #servo5_change(2)
    ser2_servo5_in = bytes.fromhex('FF FF 05 05 03 2C F4 01 D1')                            #舵机5向内夹住物体，运行时间（扭矩）为500
    ser2.write(ser2_servo5_in)
    time.sleep(3)
    ser2_servo5_in = bytes.fromhex('FF FF 05 05 03 2C 64 00 62')                            #舵机5向内夹住物体，运行时间（扭矩）为100
    ser2.write(ser2_servo5_in)
    time.sleep(2)
    #气缸缩回
    data2_to_send = bytes.fromhex('01 05 00 A0 00 00 CD E8')          #一路关
    ser3.write(data2_to_send)
    time.sleep(0.2)


# 9
#下列函数将8、9、10、11号舵机打开
def open_around() :
    servo_sport(8,1448,50,50)
    servo_sport(9,1448,50,50)
    servo_sport(10,1448,50,50)
    servo_sport(11,2048,50,50)


# 10
#下列函数将8、9、10、11号舵机闭合
def close_around() :
    servo_sport(8,2048,50,50)
    servo_sport(9,2048,50,50)
    servo_sport(10,2048,50,50)
    servo_sport(11,1448,50,50)


# 11
#下列函数用于将电机1、2移动到目标垃圾桶的上方
#x、y为垃圾的位置
def goto_garbage_can_position(x,y,garbage_can_number) :
    if garbage_can_number == 1 :
        x = 25000 - x
        y = 30000 - y
        motor_sport(1,2004,x)
        motor_sport(2,2004,y)
    if garbage_can_number == 2 :
        x = 25000 - x
        y = y
        motor_sport(1,2004,x)
        motor_sport(2,2006,y)
    if garbage_can_number == 3 :
        x = x
        y = y
        motor_sport(1,2006,x)
        motor_sport(2,2006,y)
    if garbage_can_number == 4 :
        x = x
        y = 30000-y
        motor_sport(1,2006,x)
        motor_sport(2,2004,y)

    time.sleep(0.05)


# 12
#该函数使得舵机5向外侧移动，将物体放下
def put_down() :
    servo_sport(7,1536,30,50)
    time.sleep(0.5)
    #servo5_change(2)
    ser2_servo5_in = bytes.fromhex('FF FF 05 05 03 2C F4 05 CD')                            #舵机5向外扩展，运行时间（扭矩）为500（+1024）
    ser2.write(ser2_servo5_in)                   
    time.sleep(3)
    ser2_servo5_in = bytes.fromhex('FF FF 05 05 03 2C 64 04 5E')                            #舵机5向外扩展，运行时间（扭矩）为100（+1024）
    ser2.write(ser2_servo5_in)                     
    time.sleep(1)
    servo_sport(7,2048,30,50)


# 13
def pan_tilt(garbage_can_number) :
    # 6号舵机移动到1748，7号舵机移动到2348
    if garbage_can_number == 1 :
        servo_sport(7,2048,30,50)
        time.sleep(0.8)
        servo_sport(6,2448,30,50)

    # 6号舵机移动到2348，7号舵机移动到2348
    if garbage_can_number == 2 :
        servo_sport(7,1024,30,50)
        time.sleep(0.8)
        servo_sport(6,1648,30,50)


    # 6号舵机移动到2348，7号舵机移动到1748
    if garbage_can_number == 3 :
        servo_sport(7,2048,30,50)
        time.sleep(0.8)
        servo_sport(6,1648,30,50)

    # 6号舵机移动到1748，7号舵机移动到1748
    if garbage_can_number == 4 :
        servo_sport(7,1024,30,50)
        time.sleep(0.8)
        servo_sport(6,2448,30,50)


    time.sleep(0.8)
    # 6、7号舵机全都移动到2048，即中心位置
    servo_sport(6,2048,30,50)
    servo_sport(7,2048,30,50)


# 14
#下列函数使得电机1、2复位，光电开关
def motor_reset1() :
    ser1_data1_to_send = bytes.fromhex('01 06 07 D0 00 01 48 87')
    ser1_data2_to_send = bytes.fromhex('02 06 07 D0 00 01 48 B4')
    ser1.write(ser1_data1_to_send)
    time.sleep(0.1)
    ser1.write(ser1_data2_to_send)
    time.sleep(0.05)


#14
#下列函数使得电机1、2复位，正反转
def motor_reset2(garbage_can_number) :
    if garbage_can_number == 1 :
        motor_sport(1,2006,25000)
        motor_sport(2,2006,30000)

    if garbage_can_number == 2 :
        motor_sport(1,2006,25000)
        motor_sport(2,2006,1)

    if garbage_can_number == 3 :
        motor_sport(1,2006,1)
        motor_sport(2,2006,1)

    if garbage_can_number == 4 :
        motor_sport(1,2006,1)
        motor_sport(2,2006,30000)

# 15
def whole_process_grab(x,y,angle,garbage_can_number) :
    goto_garbage_position(x,y)
    time.sleep(1.5)
    servo_2_rotate(angle)
    time.sleep(0.5)
    pneumatic_grab()
    open_around()
    time.sleep(0.5)
    goto_garbage_can_position(x,y,garbage_can_number)
    time.sleep(0.8)
    servo_2_rotate(0)
    time.sleep(0.8)
    put_down()
    close_around()
    time.sleep(1)
    motor_reset2(garbage_can_number)


# 16
def whole_process_pan(garbage_can_number) :
    open_around()
    time.sleep(0.5)
    pan_tilt(garbage_can_number)
    time.sleep(0.2)
    close_around()


# 17
def ser_close() :
    if ser1.is_open:
        ser1.close()
    if ser2.is_open:
        ser2.close()
    if ser3.is_open:
        ser3.close()


for i in range(1,50):
    ser1 = serial.Serial('COM6',9600)                                                     #打开电机的串口COM6，波特率为9600
    ser2 = serial.Serial('COM11',115200)                                                  #打开舵机的串口COM11，波特率为9600
    ser3 = serial.Serial('COM7',9600)                                                     #打开气动的串口COM7，波特率为9600
    x=random.randint(10000,20000)
    y=random.randint(10000,25000)
    j=random.randint(-90,90)
    k=random.randint(1,4)
    ser1.flushInput()  # 清空输入缓冲区  
    ser1.flushOutput() # 清空输出缓冲区
    ser2.flushInput()  # 清空输入缓冲区  
    ser2.flushOutput() # 清空输出缓冲区
    ser3.flushInput()  # 清空输入缓冲区  
    ser3.flushOutput() # 清空输出缓冲区
    servo_sport(6,2048,30,50)
    servo_sport(7,2048,30,50)
    whole_process_grab(x,y,j,k)
    time.sleep(4)
    ser_close()
    time.sleep(0.5)

