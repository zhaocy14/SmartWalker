import serial.tools.list_ports
import os,sys
import serial
import time
import numpy as np
import threading
pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)
data_path = os.path.abspath(
    os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".."  +
    os.path.sep + "data")

class IMU(object):

    def __init__(self, baud_rate=115200,name:str=""):
        """serial information"""
        self.baud_rate = baud_rate
        self.port_name, self.port_list = self.detect_serials("3-2.2.2") #USB-SERIAL CH340   1-3.3
        print(self.port_name)
        self.serial = serial.Serial(self.port_name, self.baud_rate, timeout=None)

        """data processing"""
        self.ACCData = [0.0] * 8
        self.GYROData = [0.0] * 8
        self.AngleData = [0.0] * 8
        self.GEOMAGNETData = [0.0] * 8
        self.FrameState = 0  # 通过0x后面的值判断属于哪一种情况
        self.Bytenum = 0  # 读取到这一段的第几位
        self.CheckSum = 0  # 求和校验位

        self.a = [0.0] * 3
        self.w = [0.0] * 3
        self.Angle = [0.0] * 3
        self.GEOMagnet = [0.0] * 3

        self.one_flame_data = []
        """for txt saving"""
        self.name = name
    """Print the port information"""

    def print_serial(self, port):
        print("---------------[ %s ]---------------" % port.name)
        print("Path: %s" % port.device)
        print("Descript: %s" % port.description)
        print("HWID: %s" % port.hwid)
        if not None == port.manufacturer:
            print("Manufacture: %s" % port.manufacturer)
        if not None == port.product:
            print("Product: %s" % port.product)
        if not None == port.interface:
            print("Interface: %s" % port.interface)
        print()

    """list all the port"""

    def detect_serials(self, description="1-3.3", vid=0x10c4, pid=0xea60):
        ports = serial.tools.list_ports.comports()
        port_cnt = 0
        port_list = []
        for port in ports:
            self.print_serial(port)
            if port.location.__contains__(description):
                port_list = port.description
                port_path = port.device
                self.print_serial(port)
                # print(port.name)
                return port_path, port_list
        print("Cannot find the device: IMU")
        return None, None

    """ Collect all of the original data"""
    def collect_all(self,show=False):  # 新增的核心程序，对读取的数据进行划分，各自读到对应的数组里
        self.serial.flushInput()
        inputdata = self.serial.read(44)
        for data in inputdata:  # 在输入的数据进行遍历
            # data = ord(data)  # python 自动转换了
            if self.FrameState == 0:  # 当未确定状态的时候，进入以下判断
                if data == 0x55 and self.Bytenum == 0:  # 0x55位于第一位时候，开始读取数据，增大bytenum
                    self.CheckSum = data
                    self.Bytenum = 1
                    continue
                elif data == 0x51 and self.Bytenum == 1:  # 在byte不为0 且 识别到 0x51 的时候，改变frame
                    """acceleration"""
                    self.CheckSum += data
                    self.FrameState = 1
                    self.Bytenum = 2
                elif data == 0x52 and self.Bytenum == 1:  # 同理
                    """omega"""
                    self.CheckSum += data
                    self.FrameState = 2
                    self.Bytenum = 2
                elif data == 0x53 and self.Bytenum == 1:
                    """angle"""
                    self.CheckSum += data
                    self.FrameState = 3
                    self.Bytenum = 2
                elif data == 0x54 and self.Bytenum == 1:
                    """geomagnetism"""
                    self.CheckSum += data
                    self.FrameState = 4
                    self.Bytenum = 2
            elif self.FrameState == 1:  # acc    #已确定数据代表加速度
                if self.Bytenum < 10:  # 读取8个数据
                    self.ACCData[self.Bytenum - 2] = data  # 从0开始
                    self.CheckSum += data
                    self.Bytenum += 1
                else:
                    if data == (self.CheckSum & 0xff):  # 假如校验位正确
                        self.a = self.get_acc(self.ACCData)
                    self.CheckSum = 0  # 各数据归零，进行新的循环判断
                    self.Bytenum = 0
                    self.FrameState = 0
            elif self.FrameState == 2:  # gyro 角速度
                if self.Bytenum < 10:
                    self.GYROData[self.Bytenum - 2] = data
                    self.CheckSum += data
                    self.Bytenum += 1
                else:
                    if data == (self.CheckSum & 0xff):
                        self.w = self.get_gyro(self.GYROData)
                    self.CheckSum = 0
                    self.Bytenum = 0
                    self.FrameState = 0
            elif self.FrameState == 3:  # angle
                if self.Bytenum < 10:  # 读取8个数据
                    self.AngleData[self.Bytenum - 2] = data  # 从0开始
                    self.CheckSum += data
                    self.Bytenum += 1
                else:
                    if data == (self.CheckSum & 0xff):  # 假如校验位正确
                        self.Angle = self.get_angle(self.AngleData)
                    self.CheckSum = 0  # 各数据归零，进行新的循环判断
                    self.Bytenum = 0
                    self.FrameState = 0
            elif self.FrameState == 4:  # 地磁
                if self.Bytenum < 10:
                    self.GEOMAGNETData[self.Bytenum - 2] = data
                    self.CheckSum += data
                    self.Bytenum += 1
                else:
                    if data == (self.CheckSum & 0xff):
                        self.GEOMagnet = self.get_geo_magnet(self.GEOMAGNETData)
                    self.CheckSum = 0
                    self.Bytenum = 0
                    self.FrameState = 0
            if show:
                try:
                    d = self.a + self.w + self.Angle + self.GEOMagnet
                    print("Acceleration(g):%10.3f %10.3f %10.3f Omega(deg/s):%10.3f %10.3f %10.3f Angle(deg):%10.3f %10.3f %10.3f GeoMagnet:%10.3f %10.3f %10.3f" % d)
                except:
                    continue
            # print(self.a, self.w, self.Angle, self.GEOMagnet,"\r",end="")

    def collect_all_bluetooth_version(self,show:bool=False):
        self.serial.flushInput()
        magnet_data = b'\xFF\xAA\x27\x3A\x00'
        # self.serial.write(magnet_data)
        inputdata = self.serial.read(20)
        for data in inputdata:
            # print("%#x"%data)
            # print(self.FrameState)
            if self.FrameState == 0:  # 当未确定状态的时候，进入以下判断
                if data == 0x55 and self.Bytenum == 0:  # 0x55位于第一位时候，开始读取数据，增大bytenum
                    self.Bytenum = 1
                    continue
                elif data == 0x61 and self.Bytenum == 1:  # 在byte不为0 且 识别到 0x51 的时候，改变frame
                    """acceleration omega angle """
                    self.FrameState = 1
                    self.Bytenum = 2
                elif data == 0x71 and self.Bytenum == 1:
                    """geo_magnet"""
                    self.FrameState = 2
                    self.Bytenum = 2
            elif self.FrameState == 1:
                if self.Bytenum < 8:  # 读取6个数据 acc
                    self.ACCData[self.Bytenum - 2] = data  # 从0开始
                    self.Bytenum += 1
                elif self.Bytenum < 14:
                    self.GYROData[self.Bytenum - 8] = data  # 从0开始
                    self.Bytenum += 1
                elif self.Bytenum < 20:
                    self.AngleData[self.Bytenum - 14] = data
                    self.Bytenum += 1
                else:
                    self.a = self.get_acc(self.ACCData)
                    self.w = self.get_gyro(self.GYROData)
                    self.Angle = self.get_angle(self.AngleData)
                    self.FrameState = 0
                    self.Bytenum = 0
        if show:
            print(self.a,self.w,self.Angle)
            pass

    """Calculate the accerelation"""
    def get_acc(self,datahex):
        axl = datahex[0]
        axh = datahex[1]
        ayl = datahex[2]
        ayh = datahex[3]
        azl = datahex[4]
        azh = datahex[5]

        k_acc = 16.0

        acc_x = (axh << 8 | axl) / 32768.0 * k_acc
        acc_y = (ayh << 8 | ayl) / 32768.0 * k_acc
        acc_z = (azh << 8 | azl) / 32768.0 * k_acc
        if acc_x >= k_acc:
            acc_x -= 2 * k_acc
        if acc_y >= k_acc:
            acc_y -= 2 * k_acc
        if acc_z >= k_acc:
            acc_z -= 2 * k_acc

        return acc_x, acc_y, acc_z


    def get_gyro(self,datahex):
        wxl = datahex[0]
        wxh = datahex[1]
        wyl = datahex[2]
        wyh = datahex[3]
        wzl = datahex[4]
        wzh = datahex[5]
        k_gyro = 2000.0

        gyro_x = (wxh << 8 | wxl) / 32768.0 * k_gyro
        gyro_y = (wyh << 8 | wyl) / 32768.0 * k_gyro
        gyro_z = (wzh << 8 | wzl) / 32768.0 * k_gyro
        if gyro_x >= k_gyro:
            gyro_x -= 2 * k_gyro
        if gyro_y >= k_gyro:
            gyro_y -= 2 * k_gyro
        if gyro_z >= k_gyro:
            gyro_z -= 2 * k_gyro
        return gyro_x, gyro_y, gyro_z


    def get_angle(self,datahex):
        rxl = datahex[0]
        rxh = datahex[1]
        ryl = datahex[2]
        ryh = datahex[3]
        rzl = datahex[4]
        rzh = datahex[5]
        k_angle = 180.0

        angle_x = (rxh << 8 | rxl) / 32768.0 * k_angle
        angle_y = (ryh << 8 | ryl) / 32768.0 * k_angle
        angle_z = (rzh << 8 | rzl) / 32768.0 * k_angle
        if angle_x >= k_angle:
            angle_x -= 2 * k_angle
        if angle_y >= k_angle:
            angle_y -= 2 * k_angle
        if angle_z >= k_angle:
            angle_z -= 2 * k_angle
        return angle_x, angle_y, angle_z

    def get_geo_magnet(self,datahex):
        gxl = datahex[0]
        gxh = datahex[1]
        gyl = datahex[2]
        gyh = datahex[3]
        gzl = datahex[4]
        gzh = datahex[5]
        k_geo_magnet = 1
        geo_magnet_x = (gxh << 8 | gxl)
        geo_magnet_y = (gyh << 8 | gyl)
        geo_magnet_z = (gzh << 8 | gzl)
        # if geo_magnet_x >= k_geo_magnet:
        #     geo_magnet_x -= 2 * k_geo_magnet
        # if geo_magnet_y >= k_geo_magnet:
        #     geo_magnet_y -= 2 * k_geo_magnet
        # if geo_magnet_z >= k_geo_magnet:
        #     geo_magnet_z -= 2 * k_geo_magnet
        return geo_magnet_x, geo_magnet_y, geo_magnet_z

    def read_record(self, time_delay=0,show=False, file_path=data_path, bt_vetsion:bool=False):
        IMU_data_path = file_path + os.path.sep + "IMU" + str(self.name) + ".txt"
        file_IMU = open(IMU_data_path, "w")
        while True:
            if bt_vetsion:
                self.collect_all_bluetooth_version(show=show)
            else:
                self.collect_all(show=show)
            # Add time stamp
            combine_data = list([time.time()]) + list(self.a) + list(self.w) + list(self.Angle)
            # print(combine_data)
            file_IMU.write(str(combine_data) + "\n")
            file_IMU.flush()
            time.sleep(time_delay)

    def collect_data(self,time_delay=0,show=False):
        while True:
            self.collect_all(show)
            # Add time stamp
            self.one_flame_data = list([time.time()]) + list(self.a) + list(self.w) + list(self.Angle)
            # print(self.one_flame_data)
            time.sleep(time_delay)

if __name__ == '__main__':
    IMU_1 = IMU()
    time.sleep(2)
    IMU_1.read_record(show=True,bt_vetsion=True)


