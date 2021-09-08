#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   Record_data.py
@Contact :   liumingshanneo@163.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/10/8 19:46   msliu      1.0         None
"""

import serial.tools.list_ports
import os,sys
import time
import numpy as np
import cv2
from PIL import Image
pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)
data_path = os.path.abspath(
    os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".."  +
    os.path.sep + "data")
# resource_video = resource + os.path.sep + "output.avi"
# resource_img = resource + os.path.sep + "output_npdata"

class IRCamera(object):
    # port_name = ''
    # baud_rate = 460800  # sometimes it could be 115200

    """Find the target port"""
    def __init__(self, baud_rate=460800):
        """serial information"""
        self.baud_rate = baud_rate
        self.port_name, self.port_list = self.detect_serials("USB2.0-Serial") #USB-SERIAL CH340       USB2.0-Serial
        self.serial = serial.Serial(self.port_name, self.baud_rate, timeout=None)
        print(self.port_name, self.baud_rate)

        """data processing"""
        self.head_size = 4
        self.head_self = []
        self.data_self = []

        """data output"""
        self.temperature = []
        return

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
    def detect_serials(self, description, vid=0x10c4, pid=0xea60):
        ports = serial.tools.list_ports.comports()
        port_cnt = 0
        port_list = []
        for port in ports:
            self.print_serial(port)

            if port.description.__contains__(description):
                port_list = port.description
                port_path = port.device
                return port_path, port_list
            else:
                print("Cannot find the device: IR Camera")


            # print("%x and %x" % (port.vid, port.pid))
            # if vid == port.vid and port.pid == pid:
            #     port_list.append(port)
            #     port_cnt += 1
        #     这里我还不知道vid和pid是什么东西
        return None, None

    def get_portname_baudrate(self):
        return self.port_name, self.baud_rate

    def check_head_data(self, head_data):
        """
        This function is to detect the head of frame from IR Camera
        :param head_data: The read data could be frame.
        :return:
        """
        if head_data is None:
            print("check_head_data: the head data is None")
            head_data = []
        head = [0x5A, 0x5A, 0x02, 0x06]
        for i in range(self.head_size):
            if head_data[i] != head[i]:
                # print("The head is not caught")
                return False

        return True

    def __fix_pixel(self, ir_list=[]):
        """
        存在可能，某个pixel的数据因为坏掉丢失，用周围的数据进行插值
        :param ir_list: temperature list, the ir data source
        :param x: the position x from 1
        :param y: the position y from 1
        :return: the average temperature to fix one pixel

        """
        # 768个像素依次排列的坐标，经验所得：坏的像素会因环境变化突出，在白背景下
        # 坏的像素的值会超过600
        need_fix_index = [183, 202, 303, 601]
        for i in need_fix_index:
            x = i % 32
            y = i // 32
            temp = (ir_list[x - 1 + (y - 1) * 32] +
                    ir_list[x + (y - 1) * 32] +
                    ir_list[x + 1 + (y - 1) * 32] +
                    ir_list[x - 1 + y * 32] +
                    ir_list[x + 1 + y * 32] +
                    ir_list[x - 1 + (y + 1) * 32] +
                    ir_list[x + (y + 1) * 32] +
                    ir_list[x + 1 + (y + 1) * 32]) / 8
            ir_list.insert(x + y * 32, temp)
            ir_list.pop(x + y * 32 + 1)
        return ir_list

    def get_irdata_once(self, time_index=False):
        temperature = []
        rest_num = 5

        while True:
            s = self.serial.read(1).hex()
            if s != "":
                s = int(s, 16)
            self.head_self.append(s)
            # 对头数据必须如此嵌套，否则无法区分上一帧的数据
            if len(self.head_self) == self.head_size:
                if self.check_head_data(self.head_self):
                    temp = self.serial.read(1540)
                    self.data_self.append(temp.hex())
                    self.head_self.clear()
                else:
                    self.head_self.pop(0)

                if len(self.data_self) == rest_num:
                    ir_data = self.data_self[rest_num - 1]
                    if len(ir_data) != 1540 * 2:
                        # 正常传过来一个字节 0xa5 是一个字节，一个元素表示4位， 然后用string表示一个字母就是一个字节
                        print("the array of ir_data is not 1540", len(ir_data))

                    for i in range(769):
                        t = (int(ir_data[i * 4 + 2:i * 4 + 4], 16) * 256 + int(ir_data[i * 4:i * 4 + 2], 16)) / 100
                        temperature.append(t)

                    """环境温度"""
                    temperature.pop()
                    temperature = self.__fix_pixel(temperature)
                    """插入时间戳"""
                    if time_index:
                        time_index = time.time()
                        temperature.insert(0, time_index)
                    # print(str(temperature))
                    self.data_self.pop(rest_num - 1)
                    self.data_self.pop(0)
                if len(temperature) > 0:
                    self.temperature = temperature
                    # time.sleep(0.2)
                    break
            self.demonstrate_data()
        return temperature

    def record_write(self, write = False, time_index=True, file_path=data_path, demo=False):
        head = []
        data = []
        rest_num = 5
        ir_data_path = file_path + os.path.sep + "ir_data.txt"
        if write:
            file_ir = open(ir_data_path, "w")
        # time_previous = time.time()
        while True:
            s = self.serial.read(1).hex()
            if s != "":
                s = int(s, 16)
            head.append(s)

            if len(head) == self.head_size:
                if self.check_head_data(head):
                    temp = self.serial.read(1540)
                    data.append(temp.hex())
                    head.clear()
                else:
                    head.pop(0)

                # 将读到的数据进行展示
                if len(data) == rest_num:
                    ir_data = data[rest_num - 1]
                    if len(ir_data) != 1540 * 2:
                        # 正常传过来一个字节 0xa5 是一个字节，一个元素表示4位， 然后用string表示一个字母就是一个字节
                        print("the array of ir_data is not 1540", len(ir_data))

                    temperature = []

                    for i in range(769):
                        t = (int(ir_data[i * 4 + 2:i * 4 + 4], 16) * 256 + int(ir_data[i * 4:i * 4 + 2], 16)) / 100
                        temperature.append(t)

                    """环境温度"""
                    temperature.pop()
                    temperature = self.__fix_pixel(temperature)
                    """插入时间戳"""
                    if time_index:
                        time_index = time.time()
                        temperature.insert(0, time_index)
                    if write:
                        file_ir.write(str(temperature) + "\n")
                        file_ir.flush()
                    self.temperature = temperature
                    data.pop(rest_num - 1)
                    data.pop(0)
                    # "查看接收数据频率"
                    # time_new = time.time()
                    # print("frequency:",1/(time_new-time_previous))
                    # time_previous = time_new
                    if demo:
                        self.demonstrate_data()
        if write:
            file_ir.close()

    def demonstrate_data(self, scope=10):
        temperature = []

        if len(self.temperature) != 0:
            for i in self.temperature:
                temperature.append(i)
            if len(self.temperature) == 769:
                """leave time stamp"""
                temperature.pop(0)
            maxtemp = max(temperature)
            mintemp = min(temperature)
            for i in range(len(temperature)):
                temperature[i] = (temperature[i] - mintemp) / (maxtemp - mintemp)
            # for i in range(len(temperature)):
            #     if temperature[i] <= 25:
            #         temperature[i] = 0
            #     else:
            #         temperature[i] = 1
            # npdata = np.array(self.temperature).reshape(24, 32)
            # threshold0 = 28
            # threshold1 = 27
            # for i in range(len(temperature)):
            #     if temperature[i] >= threshold0:
            #         temperature[i] = 1
            #     elif temperature[i] >= threshold1:
            #         temperature[i] = 0.5
            #     else:
            #         temperature[i] = 0
            temperature = np.array(temperature, np.float32).reshape(24, 32)
            im = Image.fromarray(temperature)
            im = im.resize((32 * scope, 24 * scope), Image.BILINEAR)
            im = np.array(im)
            cv2.imshow("Foot", im)
            cv2.waitKey(1)
            return temperature


if __name__ == '__main__':
    ir_data = IRCamera()
    ir_data.record_write()
    # # portname, baudrate = ir_data.get_portname_baudrate()
    # # print(portname, baudrate)
    # for i in range(100):
    #     a = ir_data.get_irdata_once()
    #     print(a)
    #     # print(time.time())IRCamera.py:43
    # while True:
    #     print(ir_data.get_irdata_once())
