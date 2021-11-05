import serial
import serial.tools.list_ports
import numpy as np
import math
import threading
import re
import os
import sys
import time
import matplotlib.pyplot as plt

pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)
data_path = os.path.abspath(
    os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".."  +
    os.path.sep + "data")

def print_serial(port):
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


def detect_serials(description="target device", vid=0x10c4, pid=0xea60):
    ports = serial.tools.list_ports.comports()
    for port in ports:
        print_serial(port)

        if port.description.__contains__(description):
            port_path = port.device
            return port_path
        else:
            print("Cannot find the target device: %s" % description)
    return None


class SoftSkin(object):
    def __init__(self):
        port_name = detect_serials(description="ttyACM0")  # Arduino Mega 2560 ttyACM0
        baud_rate = 115200
        print(port_name, baud_rate)
        self.pwd = os.path.abspath(os.path.abspath(__file__))
        self.father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
        self.serial = serial.Serial(port_name, baud_rate, timeout=None)
        self.raw_data = []  # 保存一帧数据
        self.base_data = []  # 建立一组基准值用于初始化
        self.temp_data = []
        self.port_num = 32
        self.average_length = 10
        self.average_buffer = np.zeros((self.average_length,self.port_num))
        self.max_pressure = 0


        pass

    def read_data(self, is_shown=1):
        try:
            one_line_data = self.serial.readline().decode("utf-8")
            # print(one_line_data)
            one_line_data = one_line_data.strip('SS')
            one_line_data = one_line_data.strip('\n')
            one_line_data = one_line_data.strip('\r')
            one_line_data = one_line_data.split('|')
            # print(one_line_data)
            if is_shown == 1:
                print(one_line_data)
            if len(one_line_data) == self.port_num:
                one_line_data = list(map(float, one_line_data))
                one_line_data = list(map(int, one_line_data))
                self.raw_data = one_line_data
                # print(self.raw_data, type(self.raw_data), type(self.raw_data[0]))
        except BaseException as be:
            print("Data Error:", be)

    def build_base_line_data(self, initial_size=20):
        """
        1.建立一组基准数值
            检测异常值
            取平均值
        :return:
        """
        base_list = []
        for i in range(initial_size):
            # time.sleep(0.01)
            # self.serial.flushInput()
            self.read_data(0)
            if len(self.raw_data) == self.port_num:
                # print(self.raw_data)
                temp_raw_data = self.raw_data
                base_list += temp_raw_data
        mean_base_list = np.array(base_list).reshape([-1, self.port_num])
        add_col = np.ones(mean_base_list.shape[0]).reshape([1, -1])
        mean_base_list = add_col.dot(mean_base_list) / mean_base_list.shape[0]
        self.base_data = mean_base_list.tolist()[0]
        # print(self.base_data)
        self.base_data = list(map(lambda x: int(x) - 1, self.base_data))
        # print(self.base_data, type(self.base_data))
        print("base line data: ", self.base_data)
        pass

    def read_and_record(self, record=False,  show=False, plot=False, plot_num=30,):
        file_path = data_path + os.path.sep + "Softskin.txt"
        plot_array = np.zeros((plot_num, self.port_num))
        if record:
            file = open(file_path, 'w')
        while True:
            try:
                # self.serial.flushInput()
                self.read_data(0)
                if len(self.raw_data) == len(self.base_data):
                    temp_data = np.array(self.raw_data) - np.array(self.base_data)
                    if show:
                        print(temp_data)
                        print(self.max_pressure)
                    if record:
                        time_index = time.time()
                        write_data = temp_data.tolist()
                        write_data.insert(0, time_index)
                        file.write(str(write_data) + '\n')
                        file.flush()
                    self.temp_data = temp_data
                    self.max_pressure = max(max(self.temp_data),self.max_pressure)
                    # time.sleep(0.08)
                    if plot:
                        # plt.ion()
                        plot_array[0:plot_num - 1, :] = plot_array[1:plot_num, :]
                        plot_array[plot_num - 1, :] = np.array(temp_data)
                        plt.clf()
                        plt.xlabel('Time')
                        plt.ylabel('pressure')
                        plt.ylim((-10, 270))
                        plt.plot(range(0, plot_num), plot_array)
                        # plt.ioff()
                        # plt.show()
                        # plt.draw()
                        plt.pause(0.0000000001)
            except BaseException as be:
                print("Data Error:", be)


if __name__ == '__main__':
    from Driver import ControlOdometryDriver as CD


    softskin = SoftSkin()
    softskin.build_base_line_data()
    # while True:
    #     softskin.read_data(0)
    #     print(np.array(softskin.raw_data) - np.array(softskin.base_data))
    def little_test(sk:SoftSkin, driver:CD.ControlDriver):
        while True:
            # print(sk.max_pressure)
            # if sk.max_pressure>=70:
            #     driver.speed = 0
            #     time.sleep(5)
            # driver.speed = 0.3
            # time.sleep(0.5)
            driver.speed = 0.3
            time.sleep(3)
            driver.speed = 0
            time.sleep(3)


    driver = CD.ControlDriver(left_right=0)
    thread_test = threading.Thread(target=little_test,args=(softskin,driver,))
    thread_test.start()

    driver.start()
    softskin.read_and_record(show=True, record=True,plot=False)
    # softskin.record_label()


