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


def detect_serials(location="1-1.1:1.0", vid=0x10c4, pid=0xea60):
    ports = serial.tools.list_ports.comports()
    for port in ports:
        print_serial(port)

        if port.location.__contains__(location):
            port_path = port.device
            return port_path
        else:
            print("Cannot find the target device: %s" % location)
    return None


class SoftSkin(object):

    def __init__(self, is_STM32: bool = True):

        port_name = detect_serials("1-1.3:1.0")  # Arduino Mega 2560 ttyACM0
        baud_rate = 115200
        print(port_name, baud_rate)
        self.serial = serial.Serial(port_name, baud_rate, timeout=None)
        self.pwd = os.path.abspath(os.path.abspath(__file__))
        self.father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
        self.serial = serial.Serial(port_name, baud_rate, timeout=None)
        self.raw_data = []  # 保存一帧数据
        self.base_data = []  # 建立一组基准值用于初始化
        self.temp_data = []
        self.port_num = 32
        self.average_length = 10
        self.average_buffer = np.zeros((self.average_length, self.port_num))

        # detect abnormal signal
        self.max_pressure = 0
        self.safe_change_rate = 10
        self.emergency_change_rate = 50
        self.detect_length = 10
        self.detect_buffer = np.zeros((self.detect_length, self.port_num))
        self.skin_unlock_event = threading.Event()
        self.skin_unlock_event.clear()

        self.build_base_line_data()
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

    def build_base_line_data(self, initial_size=10):
        """
        expired, no use
        1.建立一组基准数值
            检测异常值
            取平均值
        :return:
        not in use because the original signals are stable enough
        """
        base_list = []
        for i in range(initial_size):
            self.read_data(0)
            if len(self.raw_data) == self.port_num:
                temp_raw_data = self.raw_data
                base_list += temp_raw_data
        mean_base_list = np.array(base_list).reshape([-1, self.port_num])
        add_col = np.ones(mean_base_list.shape[0]).reshape([1, -1])
        mean_base_list = add_col.dot(mean_base_list) / mean_base_list.shape[0]
        self.base_data = mean_base_list.tolist()[0]
        self.base_data = list(map(lambda x: int(x) - 1, self.base_data))
        print("base line data: ", self.base_data)
        pass

    def read_and_record(self, record=False, show=False, plot=False, plot_num=30):
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
                    self.max_pressure = self.temp_data.max()
                    self.detect_buffer[0:-1, :] = self.detect_buffer[1:self.detect_length, :]
                    self.detect_buffer[-1, :] = np.array(self.temp_data)

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

    def update_from_STM32(self, STM32_data: np.ndarray):
        try:
            self.raw_data = STM32_data
        except:
            pass

    def unlock(self):
        while True:
            change_rate = self.detect_buffer[-1, :] - self.detect_buffer[0, :]
            change_rate = change_rate.max()
            if self.safe_change_rate <= change_rate < self.emergency_change_rate:
                print("unlock!")
                break
            time.sleep(0.1)


if __name__ == '__main__':
    skin = SoftSkin()
    # skin.build_base_line_data()
    thread_reading = threading.Thread(target=skin.read_and_record, args=())

    time.sleep(1)
    thread_reading.start()

    skin.unlock()

