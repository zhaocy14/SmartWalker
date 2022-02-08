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
            # print_serial(port)
            return port_path
        else:
            print("Cannot find the target device: %s" % location)
    return None


class SoftSkin(object):

    def __init__(self, is_STM32: bool = True):
        port_name = detect_serials("3-3.1")  # Arduino Mega 2560 ttyACM0
        baud_rate = 115200
        print(port_name, baud_rate)

        # serial
        self.serial = serial.Serial(port_name, baud_rate, timeout=None)
        self.pwd = os.path.abspath(os.path.abspath(__file__))
        self.father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")

        # sensor number
        self.sensor_num = 7

        # data list
        self.data_list = []
        self.pressure_data = np.zeros((self.sensor_num))

        # average filter
        self.average_length = 10
        self.average_buffer = np.zeros((self.average_length, self.sensor_num))

        # detect abnormal signal
        self.max_pressure = 0

        # detect change rate
        self.safe_change_rate = 10
        self.emergency_change_rate = 50
        self.detect_length = 10
        self.detect_buffer = np.zeros((self.detect_length, self.sensor_num))


        self.skin_unlock_event = threading.Event()
        self.skin_unlock_event.clear()

        # self.build_base_line_data()

    def read_data(self, is_shown=1):
        try:
            while True:
                data = self.serial.read(20).hex()
                data = data.encode("utf-8")
                print(data)

        except BaseException as be:
            print("Data Error:", be)

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
    skin.read_data()
