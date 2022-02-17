import serial
import serial.tools.list_ports
from Sensors.SensorFunctions import *
from Sensors.SensorConfig import *
import numpy as np
import math
import threading
import re
import os
import sys
import time

class SoftSkin(object):

    def __init__(self, is_STM32: bool = True):

        # serial
        port_name = detect_serials(port_key=SOFTSKIN_LOCATION, sensor_name="Softskin")  # Arduino Mega 2560 ttyACM0
        self.serial = serial.Serial(port_name, SOFTSKIN_BAUDRATE, timeout=None)

        # sensor number
        self.sensor_num = SKIN_SENSOR_NUM

        # data list
        self.data_list = []
        self.pressure_data = np.zeros((self.sensor_num))

        # average filter
        self.average_length = 10
        self.average_buffer = np.zeros((self.average_length, self.sensor_num))

        # detect abnormal signal
        self.max_pressure = 0

        # detect change rate
        self.detect_length = 10
        self.detect_buffer = np.zeros((self.detect_length, self.sensor_num))

        self.skin_unlock_event = threading.Event()
        self.skin_unlock_event.clear()

        self.convert_table = np.zeros((2, 14))
        self.initialize_table()

        # self.build_base_line_data()

    def initialize_table(self):
        self.convert_table[0,:] = np.array(SKIN_TABLE_AC)
        self.convert_table[1,:] = np.array(SKIN_TABLE_PRESSURE)

    def convert_to_force(self):
        for i in range(self.sensor_num):
            for j in range(self.convert_table.shape[1]):
                if self.pressure_data[i] >= self.convert_table[0, j]:
                    if j < self.convert_table.shape[1]-1:
                        ratio = (self.convert_table[1, j+1] - self.convert_table[1, j]) /\
                                (self.convert_table[0, j+1] - self.convert_table[0, j])
                    else:
                        ratio = (self.convert_table[1, j] - self.convert_table[1, j - 1]) / \
                                (self.convert_table[0, j] - self.convert_table[0, j - 1])
                    self.pressure_data[i] = ratio*(self.pressure_data[i] - self.convert_table[0, j]) + self.convert_table[1, j]
                    break

    def read_data(self, is_shown:bool=False):
        try:
            while True:
                data = self.serial.read(20).hex()
                data = data.encode("utf-8")
                print(data)
                print(data.hex())
                for i in range(3, self.sensor_num*2 + 3, 2):
                    self.data_list.append(int.from_bytes(data[i:i + 2], byteorder='big', signed=False))
                self.pressure_data = np.array(self.data_list)
                self.data_list = []
                if is_shown:
                    print(self.pressure_data)
        except BaseException as be:
            print("Data Error:", be)

    def read_and_record(self, record=False, show=False, plot_num=30):
        file_path = DATA_PATH + os.path.sep + "Softskin.txt"
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
