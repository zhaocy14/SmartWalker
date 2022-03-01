import serial
import serial.tools.list_ports
import numpy as np
import time
from Sensors.SensorConfig import *
from Sensors.SensorFunctions import *
import matplotlib.pyplot as plt


class Infrared_Sensor(object):

    def __init__(self, sensor_num: int = 5, baud_rate: int = 9600, is_STM32: bool = False):
        port_name, _ = detect_serials(port_key=INFRARED_LOCATION,sensor_name="Infrared_Sensor")
        self.serial = serial.Serial(port_name, baud_rate, timeout=None)
        # sensor_num is the number of the sensors
        self.sensor_num = sensor_num
        self.distance_data = np.zeros((sensor_num))
        # buffer is a time window for filtering data
        self.buffer_length = 50
        self.buffer = np.zeros((self.buffer_length, self.sensor_num))
        # status: whether the sensor is out of range
        self.status = np.zeros((1, self.sensor_num))
        # table: first row is voltage, second row is distance
        self.table_150 = [[20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150],
                          [2.5, 2, 1.55, 1.25, 1.1, 0.85, 0.8, 0.73, 0.7, 0.65, 0.6, 0.5, 0.45, 0.4]]
        self.table_80 = [[8, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80],
                         [2.75, 2.3, 1.65, 1.3, 0.9, 0.8, 0.75, 0.67, 0.6, 0.51, 0.4]]
        self.table_150 = np.array(self.table_150)
        self.table_80 = np.array(self.table_80)
        pass

    def my_inter(self, x, x0, x1, y0, y1):
        #
        return (x - x1) / (x0 - x1) * y0 + (x - x0) / (x1 - x0) * y1

    def turn_to_distance(self, sensor_range: int = 80):
        self.distance_data = self.distance_data / 1024 * 5
        # print(self.distance_data)
        if sensor_range == 150:
            for j in range(self.sensor_num):
                for i in range(self.table_150.shape[1]):
                    if self.distance_data[j] >= self.table_150[1, i]:
                        break
                if i == 0:
                    self.distance_data[j] = 20
                elif i >= self.table_150.shape[1] - 1:
                    self.distance_data[j] = 150
                else:
                    self.distance_data[j] = self.my_inter(self.distance_data[j], self.table_150[1, i - 1],
                                                          self.table_150[1, i],
                                                          self.table_150[0, i - 1], self.table_150[0, i])
        elif sensor_range == 80:
            for j in range(self.sensor_num):
                for i in range(self.table_80.shape[1]):
                    if self.distance_data[j] >= self.table_80[1, i]:
                        break
                if i == 0:
                    self.distance_data[j] = 10
                elif i >= self.table_80.shape[1] - 1:
                    self.distance_data[j] = 80
                else:
                    self.distance_data[j] = self.my_inter(self.distance_data[j], self.table_80[1, i - 1],
                                                          self.table_80[1, i],
                                                          self.table_80[0, i - 1], self.table_80[0, i])

    def read_data(self, is_shown:bool=False, is_record:bool=False, is_average:bool=False):
        if is_record:
            file_path = DATA_PATH + os.path.sep + "infrared.txt"
            file = open(file_path, "w")
        while True:
            try:
                # read data from the Arduino
                one_line_data = self.serial.readline().decode("utf-8")
                one_line_data = one_line_data.strip('\n')
                one_line_data = one_line_data.strip('\r')
                one_line_data = one_line_data.split('|')
                # print(one_line_data)
                if len(one_line_data) == self.sensor_num:
                    one_line_data = list(map(float, one_line_data))
                    self.buffer[0:-1, :] = self.buffer[1:self.buffer_length, :]
                    self.buffer[-1, :] = np.array(one_line_data).reshape(self.distance_data.shape)
                    if is_average:
                        self.distance_data = np.mean(self.buffer,axis=0)
                    else:
                        self.distance_data = np.array(one_line_data).reshape(self.distance_data.shape)
                    # change the value into real voltage:
                    self.turn_to_distance()
                    # self.distance_data = self.distance_data / 1024 * 5
                    # print(self.raw_data, type(self.raw_data), type(self.raw_data[0]))
                    if is_shown:
                        print(self.distance_data)
                        # self.count()
                    if is_record:
                        write_data = self.distance_data.tolist()
                        write_data.insert(0, time.time())
                        file.write(str(write_data)+"\n")
                        file.flush()
            except BaseException as be:
                print("Infrared Error:", be)

    def update_from_STM32(self, STM32_data:np.ndarray, is_average:bool = True):
        try:
            self.buffer[0:-1, :] = self.buffer[1:self.buffer_length, :]
            self.buffer[-1, :] = STM32_data.reshape(self.distance_data.shape)
            if is_average:
                self.distance_data = np.mean(self.buffer, axis=0)
            else:
                self.distance_data = STM32_data
        except:
            pass

if __name__ == '__main__':
    infrared = Infrared_Sensor(sensor_num=6)
    infrared.read_data(is_shown=True,is_average=True)
    # softskin.record_label()
