import serial
import serial.tools.list_ports
import numpy as np
import os,sys
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


class Infrared_Sensor(object):
    def __init__(self, sensor_num:int=1, baud_rate:int=14400, is_windows:bool=False):
        if is_windows:
            port_name = detect_serials(description="Arduino Mega 2560")
        else:
            port_name = detect_serials(description="ttyACM0")
        print(port_name, baud_rate)
        self.pwd = os.path.abspath(os.path.abspath(__file__))
        self.father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
        self.serial = serial.Serial(port_name, baud_rate, timeout=None)
        # sensor_num is the number of the sensors
        self.sensor_num = sensor_num
        self.distance_data = np.zeros((sensor_num))
        # buffer is a time window for filtering data
        self.buffer_length = 5
        self.buffer = np.zeros((self.buffer_length,self.sensor_num))
        self.average_weight = np.ones((1,self.buffer_length))/self.buffer_length
        # status: whether the sensor is out of range
        self.status = np.zeros((1,self.sensor_num))
        # table: first row is voltage, second row is distance
        self.table = [[20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150],
                      [2.5, 2, 1.55, 1.25, 1.1, 0.85, 0.8, 0.73, 0.7, 0.65, 0.6, 0.5, 0.45, 0.4]]
        self.table = np.array(self.table)
        pass

    def check_stability(self):
        mean = np.mean(self.buffer,axis=0)
        print(mean)
        print(self.buffer)
        for i in range(self.sensor_num):
            print(self.buffer[:,i]-mean[i])
            self.status[0,i] = np.mean((self.buffer[:,i]-mean[i])**2)
        self.status = np.sqrt(self.status)
        print("status:",self.status)

    def my_inter(self,x,x0,x1,y0,y1):
        return (x-x1)/(x0-x1)*y0 + (x-x0)/(x1-x0)*y1

    def turn_to_distance(self):
        for i in range(self.sensor_num):
            for i in range(self.table.shape[1]):
                if self.distance_data[j] >= self.table[1,i]:
                    break
            if i == 0:
                self.distance_data[j] = 20
            elif i == self.table.shape[1]:
                self.distance_data[j] = 150
            else:
                self.distance_data[j] = self.my_inter(self.distance_data[j],self.table[1,i-1],self.table[1,i],
                                                      self.table[0,i-1],self.table[0,i])



    def read_data(self, is_shown:bool=False, is_record:bool=False, is_average:bool=False):
        # current_time = time.time()
        while True:
            try:
                if is_record:
                    file_path = data_path + os.path.sep + "infrared.txt"
                    file = open(file_path,"w")
                # self.serial.flushInput()
                one_line_data = self.serial.readline().decode("utf-8")
                # print("original:",one_line_data)
                one_line_data = one_line_data.strip('\n')
                one_line_data = one_line_data.strip('\r')
                one_line_data = one_line_data.split('|')
                # print("strip:",one_line_data)
                if len(one_line_data) == self.sensor_num:
                    one_line_data = list(map(float, one_line_data))
                    self.buffer[0:-1, :] = self.buffer[1:self.buffer_length, :]
                    self.buffer[-1, :] = np.array(one_line_data).reshape(self.distance_data.shape)
                    # maybe we can use standard deviation to check the consecutiveness
                    # self.check_stability()
                    if is_average:
                        # self.distance_data = np.matmul(self.average_weight,self.buffer)[0]
                        self.distance_data = np.mean(self.buffer,axis=0)
                    else:
                    # one_line_data = list(map(int, one_line_data))
                        self.distance_data = self.buffer[-1,:]
                    # change the value into real voltage:
                    self.distance_data = self.distance_data/1024*5


                    # print(self.raw_data, type(self.raw_data), type(self.raw_data[0]))
                    if is_shown:
                        print(self.distance_data)
                    if is_record:
                        write_data = self.distance_data[0].tolist()
                        write_data.insert(0, time.time())
                        file.write(str(write_data)+"\n")
                # new_time = time.time()
                # print("frequency:%f"%(1/(new_time-current_time)))
                # current_time=new_time
            except BaseException as be:
                print("Data Error:", be)


if __name__ == '__main__':
    infrared = Infrared_Sensor(sensor_num=7,baud_rate=115200, is_windows=True)
    infrared.read_data(is_shown=True,is_average=True)
    # softskin.record_label()
