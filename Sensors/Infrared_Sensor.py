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
    def __init__(self, sensor_num:int=1, is_windows:bool=False):
        if is_windows:
            port_name = detect_serials(description="Arduino Mega 2560")
        else:
            port_name = detect_serials(description="ttyACM0")
        baud_rate = 9600
        print(port_name, baud_rate)
        self.pwd = os.path.abspath(os.path.abspath(__file__))
        self.father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
        self.serial = serial.Serial(port_name, baud_rate, timeout=None)
        self.sensor_num = sensor_num
        self.distance_data = np.zeros((sensor_num))
        pass

    def read_data(self, is_shown:bool=False, is_record:bool=False):
        while True:
            try:
                if is_record:
                    file_path = data_path + os.path.sep + "infrared.txt"
                    file = open(file_path,"w")
                self.serial.flushInput()
                one_line_data = self.serial.readline().decode("utf-8")
                # print(one_line_data)
                one_line_data = one_line_data.strip('\n')
                one_line_data = one_line_data.strip('\r')
                one_line_data = one_line_data.split('|')
                # print(one_line_data)
                if len(one_line_data) == self.sensor_num:
                    one_line_data = list(map(float, one_line_data))
                    # one_line_data = list(map(int, one_line_data))
                    self.distance_data = np.array(one_line_data).reshape(self.distance_data.shape)
                    # print(self.raw_data, type(self.raw_data), type(self.raw_data[0]))
                    if is_shown:
                        print(self.distance_data.shape)
                    if is_record:
                        write_data = self.distance_data[0].tolist()
                        write_data.insert(0, time.time())
                        file.write(str(write_data)+"\n")
            except BaseException as be:
                print("Data Error:", be)


if __name__ == '__main__':
    infrared = Infrared_Sensor(sensor_num=1,is_windows=True)
    infrared.read_data(is_shown=True)
    # softskin.record_label()
