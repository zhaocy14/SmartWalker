import serial
import serial.tools.list_ports
import math
import threading
import os
import sys
import time
import matplotlib.pyplot as plt
import rplidar

pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)
data_path = os.path.abspath(
    os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".." +
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
    if not None == port.vid:
        print("Vid:",port.vid)
    if not None == port.pid:
        print("Pid:",port.pid)
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


class lidar(object):

    def __init__(self,is_zmq:bool=False):
        super().__init__()
        self.port_name = detect_serials(description="CP2102 USB")
        if not is_zmq:
            self.lidar_0 = rplidar.RPLidar(self.port_name)
        else:
            self.lidar_0 = []
        self.scan_data_list = []

    def rplidar_scan_procedure(self,is_show:bool=False):
        # present_time = time.time()
        while True:
            # try:
                info = self.lidar_0.get_info()
                health = self.lidar_0.get_health()
                print(info)
                print(health)
                for i, scan in enumerate(self.lidar_0.iter_scans(max_buf_meas=5000)):
                    self.scan_data_list = scan
                    if is_show:
                        print(self.scan_data_list)
            # except BaseException as be:
            #     self.lidar_0.clean_input()
                # self.lidar_0.stop()
                # self.lidar_0.stop_motor()

    def zmq_scan(self,is_show:bool=False):
        while True:
            try:
                pass
            except Exception as be:
                pass

if __name__ == "__main__":
    lidar_instance = lidar()
    lidar_instance.rplidar_scan_procedure(True)

