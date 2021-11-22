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


class lidar(rplidar.RPLidar):

    def __init__(self):
        self.port_name = detect_serials(description="CP2102 USB")
        super().__init__(self.port_name)
        self.scan_data_list = []

    def scan_procedure(self,is_show:bool=False):
        # present_time = time.time()
        while True:
            try:
                info = self.get_info()
                print(info)
                health = self.get_health()
                print(health)
                for i, scan in enumerate(self.iter_scans(max_buf_meas=5000)):
                    # new_time = time.time()
                    # print("frequency:",1/(new_time-present_time))
                    # present_time = new_time
                    self.scan_data_list = scan
                    if is_show:
                        print(self.scan_data_list)
            except BaseException as be:
                self.clean_input()
                # self.stop()
                # self.stop_motor()

if __name__ == "__main__":
    lidar_instance = lidar()
    thread_lidar_scan = threading.Thread(target=lidar_instance.scan_procedure,args=())
    thread_lidar_scan.start()