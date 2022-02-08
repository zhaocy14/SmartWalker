import serial
import serial.tools.list_ports
import math
import threading
import os
import sys
import time
import matplotlib.pyplot as plt
import rplidar
from Sensors.SensorConfig import *
from Sensors.SensorFunctions import *


class lidar(object):

    def __init__(self,is_zmq:bool=False):
        super().__init__()
        self.port_name = detect_serials(port_key=LIDAR_DISCRIPTION, sensor_name="LiDAR")
        if not is_zmq:
            self.python_lidar = rplidar.RPLidar(self.port_name)
        else:
            self.python_lidar = []
        self.scan_data_list = []

    def rplidar_scan_procedure(self,is_show:bool=False):
        # present_time = time.time()
        while True:
            # try:
                info = self.python_lidar.get_info()
                health = self.python_lidar.get_health()
                print(info)
                print(health)
                for i, scan in enumerate(self.python_lidar.iter_scans(max_buf_meas=5000)):
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

