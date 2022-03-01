import serial
import serial.tools.list_ports
import math
import threading
import os
import sys
import time
import matplotlib.pyplot as plt
import rplidar
from Communication.Modules.Receive import ReceiveZMQ
from Sensors.SensorConfig import *
from Sensors.SensorFunctions import *


class LiDAR(object):

    def __init__(self,is_zmq:bool=True):
        super().__init__()
        self.port_name, _ = detect_serials(port_key=LIDAR_DISCRIPTION, sensor_name="LiDAR")
        if not is_zmq:
            self.python_lidar = rplidar.RPLidar(self.port_name)
        else:
            self.python_lidar = rplidar.RPLidar()
        # store the data
        self.scan_data_list = []
        # zmq part
        self.rzo = ReceiveZMQ.get_instance()
        self.zmq_refresh_theta = 0
        self.zmq_temp_list = []
        self.zmq_scan_list = []
        self.theta_flag = 0

    def python_scan(self,is_show:bool=False):
        # present_time = time.time()
        while True:
            try:
                info = self.python_lidar.get_info()
                health = self.python_lidar.get_health()
                print(info)
                print(health)
                for i, scan in enumerate(self.python_lidar.iter_scans(max_buf_meas=5000)):
                    self.scan_data_list = scan
                    if is_show:
                        print(self.scan_data_list)
            except BaseException as be:
                self.python_lidar.clean_input()
                self.python_lidar.stop()
                self.python_lidar.stop_motor()

    def zmq_get_one_round(self, zmq_data: dict):
        """
        use zmq to get lidar data from C++
        get scan list data per round
        """
        theta = float(zmq_data["theta"])
        dist = float(zmq_data["dist"])
        quality = float(zmq_data["q"])
        if theta < self.zmq_refresh_theta:
            #   if theta become 0 degree from 360 degree
            self.zmq_scan_list = self.zmq_scan_list
            self.zmq_scan_list = []
        self.zmq_refresh_theta = theta
        self.zmq_scan_list.append([quality, theta, dist])

    def zmq_scan(self,is_show:bool=False):
        while True:
            try:
                for scan in self.rzo.startLidar():
                    self.zmq_get_one_round(scan)
                    if len(self.zmq_temp_list) == 1:
                        self.scan_raw_data = np.array(self.zmq_scan_list)
                        img = self.turn_to_img(self.zmq_scan_list)
                        self.detect_leg_boundary_version(self.kmeans, img, show=show)
                        self.detect_obstacle(img=img)
            except BaseException as be:
                pass

if __name__ == "__main__":
    lidar_instance = LiDAR()
    lidar_instance.rplidar_scan_procedure(True)

