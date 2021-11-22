import os, sys
pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)
from signal import signal, SIGINT
from sys import exit
import docker
import time

from Communication.Modules.Driver_recv import DriverRecv


class CppCommand(object):
    _instance = None
    _sensors_running = False
    _nav_running = False
    _draw_running = False
    
    @staticmethod
    def get_instance(lidar_port="/dev/ttyUSB0", imu_port="/dev/ttyUSB1"):
        if CppCommand._instance is None:
            CppCommand(lidar_port=lidar_port, imu_port=imu_port)
        return CppCommand._instance


    def get_id(self):
        return self._id

    
    def handler(self, signal_received, frame):
        # Handle any cleanup here
        print('SIGINT or CTRL-C detected. Exiting gracefully')
        self.stop_navigation()
        self.stop_drawing()
        self.stop_sensors()
        time.sleep(3)
        exit(0)


    def __init__(self, lidar_port="/dev/ttyUSB0", imu_port="/dev/ttyUSB4"):
        if CppCommand._instance is not None:
            raise Exception('only one instance can exist')
        else:
            self._id = id(self)
            CppCommand._instance = self
        self.client = docker.from_env()
        self.container = self.client.containers.get('SMARTWALKER_CARTO')
        self.lidar_port = lidar_port
        self.imu_port = imu_port
        # Initialize the driver receiver object
        self.drvObj = DriverRecv(mode="offline")
        signal(SIGINT, self.handler)


    """
        online mode is for realtime sensor data generation
        offline mode is for recording the current sensor data, cannot be used for navigation or map drawing
        stdout is to determine the output of console, need to run in a new thread in order to unblock the process
    """
    def start_sensors(self, mode="online", stdout=False):
        if not self._sensors_running:
            _, stream = self.container.exec_run("/app/smartwalker_cartomap/build/start_record %s %s %s" % (self.lidar_port, self.imu_port, mode), stream=True)
            if stdout:
                for data in stream:
                    print(data.decode(), end="")
            self._sensors_running = True
        else:
            print("Sensors already running ...")


    def stop_sensors(self):
        self.container.exec_run("pkill -INT -f 'start_record*'")
        self._sensors_running = False


    """
        online mode is for realtime navigation
        offline mode is mainly for testing purpose
        stdout is to determine the output of console, need to run in a new thread in order to unblock the process
    """
    def start_navigation(self, map_file="latest", mode="online", testing="", stdout=False, driver_ctrl=False):
        filter = "60"
        if mode == "online":
            # Start the sensors if it is not runing
            if not self._sensors_running:
                self.start_sensors()
            if driver_ctrl:
                self.drvObj.start(use_thread=True)

        if not self._nav_running:
            _, stream = self.container.exec_run("/app/smartwalker_cartomap/build/start_navigation %s %s %s %s" % (map_file, filter, mode, testing), stream=True)
            if stdout:
                for data in stream:
                    print(data.decode(), end="")
            self._nav_running = True
        else:
            print("Navigation already running ...")


    def stop_navigation(self):
        self.container.exec_run("pkill -INT -f 'start_navigation*'")
        self._nav_running = False


    """
        stdout is to determine the output of console, need to run in a new thread in order to unblock the process
    """
    def start_drawing(self, stdout=False):
        if not self._sensors_running:
            self.start_sensors()
            
        if not self._draw_running:
            _, stream = self.container.exec_run("/app/smartwalker_cartomap/build/draw_map_realtime", stream=True)
            if stdout:
                for data in stream:
                    print(data.decode(), end="")
            self._draw_running = True
        else:
            print("Drawing already running ...")


    def stop_drawing(self):
        self.container.exec_run("pkill -INT -f 'draw_map_realtime*'")
        self._draw_running = False