import numpy as np
import time
import threading


from Sensors import IRCamera, softskin
from Preprocessing import Leg_detector
from Driver import  ControlOdometryDriver as cd
"""portal num"""
# camera_portal = '/dev/ttyUSB2'
lidar_portal = '/dev/ttyUSB0'
IMU_walker_portal = '/dev/ttyUSB3'


Camera = IRCamera.IRCamera()
LD = Leg_detector.Leg_detector(lidar_portal)
CD = cd.ControlDriver()


thread_leg = threading.Thread(target=LD.scan_procedure, args=())
thread_cd = threading.Thread(target=CD.control_part,args=())


def position_calculation(left_leg:np.ndarray, right_leg:np.ndarray, position_buffer:np.ndarray):
    human_position = (left_leg+right_leg)/2


def main_FFL(CD:cd.ControlDriver, LD:Leg_detector.Leg_detector):
    while True:
        time.sleep(0.2)

