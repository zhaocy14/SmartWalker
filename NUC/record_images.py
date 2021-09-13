import time
import numpy as np
import serial
import threading
import os, sys

pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)
from Sensors import IRCamera
from Preprocessing import Leg_detector


data_path = os.path.abspath(
    os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".."  +
    os.path.sep + "data")
# print(resource)
"""portal num"""
camera_portal = '/dev/ttyUSB0'
# lidar_portal = '/dev/ttyUSB1'


"""Camera part"""
Camera = IRCamera.IRCamera()
"""leg detector part"""
# Ld = Leg_detector.Leg_detector(lidar_portal)

thread_camera = threading.Thread(target=Camera.record_write, args=(True, True, data_path, True))
# thread_leg = threading.Thread(target=Ld.scan_procedure, args=(False, True))


thread_camera.start()
# thread_leg.start()