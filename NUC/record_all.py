import time
import numpy as np
import serial
import threading
import os, sys

pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)

from Sensors import IMU, IRCamera, softskin
from FrontFollowing.Preprocessing import Leg_detector
from Driver import ControlOdometryDriver as CD

data_path = os.path.abspath(
    os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".."  +
    os.path.sep + "data")

print(data_path)

"""portal num"""
camera_portal = '/dev/ttyUSB0'
lidar_portal = '/dev/ttyUSB2'
IMU_walker_portal = '/dev/ttyUSB1'

"""IMU part"""
# IMU_human = IMU.IMU()
# IMU_human.open_serial("/dev/ttyUSB1")
IMU_walker = IMU.IMU(name="")
IMU_walker.open_serial(IMU_walker_portal)
"""Camera part"""
Camera = IRCamera.IRCamera()
"""skin part"""
Skin = softskin.SoftSkin()
"""control driver(record mode)"""
Cd = CD.ControlDriver(record_mode=True)
"""leg detector part"""
Ld = Leg_detector.Leg_detector(lidar_portal)
"""initiate skin part"""
time.sleep(1)
Skin.build_base_line_data()

seperately_recording = True

if seperately_recording:
    # thread_skin = threading.Thread(target=Skin.read_and_record, args=(True,))
    thread_camera = threading.Thread(target=Camera.record_write, args=(True, True, data_path, True))
    # thread_IMU_human = threading.Thread(target=IMU_human.read_record,args=())
    thread_IMU_walker = threading.Thread(target=IMU_walker.read_record, args=(0,False,data_path))
    thread_cd = threading.Thread(target=Cd.control_part, args=())
    thread_leg = threading.Thread(target=Ld.scan_procedure, args=(False, True,))

    # thread_skin.start()
    thread_camera.start()
    # thread_IMU_human.start()
    thread_IMU_walker.start()
    thread_cd.start()
    thread_leg.start()

else:
    pass
    # thread_skin = threading.Thread(target=Skin.read_and_record, args=())
    # # thread_IMU_human = threading.Thread(target=IMU_human.read_record,args=())
    # thread_IMU_walker = threading.Thread(target=IMU_walker.read_record, args=())
    # thread_cd = threading.Thread(target=Cd.control_part, args=())
    # thread_leg = threading.Thread(target=Ld.main_procedure, args=())
    #
    # thread_skin.start()
    # # thread_IMU_human.start()
    # thread_IMU_walker.start()
    # thread_cd.start()
    # thread_leg.start()
    #
    # file_path = resource + os.path.sep + "all_data.txt"
    # file_data = open(file_path, "w")
    # while True:
    #     # present_time = time.time()
    #     Camera.get_irdata_once()
    #     data = []
    #     if len(Camera.temperature) == 768:
    #         time_index = time.time()
    #         data.append(time_index)
    #         data = data + Camera.temperature
    #         data = data +
    #         pass
