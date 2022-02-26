import os, sys
import numpy as np
#   DATA PATH
PWD = os.path.abspath(os.path.abspath(__file__))
FATHER_PATH = os.path.abspath(os.path.dirname(PWD) + os.path.sep + "..")
sys.path.append(FATHER_PATH)
DATA_PATH = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".." + os.path.sep + "data")

#   PORT Location or description
CAMERA_LOCATION = "3.2-1"
CAMERA_BAUDRATE = 460800
IMU_LOCATION = "3-2.2.2"
IMU_BAUDRATE = 9600
STM32_SERIAL_NUM = "0669"
STM32_BAUDRATE = 115200
STM32_SERIAL_DESCRIPTION = "STM32 STLink - ST-Link VCP Ctrl"
LIDAR_DISCRIPTION = "CP2102 USB"
SOFTSKIN_LOCATION = "COM3"
SOFTSKIN_BAUDRATE = 115200
INFRARED_LOCATION = "3-2.3"

#   Softskin Configuration
SKIN_SENSOR_NUM = 7
SKIN_TABLE_PRESSURE = [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # for AC to pressure converting
SKIN_TABLE_AC = [51, 1912, 2724, 3011, 3163, 3340, 3455, 3522, 3572, 3608, 3633, 3656, 3680, 3697]
SKIN_MAX_THRESHOLD = 8  # Abnormal maximum pressure
SKIN_SAFE_CHANGE_RATE = 10  # Safe pressure change rate for unlocking the walker
SKIN_EMERGENCY_CHANGE_RATE = 50     # Abnormal pressure change rate for locking the walker

#   LiDAR Configuration
# scanning configurations
SCAN_SIZE = 300
HALF_SIZE = int(SCAN_SIZE / 2)
# old version of filtering useless data
COLUMN_BOUNDARY = HALF_SIZE - 20
BOTTOM_BOUNDARY = HALF_SIZE - 100
FILTER_THETA = 150
# new version of filtering useless data
WALKER_TOP_BOUNDARY = 12
WALKER_BOTTOM_BOUNDARY = 50
WALKER_LEFT_BOUNDARY = 29
WALKER_RIGHT_BOUNDARY = 27
# center point is the geometry center of the walker
CENTER_POINT = np.array([HALF_SIZE + 45, HALF_SIZE])
# obstacle part
OBSTACLE_DISTANCE = 15  # 15 cm detection
FRONT_OBSTACLE_DISTANCE = OBSTACLE_DISTANCE
SIDE_OBSTACLE_DISTANCE = OBSTACLE_DISTANCE

#   Speaker USB port
SPEAKER_LOCATION_1 = "3-2.1.1"
SPEAKER_LOCATION_2 = "3-2.1.2"
SPEAKER_LOCATION_3 = "3-2.1.3"
SPEAKER_LOCATION_4 = "3-2.1.4"

#   Speaker Command
PLAY_SONG = b'\xAA\x07\x02\x00'
PLAY = b'\xAA\x02\x00\xAC'
MIC_INI_TIME_GAP = 1.5 # unit:second
MAX_VOLUME = 30
MIN_VOLUME = 1
VOLUME_SET = b'\xAA\x13\x01'
