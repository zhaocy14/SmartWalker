import os, sys

#   DATA PATH
pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)
DATA_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".." + os.path.sep + "data")
print(sys.path)

"""PORT Location or description"""
CAMERA_LOCATION = "3.2-1"
IMU_LOCATION = "3.2-1"
STM32_SERIAL_NUM = "STM32 STLink - ST-Link VCP Ctrl"
LIDAR_DISCRIPTION = "CP2102 USB"

#   Softskin Configuration
SKIN_SENSOR_NUM = 7

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
