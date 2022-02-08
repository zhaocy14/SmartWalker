import serial.tools.list_ports
from Sensors.SensorConfig import *
from Sensors.SensorFunctions import *
import time
import numpy as np

class Speaker(object):
    # port_name = ''
    # baud_rate = 460800  # sometimes it could be 115200
    """Find the target port"""
    def __init__(self, baud_rate=9600):
        """serial information"""
        self.baud_rate = baud_rate
        self.speaker_1_port, _ = detect_serials(port_key=SPEAKER_LOCATION_1,sensor_name="speaker1")
        self.speaker_2_port, _ = detect_serials(port_key=SPEAKER_LOCATION_2,sensor_name="speaker2")
        self.speaker_3_port, _ = detect_serials(port_key=SPEAKER_LOCATION_3,sensor_name="speaker3")
        self.speaker_4_port, _ = detect_serials(port_key=SPEAKER_LOCATION_4,sensor_name="speaker4")
        self.speaker_1 = serial.Serial(self.speaker_1_port,baud_rate)
        self.speaker_2 = serial.Serial(self.speaker_2_port, baud_rate)
        self.speaker_3 = serial.Serial(self.speaker_3_port, baud_rate)
        self.speaker_4 = serial.Serial(self.speaker_4_port, baud_rate)


    def play_clips(self, clip_num:int = 0):

        self.serial.write()
        pass

    def set_volumn(self, volumn:int = 15):
        pass

if __name__ == "__main__":
    speaker_instance = Speaker