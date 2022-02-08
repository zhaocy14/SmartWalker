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
        self.speaker_1_port, _ = detect_serials(port_key=SPEAKER_LOCATION_1, sensor_name="speaker1")
        self.speaker_2_port, _ = detect_serials(port_key=SPEAKER_LOCATION_2, sensor_name="speaker2")
        self.speaker_3_port, _ = detect_serials(port_key=SPEAKER_LOCATION_3, sensor_name="speaker3")
        self.speaker_4_port, _ = detect_serials(port_key=SPEAKER_LOCATION_4, sensor_name="speaker4")
        self.speaker_1 = serial.Serial(self.speaker_1_port, baud_rate)
        self.speaker_2 = serial.Serial(self.speaker_2_port, baud_rate)
        self.speaker_3 = serial.Serial(self.speaker_3_port, baud_rate)
        self.speaker_4 = serial.Serial(self.speaker_4_port, baud_rate)

    def set_one_speaker(self, command: bytes, speaker_num: int):
        """set one speaker"""
        if speaker_num == 1:
            self.speaker_1.write(command)
        elif speaker_num == 2:
            self.speaker_2.write(command)
        elif speaker_num == 3:
            self.speaker_3.write(command)
        elif speaker_num == 4:
            self.speaker_4.write(command)

    def set_all_speaker(self, command: bytes):
        """set all the speakers"""
        for i in range(1, 5):
            self.set_one_speaker(command, speaker_num=i)

    def play_song(self, command_num: int = 0, is_all:bool=True):
        """play the specific song"""
        song_num = command_num.to_bytes(1, byteorder='little', signed=False)
        command = PLAY_SONG + song_num
        command += uchar_checksum(command).to_bytes(1, byteorder='little', signed=False)
        if is_all:
            self.set_all_speaker(command)
        else:
            self.set_one_speaker(command, 1)

    def initialize_for_microphone(self):
        """sequentially play the speaker """
        self.speaker_1.write(PLAY)
        time.sleep(MIC_INI_TIME_GAP)
        self.speaker_2.write(PLAY)
        time.sleep(MIC_INI_TIME_GAP)
        self.speaker_3.write(PLAY)
        time.sleep(MIC_INI_TIME_GAP)
        self.speaker_4.write(PLAY)

    def set_volume(self, volume: int = 20):
        """set volume for every speaker"""
        if volume <= MIN_VOLUME:
            volume = MIN_VOLUME
        elif volume >= MAX_VOLUME:
            volume = MAX_VOLUME
        command = volume.to_bytes(1, byteorder="little", signed=False)
        command = VOLUME_SET + command
        command += uchar_checksum(command).to_bytes(1, byteorder="little", signed=False)
        self.set_all_speaker(command)
        pass


if __name__ == "__main__":
    speaker_instance = Speaker()
    speaker_instance.play_clips()
