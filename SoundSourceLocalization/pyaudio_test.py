#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import sys
import pyaudio
from pyaudio import PyAudio, paInt16
import numpy as np
from datetime import datetime
import wave
from scipy.io import wavfile

GCC_LENG = 366
GCC_BIAS = 6
ACTION_SPACE = 8
CHUNK = 1024
RECORD_DEVICE_NAME = "USB Camera-B4.09.24.1"
RECORD_WIDTH = 2
CHANNELS = 4
RATE = 16000

RECORD_SECONDS = 3  # 1
FORMAT = pyaudio.paInt16

FORWARD_SECONDS = 3
STEP_SIZE = 1

pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
print(father_path)
sys.path.append(father_path)

MODEL_PATH = father_path + "/resource/model/save20.ckpt"
WAV_PATH = father_path + "/resource/wav/online"
ONLINE_MODEL_PATH = father_path + "/resource/model/online.ckpt"

# KWS parameters
KWS_WAVE_PATH = father_path + "/resource/stream_tmp"
KWS_MODEL_PATH = father_path + "/resource/Pretrained_models/DNN/follow.pb"
KWS_LABEL_PATH = father_path + "/resource/Pretrained_models/follow_labels.txt"


class recoder:
    NUM_SAMPLES = 2000  # pyaudio内置缓冲大小
    SAMPLING_RATE = 8000  # 取样频率
    LEVEL = 500  # 声音保存的阈值
    COUNT_NUM = 20  # NUM_SAMPLES个取样之内出现COUNT_NUM个大于LEVEL的取样则记录声音
    SAVE_LENGTH = 8  # 声音记录的最小长度：SAVE_LENGTH * NUM_SAMPLES 个取样
    TIME_COUNT = 60  # 录音时间，单位s
    
    Voice_String = []
    
    def __init__(self):
        RECORD_DEVICE_NAME = "USB Camera-B4.09.24.1"
        device_index = -1
        p = PyAudio()
        
        # Recognize Mic device, before loop
        # scan to get usb device
        print('num_device:', p.get_device_count())
        for index in range(p.get_device_count()):
            info = p.get_device_info_by_index(index)
            device_name = info.get("name")
            print("device_name: ", device_name)
            
            # find mic usb device
            if device_name.find(RECORD_DEVICE_NAME) != -1:
                device_index = index
                break
        
        if device_index != -1:
            print("find the device\n", p.get_device_info_by_index(device_index))
            del p
        else:
            print("don't find the device")
            exit()
        self.device_index = device_index
        self.frames = []
    
    def savewav(self, filename, ):
        wf = wave.open(filename, 'wb')
        
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(RECORD_WIDTH)
        wf.setframerate(RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()
    
    def read_and_split_channels_from_file(self, filepath):
        f = wave.open(filepath)
        params = f.getparams()
        num_channel, sample_width, fs, num_frame = params[:4]
        str_data = f.readframes(num_frame)
        f.close()
        audio = np.frombuffer(str_data, dtype=np.short)
        audio = np.reshape(audio, (-1, 4)).T
        
        return audio
    
    def split_channels_from_frames(self, num_channel):
        audio = np.frombuffer(b''.join(self.frames), dtype=np.short)
        audio = np.reshape(audio, (-1, num_channel)).T
        
        return audio
    
    def monitor_from_4mics(self):
        # print("start monitoring ... ")
        p = PyAudio()
        stream = p.open(format=p.get_format_from_width(RECORD_WIDTH),
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=self.device_index)
        # 16 data
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            # print("here")
            frames.append(data)
        # print(len(frames))
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("End monitoring ... ")
        
        return frames
    
    def recoder(self):
        self.frames = self.monitor_from_4mics()
        
        fpath = os.path.join(WAV_PATH, 'test.wav')
        self.savewav(fpath, )
        
        # 将读入的数据转换为数组
        audio = self.split_channels_from_frames(CHANNELS)
        audio_1 = self.read_and_split_channels_from_file(fpath)


if __name__ == "__main__":
    
    r = recoder()
    r.recoder()
    r.savewav("test.wav")
