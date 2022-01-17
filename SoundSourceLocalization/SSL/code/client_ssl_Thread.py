# !/usr/bin/env python
# -*- coding:utf-8 _*-
# @Author: swang
# @Contact: wang00sheng@gmail.com
# @Project Name: SmartWalker-master
# @File: ssl_loop.py
# @Time: 2022/01/02/20:34
# @Software: PyCharm

import os, sys

code_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.dirname(code_dir)
project_dir = os.path.dirname(module_dir)
sys.path.extend([module_dir, project_dir])
# print('project_dir:', project_dir)

import time
import json
import numpy as np
from scipy import stats
import threading
from multiprocessing import Process, Value, Pipe, Queue

# general lib
from SoundSourceLocalization.mylib import utils
from SoundSourceLocalization.mylib.utils import standard_normalizaion
from SoundSourceLocalization.mylib.audiolib import normalize_single_channel_audio, audio_segmenter_4_numpy, \
    audio_energy_ratio_over_threshold, audio_energy_over_threshold, audioread, audiowrite

# independent systems
from SoundSourceLocalization.SSL_Settings import *
import SoundSourceLocalization.SpeechEnhancement.code.ns_enhance_onnx as ns_enhance_onnx
from SoundSourceLocalization.SSL.code.ssl_audio_processor import *
from SoundSourceLocalization.SSL.code.ssl_feature_extractor import FeatureExtractor
from SoundSourceLocalization.SSL.code.ssl_DOA_model import DOA
from SoundSourceLocalization.SSL.code.ssl_turning import SSLturning
from SoundSourceLocalization.ReinforcementLearning.code.ssl_agent import Agent
from SoundSourceLocalization.ReinforcementLearning.code.ssl_env import MAP_ENV, ONLINE_MAP_ENV
# from SoundSourceLocalization.ReinforcementLearning.code.ssl_actor_critic import Actor, Critic
# # from Communication.Soundlocalization_socket_local import server_receive, server_transmit
# from Communication.Soundlocalization_socket import CLIENT
import Driver.ControlOdometryDriver as CD


class SSL(object):
    def __init__(self, doDenoise=True, useCD=True, seg_len='256ms', isDebug=False, ):
        super(SSL, self).__init__()
        self.seg_len = seg_len
        self.doDenoise = doDenoise
        self.useCD = useCD
        self.isDebug = isDebug
    
    def run(self, walker_client, control_driver):
        while True:
            time.sleep(0.1)
            direction = walker_client.recv(subtopic=SSL_COMMUNICATION_TOPIC)
            if direction is None:
                continue
            print(f'Direction ({direction}) is received')
            ### 接入Owen的模块，传入aim_loca
            if self.useCD:
                direction = direction[0] * 45
                SSLturning(control_driver, direction)
                control_driver.speed = STEP_SIZE / FORWARD_SECONDS
                control_driver.radius = 0
                control_driver.omega = 0
                time.sleep(FORWARD_SECONDS)
                control_driver.speed = 0
                print("movement done.")
            else:
                pass


class SSL_Thread(object):
    def __init__(self, doDenoise=True, useCD=True, seg_len='256ms', isDebug=False, left_right=0, ):
        super(SSL_Thread, self).__init__()
        self.seg_len = seg_len
        self.doDenoise = doDenoise
        self.useCD = useCD
        self.isDebug = isDebug
        self.left_right = left_right
    
    def run(self, walker_client, ):
        cd = CD.ControlDriver(left_right=self.left_right) if self.useCD else ''
        if self.useCD:
            cd_thread = threading.Thread(target=cd.control_part, args=())
            cd_thread.start()
        ssl = SSL(seg_len=self.seg_len, doDenoise=self.doDenoise, useCD=self.useCD, isDebug=self.isDebug, )
        ssl.run(walker_client=walker_client, control_driver=cd, )
