# !/usr/bin/env python
# -*- coding:utf-8 _*-
# @Author: swang
# @Contact: wang00sheng@gmail.com
# @Project Name: SmartWalker-master
# @File: ssl_loop.py
# @Time: 2022/01/02/20:34
# @Software: PyCharm

import os, sys

CRT_DIR = os.path.dirname(os.path.abspath(__file__))
F_PATH = os.path.dirname(CRT_DIR)
FF_PATH = os.path.dirname(F_PATH)
sys.path.extend([CRT_DIR, F_PATH, FF_PATH, ])
# print('sys.path:', sys.path)

import time
import json
import numpy as np
from scipy import stats
import threading

# independent systems
from SoundSourceLocalization.SSL_Settings import *
from SoundSourceLocalization.SSL.code.ssl_turning import SSLturning
import Driver.ControlOdometryDriver as CD


class SSL(object):
    def __init__(self, doDenoise=True, useCD=True, seg_len='256ms', isDebug=False, ):
        super(SSL, self).__init__()
        self.seg_len = seg_len
        self.doDenoise = doDenoise
        self.useCD = useCD
        self.isDebug = isDebug
    
    def run(self, walker_client, control_driver, SHARED_SSL_EVENT):
        Server_SSL_Wait = True
        while True:
            time.sleep(0.1)
            if (not SHARED_SSL_EVENT.is_set()) and (not Server_SSL_Wait):  # need to wait, but server doesn't
                walker_client.send(data=True, subtopic=SSL_WAIT_COMMUNICATION_TOPIC)
                Server_SSL_Wait = True
            elif SHARED_SSL_EVENT.is_set() and Server_SSL_Wait:  # doesn't need to wait, but server is waiting
                walker_client.send(data=False, subtopic=SSL_WAIT_COMMUNICATION_TOPIC)
                Server_SSL_Wait = False
            else:
                pass
            
            direction = walker_client.recv(subtopic=SSL_DOA_COMMUNICATION_TOPIC)
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
    
    def run(self, walker_client, SHARED_SSL_EVENT):
        cd = CD.ControlDriver(left_right=self.left_right) if self.useCD else ''
        if self.useCD:
            cd_thread = threading.Thread(target=cd.control_part, args=())
            cd_thread.start()
        ssl = SSL(seg_len=self.seg_len, doDenoise=self.doDenoise, useCD=self.useCD, isDebug=self.isDebug, )
        ssl.run(walker_client=walker_client, control_driver=cd, SHARED_SSL_EVENT=SHARED_SSL_EVENT)
