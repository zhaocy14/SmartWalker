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
# import Driver.ControlOdometryDriver as CD
import Sensors.STM32 as STM32


class SSL(object):
    def __init__(self, useCD=True, ):
        super(SSL, self).__init__()
        self.useCD = useCD
    
    def run(self, walker_client, control_driver, SHARED_SSL_EVENT):
        Server_SSL_Wait = False  # TODO: for debugging
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
            # direction = (16 - direction) % 8
            ### 接入Owen的模块，传入aim_loca
            if self.useCD:
                direction = direction * 45
                SSLturning(control_driver, direction)
                speed = STEP_SIZE * 150 / FORWARD_SECONDS
                control_driver.UpdateDriver(linearVelocity=speed, angularVelocity=0, distanceToCenter=0)
                time.sleep(FORWARD_SECONDS)
                control_driver.UpdateDriver(linearVelocity=0, angularVelocity=0, distanceToCenter=0)
                print("movement done.")
            else:
                pass


class SSL_Thread(object):
    def __init__(self, useCD=True, left_right=0, ):
        super(SSL_Thread, self).__init__()
        self.useCD = useCD
        self.left_right = left_right
    
    def run(self, walker_client, SHARED_SSL_EVENT):
        # cd = CD.ControlDriver(left_right=self.left_right) if self.useCD else ''
        cd = STM32.STM32Sensors() if self.useCD else ''
        # if self.useCD:
        #     cd_thread = threading.Thread(target=cd.control_part, args=())
        #     cd_thread.start()
        ssl = SSL(useCD=self.useCD, )
        ssl.run(walker_client=walker_client, control_driver=cd, SHARED_SSL_EVENT=SHARED_SSL_EVENT)
