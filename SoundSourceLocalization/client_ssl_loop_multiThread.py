# !/usr/bin/env python
# -*- coding:utf-8 _*-
# @Author: swang
# @Contact: wang00sheng@gmail.com
# @Project Name: SmartWalker-master
# @File: ssl_loop.py
# @Time: 2022/01/02/20:34
# @Software: PyCharm

import os, sys

module_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(module_dir)
sys.path.extend([module_dir, project_dir])
# print('sys.path:', sys.path)

import time
import numpy as np
from threading import Thread
from multiprocessing import Process, Value, Pipe, Queue

# general lib

# independent systems
from SoundSourceLocalization.SSL_Settings import *
from SoundSourceLocalization.SSL_Communication.WalkerClient import WalkerClient
from SoundSourceLocalization.VoiceMenu.code.client_voice_menu_multiThread import \
    VoiceMenu, KeyWordSpotting, MonitorVoice, \
    SHARED_WORD_QUEUE_MAX_LENGTH, SHARED_WORD_QUEUE, SHARED_WORD_QUEUE_UPDATA, SHARED_WORD_QUEUE_CLEAR
from SoundSourceLocalization.SSL.code.client_ssl_Thread import SSL_Thread

if __name__ == '__main__':
    print('-' * 20, 'Hello World!', '-' * 20)
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    MappingMicro = False
    isDebug = True
    useCD = False
    left_right = 0
    
    walker_client = WalkerClient()
    mv = MonitorVoice(MappingMicro=False)
    kws = KeyWordSpotting()
    vm = VoiceMenu()
    ssl = SSL_Thread(seg_len='256ms', useDenoise=True, useCD=useCD, isDebug=isDebug, left_right=left_right, )
    
    p1 = Thread(target=mv.run, args=(walker_client,))
    p2 = Thread(target=kws.run, args=(walker_client,))
    p3 = Thread(target=vm.run_forever, args=())
    p4 = Thread(target=ssl.run, args=(walker_client,))
    
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    
    p1.join()
    # p2.join()
    # p3.join()
    
    print('-' * 20, 'Brand-new World!', '-' * 20)
