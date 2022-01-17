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
import threading
from multiprocessing import Process, Value, Pipe, Queue
from multiprocessing.managers import BaseManager

# general lib

# independent systems
from SoundSourceLocalization.SSL_Settings import *
from SoundSourceLocalization.SSL_Communication.WalkerServer import WalkerServer
from SoundSourceLocalization.VoiceMenu.code.server_voice_menu_multiProcess import \
    MonitorVoice_Process, KeyWordSpotting_Process, SSL_test, \
    SHARED_AUDIO_QUEUE, SHARED_AUDIO_QUEUE_CLEAR, SHARED_SSL_AUDIO_QUEUE
from SoundSourceLocalization.SSL.code.server_ssl_Process import SSL_Process

if __name__ == '__main__':
    print('-' * 20, 'Hello World!', '-' * 20)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    MappingMicro = False
    isDebug = True
    
    manager = BaseManager()
    # 一定要在start前注册，不然就注册无效
    manager.register('WalkerServer', WalkerServer)  # 第一个参数为类型id，通常和第二个参数传入的类的类名相同，直观并且便于阅读
    manager.start()
    walker_server = manager.WalkerServer()
    mv = MonitorVoice_Process(MappingMicro=False, )
    kws = KeyWordSpotting_Process(use_stream=False, )
    ssl = SSL_Process(seg_len='256ms', useDenoise=True, isDebug=isDebug)
    # ssl = SSL_test(seg_len='256ms', useDenoise=True, isDebug=isDebug)
    
    p1 = Process(target=mv.run, args=(walker_server, SHARED_AUDIO_QUEUE, SHARED_AUDIO_QUEUE_CLEAR,))
    p2 = Process(target=kws.run,
                 args=(walker_server, SHARED_AUDIO_QUEUE, SHARED_AUDIO_QUEUE_CLEAR, SHARED_SSL_AUDIO_QUEUE,))
    p3 = Process(target=ssl.run, args=(walker_server, SHARED_SSL_AUDIO_QUEUE,))
    
    p1.start()
    p2.start()
    p3.start()
    
    p1.join()
    # p2.join()
    # p3.join()
    
    print('-' * 20, 'Brand-new World!', '-' * 20)
