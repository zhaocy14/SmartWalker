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

# general lib

# independent systems
from SoundSourceLocalization.SSL_Settings import *

from SoundSourceLocalization.VoiceMenu.code.voice_menu_multiProcess import \
    MonitorVoice_Process, KeyWordSpotting_Process, MonitorVoice_VoiceMenu_Process, VoiceMenu_Process, \
    GLOBAL_AUDIO_QUEUE, GLOBAL_AUDIO_QUEUE_CLEAR, \
    GLOBAL_WORD_QUEUE, GLOBAL_WORD_QUEUE_UPDATA, GLOBAL_WORD_QUEUE_CLEAR, GLOBAL_IN_PIPE, GLOBAL_OUT_PIPE

from SoundSourceLocalization.SSL.code.ssl_Process import SSL_Process

if __name__ == '__main__':
    print('-' * 20, 'Hello World!', '-' * 20)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    MappingMicro = False
    isDebug = True
    useCD = False
    left_right = 0
    
    mv_vm = MonitorVoice_VoiceMenu_Process(MappingMicro=MappingMicro)
    kws = KeyWordSpotting_Process(use_stream=False)
    ssl = SSL_Process(seg_len='256ms', doDenoise=True, useCD=useCD, isDebug=isDebug)
    
    p1 = Process(target=mv_vm.run_forever, args=(GLOBAL_AUDIO_QUEUE, GLOBAL_AUDIO_QUEUE_CLEAR,
                                                 GLOBAL_WORD_QUEUE, GLOBAL_WORD_QUEUE_UPDATA, GLOBAL_WORD_QUEUE_CLEAR,))
    p2 = Process(target=kws.run, args=(GLOBAL_AUDIO_QUEUE, GLOBAL_AUDIO_QUEUE_CLEAR,
                                       GLOBAL_WORD_QUEUE, GLOBAL_WORD_QUEUE_UPDATA, GLOBAL_WORD_QUEUE_CLEAR,
                                       GLOBAL_IN_PIPE,))
    p3 = Process(target=ssl.run, args=(GLOBAL_OUT_PIPE, left_right))
    
    p1.start()
    p2.start()
    p3.start()
    
    GLOBAL_IN_PIPE.close()
    GLOBAL_OUT_PIPE.close()
    
    p1.join()
    # p2.join()
    # p3.join()
    
    print('-' * 20, 'Brand-new World!', '-' * 20)
