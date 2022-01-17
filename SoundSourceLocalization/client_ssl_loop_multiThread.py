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


class Voice_Process(object):
    def __init__(self, MappingMicro=False, isDebug=True, useCD=False, left_right=0, ):
        super(Voice_Process, self).__init__()
        self.MappingMicro = MappingMicro
        self.isDebug = isDebug
        self.useCD = useCD
        self.left_right = left_right
    
    def run(self, ):
        # class TunnelBuilder(object):
        #     def __init__(self, ):
        #         super(TunnelBuilder, self).__init__()
        #
        #     def build_tunnel(self, command):
        #         os.system(command)
        #
        #     def run(self, ):
        #         import os
        #         from multiprocessing import Process
        #
        #         p1 = Process(target=self.build_tunnel,
        #                      args=('sshpass -p ZpBrwNaX ssh -tt -L 6015:net-g14:8008 swang@gatekeeper.cs.hku.hk',))
        #         p1.start()
        #         p2 = Process(target=self.build_tunnel,
        #                      args=('sshpass -p ZpBrwNaX ssh -tt -L 6016:net-g14:8080 swang@gatekeeper.cs.hku.hk',))
        #         p2.start()
        #         print('-' * 20, 'Tunnelings have been built.', '-' * 20, )
        #
        #
        # TunnelBuilder().run()
        
        walker_client = WalkerClient()
        mv = MonitorVoice(MappingMicro=False)
        kws = KeyWordSpotting()
        vm = VoiceMenu()
        ssl = SSL_Thread(seg_len='256ms', doDenoise=True, useCD=self.useCD, isDebug=self.isDebug,
                         left_right=self.left_right, )
        
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


if __name__ == '__main__':
    print('-' * 20, 'Hello World!', '-' * 20)
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    
    MappingMicro = False
    isDebug = True
    useCD = False
    left_right = 0
    
    vp = Voice_Process(MappingMicro=MappingMicro, isDebug=isDebug, useCD=useCD, left_right=left_right, )
    p1 = Process(target=vp.run, args=())
    p1.start()
    p1.join()
    
    print('-' * 20, 'Brand-new World!', '-' * 20)
