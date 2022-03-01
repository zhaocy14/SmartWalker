# !/usr/bin/env python
# -*- coding:utf-8 _*-
# @Author: swang
# @Contact: wang00sheng@gmail.com
# @Project Name: keyword_spotting_system
# @File: test.py
# @Time: 2021/11/11/21:51
# @Software: PyCharm
import os, sys

CRT_DIR = os.path.dirname(os.path.abspath(__file__))
F_PATH = os.path.dirname(CRT_DIR)
FF_PATH = os.path.dirname(F_PATH)
FFF_PATH = os.path.dirname(FF_PATH)
sys.path.extend([CRT_DIR, F_PATH, FF_PATH, FFF_PATH, ])
# print('sys.path:', sys.path)

import time
import json
import numpy as np
import threading
from threading import Thread
# import ctypes
# from multiprocessing import Process, Value, Pipe, Queue
from queue import Queue
from collections import deque  # , BlockingQueue
from pyaudio import PyAudio

from SoundSourceLocalization.SSL_Settings import *
from SoundSourceLocalization.mylib.utils import standard_normalizaion
from SoundSourceLocalization.SSL_Communication.WalkerClient import WalkerClient

SHARED_WORD_QUEUE_MAX_LENGTH = int(MAX_COMMAND_SECONDS * 1000 / KWS_WINDOW_STRIDE_MS)
# print('SHARED_WORD_QUEUE_MAX_LENGTH:', SHARED_WORD_QUEUE_MAX_LENGTH)
SHARED_WORD_QUEUE = deque(maxlen=SHARED_WORD_QUEUE_MAX_LENGTH)  # 最大长度将由 KWS 识别速度决定，只要 VoiveMenu 在不断消费就不会溢出
SHARED_WORD_QUEUE_UPDATA = False
SHARED_WORD_QUEUE_CLEAR = False

# only used for speed testing of MonitorVoice
audio_speed_test = True
audio_speed_test_frame_count = 0
audio_speed_test_start_time = None


class VoiceMenu(object):
    def __init__(self, SHARED_COMMAND_QUEUE=None):
        print('-' * 20, 'init VoiceMenu class', '-' * 20)
        super(VoiceMenu, self).__init__()
        self.SHARED_COMMAND_QUEUE = SHARED_COMMAND_QUEUE  # send out recognized commands
        self.keyword_ls = ['walker', 'voice', 'menu', 'redraw', 'the', 'map', 'charge', 'start', 'sleep',
                           'off', 'hand', 'operation', 'yes', 'no', 'help', ]
        self.walker_name = 'walker'
        self.menu_name = 'voice menu'
        self.menu_off = 'voice menu off'
        self.command_ls = ['voice menu', 'redraw map', 'charge', 'start',
                           'sleep', 'voice menu off', 'hand operation', 'yes', 'no', 'help', ]
        self.affirm_ls = ['yes', 'no']
        self.wait_action_ls = ['redraw map', 'charge', ]  # 不考虑 voice menu
        self.instant_action_ls = ['start', 'sleep', 'voice menu off', 'hand operation', 'help', ]  # 不考虑 voice menu
        self.excluded_ls = ['silence', 'unknown', ]
        self.action_ls = self.wait_action_ls + self.instant_action_ls
        self.wait_time = 10  # s  'inf' for waiting forever
        self.last_command = None
        self.last_command_time = None
        global MAX_COMMAND_SECONDS
        self.command_interval = MAX_COMMAND_SECONDS + 0.1  # s
        self.kws_stride_second = KWS_WINDOW_STRIDE_MS / 1000
    
    def detect_command(self, command_ls, wait_time):
        global SHARED_WORD_QUEUE, SHARED_WORD_QUEUE_UPDATA, SHARED_WORD_QUEUE_CLEAR
        
        # 初始化
        start_time = time.time()
        returnCommand = False
        waiting_for_off = False  # 用来区分 'voice menu' 和 'voice menu off'
        
        while True:
            if (wait_time != 'inf') and (time.time() - start_time > wait_time):  # timeout
                return None
            if SHARED_WORD_QUEUE_CLEAR:  # 因语音序列满被清零，导致 word 序列相应被清零
                SHARED_WORD_QUEUE.clear()
                SHARED_WORD_QUEUE_CLEAR = False
                SHARED_WORD_QUEUE_UPDATA = False
                # 清空之后，一切初始化
                start_time = time.time()
                returnCommand = False
                waiting_for_off = False  # 用来区分 'voice menu' 和 'voice menu off'
                continue
            elif (not SHARED_WORD_QUEUE_UPDATA):  # 若 word 无无更新
                time.sleep(self.kws_stride_second)
                continue
            
            SHARED_WORD_QUEUE_UPDATA = False
            if len(SHARED_WORD_QUEUE) == 0:
                continue
            word_ls = list(SHARED_WORD_QUEUE)
            # 生成命令
            command = self.convert_word_to_command(word_ls, command_ls)
            # 重复命令处理
            if (command is not None) and (command != self.menu_name):
                if command != self.last_command:
                    returnCommand = True
                else:
                    if time.time() - self.last_command_time > self.command_interval:
                        returnCommand = True
                    else:
                        continue
            elif command == self.menu_name:
                waiting_for_off = True
            elif command is None:
                if waiting_for_off:  # 未等到off，已结束输入
                    returnCommand = True
                    command = self.menu_name
                else:
                    continue
            else:
                print('Warning: Bad case exists!')
            
            if returnCommand:
                self.last_command = command
                self.last_command_time = time.time()
                SHARED_WORD_QUEUE.clear()
                print('command:', command)
                return command
    
    def convert_word_to_command(self, word_ls, command_ls):  # TODO 待优化，考虑有序列表
        # word_ls = [('voice', 0.9971545), ('voice', 0.99989796), ('voice', 0.99968916), ('voice', 0.983763),
        #            ('menu', 0.86595213), ('menu', 0.9521046), ('menu', 0.82160306)]
        word_ls = [i for i in word_ls if ((i[0] not in self.excluded_ls) and (i[1] > 0.7))]
        if not len(word_ls):
            return None
        
        words, probs = list(zip(*word_ls))
        words, probs = np.asarray(words), np.asarray(probs)
        uni_words = np.unique(words)
        uni_probs = []
        for wd in uni_words:
            uni_probs.append(probs[words == wd].mean())
        uni_probs = np.asarray(uni_probs)
        
        candi_cmd_ls = []
        for cmd in command_ls:
            cmd_set = set(cmd.split(' '))
            if cmd_set.issubset(uni_words):  # 不考虑顺序
                candi_cmd_ls.append(cmd)
        if ('voice menu off' in candi_cmd_ls) and ('voice menu' in candi_cmd_ls):
            candi_cmd_ls.remove('voice menu')
        if len(candi_cmd_ls) == 0:
            return None
        elif len(candi_cmd_ls) == 1:
            return candi_cmd_ls[0]
        else:
            # 从多个候选命令中挑出一个
            cmd_prob_ls = []
            for cmd in candi_cmd_ls:
                wds = cmd.split(' ')
                cmd_prob = [uni_probs[uni_words == wd] for wd in wds]
                cmd_prob_ls.append(np.mean(cmd_prob))
            return candi_cmd_ls[np.argmax(cmd_prob_ls)]
    
    def broadcast(self, command, level=None):
        '''
        Args:
            command:
            level: level of broadcasting
                    1: Sure to ... ;
                    2: Will ... automatically in half a minute. Say "No" or press the emergency button to cancel;
                    3: Complete ...
        '''
        
        if command == None:
            self.SHARED_COMMAND_QUEUE.put(self.menu_off, block=True, timeout=1)  # send out command
            print('Broadcast: Time out. And exit the voice menu automatically.')
        # elif command == self.walker_name:
        #     print(f'Broadcast: walker_name (\'{self.walker_name}\') is detected.')
        elif command == self.menu_name:
            self.SHARED_COMMAND_QUEUE.put(command, block=True, timeout=1)  # send out command
            print('Broadcast: Voice menu started.')
        elif command in self.wait_action_ls:
            if level == 1:
                print(f'Broadcast: Sure to {command} ?')
            elif level == 2:
                print(
                    f'Broadcast: Will {command} automatically in half a minute. \n\t\t\tSay "No" or press the emergency button to cancel?')
            elif level == 3:
                self.SHARED_COMMAND_QUEUE.put(command, block=True, timeout=1)  # send out command
                print(f'Broadcast: {command} was completed')
            else:
                print(f'Warning: Level ({level}) is illegal!')
        elif command in self.instant_action_ls:
            if level == 1:
                print(f'Broadcast: Sure to {command} ?')
            elif level == 2:
                self.SHARED_COMMAND_QUEUE.put(command, block=True, timeout=1)  # send out command
                print(f'Broadcast: {command} was completed')
            else:
                print(f'Warning: Level ({level}) is illegal!')
        else:
            print('-' * 20, f'Warning: Unknow command -> {command}!', '-' * 20)
    
    def run(self, ):
        global SHARED_WORD_QUEUE, SHARED_WORD_QUEUE_UPDATA, SHARED_WORD_QUEUE_CLEAR
        
        # KWS model detects keywords all the time
        while True:
            time.sleep(0.1)
            name = self.detect_command([self.menu_name, ], 'inf')
            if name != self.menu_name:
                # print(f'Warning: Will skip \'{name}\' while waiting for menu_name({self.menu_name})')
                continue
            
            while True:  # voice menu started
                self.broadcast(self.menu_name, )
                action = self.detect_command(self.action_ls + [self.menu_name], self.wait_time)
                
                if action == None:  # 超时，返回监听 voice menu
                    self.broadcast(action, )
                    break
                elif action == self.menu_name:
                    continue
                elif action in self.instant_action_ls:
                    self.broadcast(action, level=1)
                    affirm = self.detect_command(self.affirm_ls + [self.menu_name], self.wait_time)
                    if affirm == 'yes':
                        self.broadcast(action, level=2)
                        return action
                    elif affirm in ['no', self.menu_name, None]:
                        continue
                    else:
                        print(f'Warning: Error detection -> \'{affirm}\' \
                        while detecing affirm({self.affirm_ls + [self.menu_name]})')
                elif action in self.wait_action_ls:
                    self.broadcast(action, level=1)
                    affirm = self.detect_command(self.affirm_ls + [self.menu_name], self.wait_time)
                    if affirm in ['no', self.menu_name, None]:
                        continue
                    elif affirm == 'yes':
                        self.broadcast(action, level=2)
                        reaffirm = self.detect_command(['no'] + [self.menu_name], self.wait_time)
                        if reaffirm in ['no', self.menu_name]:
                            continue
                        elif reaffirm == None:
                            self.broadcast(action, level=3)
                            return action
                        else:
                            print(f'Warning: Error detection -> \'{reaffirm}\' while detecing reaffirm')
                    else:
                        print(f'Warning: Error detection -> \'{affirm}\' while detecing affirm({self.affirm_ls})')
                else:
                    print(f'Warning: Error detection -> \'{action}\' while detecing action({self.action_ls})')
    
    def run_forever(self, ):
        while True:
            self.run()


class KeyWordSpotting(object):
    def __init__(self, ):
        # print('-' * 20, 'init KeyWordSpotting class', '-' * 20)
        super(KeyWordSpotting, self).__init__()
    
    def run(self, walker_client, ):
        global SHARED_WORD_QUEUE, SHARED_WORD_QUEUE_UPDATA, SHARED_WORD_QUEUE_CLEAR
        print('-' * 20, 'KWS is running', '-' * 20)
        while True:
            if walker_client.recv(subtopic=WORD_QUEUE_CLEAR_COMMUNICATION_TOPIC):
                print('-' * 20, 'SHARED_WORD_QUEUE_CLEAR is received', '-' * 20)
                SHARED_WORD_QUEUE_CLEAR = True
            
            y, prob = walker_client.recv(subtopic=KWS_COMMUNICATION_TOPIC)
            y = y.decode('utf-8')
            # print('y, prob:', y, prob)
            SHARED_WORD_QUEUE.append((y, prob))
            SHARED_WORD_QUEUE_UPDATA = True
            # if (y not in ['silence', 'unknown', ]) and prob > 0.70:
            #     # print('y & prob:', y, round(prob, 3), end='\t')
            #     print(y, round(prob, 3), end='\t')


class MonitorVoice(object):
    def __init__(self, MappingMicro=False):
        print('-' * 20, 'init MonitorVoice class', '-' * 20)
        super(MonitorVoice, self).__init__()
        self.MappingMicro = MappingMicro
        self.micro_mapping = np.arange(CHANNELS)
        self.CompleteMappingMicro = False
        
        self.device_index = self.__get_device_index__()
        assert CHUNK_SIZE == 16  # 1ms的采样点数，此参数可以使得语音队列中每一个值对应1ms的音频
        self.init_micro_mapping_deque_len = 1000
        self.init_micro_mapping_deque = deque(maxlen=self.init_micro_mapping_deque_len)
    
    def __get_device_index__(self, ):
        # print('-' * 20 + 'Looking for microphones...' + '-' * 20)
        device_index = -1
        
        # scan to get usb device
        p = PyAudio()
        # print('num_device:', p.get_device_count())
        for index in range(p.get_device_count()):
            info = p.get_device_info_by_index(index)
            device_name = info.get("name")
            # print("device_name: ", device_name)
            
            # find mic usb device
            if device_name.find(RECORD_DEVICE_NAME) != -1:
                device_index = index
                break
        
        if device_index != -1:
            print('-' * 20, 'Find the microphones:', p.get_device_info_by_index(device_index)['name'], '-' * 20, )
            del p
        else:
            print('-' * 20, 'Cannot find the microphones', '-' * 20)
            exit(-1)
        
        return device_index
    
    def __init_micro_mapping__(self, ):
        if not self.MappingMicro:  # 开线程初始化映射函数
            self.micro_mapping = np.arange(CHANNELS)
            self.CompleteMappingMicro = True
            del self.init_micro_mapping_deque, self.init_micro_mapping_deque_len
        else:
            print('Please tap each microphone clockwise from the upper left corner ~ ')
            mapping = [None, ] * CHANNELS
            while True:
                for i in range(CHANNELS):
                    while True:
                        # acquire audio
                        while True:
                            if len(self.init_micro_mapping_deque) == self.init_micro_mapping_deque_len:
                                audio = np.concatenate(self.init_micro_mapping_deque, axis=1)
                                break
                        # calculate the energy_ratio
                        energy = np.sum(standard_normalizaion(audio) ** 2, axis=1).reshape(-1)
                        energy_ratio = energy / energy.sum()
                        
                        # identity the index
                        idx = np.where(energy_ratio > 0.5)[0]
                        if len(idx) == 1 and (idx[0] not in mapping):
                            mapping[i] = idx[0]
                            print(f'Logical channel {i} has been set as physical channel {mapping[i]}.', '-' * 4,
                                  'Energy ratio:', str([round(e, 2) for e in energy_ratio]))
                            break
                print('Final mapping:')
                print('Logical channel:', list(range(CHANNELS)))
                print('Physical channel:', mapping)
                break
                
                confirm_info = input('Confirm or Reset the mapping? Press [y]/n :')
                if confirm_info in ['y', '', 'yes', 'Yes']:
                    break
                else:
                    print('The system will reset the mapping')
                    continue
            
            # set class variables
            self.micro_mapping = np.asarray(mapping)
            self.CompleteMappingMicro = True
            del self.init_micro_mapping_deque, self.init_micro_mapping_deque_len
    
    def split_channels_from_frame(self, frame, num_channel=CHANNELS, mapping_flag=False, micro_mapping=None):
        audio = np.frombuffer(frame, dtype=np.short)
        audio = np.reshape(audio, (-1, num_channel)).T
        if mapping_flag:
            audio = audio[micro_mapping]
        audio = np.asarray(audio, dtype=np.float64) / 32768.
        return audio
    
    def monitor_from_4mics(self, walker_client, speed_test=True):
        '''
        采集声音信号，若麦克风未完成初始化，则将采集到的语音放入 self.init_micro_mapping_queue 中；
        否则，发送至云端
        '''
        print('-' * 20, "Monitoring microphones...", '-' * 20)
        
        self.time_queue = deque(maxlen=50)
        
        def print_time():
            last_time = None
            while True:
                if len(self.time_queue) < 50:
                    continue
                audio = np.mean(np.concatenate(self.time_queue, axis=1), axis=0)
                energy = np.average(audio ** 2)
                if energy > ENERGY_THRESHOLD:
                    crt_time = time.time()
                    if last_time is None:
                        print('Send:', crt_time, )
                    elif crt_time - last_time > 2:
                        print('Send:', crt_time, )
                    last_time = crt_time
        
        p1 = Thread(target=print_time, args=())
        p1.start()
        
        def PyAudioCallback(in_data, frame_count, time_info, status):
            if self.CompleteMappingMicro:
                audio = self.split_channels_from_frame(frame=in_data,
                                                       mapping_flag=True, micro_mapping=self.micro_mapping)
                walker_client.send(data=audio, subtopic=AUDIO_COMMUNICATION_TOPIC)
                self.time_queue.append(audio)
                # print('An audio frame is sent')
                # if speed_test:
                #     global audio_speed_test_frame_count, audio_speed_test_start_time
                #     audio_speed_test_frame_count += 1
                #     if audio_speed_test_start_time == None:
                #         audio_speed_test_start_time = time.time()
                #     else:
                #         print('time - frame_count (ms):',
                #               int((time.time() - audio_speed_test_start_time) * 1000) - audio_speed_test_frame_count)
            else:
                audio = self.split_channels_from_frame(frame=in_data, mapping_flag=False)
                self.init_micro_mapping_deque.append(audio, )
            return (None, pyaudio.paContinue)
        
        p = PyAudio()
        stream = p.open(format=p.get_format_from_width(RECORD_WIDTH),
                        channels=CHANNELS,
                        rate=SAMPLE_RATE,
                        input=True,
                        input_device_index=self.device_index,
                        frames_per_buffer=CHUNK_SIZE,
                        stream_callback=PyAudioCallback,
                        start=True)
        while stream.is_active():
            time.sleep(1)
            # print('Monitoring', )
            # print('Len of AUDIO_QUEUE', AUDIO_QUEUE.qsize())
            # try:
            #     print('Len of init_micro_mapping_deque', len(self.init_micro_mapping_deque))
            # except:
            #     pass
    
    def run(self, walker_client, ):
        init_micro_thread = threading.Thread(target=self.__init_micro_mapping__, args=())
        init_micro_thread.start()
        self.monitor_from_4mics(walker_client=walker_client, )


class SSL_test(object):
    def run(self, OUT_PIPE):
        self.OUT_PIPE = OUT_PIPE
        while True:
            msg = self.OUT_PIPE.recv()
            (audio, y, prob) = msg
            print('SSL: walker data is received~', )


if __name__ == '__main__':
    print('-' * 20, 'Hello World!', '-' * 20)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
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
    # ssl = SSL_test()
    
    p1 = Thread(target=mv.run, args=(walker_client,))
    p1.start()
    p2 = Thread(target=kws.run, args=(walker_client,))
    p2.start()
    p3 = Thread(target=vm.run_forever, args=())
    p3.start()
    # p4 = Thread(target=ssl.run, args=(GLOBAL_OUT_PIPE,))
    # p4.start()
    
    p1.join()
    # p2.join()
    # # p3.join()
    
    print('-' * 20, 'Brand-new World!', '-' * 20)
