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
sys.path.append(CRT_DIR)
# print('sys.path:', sys.path)

import time
import json
import numpy as np
import threading
import ctypes
from multiprocessing import Process, Value, Pipe, Queue
# from queue import Queue
from collections import deque  # , BlockingQueue
from scipy.special import softmax as scipy_softmax
from pyaudio import PyAudio

from SoundSourceLocalization.SSL_Settings import *
from SoundSourceLocalization.mylib.utils import standard_normalizaion

# KWS
import tensorflow.compat.v1 as tf
from kws_streaming.layers import modes
from kws_streaming.models import models
from kws_streaming.models import utils
from kws_streaming.train import inference

# from SoundSourceLocalization.lib.audiolib import normalize_single_channel_audio, audio_segmenter_4_numpy, \
#     audio_energy_ratio_over_threshold, audio_energy_over_threshold, audiowrite, audioread

GLOBAL_AUDIO_QUEUE = Queue(maxsize=2000)  # 每个 item 代表 1ms
GLOBAL_AUDIO_QUEUE_CLEAR = Value(ctypes.c_bool, False)
# 永远最多只有这么长，再多了直接清空，并将 AUDIO_QUEUE_CLEAR 置True; 如此也可以限制声音采集方面的延迟最多只有这么多


# GLOBAL_WORD_QUEUE_MAX_LENGTH = Value(ctypes.c_void_p, None)
GLOBAL_WORD_QUEUE = Queue()  # 最大长度将由 KWS 识别速度决定，只要 VoiveMenu 在不断消费就不会溢出
GLOBAL_WORD_QUEUE_UPDATA = Value(ctypes.c_bool, False)
GLOBAL_WORD_QUEUE_CLEAR = Value(ctypes.c_bool, False)

# SSL_AUDIO = []
# SSL_AUDIO_UPDATE = Value(ctypes.c_bool, False)

GLOBAL_OUT_PIPE, GLOBAL_IN_PIPE = Pipe(duplex=False)


class VoiceMenu(object):
    def __init__(self):
        # print('-' * 20, 'init VoiceMenu class', '-' * 20)
        super(VoiceMenu, self).__init__()
        self.keyword_ls = ['walker', 'voice', 'menu', 'redraw', 'the', 'map', 'charge', 'start', 'sleep',
                           'off', 'hand', 'operation', 'yes', 'no', ]
        self.walker_name = 'walker'
        self.menu_name = 'voice menu'
        self.command_ls = ['voice menu', 'redraw map', 'charge', 'start',
                           'sleep', 'voice menu off', 'hand operation', 'yes', 'no', ]
        self.affirm_ls = ['yes', 'no']
        self.wait_action_ls = ['redraw map', 'charge', ]  # 不考虑 voice menu
        self.instant_action_ls = ['start', 'sleep', 'voice menu off', 'hand operation', ]  # 不考虑 voice menu
        self.excluded_ls = ['silence', 'unknown', ]
        self.action_ls = self.wait_action_ls + self.instant_action_ls
        self.wait_time = 10  # s  'inf' for waiting forever
        self.last_command = None
        self.last_command_time = None
        global MAX_COMMAND_SECONDS
        self.command_interval = MAX_COMMAND_SECONDS + 0.1  # s
        self.kws_stride_second = KWS_WINDOW_STRIDE_MS / 1000
        self.local_word_queue_max_length = MAX_COMMAND_SECONDS * 1000 // KWS_WINDOW_STRIDE_MS
        self.local_word_queue = deque(maxlen=self.local_word_queue_max_length)
    
    def clear_WORD_QUEUE(self, WORD_QUEUE):
        while not WORD_QUEUE.empty():
            try:
                WORD_QUEUE.get_nowait()
            except:
                pass
        print('-' * 20, 'WORD_QUEUE is cleared as AUDIO_QUEUE is cleared.', '-' * 20)
    
    def get_WORD(self, WORD_QUEUE):
        while not WORD_QUEUE.empty():
            try:
                self.local_word_queue.append(WORD_QUEUE.get_nowait())
            except:
                pass
        return self.local_word_queue
    
    def detect_command(self, command_ls, wait_time):
        # 初始化
        start_time = time.time()
        returnCommand = False
        waiting_for_off = False  # 用来区分 'voice menu' 和 'voice menu off'
        
        while True:
            if (wait_time != 'inf') and (time.time() - start_time > wait_time):  # timeout
                return None
            if self.WORD_QUEUE_CLEAR.value:  # 因语音序列满被清零，导致 word 序列相应被清零
                self.clear_WORD_QUEUE(self.WORD_QUEUE)
                self.WORD_QUEUE_CLEAR.value = False
                self.WORD_QUEUE_UPDATA.value = False
                # 清空之后，一切初始化
                start_time = time.time()
                returnCommand = False
                waiting_for_off = False  # 用来区分 'voice menu' 和 'voice menu off'
                continue
            elif (not self.WORD_QUEUE_UPDATA.value):  # 若 word 无无更新
                time.sleep(self.kws_stride_second)
                continue
            
            self.WORD_QUEUE_UPDATA.value = False
            if self.WORD_QUEUE.empty():
                continue
            self.get_WORD(self.WORD_QUEUE)
            # print('len(self.local_word_queue):', len(self.local_word_queue))
            word_ls = list(self.local_word_queue)
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
                self.local_word_queue.clear()
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
            print('Broadcast: Time out. And exit the voice menu automatically.')
        # elif command == self.walker_name:
        #     print(f'Broadcast: walker_name (\'{self.walker_name}\') is detected.')
        elif command == self.menu_name:
            print('Broadcast: Voice menu started.')
        elif command in self.wait_action_ls:
            if level == 1:
                print(f'Broadcast: Sure to {command} ?')
            elif level == 2:
                print(
                    f'Broadcast: Will {command} automatically in half a minute. \n\t\t\tSay "No" or press the emergency button to cancel?')
            elif level == 3:
                print(f'Broadcast: {command} was completed')
            else:
                print(f'Warning: Level ({level}) is illegal!')
        elif command in self.instant_action_ls:
            if level == 1:
                print(f'Broadcast: Sure to {command} ?')
            elif level == 2:
                print(f'Broadcast: {command} was completed')
            else:
                print(f'Warning: Level ({level}) is illegal!')
        else:
            print('-' * 20, f'Warning: Unknow command -> {command}!', '-' * 20)
    
    def run(self, WORD_QUEUE, WORD_QUEUE_UPDATA, WORD_QUEUE_CLEAR, ):
        # KWS model detects keywords all the time
        self.WORD_QUEUE, self.WORD_QUEUE_UPDATA, self.WORD_QUEUE_CLEAR = WORD_QUEUE, WORD_QUEUE_UPDATA, WORD_QUEUE_CLEAR
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
    
    def run_forever(self, WORD_QUEUE, WORD_QUEUE_UPDATA, WORD_QUEUE_CLEAR, ):
        while True:
            self.run(WORD_QUEUE, WORD_QUEUE_UPDATA, WORD_QUEUE_CLEAR)


class KeyWordSpotting(object):
    def __init__(self, use_stream=False):
        
        # print('-' * 20, 'init KeyWordSpotting class', '-' * 20)
        super(KeyWordSpotting, self).__init__()
        
        self.use_stream = use_stream
        assert self.use_stream == False, 'Streaming model has not been tested'  # 暂时不考虑流式模型
        
        self.model_name = 'ds_tc_resnet_cpu_causal_20211231-200734'
        self.model_dir = os.path.abspath(os.path.join(CRT_DIR, '../model', self.model_name, ))
        print('KWS_model_dir:', self.model_dir)
        self.flags_path = os.path.join(self.model_dir, 'flags.json')
        self.flags = self.__load__flags__()
        self.flags.batch_size = 1
        
        print('-' * 20, 'Loading KWS non_stream_model...', '-' * 20, )
        self.non_stream_model = self.__load_non_stream_model__(weights_name='last_weights')
        if self.use_stream:  # TODO 保存流式模型，直接加载？而非每次都要转换，还挺耗时的
            self.stream_model = self.__convert_2_stream_model__()
        self.labels = np.array(['silence', 'unknown', ] + self.flags.wanted_words.split(','))
        self.walker_name = self.labels[2]
        print('-' * 20, 'KWS labels:', ' '.join(self.labels), '-' * 20)
        print('-' * 20, 'KWS walker_name:', self.walker_name, '-' * 20)
        
        self.clip_duration_ms = int(self.flags.clip_duration_ms)
        assert self.clip_duration_ms == int(CLIP_MS)
        if self.use_stream:
            self.window_stride_ms = int(self.flags.window_stride_ms)
        else:
            self.window_stride_ms = KWS_WINDOW_STRIDE_MS
    
    def __load__flags__(self, ):
        with open(self.flags_path, 'r') as load_f:
            flags_json = json.load(load_f)
        
        class KWS_FLAGS_DictStruct(object):
            def __init__(self, **entries):
                self.__dict__.update(entries)
        
        return KWS_FLAGS_DictStruct(**flags_json)
    
    def __load_non_stream_model__(self, weights_name):
        tf.reset_default_graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        tf.keras.backend.set_session(sess)
        # self.audio_processor = input_data.AudioProcessor(self.flags)
        tf.keras.backend.set_learning_phase(0)
        # tf.disable_eager_execution()
        # print('tf.keras.backend.image_data_format():', tf.keras.backend.image_data_format())
        tf.keras.backend.set_image_data_format('channels_last')
        non_stream_model = models.MODELS[self.flags.model_name](self.flags)
        weight_path = os.path.join(self.model_dir, weights_name, )
        non_stream_model.load_weights(weight_path).expect_partial()
        # non_stream_model.summary()
        
        # tf.keras.utils.plot_model(
        #     non_stream_model,
        #     show_shapes=True,
        #     show_layer_names=True,
        #     expand_nested=True,
        #     to_file=os.path.join('./', self.model_name + '_non_stream.png'), )
        #
        return non_stream_model
    
    def __convert_2_stream_model__(self, ):
        print('tf stream model state internal without state resetting between testing sequences')
        self.flags.data_shape = modes.get_input_data_shape(self.flags, modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
        stream_model = utils.to_streaming_inference(
            self.non_stream_model, self.flags, modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
        # stream_model.summary()
        
        # tf.keras.utils.plot_model(
        #     stream_model,
        #     show_shapes=True,
        #     show_layer_names=True,
        #     expand_nested=True,
        #     to_file=os.path.join('./', self.model_name + '_stream.png'),
        # )
        
        return stream_model
    
    def predict(self, x, use_stream=None):
        use_stream = self.use_stream if (use_stream is None) else use_stream
        if use_stream:
            y_pred = inference.run_stream_inference_classification(self.flags, self.stream_model, x)
        else:
            y_pred = self.non_stream_model.predict(x)
        y_pred = scipy_softmax(y_pred, axis=-1)
        label = np.argmax(y_pred, axis=-1)
        
        return label, y_pred[:, label].squeeze(axis=0)
    
    def run(self, AUDIO_QUEUE, AUDIO_QUEUE_CLEAR, WORD_QUEUE, WORD_QUEUE_UPDATA, WORD_QUEUE_CLEAR, IN_PIPE, ):
        print('-' * 20, 'KWS is running', '-' * 20)
        if not self.use_stream:
            local_audio_frames = deque(maxlen=self.clip_duration_ms)
            while True:
                for i in range(self.window_stride_ms):
                    if AUDIO_QUEUE_CLEAR.value:
                        AUDIO_QUEUE_CLEAR.value = False
                        WORD_QUEUE_CLEAR.value = True
                        local_audio_frames.clear()
                    try:
                        local_audio_frames.append(AUDIO_QUEUE.get_nowait())
                    except:
                        pass
                if len(local_audio_frames) != self.clip_duration_ms:
                    continue
                ################################ predict ########################################
                audio = np.concatenate(local_audio_frames, axis=1)
                # audio = normalize_single_channel_to_target_level(audio, )
                x = np.array(audio[0], dtype=np.float64)[np.newaxis, :]
                y, prob = self.predict(x, use_stream=self.use_stream)
                y, prob = self.labels[y[0]], prob[0]
                # print('y, prob:', y, prob)
                WORD_QUEUE.put((y, prob), block=True, timeout=3)
                WORD_QUEUE_UPDATA.value = True
                # if (y not in ['silence', 'unknown', ]) and prob > 0.70:
                #     # print('y & prob:', y, round(prob, 3), end='\t')
                #     print(y, round(prob, 3), end='\t')
                if y == self.walker_name:  # TODO 监听到 walker_name，使用 Pipe 将音频传给声源定位模块
                    IN_PIPE.send((audio, y, prob, time.time()))
                    # print(f'KWS: walker_name (\'{self.walker_name}\') is detected.')
                    # SSL_AUDIO = (audio, y, prob)  # （音频，文本，概率）
                    # SSL_AUDIO_UPDATE = True
        else:
            assert False, 'Streaming model has not been tested'
            local_audio_frames = deque(maxlen=self.window_stride_ms)
            while True:
                for i in range(self.window_stride_ms):
                    local_audio_frames.append(AUDIO_QUEUE.get(block=True, timeout=None))
                if len(local_audio_frames) != self.clip_duration_ms:
                    continue
                ################################ predict ########################################
                audio = np.concatenate(local_audio_frames, axis=1)
                # audio = normalize_single_channel_to_target_level(audio, )
                x = np.array(audio[0], dtype=np.float64)[np.newaxis, :]
                y, prob = self.predict(x, use_stream=self.use_stream)
                y, prob = self.labels[y[0]], prob[0]
                # WORD_QUEUE.put((y, prob))
                # WORD_QUEUE_UPDATA = True
                # if (y not in ['silence', 'unknown', ]) and prob > 0.70:
                #     # print('y & prob:', y, round(prob, 3), end='\t')
                #     print(y, round(prob, 3), end='\t')
                if y == self.walker_name:  # TODO 监听到 walker_name，使用 Pipe 将音频传给声源定位模块
                    print(f'KWS: walker_name (\'{self.walker_name}\') is detected.')
                    # SSL_AUDIO = (audio, y, prob)  # （音频，文本，概率）
                    # SSL_AUDIO_UPDATE = True


class MonitorVoice(object):
    def __init__(self, MappingMicro=False):
        # print('-' * 20, 'init MonitorVoice class', '-' * 20)
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
    
    def clear_AUDIO_QUEUE(self, AUDIO_QUEUE):
        while not AUDIO_QUEUE.empty():
            try:
                AUDIO_QUEUE.get_nowait()
            except:
                pass
        print('-' * 20, 'AUDIO_QUEUE is cleared as it is full.', '-' * 20)
    
    def monitor_from_4mics(self, AUDIO_QUEUE, AUDIO_QUEUE_CLEAR, ):
        '''
        采集声音信号，若麦克风未完成初始化，则将采集到的语音放入 self.init_micro_mapping_queue 中；
        否则，放入全局变量 AUDIO_QUEUE 中.
        :return:
        '''
        print('-' * 20, "Monitoring microphones...", '-' * 20)
        
        def PyAudioCallback(in_data, frame_count, time_info, status):
            # try:
            if self.CompleteMappingMicro:
                audio = self.split_channels_from_frame(frame=in_data,
                                                       mapping_flag=True, micro_mapping=self.micro_mapping)
                if AUDIO_QUEUE.full():
                    self.clear_AUDIO_QUEUE(AUDIO_QUEUE, )
                    AUDIO_QUEUE_CLEAR.value = True
                AUDIO_QUEUE.put(audio, block=True, timeout=3)
            else:
                audio = self.split_channels_from_frame(frame=in_data, mapping_flag=False)
                self.init_micro_mapping_deque.append(audio, )
            # except Exception as e:
            #     print('Capture error:', e)
            #     print('-' * 20, 'AUDIO_QUEUE is FULL', '-' * 20)
            #     exit(-1)
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
    
    def run(self, AUDIO_QUEUE, AUDIO_QUEUE_CLEAR, ):
        init_micro_thread = threading.Thread(target=self.__init_micro_mapping__, args=())
        init_micro_thread.start()
        
        self.monitor_from_4mics(AUDIO_QUEUE, AUDIO_QUEUE_CLEAR, )


class MonitorVoice_Process(object):
    def __init__(self, MappingMicro=False):
        super(MonitorVoice_Process, self).__init__()
        self.MappingMicro = MappingMicro
    
    def run(self, AUDIO_QUEUE, AUDIO_QUEUE_CLEAR, ):
        mv = MonitorVoice(MappingMicro=self.MappingMicro)
        mv.run(AUDIO_QUEUE, AUDIO_QUEUE_CLEAR, )


class KeyWordSpotting_Process(object):
    def __init__(self, use_stream=False):
        super(KeyWordSpotting_Process, self).__init__()
        self.use_stream = use_stream
    
    def run(self, AUDIO_QUEUE, AUDIO_QUEUE_CLEAR, WORD_QUEUE, WORD_QUEUE_UPDATA, WORD_QUEUE_CLEAR, IN_PIPE):
        kws = KeyWordSpotting(use_stream=self.use_stream)
        kws.run(AUDIO_QUEUE, AUDIO_QUEUE_CLEAR, WORD_QUEUE, WORD_QUEUE_UPDATA, WORD_QUEUE_CLEAR, IN_PIPE)


class VoiceMenu_Process(object):
    def __init__(self, ):
        super(VoiceMenu_Process, self).__init__()
    
    def run_forever(self, WORD_QUEUE, WORD_QUEUE_UPDATA, WORD_QUEUE_CLEAR, ):
        vm = VoiceMenu()
        vm.run_forever(WORD_QUEUE, WORD_QUEUE_UPDATA, WORD_QUEUE_CLEAR)


class SSL_test(object):
    def run(self, OUT_PIPE):
        self.OUT_PIPE = OUT_PIPE
        while True:
            msg = self.OUT_PIPE.recv()
            (audio, y, prob) = msg
            print('SSL: walker data is received~', )


class MonitorVoice_VoiceMenu_Process():
    def __init__(self, MappingMicro=False):
        super(MonitorVoice_VoiceMenu_Process, self).__init__()
        self.MappingMicro = MappingMicro
    
    def run_forever(self, AUDIO_QUEUE, AUDIO_QUEUE_CLEAR, WORD_QUEUE, WORD_QUEUE_UPDATA, WORD_QUEUE_CLEAR, ):
        vm = VoiceMenu()
        vm_thread = threading.Thread(target=vm.run_forever, args=(WORD_QUEUE, WORD_QUEUE_UPDATA, WORD_QUEUE_CLEAR,))
        vm_thread.start()
        
        mv = MonitorVoice(MappingMicro=self.MappingMicro)
        mv.run(AUDIO_QUEUE, AUDIO_QUEUE_CLEAR, )


if __name__ == '__main__':
    print('-' * 20, 'Hello World!', '-' * 20)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    mv = MonitorVoice_Process(MappingMicro=False)
    kws = KeyWordSpotting_Process(use_stream=False)
    vm = VoiceMenu_Process()
    ssl = SSL_test()
    
    p1 = Process(target=mv.run, args=(GLOBAL_AUDIO_QUEUE, GLOBAL_AUDIO_QUEUE_CLEAR,))
    p1.start()
    p2 = Process(target=kws.run, args=(GLOBAL_AUDIO_QUEUE, GLOBAL_AUDIO_QUEUE_CLEAR,
                                       GLOBAL_WORD_QUEUE, GLOBAL_WORD_QUEUE_UPDATA, GLOBAL_WORD_QUEUE_CLEAR,
                                       GLOBAL_IN_PIPE,))
    p2.start()
    p3 = Process(target=vm.run_forever, args=(GLOBAL_WORD_QUEUE, GLOBAL_WORD_QUEUE_UPDATA, GLOBAL_WORD_QUEUE_CLEAR,))
    p3.start()
    p4 = Process(target=ssl.run, args=(GLOBAL_OUT_PIPE,))
    p4.start()
    
    GLOBAL_IN_PIPE.close()
    GLOBAL_OUT_PIPE.close()
    
    p1.join()
    # p2.join()
    # # p3.join()
    
    print('-' * 20, 'Brand-new World!', '-' * 20)
