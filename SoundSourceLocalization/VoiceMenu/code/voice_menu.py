# !/usr/bin/env python
# -*- coding:utf-8 _*-
# @Author: swang
# @Contact: wang00sheng@gmail.com
# @Project Name: keyword_spotting_system
# @File: test.py
# @Time: 2021/11/11/21:51
# @Software: PyCharm
import os, sys
import time
import json
import numpy as np
import threading
import multiprocessing

from pyaudio import PyAudio
from collections import deque  # , BlockingQueue
from queue import Queue
from SoundSourceLocalization.SSL_Settings import *
from SoundSourceLocalization.lib.utils import standard_normalizaion
# from SoundSourceLocalization.lib.audiolib import normalize_single_channel_audio, audio_segmenter_4_numpy, \
#     audio_energy_ratio_over_threshold, audio_energy_over_threshold, audiowrite, audioread
import tensorflow.compat.v1 as tf
from kws_streaming.layers import modes
from kws_streaming.models import models
from kws_streaming.models import utils
from kws_streaming.train import inference
from scipy.special import softmax as scipy_softmax

WORD_QUEUE_MAX_LENGTH = None
WORD_QUEUE = deque()  # 最大长度将在KWS中获得检测步长后修正
WORD_QUEUE_UPDATA = False

AUDIO_QUEUE = Queue()  # (maxsize=3 * self.clip_duration_ms)   # TODO 存在内存溢出风险

SSL_AUDIO = []
SSL_AUDIO_UPDATE = False


class VoiceMenu(object):
    def __init__(self):
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
    
    def detect_command(self, command_ls, wait_time):
        # menu_name is detected by default, even if it isn't in command_ls
        global WORD_QUEUE, WORD_QUEUE_UPDATA, WORD_QUEUE_MAX_LENGTH
        # command_ls = command_ls if ('voice menu off' in command_ls) else command_ls + ['voice menu off']
        start_time = time.time()
        returnCommand = False
        waiting_off = False  # 用来区分 'voice menu' 和 'voice menu off'
        
        while True:
            if (wait_time != 'inf') and (time.time() - start_time > wait_time):
                return None
            if not (WORD_QUEUE_UPDATA and (len(WORD_QUEUE) == WORD_QUEUE_MAX_LENGTH)):
                time.sleep(0.1)
            WORD_QUEUE_UPDATA = False
            word_ls = list(WORD_QUEUE)
            # 生成命令
            command = self.convert_word_to_command(word_ls, command_ls)
            # 重复命令处理
            if (command is not None) and (command != self.menu_name):
                if command != self.last_command:
                    returnCommand = True
                else:
                    assert self.last_command_time is not None, ('param {last_command_time} was not updated by mistake')
                    if time.time() - self.last_command_time > self.command_interval:
                        returnCommand = True
                    else:
                        continue
            elif command == self.menu_name:
                waiting_off = True
            elif command is None:
                if waiting_off:  # 未等到off，已结束输入
                    returnCommand = True
                    command = self.menu_name
                else:
                    continue
            else:
                print('Warning: Bad case exists!')
            
            if returnCommand:
                self.last_command = command
                self.last_command_time = time.time()
                WORD_QUEUE.clear()
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
    
    def run(self):
        # streaming KWS model detects keywords all the time
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
    def __init__(self, use_stream=False):
        super(KeyWordSpotting, self).__init__()
        self.use_stream = use_stream
        assert self.use_stream == False  # 暂时不考虑流式模型
        
        self.model_name = 'ds_tc_resnet_cpu_causal_20211231-200734'
        self.model_dir = os.path.join('../model', self.model_name, )
        self.flags_path = os.path.join(self.model_dir, 'flags.json')
        self.flags = self.__load__flags__()
        self.flags.batch_size = 1
        
        self.non_stream_model = self.__load_non_stream_model__(weights_name='last_weights')
        if self.use_stream:  # TODO 保存流式模型，直接加载？而非每次都要转换，还挺耗时的
            self.stream_model = self.__convert_2_stream_model__()
        self.labels = np.array(['silence', 'unknown', ] + self.flags.wanted_words.split(','))
        self.walker_name = self.labels[2]
        print('-' * 10, 'labels:', str(self.labels), '-' * 10)
        print('-' * 10, 'walker_name:', str(self.walker_name), '-' * 10)
        
        self.clip_duration_ms = int(self.flags.clip_duration_ms)
        assert self.clip_duration_ms == int(CLIP_MS)
        if self.use_stream:
            self.window_stride_ms = int(self.flags.window_stride_ms)
        else:
            self.window_stride_ms = WINDOW_STRIDE_MS
        
        global WORD_QUEUE, WORD_QUEUE_MAX_LENGTH
        WORD_QUEUE_MAX_LENGTH = MAX_COMMAND_SECONDS * 1000 // self.window_stride_ms
        WORD_QUEUE = deque(maxlen=WORD_QUEUE_MAX_LENGTH)
    
    def __load__flags__(self, ):
        with tf.compat.v1.gfile.Open(self.flags_path, 'r') as fd:
            flags_json = json.load(fd)
        
        class DictStruct(object):
            def __init__(self, **entries):
                self.__dict__.update(entries)
        
        self.flags = DictStruct(**flags_json)
        
        return self.flags
    
    def __load_non_stream_model__(self, weights_name):
        tf.reset_default_graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        tf.keras.backend.set_session(sess)
        # self.audio_processor = input_data.AudioProcessor(self.flags)
        tf.keras.backend.set_learning_phase(0)
        # tf.disable_eager_execution()
        
        non_stream_model = models.MODELS[self.flags.model_name](self.flags)
        print('non_stream_model_weight:', os.path.join(self.model_dir, weights_name, ))
        non_stream_model.load_weights(os.path.join(self.model_dir, weights_name, )).expect_partial()
        non_stream_model.summary()
        
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
        stream_model.summary()
        
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
    
    def run(self):
        global WORD_QUEUE, WORD_QUEUE_UPDATA, AUDIO_QUEUE, SSL_AUDIO, SSL_AUDIO_UPDATE
        
        if not self.use_stream:
            local_audio_frames = deque(maxlen=self.clip_duration_ms)
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
                WORD_QUEUE.append((y, prob))
                WORD_QUEUE_UPDATA = True
                # if (y not in ['silence', 'unknown', ]) and prob > 0.70:
                #     # print('y & prob:', y, round(prob, 3), end='\t')
                #     print(y, round(prob, 3), end='\t')
                if y == self.walker_name:  # 监听到 walker_name，将音频传给声源定位模块
                    print(f'KWS: walker_name (\'{self.walker_name}\') is detected.')
                    SSL_AUDIO = (audio, y, prob)  # （音频，文本，概率）
                    SSL_AUDIO_UPDATE = True
        else:
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
                WORD_QUEUE.append((y, prob))
                WORD_QUEUE_UPDATA = True
                # if (y not in ['silence', 'unknown', ]) and prob > 0.70:
                #     # print('y & prob:', y, round(prob, 3), end='\t')
                #     print(y, round(prob, 3), end='\t')
                if y == self.walker_name:  # 监听到 walker_name，将音频传给声源定位模块
                    print(f'KWS: walker_name (\'{self.walker_name}\') is detected.')
                    SSL_AUDIO = (audio, y, prob)  # （音频，文本，概率）
                    SSL_AUDIO_UPDATE = True


class MonitorVoice(object):
    def __init__(self, MappingMicro=False):
        super(MonitorVoice, self).__init__()
        print('-' * 20 + 'init MonitorVoice class' + '-' * 20)
        self.MappingMicro = MappingMicro
        self.micro_mapping = np.arange(CHANNELS)
        self.CompleteMappingMicro = False
        
        self.device_index = self.__get_device_index__()
        assert CHUNK_SIZE == 16  # 1ms的采样点数，此参数可以使得语音队列中每一个值对应1ms的音频
        self.audio_queue = AUDIO_QUEUE
        self.init_micro_mapping_deque_len = 1000
        self.init_micro_mapping_deque = deque(maxlen=self.init_micro_mapping_deque_len)
    
    def __get_device_index__(self):
        device_index = -1
        
        # scan to get usb device
        p = PyAudio()
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
            print('-' * 20 + 'Find the device' + '-' * 20 + '\n', p.get_device_info_by_index(device_index), '\n')
            del p
        else:
            print('-' * 20 + 'Cannot find the device' + '-' * 20 + '\n')
            exit()
        
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
    
    def monitor_from_4mics(self, ):
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
                self.audio_queue.put(audio, block=False, )
            else:
                audio = self.split_channels_from_frame(frame=in_data, mapping_flag=False)
                self.init_micro_mapping_deque.append(audio, )
            # except Exception as e:
            #     print('Capture error:', e)
            #     print('-' * 20, 'audio_queue is FULL', '-' * 20)
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
    
    def split_channels_from_frame(self, frame, num_channel=CHANNELS, mapping_flag=False, micro_mapping=None):
        audio = np.frombuffer(frame, dtype=np.short)
        audio = np.reshape(audio, (-1, num_channel)).T
        if mapping_flag:
            audio = audio[micro_mapping]
        audio = np.asarray(audio, dtype=np.float64) / 32768.
        return audio
    
    def run(self):
        init_micro_thread = threading.Thread(target=self.__init_micro_mapping__, args=())
        init_micro_thread.start()
        
        self.monitor_from_4mics()


if __name__ == '__main__':
    print('-' * 20, 'Hello World!', '-' * 20)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    mv = MonitorVoice(MappingMicro=False)
    kws = KeyWordSpotting(use_stream=False)
    vm = VoiceMenu()
    
    p1 = threading.Thread(target=mv.run, args=())
    p1.start()
    p2 = threading.Thread(target=kws.run, args=())
    p2.start()
    p3 = threading.Thread(target=vm.run_forever, args=())
    p3.start()
    # p1 = multiprocessing.Process(target=mv.run, args=())
    # p1.start()
    # p2 = multiprocessing.Process(target=kws.run, args=())
    # p2.start()
    # p3 = multiprocessing.Process(target=vm.run_forever, args=())
    # p3.start()
    
    p1.join()
    # p2.join()
    # # p3.join()
    
    print('-' * 20, 'Brand-new World!', '-' * 20)
