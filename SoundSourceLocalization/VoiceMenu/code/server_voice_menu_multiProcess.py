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
import requests
import numpy as np
import threading
import ctypes
from multiprocessing import Process, Value, Pipe, Queue
from multiprocessing.managers import BaseManager
# from queue import Queue
from collections import deque  # , BlockingQueue
from scipy.special import softmax as scipy_softmax
from pyaudio import PyAudio

from SoundSourceLocalization.SSL_Settings import *
from SoundSourceLocalization.mylib import utils, audiolib
from SoundSourceLocalization.SSL_Communication.WalkerServer import WalkerServer

# KWS
import tensorflow.compat.v1 as tf
from kws_streaming.layers import modes
from kws_streaming.models import models
from kws_streaming.models import utils
from kws_streaming.train import inference

# from SoundSourceLocalization.lib.audiolib import normalize_single_channel_audio, audio_segmenter_4_numpy, \
#     audio_energy_ratio_over_threshold, audio_energy_over_threshold, audiowrite, audioread

SHARED_AUDIO_QUEUE = Queue(maxsize=2000)  # 每个 item 代表 1ms
SHARED_AUDIO_QUEUE_CLEAR = Value(ctypes.c_bool, False)
# 永远最多只有这么长，再多了直接清空，并将 AUDIO_QUEUE_CLEAR 置True; 如此也可以限制声音采集方面的延迟最多只有这么多


SHARED_SSL_AUDIO_QUEUE = Queue(maxsize=100)


# SSL_AUDIO_UPDATE = Value(ctypes.c_bool, False)
# SHARED_OUT_PIPE, SHARED_IN_PIPE = Pipe(duplex=False)


class KeyWordSpotting(object):
    def __init__(self, use_stream=False, ):
        
        # print('-' * 20, 'init KeyWordSpotting class', '-' * 20)
        super(KeyWordSpotting, self).__init__()
        
        self.use_stream = use_stream
        assert self.use_stream == False, 'Streaming model has not been tested'  # 暂时不考虑流式模型
        
        self.model_name = 'ds_tc_resnet_cpu_causal_20220110-143343'
        self.model_dir = os.path.abspath(os.path.join(CRT_DIR, '../model', self.model_name, ))
        print('KWS_model_dir:', self.model_dir)
        self.flags_path = os.path.join(self.model_dir, 'flags.json')
        self.flags = self.__load__flags__()
        self.flags.batch_size = 1
        
        print('-' * 20, 'Loading KWS non_stream_model...', '-' * 20, )
        self.non_stream_model = self.__load_non_stream_model__(weights_name='last_weights')
        if self.use_stream:  # 保存流式模型，直接加载，而非每次都要转换，还挺耗时的
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
    
    def __load_non_stream_model__(self, weights_name, ):
        tf.reset_default_graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        tf.keras.backend.set_session(sess)
        # tf.compat.v1.keras.backend.set_session(sess)
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
    
    def clear_Queue(self, Queue, description=None):
        while not Queue.empty():
            try:
                Queue.get_nowait()
            except:
                pass
        if description is not None:
            print('-' * 20, str(description), '-' * 20)
    
    def predict(self, x, use_stream=None, ):
        use_stream = self.use_stream if (use_stream is None) else use_stream
        if use_stream:
            y_pred = inference.run_stream_inference_classification(self.flags, self.stream_model, x)
        else:
            y_pred = self.non_stream_model.predict(x)
        y_pred = scipy_softmax(y_pred, axis=-1)
        label = np.argmax(y_pred, axis=-1)
        
        return label, y_pred[:, label].squeeze(axis=0)
    
    def save_audio(self, AUDIO_QUEUE, AUDIO_QUEUE_CLEAR, ):
        audio_frames = deque()
        while True:
            if AUDIO_QUEUE_CLEAR.value:
                AUDIO_QUEUE_CLEAR.value = False
                audio_frames.clear()
            try:
                audio_frames.append(AUDIO_QUEUE.get(block=True, timeout=1))
            except Exception as e:
                # print('Error when KWS get frames from AUDIO_QUEUE:', e)
                pass
            if len(audio_frames) >= 10000:
                audio = np.concatenate(audio_frames, axis=1)
                audio_frames.clear()
                save_dir = os.path.join('./', str(time.strftime("%Y%m%d-%H%M%S")))
                audiolib.save_multi_channel_audio(audio=audio, fs=16000, des_dir=save_dir, norm=False, )
    
    def send_help_request(self, prob):
        url = 'http://smartwalker.cs.hku.hk/smartwalker-backend/api/v1/notification/help'
        s = json.dumps({
            'from'       : 'SW000001',
            'to'         : 'smartwalker-demo-user',
            'probability': prob,
        })
        requests.post(url, data=s)
    
    def run(self, walker_server, AUDIO_QUEUE, AUDIO_QUEUE_CLEAR, SSL_AUDIO_QUEUE, ):
        print('-' * 20, 'KWS is running', '-' * 20)
        if not self.use_stream:
            local_audio_frames = deque(maxlen=self.clip_duration_ms)
            while True:
                for i in range(self.window_stride_ms):
                    if AUDIO_QUEUE_CLEAR.value:
                        AUDIO_QUEUE_CLEAR.value = False
                        # WORD_QUEUE_CLEAR.value = True
                        walker_server.send(data=True, subtopic=WORD_QUEUE_CLEAR_COMMUNICATION_TOPIC)
                        local_audio_frames.clear()
                    try:
                        local_audio_frames.append(AUDIO_QUEUE.get(block=True, timeout=1))
                    except Exception as e:
                        # print('Error when KWS get frames from AUDIO_QUEUE:', e)
                        pass
                if len(local_audio_frames) != self.clip_duration_ms:
                    continue
                ################################ predict ########################################
                audio = np.concatenate(local_audio_frames, axis=1)
                # audio = normalize_single_channel_to_target_level(audio, )
                x = np.array(audio[0], dtype=np.float64)[np.newaxis, :]
                y, prob = self.predict(x, use_stream=self.use_stream)
                y, prob = self.labels[y[0]], float(prob[0])
                walker_server.send(data=(y, prob), subtopic=KWS_COMMUNICATION_TOPIC)
                # print('y, prob:', y, prob)
                
                # if (y not in ['silence', 'unknown', ]) and prob > 0.70:
                #     print('y & prob:', y, round(prob, 3), end='\t')
                # print(y, round(prob, 3), end='\t')  # TODO: for debugging
                if y == self.walker_name:
                    if SSL_AUDIO_QUEUE.full():
                        self.clear_Queue(SSL_AUDIO_QUEUE, description='SSL_AUDIO_QUEUE is cleared as it is full.')
                    SSL_AUDIO_QUEUE.put((audio, y, prob, time.time()), block=True, timeout=1)
                    # print(f'KWS: walker_name (\'{self.walker_name}\') is detected.')
                    # SSL_AUDIO = (audio, y, prob)  # （音频，文本，概率）
                    # SSL_AUDIO_UPDATE = True
                elif y == 'help':  # send help request
                    self.send_help_request(prob=prob)
        
        else:
            assert False, 'Streaming model has not been tested'
            # local_audio_frames = deque(maxlen=self.window_stride_ms)
            # while True:
            #     for i in range(self.window_stride_ms):
            #         local_audio_frames.append(AUDIO_QUEUE.get(block=True, timeout=None))
            #     if len(local_audio_frames) != self.clip_duration_ms:
            #         continue
            #     ################################ predict ########################################
            #     audio = np.concatenate(local_audio_frames, axis=1)
            #     # audio = normalize_single_channel_to_target_level(audio, )
            #     x = np.array(audio[0], dtype=np.float64)[np.newaxis, :]
            #     y, prob = self.predict(x, use_stream=self.use_stream)
            #     y, prob = self.labels[y[0]], prob[0]
            #     # WORD_QUEUE.put((y, prob))
            #     # WORD_QUEUE_UPDATA = True
            #     # if (y not in ['silence', 'unknown', ]) and prob > 0.70:
            #     #     # print('y & prob:', y, round(prob, 3), end='\t')
            #     #     print(y, round(prob, 3), end='\t')
            #     if y == self.walker_name:
            #         print(f'KWS: walker_name (\'{self.walker_name}\') is detected.')
            #         # SSL_AUDIO = (audio, y, prob)  # （音频，文本，概率）
            #         # SSL_AUDIO_QUEUE_UPDATE = True


class MonitorVoice(object):
    def __init__(self, ):
        print('-' * 20, 'init MonitorVoice class', '-' * 20)
        super(MonitorVoice, self).__init__()
    
    def clear_Queue(self, Queue, description=None):
        while not Queue.empty():
            try:
                Queue.get_nowait()
            except:
                pass
        if description is not None:
            print('-' * 20, str(description), '-' * 20)
    
    def run(self, walker_server, AUDIO_QUEUE, AUDIO_QUEUE_CLEAR, ):
        while True:
            audio = walker_server.recv(subtopic=AUDIO_COMMUNICATION_TOPIC)
            audio = np.asarray(audio)
            # time.sleep(0.001)
            # audio = np.random.random((4, 16))
            # print('An audio frame is received')
            if AUDIO_QUEUE.full():
                self.clear_Queue(AUDIO_QUEUE, description='AUDIO_QUEUE is cleared as it is full.', )
                AUDIO_QUEUE_CLEAR.value = True
            AUDIO_QUEUE.put(audio, block=True, timeout=1)


class MonitorVoice_Process(object):
    def __init__(self, MappingMicro=False, ):
        super(MonitorVoice_Process, self).__init__()
        self.MappingMicro = MappingMicro
    
    def run(self, walker_server, AUDIO_QUEUE, AUDIO_QUEUE_CLEAR, ):
        mv = MonitorVoice()
        mv.run(walker_server, AUDIO_QUEUE, AUDIO_QUEUE_CLEAR, )


class KeyWordSpotting_Process(object):
    def __init__(self, use_stream=False, ):
        super(KeyWordSpotting_Process, self).__init__()
        self.use_stream = use_stream
    
    def run(self, walker_server, AUDIO_QUEUE, AUDIO_QUEUE_CLEAR, SSL_AUDIO_QUEUE, ):
        kws = KeyWordSpotting(use_stream=self.use_stream, )
        kws.run(walker_server, AUDIO_QUEUE, AUDIO_QUEUE_CLEAR, SSL_AUDIO_QUEUE, )


class SSL_test(object):
    def __init__(self, **kwargs):
        super(SSL_test, self).__init__()
    
    def run(self, walker_server, SSL_AUDIO_QUEUE, **kwargs):
        self.SSL_AUDIO_QUEUE = SSL_AUDIO_QUEUE
        while True:
            msg = SSL_AUDIO_QUEUE.get(block=True, timeout=None)
            (audio, y, prob, time) = msg
            print('SSL: walker data is received~', )
            direction = 0
            walker_server.send(data=direction, subtopic=SSL_NAV_COMMUNICATION_TOPIC)
            print(f'SSL: Direction ({direction}) is received')


if __name__ == '__main__':
    print('-' * 20, 'Hello World!', '-' * 20)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    manager = BaseManager()
    # 一定要在start前注册，不然就注册无效
    manager.register('WalkerServer', WalkerServer)  # 第一个参数为类型id，通常和第二个参数传入的类的类名相同，直观并且便于阅读
    manager.start()
    walker_server = manager.WalkerServer()
    
    mv = MonitorVoice_Process(MappingMicro=False, )
    kws = KeyWordSpotting_Process(use_stream=False, )
    ssl = SSL_test()
    
    p1 = Process(target=mv.run, args=(walker_server, SHARED_AUDIO_QUEUE, SHARED_AUDIO_QUEUE_CLEAR,))
    p1.start()
    p2 = Process(target=kws.run,
                 args=(walker_server, SHARED_AUDIO_QUEUE, SHARED_AUDIO_QUEUE_CLEAR, SHARED_SSL_AUDIO_QUEUE,))
    p2.start()
    # p3 = Process(target=vm.run_forever, args=(SHARED_WORD_QUEUE, SHARED_WORD_QUEUE_UPDATA, SHARED_WORD_QUEUE_CLEAR,))
    # p3.start()
    p4 = Process(target=ssl.run, args=(walker_server, SHARED_SSL_AUDIO_QUEUE,))
    p4.start()
    
    # SHARED_IN_PIPE.close()
    # SHARED_OUT_PIPE.close()
    
    p1.join()
    # p2.join()
    # # p3.join()
    
    print('-' * 20, 'Brand-new World!', '-' * 20)
