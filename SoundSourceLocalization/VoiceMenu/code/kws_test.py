# !/usr/bin/env python
# -*- coding:utf-8 _*-
# @Author: swang
# @Contact: wang00sheng@gmail.com
# @Project Name: keyword_spotting_system
# @File: test.py
# @Time: 2021/11/11/21:51
# @Software: PyCharm
import os
import sys
import time
import random
import warnings
import numpy as np
from copy import deepcopy
import wave
from ssl_setup import *
from pyaudio import PyAudio
import threading
from collections import deque  # , BlockingQueue
from queue import Queue
from SoundSourceLocalization.lib.audiolib import normalize_single_channel_audio, audio_segmenter_4_numpy, \
    audio_energy_ratio_over_threshold, audio_energy_over_threshold, audiowrite, audioread

sys.path.append("./kws_streaming/train")
sys.path.append("./kws_streaming/train/model_ckpt")
from absl import logging
import tensorflow.compat.v1 as tf
# import kws_streaming.data.input_data as input_data
from kws_streaming.layers import modes
from kws_streaming.models import models
from kws_streaming.models import utils
from kws_streaming.train import inference
import pickle
from scipy.special import softmax
import json


# logging.info('TF Final test accuracy on non stream model = %.2f%% (N=%d)', *(total_accuracy * 100, set_size))


class KWS(object):
    def __init__(self, use_stream=False):
        print('-' * 20 + 'init KWS class' + '-' * 20)
        self.use_stream = use_stream
        
        self.model_name = 'ds_tc_resnet_cpu_causal_20211231-200734'
        # ds_tc_resnet_cpu_causal_20211220-131242/ds_tc_resnet_cpu_causal_20211220-154022/ds_tc_resnet_cpu_causal_20211220-163449/ds_tc_resnet_cpu_causal_20211220-163457/ds_tc_resnet_cpu_causal_20211220-170935/ds_tc_resnet_cpu_causal_20211220-170943/ds_tc_resnet_cpu_causal_20211220-170948/
        # ds_tc_resnet_cpu_causal_20211231-194536/ds_tc_resnet_cpu_causal_20211231-200704/ds_tc_resnet_cpu_causal_20211231-200709/ds_tc_resnet_cpu_causal_20211231-200721/ds_tc_resnet_cpu_causal_20211231-200734/ds_tc_resnet_cpu_causal_20211231-200742
        self.train_dir = './kws_streaming/train/'
        self.model_dir = os.path.join(self.train_dir, 'model_ckpt', self.model_name, )
        self.flags = self.__load__flags__()
        self.flags.batch_size = 1
        self.non_stream_model = self.__load_non_stream_model__(weights_name='last_weights')
        if self.use_stream:
            self.stream_model = self.__convert_2_stream_model__()
        self.labels = np.array(['silence', 'unknown', ] + self.flags.wanted_words.split(','))
        print('-' * 10, 'labels:', self.labels, '-' * 10)
        self.clip_duration_ms = int(self.flags.clip_duration_ms)
        self.window_stride_ms = int(self.flags.window_stride_ms)
        
        self.micro_mapping = np.array(range(CHANNELS), dtype=np.int)
        self.device_index = self.__get_device_index__()
        self.chunk = 16  # 1ms的采样点数，此参数可以使得队列中每一个值对应1ms的音频
        self.frames = Queue()  # (maxsize=3 * self.clip_duration_ms)  # []
        # self.fs = SAMPLE_RATE
        self.start_time = None
    
    def __load__flags__(self):
        # print('flags_path:', os.path.abspath(flags_path))
        # with open(flags_path, "rb") as fo:
        #     self.flags = pickle.load(fo)
        
        with tf.compat.v1.gfile.Open(os.path.join(self.model_dir, 'flags.json'), 'r') as fd:
            flags_json = json.load(fd)
        
        class DictStruct(object):
            def __init__(self, **entries):
                self.__dict__.update(entries)
        
        self.flags = DictStruct(**flags_json)
        
        return self.flags
    
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
    
    def __load_non_stream_model__(self, weights_name):
        tf.reset_default_graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        tf.keras.backend.set_session(sess)
        # self.audio_processor = input_data.AudioProcessor(self.flags)
        tf.keras.backend.set_learning_phase(0)
        # tf.disable_eager_execution()
        
        self.non_stream_model = models.MODELS[self.flags.model_name](self.flags)
        print('non_stream_model_weight:',
              os.path.join(self.train_dir, os.path.join(self.flags.train_dir, weights_name, )))
        self.non_stream_model.load_weights(
            os.path.join(self.train_dir, self.flags.train_dir, weights_name, )).expect_partial()
        self.non_stream_model.summary()
        
        # tf.keras.utils.plot_model(
        #     self.non_stream_model,
        #     show_shapes=True,
        #     show_layer_names=True,
        #     expand_nested=True,
        #     to_file=os.path.join('./', self.model_name + '_non_stream.png'), )
        #
        return self.non_stream_model
    
    def __convert_2_stream_model__(self):
        print('tf stream model state internal without state resetting between testing sequences')
        self.flags.data_shape = modes.get_input_data_shape(self.flags, modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
        self.stream_model = utils.to_streaming_inference(
            self.non_stream_model, self.flags, modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
        self.stream_model.summary()
        
        # tf.keras.utils.plot_model(
        #     self.stream_model,
        #     show_shapes=True,
        #     show_layer_names=True,
        #     expand_nested=True,
        #     to_file=os.path.join('./', self.model_name + '_stream.png'),
        # )
        
        return self.stream_model
    
    def monitor_from_4mics(self, ):
        print('-' * 20, "Monitoring microphones...", '-' * 20)
        
        def PyAudioCallback(in_data, frame_count, time_info, status):
            if self.start_time is None:
                self.start_time = time.time()
            try:
                self.frames.put(in_data, block=False, timeout=None)
            except:
                print('-' * 20, 'self.frames is FULL', '-' * 20)
                exit(-1)
            # exit(0)
            # print('frame_count:', )
            # print('in_data_len:', len(in_data))
            return (None, pyaudio.paContinue)
        
        p = PyAudio()
        stream = p.open(format=p.get_format_from_width(RECORD_WIDTH),
                        channels=CHANNELS,
                        rate=SAMPLE_RATE,
                        input=True,
                        input_device_index=self.device_index,
                        frames_per_buffer=self.chunk,
                        stream_callback=PyAudioCallback,
                        start=True)
        while stream.is_active():
            time.sleep(1)
    
    def split_channels_from_frames(self, frames=None, num_channel=CHANNELS, mapping_flag=True):
        if frames is None:
            frames = self.frames
        audio = np.frombuffer(b''.join(frames), dtype=np.short)
        audio = np.reshape(audio, (-1, num_channel)).T
        if mapping_flag:
            audio = audio[self.micro_mapping]
        return audio
    
    def time_len(self):
        while True:
            time_len = len(self.frames) * self.chunk / SAMPLE_RATE
            print('frame_len: {:.3f}'.format(time_len))
            # if self.start_time is not None:
            #     end_time = time.time()
            #     print('time_sub:', (end_time - self.start_time) - time_len)
            time.sleep(1)
    
    def save_multi_channel_audio(self, des_dir, audio, fs=SAMPLE_RATE, norm=True, ):
        for i in range(len(audio)):
            file_path = os.path.join(des_dir, 'test_mic%d.wav' % i)
            audiowrite(file_path, audio[i], sample_rate=fs, norm=norm, target_level=-25, clipping_threshold=0.99)
    
    def savewav_from_frames(self, filename, frames=None):
        if frames is None:
            frames = self.frames
        
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(RECORD_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
    
    def save_audio_test(self, counters=''):
        ini_dir = os.path.join('./test', 'wav', 'ini_signal' + str(counters))
        # self.save_multi_channel_audio(ini_dir, ini_signals, fs=SAMPLE_RATE, norm=False, )
        ini_path = ini_dir + '.wav'
        self.savewav_from_frames(ini_path, frames=self.frames)
    
    # @staticmethod
    def non_stream_kws(self, x):
        y_pred = self.non_stream_model.predict(x)
        y_pred = softmax(y_pred, axis=-1)
        label = np.argmax(y_pred, axis=-1)
        
        return label, y_pred[:, label].squeeze(axis=0)
    
    # @staticmethod
    def stream_kws(self, x):
        y_pred = inference.run_stream_inference_classification(self.flags, self.stream_model, x)
        y_pred = softmax(y_pred, axis=-1)
        label = np.argmax(y_pred, axis=-1)
        
        return label, y_pred[:, label].squeeze(axis=0)
    
    def speed_test(self):
        import time
        data = np.random.random((1, 16000))
        start_time = time.time()
        for _ in range(10000):
            self.non_stream_kws(x=data)
        end_time = time.time()
        time_non_stream = end_time - start_time
        print('time_non_stream:', time_non_stream)
        
        data = np.random.random((1, 160))
        start_time = time.time()
        for _ in range(10000):
            self.stream_kws(x=data)
        end_time = time.time()
        time_stream = end_time - start_time
        print('time_stream:', time_stream)
        
        print('runtime ratio of non_stream_kws/stream_kws:', time_non_stream / time_stream)
    
    def run(self):
        if not self.use_stream:
            local_frames = deque(maxlen=self.clip_duration_ms)
            while True:
                for i in range(self.window_stride_ms):
                    local_frames.append(self.frames.get(block=True, timeout=3 * self.clip_duration_ms / 1000))
                if len(local_frames) != self.clip_duration_ms:
                    continue
                ################################ predict ########################################
                ini_signals = self.split_channels_from_frames(frames=local_frames, num_channel=CHANNELS,
                                                              mapping_flag=False)
                # Normalize short ints to floats in range [-1..1).
                audio = np.array(ini_signals[0], dtype=np.float64)[np.newaxis, :] / 32768.0
                # audio = normalize_single_channel_to_target_level(audio, )
                y, prob = self.non_stream_kws(audio)
                y, prob = self.labels[y[0]], prob[0]
                if prob > 0.8 and (y not in ['silence', 'unknown', ]):
                    # print('y & prob:', y, round(prob, 3), end='\t')
                    print(y, ':', round(prob, 3), end='\t')
                # time.sleep(0.1)
                # test_fingerprints, test_ground_truth = np.random.random((1, 16000)), [0.]
                # y = self.non_stream_kws(test_fingerprints)
                # print('y:', y)
        else:
            local_frames = deque(maxlen=self.window_stride_ms)
            while True:
                for i in range(self.window_stride_ms):
                    local_frames.append(self.frames.get(block=True, timeout=3 * self.clip_duration_ms / 1000))
                if len(local_frames) != self.window_stride_ms:
                    continue
                ################################ predict ########################################
                ini_signals = self.split_channels_from_frames(frames=local_frames, num_channel=CHANNELS,
                                                              mapping_flag=False)
                # Normalize short ints to floats in range [-1..1).
                audio = np.array(ini_signals[0], dtype=np.float64)[np.newaxis, :] / 32768.0
                # audio = normalize_single_channel_to_target_level(audio, )
                y, prob = self.stream_kws(audio)
                y, prob = self.labels[y[0]], prob[0]
                if prob > 0.90:
                    print('y & prob:', y, round(prob, 3))
                # time.sleep(0.1)
                # test_fingerprints, test_ground_truth = np.random.random((1, 16000)), [0.]
                # y = self.non_stream_kws(test_fingerprints)
                # print('y:', y)


if __name__ == '__main__':
    print('-' * 20, 'Hello World!', '-' * 20)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    kws = KWS(use_stream=False)
    # kws.speed_test()
    p1 = threading.Thread(target=kws.monitor_from_4mics, args=())
    p1.start()
    # p2 = threading.Thread(target=kws.run, args=())
    # p2.start()
    # # p3 = threading.Thread(target=kws.time_len, args=())
    # # p3.start()
    #
    # p1.join()
    # p2.join()
    # # p3.join()
    kws.run()
    
    print('-' * 20, 'Brand-new World!', '-' * 20)
