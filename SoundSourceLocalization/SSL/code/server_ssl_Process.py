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
from scipy.signal import get_window

import threading
from multiprocessing import Process, Value, Pipe, Queue

# general lib
from SoundSourceLocalization.mylib import utils
from SoundSourceLocalization.mylib.audiolib import normalize_single_channel_audio, audio_segmenter_4_numpy, \
    audio_energy_ratio_over_threshold, audio_energy_over_threshold, audioread, audiowrite

# independent systems
from SoundSourceLocalization.SSL_Settings import *
import SoundSourceLocalization.SpeechEnhancement.code.ns_enhance_onnx as ns_enhance_onnx
from SoundSourceLocalization.SSL.code.ssl_audio_processor import *
from SoundSourceLocalization.SSL.code.ssl_DOA_model import DOA
from SoundSourceLocalization.SSL.code.ssl_turning import SSLturning


# from SoundSourceLocalization.ReinforcementLearning.code.ssl_agent import Agent
# from SoundSourceLocalization.ReinforcementLearning.code.ssl_env import MAP_ENV, ONLINE_MAP_ENV


class SSL(object):
    def __init__(self, doDenoise=True, seg_len='1s', isDebug=False):
        print('-' * 20 + 'init SSL class' + '-' * 20)
        self.isDebug = isDebug
        self.seg_len = seg_len
        self.doDenoise = doDenoise  # useless
        self.doDrop = True
        
        '------------------------------ preprocessing parameters ------------------------------'
        segment_para_set = {
            '32ms' : {
                'name'          : '32ms',
                'time_len'      : 32 / 1000,
                'threshold'     : 100,
                'stepsize_ratio': 0.5
            },
            '50ms' : {
                'name'          : '50ms',
                'time_len'      : 50 / 1000,
                'threshold'     : 100,
                'stepsize_ratio': 0.5
            },
            '64ms' : {
                'name'          : '64ms',
                'time_len'      : 64 / 1000,
                'threshold'     : 100,
                'stepsize_ratio': 0.5
            },
            '128ms': {
                'name'          : '128ms',
                'time_len'      : 128 / 1000,
                'threshold'     : 200,  # 100?
                'stepsize_ratio': 0.5
            },
            '256ms': {
                'name'          : '256ms',
                'time_len'      : 256 / 1000,
                'threshold'     : 400,
                'stepsize_ratio': 256 / 1000 / 2
            },
            '1s'   : {
                'name'          : '1s',
                'time_len'      : 1,  # total length of an audio clip
                'threshold'     : 800,  # threshold for filter clips with low volume and improper frequency ratio.
                'stepsize_ratio': 0.5,  # the ratio of stepsize over time_len
                # 'stepsize'      : 0.5,  # stepsize (in seconds)
            },
        }
        self.seg_para = segment_para_set[seg_len]
        self.seg_para['seg_points'] = int(self.seg_para['time_len'] * SAMPLE_RATE)
        self.seg_para['stepsize_points'] = int(self.seg_para['seg_points'] * self.seg_para['stepsize_ratio'])
        self.window_type = 'hann'
        self.window_filter = get_window(window=self.window_type, Nx=self.seg_para['seg_points'], )
        
        # ref_audio, _ = audioread(os.path.abspath('./reference_wav.wav'))
        # self.ref_audio = normalize_single_channel_audio(ref_audio)
        # self.ref_audio_threshold = (self.ref_audio ** 2).sum() / len(self.ref_audio) / 500
        # del ref_audio, self.ref_audio,
        self.ref_audio_energy_threshold = 6.324555320336738e-06  # get from reference audio
        # feature paras
        self.num_gcc_bin = 128
        self.num_mel_bin = 128
        self.fft_len = self.seg_para['seg_points']
        self.fft_seg_len = 0.064
        self.fft_stepsize_ratio = 0.5
        '----------------------------------------------------------------------------------------------------'
        
        # SpeechEnhancement
        print('-' * 20, 'Loading denoising model...', '-' * 20, )
        self.denoise_model, _ = ns_enhance_onnx.load_onnx_model()
        
        # DOA
        print('-' * 20, 'Loading DOA model...', '-' * 20, )
        self.num_action = 8
        doa_model_path = os.path.abspath(os.path.join(CRT_DIR, '../model/ResCNN/base_model/ckpt'))
        self.doa = DOA(model_dir=doa_model_path, num_gcc_bin=self.num_gcc_bin, num_mel_bin=self.num_mel_bin,
                       fft_len=self.fft_len, fft_seg_len=self.fft_seg_len, fft_stepsize_ratio=self.fft_stepsize_ratio,
                       fs=SAMPLE_RATE, )
        
        # RL
        # self.save_model_steps = 3
        # self.env = ONLINE_MAP_ENV()
        # self.save_ac_model = './model/ac_model'
        # self.agent = Agent(alpha=1., num_action=self.num_action, gamma=0.99, ac_model_dir=self.save_ac_model,
        #                    load_ac_model=True, save_model_steps=self.save_model_steps)
    
    def drop_audio_per_seg(self, signal, ):
        '''
        return flag to determine whether a audio segment should be dropped or not, based on two standards:
        1. audio_energy_ratio
        2. audio_energy_over_threshold
        :param signal: a multi-channel signal clip
        :return: boolean flag
        '''
        signal = np.array(signal)
        for channel in signal:
            if not (audio_energy_over_threshold(channel, threshold=self.ref_audio_energy_threshold, ) and
                    audio_energy_ratio_over_threshold(channel, fs=SAMPLE_RATE, threshold=self.seg_para['threshold'], )):
                return True
        return False
    
    # def drop_audio_clips(self, signal_segments, ):
    #     # print('Number of segments before dropping: ', len(signal_segments))
    #     audio_segments = []
    #     drop_flag = []
    #     for i in range(len(signal_segments)):
    #         drop_flag.append(self.drop_audio_per_seg_point(signal_segments[i]))
    #         if not drop_flag[-1]:
    #             audio_segments.append(signal_segments[i])
    #         else:
    #             continue
    #             # audio_segments.append([])
    #     # print('Number of segments after dropping: ', len(audio_segments))
    #
    #     return np.array(audio_segments), drop_flag
    
    # def normalize_batch_audio(self, audio_batch):
    #     '''
    #     For every audio in a batch, normalize its channels respectively.
    #     :param audio_batch:
    #     :return:
    #     '''
    #     res_audio = []
    #     for audio_channels in audio_batch:
    #         norm_audio_channels = []
    #         for audio in audio_channels:
    #             norm_audio_channels.append(normalize_single_channel_audio(audio))
    #         res_audio.append(norm_audio_channels)
    #
    #     return np.asarray(res_audio)
    
    # def denoise_batch_audio(self, audio_batch):
    #     '''
    #      For every audio in a batch, denoise its channels respectively.
    #     :param audio_batch:
    #     :return:
    #     '''
    #     res_audio = []
    #     for audio_channels in audio_batch:
    #         denoised_channels = []
    #         for audio in audio_channels:
    #             denoised_channels.append(
    #                 ns_enhance_onnx.denoise_nsnet2(audio=audio, fs=SAMPLE_RATE, model=self.denoise_model, ))
    #         res_audio.append(denoised_channels)
    #
    #     return np.asarray(res_audio)
    # def get_audio_from_pipe(self, RECV_PIPE):
    #     ''' 简单起见，目前只选取最新的数据，且距离发送时间不超过0.5s
    #     实际测试发现，KWS传过来的声音对于单个单词持续时间在 [0.2, 0.5]s 之间
    #     :return audio
    #     '''
    #     res = []
    #     noData = True
    #     while noData:
    #         start_time = time.time()
    #         while RECV_PIPE.poll():
    #             msg = RECV_PIPE.recv()
    #             (audio, y, prob, send_time) = msg
    #             if abs(start_time - send_time) < KWS_TIMEOUT_SECONDS:
    #                 res.append(msg)
    #                 noData = False
    #     print('SSL: walker data is received~', )
    #     return res[-1][0]
    # def drop_audio_clips(self, signal_segments, ):
    #     # print('Number of segments before dropping: ', len(signal_segments))
    #     audio_segments = []
    #     drop_flag = []
    #     for i in range(len(signal_segments)):
    #         drop_flag.append(self.drop_audio_per_seg_point(signal_segments[i]))
    #         if not drop_flag[-1]:
    #             audio_segments.append(signal_segments[i])
    #         else:
    #             continue
    #             # audio_segments.append([])
    #     # print('Number of segments after dropping: ', len(audio_segments))
    #
    #     return np.array(audio_segments), drop_flag
    
    # def normalize_batch_audio(self, audio_batch):
    #     '''
    #     For every audio in a batch, normalize its channels respectively.
    #     :param audio_batch:
    #     :return:
    #     '''
    #     res_audio = []
    #     for audio_channels in audio_batch:
    #         norm_audio_channels = []
    #         for audio in audio_channels:
    #             norm_audio_channels.append(normalize_single_channel_audio(audio))
    #         res_audio.append(norm_audio_channels)
    #
    #     return np.asarray(res_audio)
    
    # def denoise_batch_audio(self, audio_batch):
    #     '''
    #      For every audio in a batch, denoise its channels respectively.
    #     :param audio_batch:
    #     :return:
    #     '''
    #     res_audio = []
    #     for audio_channels in audio_batch:
    #         denoised_channels = []
    #         for audio in audio_channels:
    #             denoised_channels.append(
    #                 ns_enhance_onnx.denoise_nsnet2(audio=audio, fs=SAMPLE_RATE, model=self.denoise_model, ))
    #         res_audio.append(denoised_channels)
    #
    #     return np.asarray(res_audio)
    # def get_audio_from_pipe(self, RECV_PIPE):
    #     ''' 简单起见，目前只选取最新的数据，且距离发送时间不超过0.5s
    #     实际测试发现，KWS传过来的声音对于单个单词持续时间在 [0.2, 0.5]s 之间
    #     :return audio
    #     '''
    #     res = []
    #     noData = True
    #     while noData:
    #         start_time = time.time()
    #         while RECV_PIPE.poll():
    #             msg = RECV_PIPE.recv()
    #             (audio, y, prob, send_time) = msg
    #             if abs(start_time - send_time) < KWS_TIMEOUT_SECONDS:
    #                 res.append(msg)
    #                 noData = False
    #     print('SSL: walker data is received~', )
    #     return res[-1][0]
    
    def preprocess_ini_signal(self, ini_signals):
        '''
        preprocess the received signals. Must be consistent with the dataset preprocessing method.
        :param ini_signals: a multi-channel signal clip
        :return:
        '''
        ini_signals = np.array(ini_signals, dtype=np.float32)
        
        # win
        win_signals = ini_signals * self.window_filter
        # norm
        norm_res = []
        for channel in win_signals:
            norm_res.append(normalize_single_channel_audio(channel, returnScalar=True))
        norm_audio, norm_scalar = list(zip(*norm_res))
        # denoise
        de_norm_audio = np.asarray([self.denoise_model(sigIn=i, inFs=SAMPLE_RATE) for i in norm_audio])
        # restore
        de_audio = de_norm_audio / np.reshape(norm_scalar, (4, -1))
        
        # drop
        # doDrop = self.drop_audio_per_seg(signal=de_audio)
        # return de_audio if (not doDrop) else None
        return de_audio
    
    def convert_owen_dir_2_digit(self, rad):
        rad = rad if (rad >= 0) else (rad + 2 * np.pi)
        degree = rad * 180 / np.pi
        dir_digit = (int(degree + 22.5) // 45 + 8 - 2) % 8
        print('degree: ', degree, 'dir_digit: ', dir_digit)
        
        return dir_digit
    
    def convert_owen_location_2_map(self, location):
        location = [location[0] - 40, location[1] - 12]
        return location
    
    def convert_map_location_2_owen(self, location):
        if np.allclose(location, [60, 425]):  # 1
            location = [120, 440]
        elif np.allclose(location, [160, 320]):  # 2
            location = [196, 326]
        elif np.allclose(location, [220, 15]):  # 9
            location = [246, 30]
        elif np.allclose(location, [530, 220]):  # 18
            location = [560, 232]
        else:
            location = [location[0] + 40, location[1] + 12]
        return location
    
    def get_crt_position(self):
        # message = '[320.5940246582031,201.4725799560547,-1.5714188814163208]'
        while True:
            message = self.client.receive()
            if message != '':
                break
        print('End receiving: ', message)
        message = json.loads(message)
        location = self.convert_owen_location_2_map(message[0:2])
        dir_digit = self.convert_owen_dir_2_digit(message[2])
        
        return location, dir_digit
    
    def send_crt_position(self, position, ):
        (y, x) = self.convert_map_location_2_owen(position)
        message = [int(y), int(x)]
        print('Starting to send')
        self.client.transmit(message=message)
        print('End sending: ', message)
    
    def clear_Queue(self, Queue, description=None):
        while not Queue.empty():
            try:
                Queue.get_nowait()
            except:
                pass
        if description is not None:
            print('-' * 20, str(description), '-' * 20)
    
    def run(self, walker_server, SSL_AUDIO_QUEUE, ):
        # initialize models
        num_step = 0
        Event_Wait = False  # control the running state of SSL # TODO: for debugging
        last_ssl_time = None
        # steps
        while True:
            temp_wait = walker_server.recv(subtopic=SSL_WAIT_COMMUNICATION_TOPIC, )
            if temp_wait is not None:
                Event_Wait = temp_wait
                self.clear_Queue(Queue=SSL_AUDIO_QUEUE,
                                 description='SSL_AUDIO_QUEUE is cleared due to the change of SSL state.')
            if Event_Wait:
                time.sleep(0.1)
                continue
            
            # Detecting for walker_name
            # ini_signals = self.get_audio_from_pipe(RECV_PIPE)
            (ini_signals, y, prob, send_time) = SSL_AUDIO_QUEUE.get(block=True, timeout=None)
            if prob <= 0.6:
                print('walker will be dropped due to low confidence.')
                continue
            if time.time() - send_time >= 1:
                print('walker will be dropped due to high latency.')
                continue
            if (last_ssl_time is not None) and abs(send_time - last_ssl_time) < 2:
                print('walker will be dropped due to other adjacent walkers.')
                continue
            last_ssl_time = send_time
            
            # preprocess initial audios (may be dropped)
            audio = self.preprocess_ini_signal(ini_signals)
            
            # not drop and will localize the source
            num_step += 1
            print('-' * 20, 'SSL step:', num_step, '-' * 20)
            
            # extract features
            feature = self.doa.extract_stft(audio=audio, )
            # SSL predict
            _, direction = self.doa.predict([feature])
            print("Producing action ...\n", 'Direction', direction)
            walker_server.send(data=direction, subtopic=SSL_DOA_COMMUNICATION_TOPIC)
    
    def run_RL(self, walker_server, SSL_AUDIO_QUEUE, ):
        # initialize models
        doa = self.doa
        # configuration for RL
        env = self.env
        agent = self.agent
        state, state_, = None, None,
        node, node_ = None, None
        action, action_ = None, None
        reward, reward_ = None, None
        done = False
        num_step = 0
        reward_history = []
        position = None
        
        # steps
        while True:
            time.sleep(0.5)
            # Detecting for walker_name
            if VoiceMenu.SSL_AUDIO_UPDATE:
                ini_signals, y, prob = VoiceMenu.SSL_AUDIO
                VoiceMenu.SSL_AUDIO_UPDATE = False
                # save data
                # ini_dir = os.path.join(WAV_PATH, save_dir, 'ini_signal')
                # self.save_multi_channel_audio(ini_dir, ini_signals, fs=SAMPLE_RATE, norm=False, )
            else:
                continue
            
            # preprocess initial audios
            audio_segments, drop_flag = self.preprocess_ini_signal(ini_signals)
            print('Number of preprocessed audio segments: ', len(audio_segments))
            
            if not (len(audio_segments) > 0):
                continue
            num_step += 1
            print('-' * 20, num_step, '-' * 20)
            '''------------------------- 获取可行方向 -----------------------------'''
            # 获取实时位置
            if isDebug:
                # crt_position = input('please input current position and direction')
                crt_position = '280 160 2'
                crt_position = list(map(float, crt_position.split(' ')))
                crt_loca, crt_abs_doa = crt_position[:2], int(crt_position[2])
            else:
                crt_loca, crt_abs_doa = self.get_crt_position()
            print('crt_location: ', crt_loca, 'crt_abs_doa: ', crt_abs_doa)
            # 获取可行方向
            crt_node = env.get_graph_node_idx(position=crt_loca)
            node_ = crt_node
            abs_availalbe_dircs = env.get_availalbe_dircs(node_idx=crt_node)  # 此处方向应该以小车为坐标系,但是获得的方向是绝对坐标系。
            # print('availalbe_dircs: ', availalbe_dircs)
            abs_dirc_mask = np.array(np.array(abs_availalbe_dircs) != None)
            rela_dirc_mask = np.roll(abs_dirc_mask, shift=-crt_abs_doa)
            # print('rela_dirc_mask: ', rela_dirc_mask)
            dirc_digit = np.where(rela_dirc_mask)
            print("crt_node: ", crt_node, 'avaliable_rela_dirc_digit: ', list(dirc_digit))
            
            '''--------------------------- 强化学习 -------------------------------'''
            # update state
            if not self.isDebug:
                gcc_feature_batch = doa.extract_gcc_phat_4_batch(audio_segments)
                gcc_feature = np.mean(gcc_feature_batch, axis=0)
                state_ = gcc_feature
            else:
                state_ = np.ones((1, 6, 128))
            ### 接入强化学习 learn
            # 选择行为前，mask掉不可行的方向
            action_ = agent.choose_action(state_, dirc_mask=rela_dirc_mask, sample=True)
            # _, direction_cate, = doa.predict(gcc_feature)
            # print(direction_prob)
            print('Predicted action_: ', action_)
            # print("Producing action ...\n", 'Direction', direction)
            aim_node = env.next_id_from_rela_action(crt_node, action=action_, abs_doa=crt_abs_doa)
            aim_loca = env.map.coordinates[aim_node]
            position = aim_loca
            print('aim_node: ', aim_node, 'aim_loca: ', aim_loca)
            
            ### 接入Owen的模块，传入aim_loca
            if self.useCD:
                SSLturning(control, action_)
                control.speed = STEP_SIZE / FORWARD_SECONDS
                control.radius = 0
                control.omega = 0
                time.sleep(FORWARD_SECONDS)
                control.speed = 0
                print("movement done.")
            else:
                self.send_crt_position(aim_loca)
            
            # 维护 done TODO
            # 强化
            if state is not None:
                # state_, reward, done, info = env.step(action)
                # reward = reward_history[-1]
                agent.learn(state, action, reward, state_, done)
            reward_ = float(input('Please input the reward for this action: '))
            
            state = state_
            node = node_
            action = action_
            reward = reward_


class SSL_Process(object):
    def __init__(self, doDenoise=True, seg_len='1s', isDebug=False, ):
        super(SSL_Process, self).__init__()
        self.seg_len = seg_len
        self.doDenoise = doDenoise
        self.isDebug = isDebug
    
    def run(self, walker_server, SSL_AUDIO_QUEUE, ):
        ssl = SSL(seg_len=self.seg_len, doDenoise=self.doDenoise, isDebug=self.isDebug, )
        ssl.run(walker_server, SSL_AUDIO_QUEUE, )


if __name__ == '__main__':
    ssl = SSL()
