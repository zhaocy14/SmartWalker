# !/usr/bin/env python
# -*- coding:utf-8 _*-
# @Author: swang
# @Contact: wang00sheng@gmail.com
# @Project Name: SmartWalker-master
# @File: ssl_loop.py
# @Time: 2022/01/02/20:34
# @Software: PyCharm

import os, sys

code_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.dirname(code_dir)
project_dir = os.path.dirname(module_dir)
sys.path.extend([module_dir, project_dir])
# print('project_dir:', project_dir)

import time
import json
import numpy as np
from scipy import stats
import threading
from multiprocessing import Process, Value, Pipe, Queue

# general lib
from SoundSourceLocalization.mylib import utils
from SoundSourceLocalization.mylib.utils import standard_normalizaion
from SoundSourceLocalization.mylib.audiolib import normalize_single_channel_audio, audio_segmenter_4_numpy, \
    audio_energy_ratio_over_threshold, audio_energy_over_threshold, audioread, audiowrite

# independent systems
from SoundSourceLocalization.SSL_Settings import *
import SoundSourceLocalization.SpeechEnhancement.code.ns_enhance_onnx as ns_enhance_onnx
from SoundSourceLocalization.SSL.code.ssl_audio_processor import *
from SoundSourceLocalization.SSL.code.ssl_feature_extractor import FeatureExtractor
from SoundSourceLocalization.SSL.code.ssl_DOA_model import DOA
from SoundSourceLocalization.SSL.code.ssl_turning import SSLturning
from SoundSourceLocalization.ReinforcementLearning.code.ssl_agent import Agent
from SoundSourceLocalization.ReinforcementLearning.code.ssl_env import MAP_ENV, ONLINE_MAP_ENV
# from SoundSourceLocalization.ReinforcementLearning.code.ssl_actor_critic import Actor, Critic
# # from Communication.Soundlocalization_socket_local import server_receive, server_transmit
# from Communication.Soundlocalization_socket import CLIENT
import Driver.ControlOdometryDriver as CD


class SSL(object):
    def __init__(self, doDenoise=True, useCD=True, seg_len='256ms', isDebug=False):
        print('-' * 20 + 'init SSL class' + '-' * 20)
        self.isDebug = isDebug
        self.doDrop = False
        self.doDenoise = doDenoise  # useless
        self.useCD = useCD
        self.frames = []
        segment_para_set = {
            '32ms' : {
                'name'     : '32ms',
                'time_len' : 32 / 1000,
                'threshold': 100,
                'stepsize' : 0.5
            },
            '50ms' : {
                'name'     : '50ms',
                'time_len' : 50 / 1000,
                'threshold': 100,
                'stepsize' : 0.5
            },
            '64ms' : {
                'name'     : '64ms',
                'time_len' : 64 / 1000,
                'threshold': 100,
                'stepsize' : 0.5
            },
            '128ms': {
                'name'     : '128ms',
                'time_len' : 128 / 1000,
                'threshold': 200,  # 100?
                'stepsize' : 0.5
            },
            '256ms': {
                'name'     : '256ms',
                'time_len' : 256. / 1000,
                'threshold': 400,
                'stepsize' : 256. / 1000 / 2
            },
            '1s'   : {
                'name'     : '1s',
                'time_len' : 1.,
                'threshold': 800,
                'stepsize' : 0.5
            },
        }
        self.seg_para = segment_para_set[seg_len]
        ref_audio, _ = audioread(os.path.abspath('./reference_wav.wav'))
        self.ref_audio = normalize_single_channel_audio(ref_audio)
        self.ref_audio_threshold = (self.ref_audio ** 2).sum() / len(self.ref_audio) / 500
        del ref_audio, self.ref_audio,
        # feature paras
        self.num_gcc_bin = 128
        self.num_mel_bin = 128
        self.fft_len = utils.next_greater_power_of_2(self.seg_para['time_len'] * SAMPLE_RATE)
        
        # SpeechEnhancement
        print('-' * 20, 'Loading denoising model...', '-' * 20, )
        self.denoise_model, _ = ns_enhance_onnx.load_onnx_model()
        
        # DOA
        print('-' * 20, 'Loading DOA model...', '-' * 20, )
        num_action = 8
        self.doa = DOA(model_dir=os.path.abspath('./model/EEGNet/ckpt'), fft_len=self.fft_len,
                       num_gcc_bin=self.num_gcc_bin, num_mel_bin=self.num_mel_bin, fs=SAMPLE_RATE, )
        # RL
        self.save_model_steps = 3
        self.env = ONLINE_MAP_ENV()
        self.save_ac_model = './model/ac_model'
        self.agent = Agent(alpha=1., num_action=num_action, gamma=0.99, ac_model_dir=self.save_ac_model,
                           load_ac_model=True, save_model_steps=self.save_model_steps)
        
        # Communication with Sensors
        # self.client = CLIENT()
        self.client = None
    
    def drop_audio_per_seg_point(self, signal_segment, ):
        '''
        return flag to determine whether a audio segment should be dropped or not, based on two standards:
        1. audio_energy_ratio
        2. audio_energy_over_threshold
        :param signal_segment:
        :return: boolean flag
        '''
        signal_mean = signal_segment.mean(axis=0)
        
        return not (audio_energy_over_threshold(signal_mean, threshold=self.ref_audio_threshold, ) and
                    audio_energy_ratio_over_threshold(signal_mean, fs=SAMPLE_RATE,
                                                      threshold=self.seg_para['threshold'], ))
    
    def save_continuous_True(self, ini_list, num=3):  # todo
        pass
    
    def drop_audio_clips(self, signal_segments, ):
        # print('Number of segments before dropping: ', len(signal_segments))
        audio_segments = []
        drop_flag = []
        for i in range(len(signal_segments)):
            drop_flag.append(self.drop_audio_per_seg_point(signal_segments[i]))
            if not drop_flag[-1]:
                audio_segments.append(signal_segments[i])
            else:
                continue
                # audio_segments.append([])
        # print('Number of segments after dropping: ', len(audio_segments))
        
        return np.array(audio_segments), drop_flag
    
    def normalize_batch_audio(self, audio_batch):
        '''
        For every audio in a batch, normalize its channels respectively.
        :param audio_batch:
        :return:
        '''
        res_audio = []
        for audio_channels in audio_batch:
            norm_audio_channels = []
            for audio in audio_channels:
                norm_audio_channels.append(normalize_single_channel_audio(audio))
            res_audio.append(norm_audio_channels)
        
        return np.asarray(res_audio)
    
    def denoise_batch_audio(self, audio_batch):
        '''
         For every audio in a batch, denoise its channels respectively.
        :param audio_batch:
        :return:
        '''
        res_audio = []
        for audio_channels in audio_batch:
            denoised_channels = []
            for audio in audio_channels:
                denoised_channels.append(
                    ns_enhance_onnx.denoise_nsnet2(audio=audio, fs=SAMPLE_RATE, model=self.denoise_model, ))
            res_audio.append(denoised_channels)
        
        return np.asarray(res_audio)
    
    def preprocess_ini_signal(self, ini_signals):
        '''
        preprocess the received signals. Must be consistent with the dataset preprocessing method.
        :param ini_signals:
        :return:
        '''
        ini_signals = np.array(ini_signals, dtype=np.float64)
        segs = np.array([audio_segmenter_4_numpy(signal, fs=SAMPLE_RATE, segment_len=self.seg_para['time_len'],
                                                 stepsize=self.seg_para['stepsize'], window='hann', padding=False,
                                                 pow_2=True) for signal in ini_signals]).transpose(1, 0, 2)
        # norm_segs = segs
        norm_segs = self.normalize_batch_audio(segs)
        
        denoised_norm_segs = self.denoise_batch_audio(audio_batch=norm_segs)
        
        if self.doDrop:
            drop_denoised_norm_segs, drop_flag = self.drop_audio_clips(signal_segments=denoised_norm_segs)
        else:
            drop_denoised_norm_segs, drop_flag = denoised_norm_segs, [False, ] * len(denoised_norm_segs)
        final_segments = self.normalize_batch_audio(drop_denoised_norm_segs)
        
        return final_segments, drop_flag
    
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
    
    def get_audio_from_pipe(self, ):
        ''' ??????????????????????????????????????????????????????????????????????????????0.5s
        ?????????????????????KWS??????????????????????????????????????????????????? [0.2, 0.5]s ??????
        :return audio
        '''
        res = []
        noData = True
        while noData:
            start_time = time.time()
            while self.RECV_PIPE.poll():
                msg = self.RECV_PIPE.recv()
                (audio, y, prob, send_time) = msg
                if abs(start_time - send_time) < KWS_TIMEOUT_SECONDS:
                    res.append(msg)
                    noData = False
        print('SSL: walker data is received~', )
        return res[-1][0]
    
    def run(self, RECV_PIPE, control, ):
        self.RECV_PIPE = RECV_PIPE
        # initialize models
        doa = self.doa
        num_step = 0
        
        # steps
        while True:
            # Detecting for walker_name
            ini_signals = self.get_audio_from_pipe()
            # preprocess initial audios
            audio_segments, drop_flag = self.preprocess_ini_signal(ini_signals)
            print('Number of preprocessed audio segments: ', len(audio_segments))
            
            if not (len(audio_segments) > 0):
                continue
            num_step += 1
            print('-' * 20, num_step, '-' * 20)
            
            # calculate features
            gcc_feature_batch = doa.extract_gcc_phat_4_batch(audio_segments)
            gcc_feature_batch = np.mean(gcc_feature_batch, axis=0)[np.newaxis, :]
            _, direction = doa.predict(gcc_feature_batch)
            print("Producing action ...\n", 'Direction', direction)
            
            ### ??????Owen??????????????????aim_loca
            if self.useCD:
                direction = direction[0] * 45
                
                SSLturning(control, direction)
                control.speed = STEP_SIZE / FORWARD_SECONDS
                control.radius = 0
                control.omega = 0
                time.sleep(FORWARD_SECONDS)
                control.speed = 0
                print("movement done.")
            else:
                pass
    
    def run_RL(self, RECV_PIPE, control, ):
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
            '''------------------------- ?????????????????? -----------------------------'''
            # ??????????????????
            if isDebug:
                # crt_position = input('please input current position and direction')
                crt_position = '280 160 2'
                crt_position = list(map(float, crt_position.split(' ')))
                crt_loca, crt_abs_doa = crt_position[:2], int(crt_position[2])
            else:
                crt_loca, crt_abs_doa = self.get_crt_position()
            print('crt_location: ', crt_loca, 'crt_abs_doa: ', crt_abs_doa)
            # ??????????????????
            crt_node = env.get_graph_node_idx(position=crt_loca)
            node_ = crt_node
            abs_availalbe_dircs = env.get_availalbe_dircs(node_idx=crt_node)  # ???????????????????????????????????????,??????????????????????????????????????????
            # print('availalbe_dircs: ', availalbe_dircs)
            abs_dirc_mask = np.array(np.array(abs_availalbe_dircs) != None)
            rela_dirc_mask = np.roll(abs_dirc_mask, shift=-crt_abs_doa)
            # print('rela_dirc_mask: ', rela_dirc_mask)
            dirc_digit = np.where(rela_dirc_mask)
            print("crt_node: ", crt_node, 'avaliable_rela_dirc_digit: ', list(dirc_digit))
            
            '''--------------------------- ???????????? -------------------------------'''
            # update state
            if not self.isDebug:
                gcc_feature_batch = doa.extract_gcc_phat_4_batch(audio_segments)
                gcc_feature = np.mean(gcc_feature_batch, axis=0)
                state_ = gcc_feature
            else:
                state_ = np.ones((1, 6, 128))
            ### ?????????????????? learn
            # ??????????????????mask?????????????????????
            action_ = agent.choose_action(state_, dirc_mask=rela_dirc_mask, sample=True)
            # _, direction_cate, = doa.predict(gcc_feature)
            # print(direction_prob)
            print('Predicted action_: ', action_)
            # print("Producing action ...\n", 'Direction', direction)
            aim_node = env.next_id_from_rela_action(crt_node, action=action_, abs_doa=crt_abs_doa)
            aim_loca = env.map.coordinates[aim_node]
            position = aim_loca
            print('aim_node: ', aim_node, 'aim_loca: ', aim_loca)
            
            ### ??????Owen??????????????????aim_loca
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
            
            # ?????? done TODO
            # ??????
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
    def __init__(self, doDenoise=True, useCD=True, seg_len='256ms', isDebug=False, ):
        super(SSL_Process, self).__init__()
        self.seg_len = seg_len
        self.doDenoise = doDenoise
        self.useCD = useCD
        self.isDebug = isDebug
    
    def run(self, RECV_PIPE, left_right):
        cd = CD.ControlDriver(left_right=left_right) if self.useCD else ''
        if self.useCD:
            cd_thread = threading.Thread(target=cd.control_part, args=())
            cd_thread.start()
        ssl = SSL(seg_len=self.seg_len, doDenoise=self.doDenoise, useCD=self.useCD, isDebug=self.isDebug, )
        ssl.run(RECV_PIPE, cd, )
