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
FFF_PATH = os.path.dirname(FF_PATH)
sys.path.extend([CRT_DIR, F_PATH, FF_PATH, FFF_PATH, ])
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
        self.num_action = 8
        
        # print('-' * 20, 'Loading DOA model...', '-' * 20, )
        # self.num_action = 8
        # doa_model_path = os.path.abspath(os.path.join(CRT_DIR, '../model/ResCNN/base_model'))
        # self.doa = DOA(model_dir=doa_model_path, num_gcc_bin=self.num_gcc_bin, num_mel_bin=self.num_mel_bin,
        #                fft_len=self.fft_len, fft_seg_len=self.fft_seg_len, fft_stepsize_ratio=self.fft_stepsize_ratio,
        #                fs=SAMPLE_RATE, loadModel=False)
        
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
        window -> norm -> denoise -> restore scales -> drop
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
    
    def receive_preprocess_audio(self, SSL_AUDIO_QUEUE, last_ssl_time):
        # ini_signals = self.get_audio_from_pipe(RECV_PIPE)
        (ini_signals, y, prob, send_time) = SSL_AUDIO_QUEUE.get(block=True, timeout=None)
        if prob <= 0.6:
            # print(f'walker will be dropped due to low confidence ({round(prob, 3)}).')
            return None, send_time
        if time.time() - send_time >= 1.:
            print('walker will be dropped due to high latency.')
            return None, send_time
        if (last_ssl_time is not None) and abs(send_time - last_ssl_time) < 2:
            print('walker will be dropped due to other adjacent walkers.')
            return None, send_time
        
        # preprocess initial audios (may be dropped)
        audio = self.preprocess_ini_signal(ini_signals)  # window
        return audio, send_time
    
    def run(self, walker_server, SSL_AUDIO_QUEUE, ):
        # DOA
        print('-' * 20, 'Loading DOA model...', '-' * 20, )
        doa_model_path = os.path.abspath(os.path.join(CRT_DIR, '../model/ResCNN/base_model_fullData_woBN'))
        self.doa = DOA(model_dir=doa_model_path, num_gcc_bin=self.num_gcc_bin, num_mel_bin=self.num_mel_bin,
                       fft_len=self.fft_len, fft_seg_len=self.fft_seg_len, fft_stepsize_ratio=self.fft_stepsize_ratio,
                       fs=SAMPLE_RATE, loadModel=True)
        
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
            
            # receive and preprocess initial audios (may be dropped)
            audio, send_time = self.receive_preprocess_audio(SSL_AUDIO_QUEUE, last_ssl_time)
            if audio is None:
                continue
            else:
                last_ssl_time = send_time
            
            # localize the source
            num_step += 1
            print('-' * 20, 'SSL step:', num_step, '-' * 20)
            direction, probs = self.doa.predict_sample(audio=audio, invalid_classes=None)
            print("Producing action ...\n", 'Direction', direction)
            print('Probs:', np.around(probs, 3))
            walker_server.send(data=direction, subtopic=SSL_NAV_COMMUNICATION_TOPIC)
    
    def run_D3QN(self, walker_server, SSL_AUDIO_QUEUE, ):
        def get_input():
            while True:
                with open('/home/swadmin/project/SmartWalker-master/SoundSourceLocalization/temp_input.txt', 'r+') as f:
                    lines = f.readlines()
                    if len(lines) > 0:
                        input_str = lines[0].strip('\n')
                        print(input_str)
                        f.truncate(0)
                        break
                time.sleep(0.5)
            return input_str
        
        from SoundSourceLocalization.SSL.code.SSL_RL.agent_d3qn import DQNAgent
        # DOA
        print('-' * 20, 'Loading DOA model...', '-' * 20, )
        doa = DOA(model_dir=None, num_gcc_bin=self.num_gcc_bin, num_mel_bin=self.num_mel_bin,
                  fft_len=self.fft_len, fft_seg_len=self.fft_seg_len, fft_stepsize_ratio=self.fft_stepsize_ratio,
                  fs=SAMPLE_RATE, loadModel=False)
        
        # initialize an agent
        D3QN_config = {
            'AGENT_CLASS'         : 'D3QN',
            'agent_learn'         : True,
            'print_interval'      : 1,
            'max_episode_steps'   : 30,  # 一个episode最多探索多少步，超过则强行终止。
            'num_update_episode'  : 1,  # update target model and reward graph & data
            'num_smooth_reward'   : 20,
            'num_save_episode'    : 1000,
            'num_plot_episode'    : 1,
            
            # -------------------------------- D3QN agent parameters ------------------------------------#
            'reward_discount_rate': 0.75,  # [0.8, 0.95]
            'lr'                  : 1e-4,  # [1e-5, 3e-4]
            'ddqn'                : True,
            'dueling'             : True,
            'softUpdate'          : True,
            'softUpdate_tau'      : 0.01,
            'learnTimes'          : 8,
            'usePER'              : False,
            'batch_size'          : 32,
            'memory_size'         : 1024,
            # 'episodes'            : 500,
            'eps_decay'           : False,
            'ini_eps'             : 1.0,
            'min_eps'             : 0.10,
            'eps_decay_rate'      : 0.999,
            'base_model_dir'      : \
                os.path.abspath(os.path.join(os.path.dirname(__file__), '../model/ResCNN/base_model_fullData_woBN')),
            'd3qn_model_dir'      : \
                os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../model/ResCNN/d3qn_model')),
            'load_d3qn_model'     : True,
            'based_on_base_model' : True,
            'd3qn_model_name'     : 'Dueling_DDQN_softUpdate__lr_0.0001_20220212-235802',
        }
        agent = DQNAgent(num_action=self.num_action, **D3QN_config)
        
        # ----------------------------------- running RL ----------------------------------------- #
        # initialize parameters
        # Event_Wait = False  # control the running state of SSL # TODO: for debugging
        total_step = 0
        episode_idx = 0
        last_ssl_time = None  # be used for dropping expired walker audio
        while True:
            # start an episode  # TODO: for debugging
            # temp_wait = walker_server.recv(subtopic=SSL_WAIT_COMMUNICATION_TOPIC, )
            # if temp_wait is not None:
            #     Event_Wait = temp_wait
            #     self.clear_Queue(Queue=SSL_AUDIO_QUEUE,
            #                      description='SSL_AUDIO_QUEUE is cleared due to the change of SSL state.')
            # if Event_Wait:
            #     time.sleep(0.1)
            #     continue
            print('-' * 20, 'will start a new episode', '-' * 20, )
            
            # run an episode
            num_step = 0
            # state, action, reward, state_, done = None, None, None, None, None
            states, actions, rewards, states_, dones = [], [], [], [], []
            
            self.clear_Queue(Queue=SSL_AUDIO_QUEUE,
                             description='SSL_AUDIO_QUEUE is cleared due to the start of a new episode.')
            while True:
                # determine if this episode ends # TODO: for debugging
                # temp_wait = walker_server.recv(subtopic=SSL_WAIT_COMMUNICATION_TOPIC, )
                # if temp_wait is not None:
                #     Event_Wait = temp_wait
                #     self.clear_Queue(Queue=SSL_AUDIO_QUEUE,
                #                      description='SSL_AUDIO_QUEUE is cleared due to the change of SSL state.')
                # if Event_Wait:
                #     break
                
                # receive state
                audio, send_time = self.receive_preprocess_audio(SSL_AUDIO_QUEUE, last_ssl_time)
                if audio is None:
                    continue
                else:
                    last_ssl_time = send_time
                
                # act
                num_step += 1
                total_step += 1
                print('-' * 20, 'SSL step(episode/total):', num_step, '/', total_step, '-' * 20)
                stft_feature = doa.get_stft_feature(audio=audio)
                direction, act_info = agent.act(stft_feature, decay_step=total_step)
                direction = int(direction)
                print('Producing action ...\n', 'Direction:', direction, 'act_info:', act_info)
                
                # process last step's experience
                if len(states) == 0:
                    states.append(audio)
                else:
                    states_.append(states[-1])
                    states.append(audio)
                actions.append(direction)
                dones.append(False)
                rewards.append(-0.05)
                
                print('Send this direction?', end='\t')
                if get_input() == 'y':  # TODO: for debugging
                    walker_server.send(data=direction, subtopic=SSL_NAV_COMMUNICATION_TOPIC)
                print('End this eposoide?', end='\t')
                if get_input() == 'y':
                    self.clear_Queue(Queue=SSL_AUDIO_QUEUE,
                                     description='SSL_AUDIO_QUEUE is cleared due to the change of SSL state.')
                    break
            
            # end an episode # if end, improve based on the rules
            if len(states) > 0:
                states_.append(None)
                print('Done?', end='\t')
                if get_input() == 'y':  # TODO: for debugging
                    dones[-1] = True
                    rewards[-1] = 1.
                else:
                    dones[-1] = False
                    rewards[-1] = -0.05
                experience_ls = [list(i) for i in zip(states, actions, rewards, states_, dones)]
                agent.remember_batch(batch_experience=experience_ls, useDiscount=True, feature_extractor=doa)
                episode_idx += 1
                print('episode_idx:', episode_idx, '\t', 'steps:', num_step, '\t', 'done:', dones[-1], '\t', )
            
            if D3QN_config['agent_learn']:
                agent.learn()
            if (episode_idx + 1) % D3QN_config['num_update_episode'] == 0:
                agent.update_target_model()
                # self.plot_and_save_rewards(episode_rewards)
            print('Saving model...')
            agent.save_model(model_dir='./temp_d3qn')
    
    def run_SAC(self, walker_server, SSL_AUDIO_QUEUE, ):
        def get_input():
            while True:
                with open('/home/swadmin/project/SmartWalker-master/SoundSourceLocalization/temp_input.txt', 'r+') as f:
                    lines = f.readlines()
                    if len(lines) > 0:
                        input_str = lines[0].strip('\n')
                        print(input_str)
                        f.truncate(0)
                        break
                time.sleep(0.5)
            return input_str
        
        from SoundSourceLocalization.SSL.code.SSL_RL.agent_sac import SACAgent
        # DOA
        print('-' * 20, 'Loading DOA model...', '-' * 20, )
        doa = DOA(model_dir=None, num_gcc_bin=self.num_gcc_bin, num_mel_bin=self.num_mel_bin,
                  fft_len=self.fft_len, fft_seg_len=self.fft_seg_len, fft_stepsize_ratio=self.fft_stepsize_ratio,
                  fs=SAMPLE_RATE, loadModel=False)
        
        # initialize an agent
        SAC_config = {
            'AGENT_CLASS'         : 'SAC',
            'agent_learn'         : True,
            
            'print_interval'      : 1,
            'num_smooth_reward'   : 20,
            'num_save_episode'    : 1000,
            'num_plot_episode'    : 1,
            
            # -------------------------------- D3QN agent parameters ------------------------------------#
            'num_Q'               : 2,
            'reward_scale'        : 1.,
            'reward_discount_rate': 0.75,
            
            'policy_lr'           : 3e-4,
            'Q_lr'                : 3e-4,
            'alpha_lr'            : 3e-4,
            
            'learnTimes'          : 8,
            'batch_size'          : 32,
            'memory_size'         : 1024,
            
            'softUpdate'          : True,
            'num_update_episode'  : 1,
            'softUpdate_tau'      : 0.01,
            
            'load_sac_model'      : True,
            'based_on_base_model' : True,
            'base_model_dir'      : \
                os.path.abspath(os.path.join(os.path.dirname(__file__), '../model/ResCNN/base_model_fullData_woBN')),
            'sac_model_dir'       : \
                os.path.abspath(os.path.join(os.path.dirname(__file__), '../model/ResCNN/sac_model')),
            'sac_model_name'      : '20220214-005038_SAC_lr-p-0.0003-Q-0.0003-a-0.0003_up-1-8-tau-0.01_mom-1024-32_rwd-1.0-0.75',
        }
        agent = SACAgent(num_action=self.num_action, **SAC_config)
        
        # ----------------------------------- running RL ----------------------------------------- #
        # initialize parameters
        # Event_Wait = False  # control the running state of SSL # TODO: for debugging
        total_step = 0
        episode_idx = 0
        last_ssl_time = None  # be used for dropping expired walker audio
        while True:
            # start an episode  # TODO: for debugging
            # temp_wait = walker_server.recv(subtopic=SSL_WAIT_COMMUNICATION_TOPIC, )
            # if temp_wait is not None:
            #     Event_Wait = temp_wait
            #     self.clear_Queue(Queue=SSL_AUDIO_QUEUE,
            #                      description='SSL_AUDIO_QUEUE is cleared due to the change of SSL state.')
            # if Event_Wait:
            #     time.sleep(0.1)
            #     continue
            print('-' * 20, 'will start a new episode', '-' * 20, )
            
            # run an episode
            num_step = 0
            # state, action, reward, state_, done = None, None, None, None, None
            states, actions, rewards, states_, dones = [], [], [], [], []
            
            self.clear_Queue(Queue=SSL_AUDIO_QUEUE,
                             description='SSL_AUDIO_QUEUE is cleared due to the start of a new episode.')
            while True:
                # determine if this episode ends # TODO: for debugging
                # temp_wait = walker_server.recv(subtopic=SSL_WAIT_COMMUNICATION_TOPIC, )
                # if temp_wait is not None:
                #     Event_Wait = temp_wait
                #     self.clear_Queue(Queue=SSL_AUDIO_QUEUE,
                #                      description='SSL_AUDIO_QUEUE is cleared due to the change of SSL state.')
                # if Event_Wait:
                #     break
                
                # receive state
                audio, send_time = self.receive_preprocess_audio(SSL_AUDIO_QUEUE, last_ssl_time)
                if audio is None:
                    continue
                else:
                    last_ssl_time = send_time
                
                # act
                num_step += 1
                total_step += 1
                print('-' * 20, 'SSL step(episode/total):', num_step, '/', total_step, '-' * 20)
                stft_feature = doa.get_stft_feature(audio=audio)
                direction, act_info = agent.act(stft_feature, decay_step=total_step)
                direction = int(direction)
                print('Producing action ...\n', 'Direction:', direction, 'act_info:', act_info)
                
                # process last step's experience
                if len(states) == 0:
                    states.append(audio)
                else:
                    states_.append(states[-1])
                    states.append(audio)
                actions.append(direction)
                dones.append(False)
                rewards.append(-0.05)
                
                print('Send this direction?', end='\t')
                if get_input() == 'y':  # TODO: for debugging
                    walker_server.send(data=direction, subtopic=SSL_NAV_COMMUNICATION_TOPIC)
                print('End this eposoide?', end='\t')
                if get_input() == 'y':
                    self.clear_Queue(Queue=SSL_AUDIO_QUEUE,
                                     description='SSL_AUDIO_QUEUE is cleared due to the change of SSL state.')
                    break
            
            # end an episode # if end, improve based on the rules
            if len(states) > 0:
                states_.append(None)
                print('Done?', end='\t')
                if get_input() == 'y':  # TODO: for debugging
                    dones[-1] = True
                    rewards[-1] = 1.
                else:
                    dones[-1] = False
                    rewards[-1] = -0.05
                experience_ls = [list(i) for i in zip(states, actions, rewards, states_, dones)]
                agent.remember_batch(batch_experience=experience_ls, useDiscount=True, feature_extractor=doa)
                episode_idx += 1
                print('episode_idx:', episode_idx, '\t', 'steps:', num_step, '\t', 'done:', dones[-1], '\t', )
            
            if SAC_config['agent_learn']:
                agent.learn()
            if (episode_idx + 1) % SAC_config['num_update_episode'] == 0:
                agent.update_target_model()
                # self.plot_and_save_rewards(episode_rewards)
            print('Saving model...')
            agent.save_model(model_dir='./temp_sac')


class SSL_Process(object):
    def __init__(self, doDenoise=True, seg_len='1s', isDebug=False, ):
        super(SSL_Process, self).__init__()
        self.seg_len = seg_len
        self.doDenoise = doDenoise
        self.isDebug = isDebug
    
    def run(self, walker_server, SSL_AUDIO_QUEUE, ):
        ssl = SSL(seg_len=self.seg_len, doDenoise=self.doDenoise, isDebug=self.isDebug, )
        # ssl.run(walker_server, SSL_AUDIO_QUEUE, )
        ssl.run_D3QN(walker_server, SSL_AUDIO_QUEUE, )
        # ssl.run_SAC(walker_server, SSL_AUDIO_QUEUE, )


if __name__ == '__main__':
    ssl = SSL()
