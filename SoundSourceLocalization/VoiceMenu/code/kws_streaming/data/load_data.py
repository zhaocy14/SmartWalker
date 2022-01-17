# !/usr/bin/env python
# -*- coding:utf-8 _*-
# @Author: Forence
# @Contact: wang00sheng@gmail.com
# @Project Name: keyword_spotting_system
# @File: load_data.py
# @Time: 2021/12/19/19:12
# @Software: PyCharm
import os
import sys
import time
import glob
import math
import shutil
import random
import warnings
import numpy as np
from pathlib import Path
from copy import deepcopy
import scipy
import scipy.signal as signal
import librosa
from lib import audiolib
from threading import Thread

EPS = np.finfo(np.float32).eps


class AudioProcessor(object):
    def __init__(self, flags):
        self.ds_root_dir = flags.data_dir
        self.kwd_dir = os.path.join(self.ds_root_dir, 'self_collected_KWS_keyword_based')
        if flags.use_google_command_as_other_words:
            self.unknown_wd_dir = os.path.join(self.ds_root_dir, 'speech_commands_v0.02')
        else:
            self.unknown_wd_dir = os.path.join(self.ds_root_dir, 'clean')
        self.noise_dir = os.path.join(self.ds_root_dir, 'noise')
        self.rir_dir = os.path.join(self.ds_root_dir, 'real_rir')
        
        self.sample_rate = flags.sample_rate
        self.kwd_ls = flags.wanted_words.split(',')
        self.silence_name = 'silence'
        self.silence_label = 0
        self.unknown_name = 'unknown'
        self.unknown_label = 1
        self.word_ls = [self.silence_name, self.unknown_name, ] + self.kwd_ls
        self.word_2_label = dict(zip(self.word_ls, list(range(len(self.word_ls)))))
        self.clip_samples = flags.clip_duration_ms * self.sample_rate // 1000
        self.unknown_ratio = flags.unknown_percentage
        self.silence_ratio = flags.silence_percentage
        self.word_prob = [self.silence_ratio, self.unknown_ratio, 1. - self.silence_ratio - self.unknown_ratio]
        
        # audio augmentation
        self.speed_interval = [0.85, 1.15]
        shift_samples = flags.time_shift_ms * self.sample_rate // 1000
        self.shift_interval = [-shift_samples, shift_samples]
        self.noise_ratio = 1.
        self.rir_ratio = 1.
        # self.rir_t60_interval = rir_t60_interval
        self.max_clip_tries = 20
        # self.clean_activity_threshold = 0.6
        # self.noise_activity_threshold = 0.
        self.noise_snr_interval = [-5, 10]
        self.noisy_level_interval = [-35, -15]
        self.active_percentage = 0.2
        
        # load various paths
        self.kwd_files = self.load_kwds_files(self.kwd_dir)
        self.unknown_wd_files = self.load_unknown_wds_files(self.unknown_wd_dir,
                                                            flags.use_google_command_as_other_words)
        self.rir_files = self.load_rir_files(self.rir_dir)
        self.noise_files = self.load_noise_files(self.noise_dir)
        # self.noise_files = self.rir_files[:10]
        self.silence_data = (self.silence_label, np.zeros((self.clip_samples,)))
    
    # rir_files = load_csv_columns(rir_path, ['wavfile', 'channel', 'T60_WB', 'isRealRIR'], header=0, index_col=0)
    # self.rir_files = self.clean_rir_files(list(zip(*rir_files)))
    #
    # def clean_rir_files(self, rir_files):
    #     # ['wavfile', 'channel', 'T60_WB', 'isRealRIR']
    #     rir_choice = self.rir_choice
    #     lower_t60, upper_t60 = self.rir_t60_interval
    #     lower_channel, upper_chanel = self.rir_channel_interval
    #
    #     if rir_choice == 1:  # real
    #         rir_files = [i for i in rir_files if (i[3] == 1)]
    #     elif rir_choice == 2:  # synthetic
    #         rir_files = [i for i in rir_files if (i[3] == 0)]
    #     elif rir_choice == 3:  # both real and synthetic
    #         rir_files = rir_files
    #     else:
    #         raise ValueError('-' * 20 + 'rir_choice must be 1, 2 or 3' + '-' * 20)
    #
    #     rir_path = []
    #     for i in rir_files:
    #         if (lower_t60 is not None) and (float(i[2]) < lower_t60):
    #             continue
    #         if (upper_t60 is not None) and (float(i[2]) > upper_t60):
    #             continue
    #         if (lower_channel is not None) and (i[1] < lower_channel):
    #             continue
    #         if (upper_chanel is not None) and (i[1] > upper_chanel):
    #             continue
    #         rir_path.append(i)
    #     rir_path = [os.path.join('impulse_responses', i[0]) for i in rir_path]
    #     return rir_path
    def load_kwds_files(self, root_dir):
        res = []
        counter = [0 for _ in range(len(self.kwd_ls))]
        for i, kwd in enumerate(self.kwd_ls):
            path = os.path.join(root_dir, kwd)
            for file in Path(path).rglob('*.wav'):
                counter[i] += 1
                res.append((kwd, file))
        print('-' * 20, 'Sample number of every keyword', '-' * 20)
        print_ls = list(zip(*[self.kwd_ls, counter]))
        [print(kwd + ':', num) for kwd, num in print_ls]
        return res
    
    def load_noise_files(self, root_dir):
        files = list(Path(root_dir).rglob('*.wav'))
        return files
    
    def load_rir_files(self, root_dir):
        files = list(Path(root_dir).rglob('*.wav'))
        return files
    
    def load_unknown_wds_files(self, root_dir, use_google_command=True):
        if use_google_command:
            files = []
            unknown_wds = next(os.walk(root_dir))[1]
            for wd in unknown_wds:
                if wd not in (['_background_noise_'] + self.kwd_ls):
                    wd_dir = os.path.join(root_dir, wd)
                    files.extend(list(Path(wd_dir).rglob('*.wav')))
                    if len(files) > 100:
                        break
        else:
            files = list(Path(root_dir).rglob('*.wav'))
        
        files = [(self.unknown_name, i) for i in files]
        return files
    
    def load_audio(self, path, first_channel=True, ):
        # audio, sr = sf.read(path, )
        audio, sr = librosa.load(path, sr=None, mono=False, )
        audio = audio[0] if (first_channel and audio.ndim > 1) else audio
        audio = audio if sr == self.sample_rate else librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        return audio
    
    def normalize_audio(self, audio, target_level=-25):
        '''Normalize the signal to the target level'''
        audio = np.array(audio)
        rms = (audio ** 2).mean() ** 0.5
        scalar = 10 ** (target_level / 20) / (rms + EPS)
        audio = audio * scalar
        return audio
    
    def band_filtering(self, audio, low_freq=50, high_freq=5000, sample_rate=16000, N=4):
        b, a = scipy.signal.butter(N=N, Wn=[low_freq, high_freq], btype='bandpass', analog=False, output='ba',
                                   fs=sample_rate, )
        return scipy.signal.filtfilt(b, a, audio)  # data为要过滤的信号
    
    def audio_energy_ratio_over_threshold(self, audio, threshold=200, fs=16000, norm=True):
        # noisy_audio = np.array(audio)
        # clean_audio = self.band_filtering(noisy_audio)  # band filtering
        #
        # clean_rms2 = (clean_audio ** 2).mean()
        # noisy_rms2 = (noisy_audio ** 2).mean()
        # noise_rms2 = noisy_rms2 - clean_rms2
        # print(clean_rms2 / noise_rms2)
        audio = self.normalize_audio(audio) if norm else audio
        rfft_amp = np.abs(rfft(audio))
        audio_energy = rfft_amp ** 2
        l_idx, h_idx = int(50 * len(audio) / fs + 1), int(500 * len(audio) / fs)
        energy_ratio = audio_energy[l_idx:h_idx].sum() / (audio_energy[1:l_idx].sum() + EPS)
        
        # print('energy_ratio', energy_ratio)
        if energy_ratio > threshold:
            return True
        else:
            return False
    
    def activity_detector(self, audio, energy_thresh=0.13, target_level=-25):
        '''Return the percentage of the time the audio signal is above an energy threshold'''
        audio = np.array(audio)
        audio = self.normalize_audio(audio, target_level)
        
        if not self.audio_energy_ratio_over_threshold(audio, norm=False):
            return 0.
        # else:
        #     return 1.
        
        audio_len = len(audio)
        window_size = 32  # in ms
        window_samples = int(self.sample_rate * window_size / 1000)
        sample_start = 0
        num_total = 0
        num_active_frame = 0
        prev_energy_prob = 0
        
        a = -1
        b = 0.2
        alpha_rel = 0.05
        alpha_att = 0.8
        
        while sample_start < audio_len:
            sample_end = min(sample_start + window_samples, audio_len)
            audio_win = audio[sample_start:sample_end]
            frame_rms = 20 * np.log10((audio_win ** 2).sum() + EPS)
            frame_energy_prob = 1. / (1 + np.exp(-(a + b * frame_rms)))
            
            if frame_energy_prob > prev_energy_prob:
                smoothed_energy_prob = frame_energy_prob * alpha_att + prev_energy_prob * (1 - alpha_att)
            else:
                smoothed_energy_prob = frame_energy_prob * alpha_rel + prev_energy_prob * (1 - alpha_rel)
            
            if smoothed_energy_prob > energy_thresh:
                num_active_frame += 1
            prev_energy_prob = frame_energy_prob
            sample_start += window_samples
            num_total += 1
        
        return num_active_frame / num_total
    
    def clip_pad_audio(self, audio, random=False):
        '''clip or pad audio'''
        audio = np.asarray(audio)
        len_audio = len(audio)
        if len_audio == self.clip_samples:
            return audio
        elif len_audio > self.clip_samples:
            if random:
                low_idx = np.random.randint(0, len_audio - self.clip_samples)
            else:
                low_idx = (len_audio - self.clip_samples) // 2
            return audio[low_idx:low_idx + self.clip_samples]
        else:
            if random:
                low_idx = np.random.randint(0, self.clip_samples - len_audio)
            else:
                low_idx = (self.clip_samples - len_audio) // 2
            aug_audio = np.zeros((self.clip_samples,))
            aug_audio[low_idx:low_idx + len_audio] = audio
            return aug_audio
    
    def sample_noise_audio(self):
        # 随机确定是否引入噪声，并加载
        if random.uniform(0, 1) < self.noise_ratio:  # 引入噪声
            noise_path = random.sample(self.noise_files, k=1)[0]  # 随机选择一个audio
            noise = self.load_audio(noise_path, first_channel=True)
            clip = self.clip_pad_audio(noise, random=True)
            return clip
        else:
            return None
    
    def sample_rir_audio(self):
        # 随机确定是否引入混响，并加载
        if random.uniform(0, 1) < self.rir_ratio:  # 引入混响
            rir_path = random.sample(self.rir_files, 1)[0]  # 随机选择一个audio
            rir = self.load_audio(rir_path, first_channel=True)
            return rir
        else:
            return None
    
    def add_reverberation(self, clean, rir):
        reverb_speech = signal.fftconvolve(clean, rir, mode="full")
        return reverb_speech[: clean.shape[0]]  # make reverb_speech same length as clean_speech
    
    def snr_audio_generator(self, clean, noise, snr, target_level=-25, useActive=False, active_percentage=0.2):
        '''Function to mix clean speech and noise at various segmental SNR levels'''
        clean, noise = np.array(clean), np.array(noise)
        noise = noise[:len(clean)] if (len(noise) >= len(clean)) \
            else np.append(noise, np.zeros(len(clean) - len(noise)))
        clean_clip_scalar = 1. / (max(abs(clean)) + EPS)
        noise_clip_scalar = 1. / (max(abs(noise)) + EPS)
        clean = clean * clean_clip_scalar
        noise = noise * noise_clip_scalar
        # clean_rms, noise_rms = audiolib.active_rms(clean=clean, noise=noise, window_size=32,
        #                                            energy_thresh=-np.inf)  # 64
        # print('noise_rms:', noise_rms)
        if useActive:
            clean_rms, noise_rms = audiolib.active_rms_percentage(clean, noise, fs=self.sample_rate, window_size=64,
                                                                  active_percentage=active_percentage, )
        else:
            clean_rms, noise_rms = (clean ** 2).mean() ** 0.5, (noise ** 2).mean() ** 0.5
        if np.allclose(noise_rms, 0.):
            clean_norm, clean_norm_scalar = clean, 1.
            noise_norm, noise_norm_scalar = noise, 1.
            # Set the noise level for a given SNR
            noise_snr_scalar = 1.
        else:
            clean_norm, clean_norm_scalar = audiolib.normalize_single_channel_audio(
                clean, rms=clean_rms, target_level=target_level, returnScalar=True)
            noise_norm, noise_norm_scalar = audiolib.normalize_single_channel_audio(
                noise, rms=noise_rms, target_level=target_level, returnScalar=True)
            # Set the noise level for a given SNR
            noise_snr_scalar = clean_rms / (10 ** (snr / 20)) / (noise_rms + EPS)
        noise_snr = noise_norm * noise_snr_scalar
        
        return clean_norm, clean_clip_scalar * clean_norm_scalar, \
               noise_snr, noise_clip_scalar * noise_norm_scalar * noise_snr_scalar
    
    def random_adjust_audio_level(self, audio, level_interval, clipping_threshold=0.99):
        # Randomly select RMS value in level_interval (dB) and normalize audio with that value
        # There is a chance of clipping that might happen with very less probability, which is not a major issue.
        rms_level = random.uniform(*level_interval)
        audio_norm, audio_scalar = audiolib.normalize_single_channel_audio(
            audio, target_level=rms_level, returnScalar=True)
        
        # Final check to see if there are any amplitudes exceeding +/- clipping_threshold.
        # If so, normalize all the signals accordingly
        if audiolib.is_clipped(audio_norm, clipping_threshold=clipping_threshold):
            audio_scalar *= (clipping_threshold - EPS) / max(abs(audio_norm))
        
        return audio * audio_scalar, audio_scalar
    
    def multi_audios_mixer(self, ref_audio, audio_ls, snr_interval, isNoise=False):
        scalar_ls = []
        mix = np.array(ref_audio)
        for audio in audio_ls:
            snr = random.uniform(*snr_interval)
            _, ref_scalar, _, audio_scalar = self.snr_audio_generator(np.array(ref_audio), np.array(audio), snr,
                                                                      useActive=isNoise)
            scalar_ls.append(audio_scalar / ref_scalar)
            mix += scalar_ls[-1] * audio
        
        return mix, scalar_ls
        # 对混合语音进行响度变换
    
    def shift_audio(self, audio):
        shift_samples = random.randint(*self.shift_interval)
        audio = np.roll(audio, shift=shift_samples)
        if shift_samples > 0:
            audio[:shift_samples] = 0
        elif shift_samples < 0:
            audio[shift_samples:] = 0
        return audio
    
    def __gen_item__(self, kwd, kwd_path, res, random_clip=False, ):
        clean = self.load_audio(kwd_path, first_channel=True, )
        noise = self.sample_noise_audio()
        rir = self.sample_rir_audio()
        
        # 综合所有素材，生成增强语音
        # norm -> 重采样 -> 平移 —> 变换回原来的长度 ->  噪音（音量调整） -> 整体响度变换 -> 输出未提取特征的音频
        # norm
        clean = self.normalize_audio(clean, target_level=-25)
        # 重采样
        speed = random.uniform(*self.speed_interval)
        clean = librosa.resample(clean, orig_sr=int(self.sample_rate * speed), target_sr=self.sample_rate, )
        # 平移
        clean = self.shift_audio(clean, )
        # 变换回原来的长度
        clean = self.clip_pad_audio(clean, random=random_clip)
        # 噪音（随机snr）
        if noise is not None:
            snr = random.uniform(*self.noise_snr_interval)
            clean_snr, _, noise_snr, _ = self.snr_audio_generator(clean, noise, snr, target_level=-25, useActive=True,
                                                                  active_percentage=self.active_percentage)
            audio = clean_snr + noise_snr
        else:
            audio = clean
        # 添加混响
        if rir is not None:
            audio = self.add_reverberation(audio, rir)
        # 整体响度变换
        audio, _ = self.random_adjust_audio_level(audio, self.noisy_level_interval)
        
        res.append((self.word_2_label[kwd], audio))
        return self.word_2_label[kwd], audio
    
    def get_batch_data(self, batch_size, **kwargs):
        sample = np.random.choice([0, 1, 2], size=batch_size, p=self.word_prob, replace=True, )
        num_silence, num_unknown, num_kwd = (sample == 0).sum(), (sample == 1).sum(), (sample == 2).sum()
        
        # sample kws_ls/unknown_ls/silence_ls
        if num_kwd > len(self.kwd_files):
            kws_ls = np.random.choice(np.arange(len(self.kwd_files)), size=num_kwd, replace=True)
            kws_ls = [self.kwd_files[i] for i in kws_ls]
        else:
            kws_ls = random.sample(self.kwd_files, k=num_kwd, )
        
        if num_unknown > len(self.unknown_wd_files):
            unknown_ls = np.random.choice(np.arange(len(self.unknown_wd_files)), size=num_unknown, replace=True)
            unknown_ls = [self.unknown_wd_files[i] for i in unknown_ls]
        else:
            unknown_ls = random.sample(self.unknown_wd_files, k=num_unknown, )
        
        silence_ls = [self.silence_data] * num_silence
        
        # batch_data = []
        # aug_ls = kws_ls + unknown_ls
        # for aug_data in aug_ls:
        #     data = self.__gen_item__(*aug_data, )
        #     batch_data.append(data)
        
        batch_data = []
        threads = []
        for aug_data in kws_ls:
            threads.append(Thread(target=self.__gen_item__, args=(*aug_data, batch_data, False)))
            threads[-1].start()
        for thread in threads:
            thread.join()
        
        threads = []
        for aug_data in unknown_ls:
            threads.append(Thread(target=self.__gen_item__, args=(*aug_data, batch_data, True)))
            threads[-1].start()
        for thread in threads:
            thread.join()
        
        batch_data.extend(silence_ls)
        random.shuffle(batch_data)
        y, x = list(zip(*batch_data))
        return np.asarray(x), np.asarray(y)
    
    def get_batch_data_single_thread(self, batch_size, **kwargs):
        sample = np.random.choice([0, 1, 2], size=batch_size, p=self.word_prob, replace=True, )
        num_silence, num_unknown, num_kwd = (sample == 0).sum(), (sample == 1).sum(), (sample == 2).sum()
        
        # sample kws_ls/unknown_ls/silence_ls
        if num_kwd > len(self.kwd_files):
            kws_ls = np.random.choice(np.arange(len(self.kwd_files)), size=num_kwd, replace=True)
            kws_ls = [self.kwd_files[i] for i in kws_ls]
        else:
            kws_ls = random.sample(self.kwd_files, k=num_kwd, )
        
        if num_unknown > len(self.unknown_wd_files):
            unknown_ls = np.random.choice(np.arange(len(self.unknown_wd_files)), size=num_unknown, replace=True)
            unknown_ls = [self.unknown_wd_files[i] for i in unknown_ls]
        else:
            unknown_ls = random.sample(self.unknown_wd_files, k=num_unknown, )
        
        silence_ls = [self.silence_data] * num_silence
        
        batch_data = []
        aug_ls = kws_ls + unknown_ls
        for aug_data in aug_ls:
            data = self.__gen_item__(*aug_data, res=[])
            batch_data.append(data)
        
        # threads = []
        # for aug_data in aug_ls:
        #     threads.append(Thread(target=self.__gen_item__, args=(*aug_data, batch_data)))
        #     threads[-1].start()
        # for thread in threads:
        #     thread.join()
        
        batch_data.extend(silence_ls)
        random.shuffle(batch_data)
        y, x = list(zip(*batch_data))
        return np.asarray(x), np.asarray(y)
    
    def set_size(self, mode):
        if mode == 'validation':
            return 1024
        elif mode == 'testing':
            return 2046
        elif mode == 'training':
            return 2046


if __name__ == '__main__':
    from kws_streaming.train import base_parser
    from kws_streaming.models import model_flags
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    parser = base_parser.base_parser()  # load params for data and features
    FLAGS, unparsed = parser.parse_known_args()  # parse params
    flags = model_flags.update_flags(FLAGS)
    
    audio_processor = AudioProcessor(flags)
    
    audio_processor.get_batch_data(batch_size=32)
    
    # # 速度测试
    # start = time.time()
    # [audio_processor.get_batch_data(batch_size=128) for _ in range(10)]
    # multi_time = time.time() - start
    # start = time.time()
    # [audio_processor.get_batch_data_single_thread(batch_size=128) for _ in range(10)]
    # single_time = time.time() - start
    # print('Ratio:', multi_time / single_time)
    
    print('Hello World!')
