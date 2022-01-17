# -*- coding: utf-8 -*-
"""
@author: chkarada
"""
import os
import numpy as np
from numpy.fft import fft, ifft, rfft
import soundfile as sf
import subprocess
import glob
import librosa
import random
import tempfile
from .utils import plot_curve, plot_hist, otsu_threshold
import shutil

EPS = np.finfo(float).eps
REF_POWER = 1e-12
np.random.seed(0)



def is_clipped(audio, clipping_threshold=0.99):
    return any(abs(audio) > clipping_threshold)


def normalize_single_channel_to_target_level(audio, target_level=-25):
    '''Normalize the signal to the target level'''
    audio = np.array(audio, dtype=np.float32)
    rms = (audio ** 2).mean() ** 0.5
    scalar = 10 ** (target_level / 20) / (rms + EPS)
    audio = audio * scalar
    return audio


def normalize_segmental_rms(audio, rms, target_level=-25):
    '''Normalize the signal to the target level
    based on segmental RMS'''
    scalar = 10 ** (target_level / 20) / (rms + EPS)
    audio = audio * scalar
    return audio


def audioread(path, norm=False, start=0, stop=None, target_level=-25):
    '''Function to read audio'''
    
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise ValueError("[{}] does not exist!".format(path))
    try:
        audio, sample_rate = sf.read(path, start=start, stop=stop)
    except RuntimeError:  # fix for sph pcm-embedded shortened v2
        print('WARNING: Audio type not supported')
    
    if len(audio.shape) == 1:  # mono
        if norm:
            rms = (audio ** 2).mean() ** 0.5
            scalar = 10 ** (target_level / 20) / (rms + EPS)
            audio = audio * scalar
    else:  # multi-channel   average all the channels
        audio = audio.T
        audio = audio.sum(axis=0) / audio.shape[0]
        if norm:
            audio = normalize_single_channel_to_target_level(audio, target_level)
    
    return audio, sample_rate


def audiowrite(destpath, audio, sample_rate=16000, norm=False, target_level=-25, clipping_threshold=0.99,
               clip_test=False):
    '''Function to write audio'''
    
    if clip_test:
        if is_clipped(audio, clipping_threshold=clipping_threshold):
            raise ValueError("Clipping detected in audiowrite()! " + \
                             destpath + " file not written to disk.")
    
    if norm:
        audio = normalize_single_channel_to_target_level(audio, target_level)
        max_amp = max(abs(audio))
        if max_amp >= clipping_threshold:
            audio = audio / max_amp * (clipping_threshold - EPS)
    
    destpath = os.path.abspath(destpath)
    destdir = os.path.dirname(destpath)
    
    if not os.path.exists(destdir):
        os.makedirs(destdir)
    
    sf.write(destpath, audio, sample_rate)
    return


def add_reverb(sasxExe, input_wav, filter_file, output_wav):
    ''' Function to add reverb'''
    command_sasx_apply_reverb = "{0} -r {1} \
        -f {2} -o {3}".format(sasxExe, input_wav, filter_file, output_wav)
    
    subprocess.call(command_sasx_apply_reverb)
    return output_wav


def add_clipping(audio, max_thresh_perc=0.8):
    '''Function to add clipping'''
    threshold = max(abs(audio)) * max_thresh_perc
    audioclipped = np.clip(audio, -threshold, threshold)
    return audioclipped


def adsp_filter(Adspvqe, nearEndInput, nearEndOutput, farEndInput):
    command_adsp_clean = "{0} --breakOnErrors 0 --sampleRate 16000 --useEchoCancellation 0 \
                    --operatingMode 2 --useDigitalAgcNearend 0 --useDigitalAgcFarend 0 \
                    --useVirtualAGC 0 --useComfortNoiseGenerator 0 --useAnalogAutomaticGainControl 0 \
                    --useNoiseReduction 0 --loopbackInputFile {1} --farEndInputFile {2} \
                    --nearEndInputFile {3} --nearEndOutputFile {4}".format(Adspvqe,
                                                                           farEndInput, farEndInput, nearEndInput,
                                                                           nearEndOutput)
    subprocess.call(command_adsp_clean)


def snr_mixer(params, clean, noise, snr, target_level=-25, clipping_threshold=0.99):
    '''Function to mix clean speech and noise at various SNR levels'''
    cfg = params['cfg']
    if len(clean) > len(noise):
        noise = np.append(noise, np.zeros(len(clean) - len(noise)))
    else:
        clean = np.append(clean, np.zeros(len(noise) - len(clean)))
    
    # Normalizing to -25 dB FS
    clean = clean / (max(abs(clean)) + EPS)
    clean = normalize_single_channel_to_target_level(clean, target_level)
    rmsclean = (clean ** 2).mean() ** 0.5
    
    noise = noise / (max(abs(noise)) + EPS)
    noise = normalize_single_channel_to_target_level(noise, target_level)
    rmsnoise = (noise ** 2).mean() ** 0.5
    
    # Set the noise level for a given SNR
    noisescalar = rmsclean / (10 ** (snr / 20)) / (rmsnoise + EPS)
    noisenewlevel = noise * noisescalar
    
    # Mix noise and clean speech
    noisyspeech = clean + noisenewlevel
    
    # Randomly select RMS value between -15 dBFS and -35 dBFS and normalize noisyspeech with that value
    # There is a chance of clipping that might happen with very less probability, which is not a major issue. 
    noisy_rms_level = np.random.randint(params['target_level_lower'], params['target_level_upper'])
    rmsnoisy = (noisyspeech ** 2).mean() ** 0.5
    scalarnoisy = 10 ** (noisy_rms_level / 20) / (rmsnoisy + EPS)
    noisyspeech = noisyspeech * scalarnoisy
    clean = clean * scalarnoisy
    noisenewlevel = noisenewlevel * scalarnoisy
    
    # Final check to see if there are any amplitudes exceeding +/- 1. If so, normalize all the signals accordingly
    if is_clipped(noisyspeech):
        noisyspeech_maxamplevel = max(abs(noisyspeech)) / (clipping_threshold - EPS)
        noisyspeech = noisyspeech / noisyspeech_maxamplevel
        clean = clean / noisyspeech_maxamplevel
        noisenewlevel = noisenewlevel / noisyspeech_maxamplevel
        noisy_rms_level = int(20 * np.log10(scalarnoisy / noisyspeech_maxamplevel * (rmsnoisy + EPS)))
    
    return clean, noisenewlevel, noisyspeech, noisy_rms_level


def segmental_snr_mixer(params, clean, noise, snr, target_level=-25, clipping_threshold=0.99):
    '''Function to mix clean speech and noise at various segmental SNR levels'''
    cfg = params['cfg']
    if len(clean) > len(noise):
        noise = np.append(noise, np.zeros(len(clean) - len(noise)))
    else:
        clean = np.append(clean, np.zeros(len(noise) - len(clean)))
    clean = clean / (max(abs(clean)) + EPS)
    noise = noise / (max(abs(noise)) + EPS)
    rmsclean, rmsnoise = active_rms(clean=clean, noise=noise)
    clean = normalize_segmental_rms(clean, rms=rmsclean, target_level=target_level)
    noise = normalize_segmental_rms(noise, rms=rmsnoise, target_level=target_level)
    # Set the noise level for a given SNR
    noisescalar = rmsclean / (10 ** (snr / 20)) / (rmsnoise + EPS)
    noisenewlevel = noise * noisescalar
    
    # Mix noise and clean speech
    noisyspeech = clean + noisenewlevel
    # Randomly select RMS value between -15 dBFS and -35 dBFS and normalize noisyspeech with that value
    # There is a chance of clipping that might happen with very less probability, which is not a major issue. 
    noisy_rms_level = np.random.randint(params['target_level_lower'], params['target_level_upper'])
    rmsnoisy = (noisyspeech ** 2).mean() ** 0.5
    scalarnoisy = 10 ** (noisy_rms_level / 20) / (rmsnoisy + EPS)
    noisyspeech = noisyspeech * scalarnoisy
    clean = clean * scalarnoisy
    noisenewlevel = noisenewlevel * scalarnoisy
    # Final check to see if there are any amplitudes exceeding +/- 1. If so, normalize all the signals accordingly
    if is_clipped(noisyspeech):
        noisyspeech_maxamplevel = max(abs(noisyspeech)) / (clipping_threshold - EPS)
        noisyspeech = noisyspeech / noisyspeech_maxamplevel
        clean = clean / noisyspeech_maxamplevel
        noisenewlevel = noisenewlevel / noisyspeech_maxamplevel
        noisy_rms_level = int(20 * np.log10(scalarnoisy / noisyspeech_maxamplevel * (rmsnoisy + EPS)))
    
    return clean, noisenewlevel, noisyspeech, noisy_rms_level


def active_rms(clean, noise, fs=16000, energy_thresh=-50):
    '''Returns the clean and noise RMS of the noise calculated only in the active portions'''
    window_size = 100  # in ms
    window_samples = int(fs * window_size / 1000)
    sample_start = 0
    noise_active_segs = []
    clean_active_segs = []
    
    while sample_start < len(noise):
        sample_end = min(sample_start + window_samples, len(noise))
        noise_win = noise[sample_start:sample_end]
        clean_win = clean[sample_start:sample_end]
        noise_seg_rms = 20 * np.log10((noise_win ** 2).mean() + EPS)
        # Considering frames with energy
        if noise_seg_rms > energy_thresh:
            noise_active_segs = np.append(noise_active_segs, noise_win)
            clean_active_segs = np.append(clean_active_segs, clean_win)
        sample_start += window_samples
    
    if len(noise_active_segs) != 0:
        noise_rms = (noise_active_segs ** 2).mean() ** 0.5
    else:
        noise_rms = EPS
    
    if len(clean_active_segs) != 0:
        clean_rms = (clean_active_segs ** 2).mean() ** 0.5
    else:
        clean_rms = EPS
    
    return clean_rms, noise_rms


def activitydetector(audio, fs=16000, energy_thresh=0.13, target_level=-25):
    '''Return the percentage of the time the audio signal is above an energy threshold'''
    
    audio = normalize_single_channel_to_target_level(audio, target_level)
    window_size = 50  # in ms
    window_samples = int(fs * window_size / 1000)
    sample_start = 0
    cnt = 0
    prev_energy_prob = 0
    active_frames = 0
    
    a = -1
    b = 0.2
    alpha_rel = 0.05
    alpha_att = 0.8
    
    while sample_start < len(audio):
        sample_end = min(sample_start + window_samples, len(audio))
        audio_win = audio[sample_start:sample_end]
        frame_rms = 20 * np.log10(sum(audio_win ** 2) + EPS)
        frame_energy_prob = 1. / (1 + np.exp(-(a + b * frame_rms)))
        
        if frame_energy_prob > prev_energy_prob:
            smoothed_energy_prob = frame_energy_prob * alpha_att + prev_energy_prob * (1 - alpha_att)
        else:
            smoothed_energy_prob = frame_energy_prob * alpha_rel + prev_energy_prob * (1 - alpha_rel)
        
        if smoothed_energy_prob > energy_thresh:
            active_frames += 1
        prev_energy_prob = frame_energy_prob
        sample_start += window_samples
        cnt += 1
    
    perc_active = active_frames / cnt
    return perc_active


def resampler(input_dir, target_sr=16000, ext='*.wav'):
    '''Resamples the audio files in input_dir to target_sr'''
    files = glob.glob(f"{input_dir}/" + ext)
    for pathname in files:
        print(pathname)
        try:
            audio, fs = audioread(pathname)
            audio_resampled = librosa.core.resample(audio, fs, target_sr)
            audiowrite(pathname, audio_resampled, target_sr)
        except:
            continue


def calculate_audio_power(audio):
    power = (audio ** 2).mean()
    power_dB = int(10 * np.log10(power / REF_POWER + EPS))
    
    return power_dB


def audio_energy_ratio_over_threshold(audio, fs=16000, threshold=None, ):
    audio_energy = single_side_fft(normalize_single_channel_to_target_level(audio)) ** 2
    l_idx, h_idx = int(50 * len(audio) / fs + 1), int(500 * len(audio) / fs)
    energy_ratio = audio_energy[l_idx:h_idx].sum() / (audio_energy[1:l_idx].sum() + EPS)
    
    # curve_name = ['voice', 'low', ]
    # curve_data = [audio_energy[1:h_idx], audio_energy[1:l_idx], ]
    # color = ['r', 'g']
    # title = path + '_' + str(int(energy_ratio))
    # img_path = os.path.join('./image', title + '.jpg', )
    # plot_curve(data=list(zip(curve_name, curve_data, color)), title=title, img_path=img_path, show=False)
    
    # print('energy_ratio', energy_ratio)
    if energy_ratio > threshold:
        return True
    else:
        return False


def audio_energy_over_threshold(audio, threshold=None):
    energy_mean = (audio ** 2).sum() / len(audio)
    # print('energy', threshold, energy_mean)
    if energy_mean > threshold:
        return True
    else:
        return False


def single_side_fft(audio, ):  # 其实可以直接用   np.abs(rfft(audio)) / fs * 2
    rfft_amp = np.abs(rfft(audio))
    return rfft_amp


def audio_segmenter_4_file(input_path, dest_dir, segment_len=10, overlap_per=0., fs=None, dropout=False,
                           threshold=None):
    '''Segments the single-channel audio clips to segment_len in secs'''
    '''
    dropout: drop some segments according to their power based on otsu
    '''
    audio, ini_fs = audioread(input_path)
    if fs is not None:
        audio = librosa.core.resample(audio, ini_fs, fs)
    else:
        fs = ini_fs
    seg_len = int(segment_len * fs)
    step_size = int(seg_len * (1 - overlap_per))
    
    # complement the audio
    if len(audio) > seg_len and (len(audio) - seg_len) % step_size != 0:
        audio = np.append(audio, audio[0: step_size - (len(audio) - seg_len) % step_size])
    elif len(audio) < seg_len:
        while len(audio) < seg_len:
            audio = np.append(audio, audio)
        audio = audio[:seg_len]
    
    # split the audio
    num_segments = int((len(audio) - seg_len) / step_size) + 1
    audio_segments = np.array([audio[i * step_size:seg_len + i * step_size] for i in range(num_segments)])
    
    file_basename = os.path.basename(input_path)
    basename, ext = os.path.splitext(file_basename)
    
    for i in range(len(audio_segments)):
        #  决定是否丢弃
        img_path = os.path.basename(os.path.dirname(dest_dir)) + basename + '_seg' + str(i)
        if dropout and (
                not audio_energy_ratio_over_threshold(audio_segments[i], fs=fs, path=img_path, threshold=threshold)):
            # save_path = os.path.join(dest_dir.replace('ini', 'ini_del'), basename + '_seg' + str(i) + ext)
            # audiowrite(save_path, audio_segments[i], fs)
            
            continue
        
        save_path = os.path.join(dest_dir, basename + '_seg' + str(i) + ext)
        audiowrite(save_path, audio_segments[i], fs)


def audio_segmenter_4_numpy(audio, fs=16000, segment_len=1., overlap_per=0., dropout=False, threshold=None):
    '''
    Segments the single-channel audio clips to segment_len in secs
    :param audio:
    :param segment_len:
    :param overlap_per:
    :param fs:
    :param dropout: drop some segments according to specific standard
    :param threshold:
    :return:
    '''
    
    seg_len = int(segment_len * fs)
    step_size = int(seg_len * (1 - overlap_per))
    
    # complement the audio
    if len(audio) > seg_len and (len(audio) - seg_len) % step_size != 0:
        audio = np.append(audio, audio[0: step_size - (len(audio) - seg_len) % step_size])
    elif len(audio) < seg_len:
        while len(audio) < seg_len:
            audio = np.append(audio, audio)
        audio = audio[:seg_len]
    
    # split the audio
    num_segments = int((len(audio) - seg_len) / step_size) + 1
    audio_segments = [audio[i * step_size:seg_len + i * step_size] for i in range(num_segments)]
    
    if dropout:
        audio_res = []
        for i in audio_segments:
            if audio_energy_ratio_over_threshold(i, fs=fs, threshold=threshold):
                audio_res.append(i)
        audio_segments = audio_res
    
    return np.array(audio_segments)
