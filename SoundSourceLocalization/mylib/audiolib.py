# -*- coding: utf-8 -*-
"""
@author: chkarada
"""
import os
import glob
import math
import wave
import logging
import librosa
import numpy as np
from numpy.fft import fft, ifft, rfft
import soundfile as sf
import subprocess
import scipy.signal.windows as windows
from scipy.signal import get_window
from .utils import plot_curve

EPS = np.finfo(float).eps
REF_POWER = 1e-12
np.random.seed(0)


def write_frames(frames, filename, channels, sample_rate, record_width):
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(record_width)
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()


def save_wav(filepath, sample_rate, audio_string):
    wf = wave.open(filepath, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate)
    wf.writeframes(np.array(audio_string).tostring())
    # wf.writeframes(self.Voice_String.decode())
    wf.close()


def save_multi_channel_audio(audio, fs, des_dir, norm=False, ):
    for i in range(len(audio)):
        file_path = os.path.join(des_dir, f'test_mic_{i}.wav')
        audiowrite(file_path, audio[i], sample_rate=fs, norm=norm, target_level=-25, clipping_threshold=0.99)


def read_multi_channel_audio(dir_path, num_channel):
    audio = []
    for i in range(num_channel):
        file_path = os.path.join(dir_path, f'test_mic_{i}.wav')
        audio_i, _ = audioread(file_path, )
        audio.append(audio_i)
    return np.asarray(audio)


def read_and_split_channels_from_file(filepath, ):
    f = wave.open(filepath)
    params = f.getparams()
    num_channel, sample_width, fs, num_frame = params[:4]
    str_data = f.readframes(num_frame)
    f.close()
    audio = np.frombuffer(str_data, dtype=np.short)
    audio = np.reshape(audio, (-1, num_channel)).T / 32768.
    
    return audio


def extract_mel_feature():
    import librosa
    y, sr = librosa.load('./train_nb.wav', sr=16000)
    # 提取 mel spectrogram feature
    melspec = librosa.feature.melspectrogram(y, sr, n_fft=1024, hop_length=512, n_mels=128)
    logmelspec = librosa.power_to_db(melspec)  # 转换到对数刻度
    
    print(logmelspec.shape)


def is_clipped(audio, clipping_threshold=0.99):
    return any(abs(audio) > clipping_threshold)


def normalize_single_channel_audio(audio, target_level=-25, rms=None, returnScalar=False):
    '''Normalize the signal to the target level (based on segmental RMS) '''
    audio = np.array(audio, )
    rms = rms if (rms is not None) else (audio ** 2).mean() ** 0.5
    scalar = 10 ** (target_level / 20) / (rms + EPS)
    audio = audio * scalar
    if returnScalar:
        return audio, scalar
    else:
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
            audio = normalize_single_channel_audio(audio, target_level=target_level, )
    
    return audio, sample_rate


def audiowrite(destpath, audio, sample_rate=16000, norm=False, target_level=-25, clipping_threshold=None,
               clip_test=False):
    '''Function to write audio'''
    
    if clip_test:
        if is_clipped(audio, clipping_threshold=clipping_threshold):
            raise ValueError("Clipping detected in audiowrite()! " + destpath + " file not written to disk.")
    
    if norm:
        audio = normalize_single_channel_audio(audio, target_level=target_level, )
    if clipping_threshold is not None:
        max_amp = max(abs(audio))
        if max_amp >= clipping_threshold:
            audio = audio / max_amp * (clipping_threshold - EPS)
    
    destpath = os.path.abspath(destpath)
    destdir = os.path.dirname(destpath)
    os.makedirs(destdir, exist_ok=True)
    
    sf.write(destpath, audio, sample_rate)


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
    clean = normalize_single_channel_audio(clean, target_level=target_level, )
    rmsclean = (clean ** 2).mean() ** 0.5
    
    noise = noise / (max(abs(noise)) + EPS)
    noise = normalize_single_channel_audio(noise, target_level=target_level, )
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


def active_rms(clean, noise, fs=16000, energy_thresh=-50, window_size=32):
    '''Returns the clean and noise RMS calculated only in the noise-active portions'''
    # window_size = 32  # in ms
    window_samples = int(fs * window_size / 1000)
    sample_start = 0
    noise_active_segs = []
    clean_active_segs = []
    
    while sample_start < len(noise):
        sample_end = min(sample_start + window_samples, len(noise))
        noise_win = noise[sample_start:sample_end]
        clean_win = clean[sample_start:sample_end]
        # noise_seg_rms = (noise_win ** 2).mean() ** 0.5
        noise_seg_rms = 20 * np.log10((noise_win ** 2).mean() + EPS)
        # Considering frames with energy
        if noise_seg_rms > energy_thresh:
            noise_active_segs.append(noise_win)
            clean_active_segs.append(clean_win)
        sample_start += window_samples
    
    if len(noise_active_segs) != 0:
        noise_active_segs = np.concatenate(noise_active_segs, axis=-1)
        noise_rms = (noise_active_segs ** 2).mean() ** 0.5
    else:
        noise_rms = 0.
    
    if len(clean_active_segs) != 0:
        clean_active_segs = np.concatenate(clean_active_segs, axis=-1)
        clean_rms = (clean_active_segs ** 2).mean() ** 0.5
    else:
        clean_rms = 0.
    
    return clean_rms, noise_rms


def active_rms_percentage(clean, noise, fs=16000, active_percentage=0.2, window_size=32):
    '''Returns the clean and noise RMS calculated only in the top active_percentage of noise portions'''
    # window_size = 32  # in ms
    clean, noise = np.array(clean), np.array(noise)
    window_samples = int(fs * window_size / 1000)
    sample_start = 0
    clean_wins, noise_wins, noise_seg_rmses = [], [], []
    while sample_start < len(noise):
        sample_end = min(sample_start + window_samples, len(noise))
        clean_win, noise_win = clean[sample_start:sample_end], noise[sample_start:sample_end]
        clean_wins.append(clean_win), noise_wins.append(noise_win)
        noise_seg_rmses.append((noise_win ** 2).mean() ** 0.5)
        sample_start += window_samples
    
    # find the threshold
    seg_rms_threshold = np.percentile(noise_seg_rmses, (1. - active_percentage) * 100)
    noise_active_segs, clean_active_segs = [], []
    for clean_win, noise_win, noise_seg_rms in zip(*[clean_wins, noise_wins, noise_seg_rmses, ]):
        if noise_seg_rms > seg_rms_threshold:
            clean_active_segs.append(clean_win), noise_active_segs.append(noise_win)
    
    if len(noise_active_segs) != 0:
        noise_active_segs = np.concatenate(noise_active_segs, axis=-1)
        noise_rms = (noise_active_segs ** 2).mean() ** 0.5
    else:
        noise_rms = 0.
    
    if len(clean_active_segs) != 0:
        clean_active_segs = np.concatenate(clean_active_segs, axis=-1)
        clean_rms = (clean_active_segs ** 2).mean() ** 0.5
    else:
        clean_rms = 0.
    
    return clean_rms, noise_rms


def activitydetector(audio, fs=16000, energy_thresh=0.13, target_level=-25):
    '''Return the percentage of the time the audio signal is above an energy threshold'''
    
    audio = normalize_single_channel_audio(audio, target_level=target_level, )
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
    norm_audio = normalize_single_channel_audio(audio)
    rfft_amp = np.abs(rfft(norm_audio))
    audio_energy = rfft_amp ** 2
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
    return energy_mean > threshold


# def single_side_fft(audio, ):  # 其实可以直接用   np.abs(rfft(audio)) / fs * 2
#     rfft_amp = np.abs(rfft(audio))
#     return rfft_amp


def next_greater_power_of_2(x):
    return 2 ** (int(x) - 1).bit_length()


def next_lower_power_of_2(x):
    return 2 ** ((int(x) - 1).bit_length() - 1)


def audio_segmenter_4_file(input_path, dest_dir, segment_len, stepsize, fs=None, window='hann', padding=False,
                           pow_2=False, save2segFolders=False):
    '''
    Segment single-channel audio into clips, and save them in seg_{i} folder.
    :param input_path: 待clip的声音文件路径
    :param dest_dir: 保存片段的文件夹
    :param segment_len: 声音片段的长度(单位 s)
    :param stepsize: 相邻clip间的步长大小(单位 s)
    :param fs: 目标采样率，若为None，则不对声音做任何采样处理；若与原声音不同，则对原声音重采样
    :param window: 加窗类型
    :param padding: 当最后一个clip不够长时，是否补足
    :param pow_2: 是否将clip采样点数向上取整至2的整数次幂
    :param save2segFolders: 若为True，则将片段保存至单独的文件夹；否则，添加后缀 _seg_{i}，并保存
    :return:
    '''
    audio, ini_fs = audioread(input_path)
    fs, ini_fs = int(fs), int(ini_fs)
    if (fs is not None) and (fs != ini_fs):
        audio = librosa.core.resample(audio, ini_fs, fs)
    else:
        fs = ini_fs
    
    audio_segments = audio_segmenter_4_numpy(audio, segment_len=segment_len, stepsize=stepsize, fs=fs,
                                             window=window, padding=padding, pow_2=pow_2)
    file_basename = os.path.basename(input_path)
    basename, ext = os.path.splitext(file_basename)
    os.makedirs(dest_dir, exist_ok=True)
    if save2segFolders:
        for i, audio_seg in enumerate(audio_segments):
            save_path = os.path.join(dest_dir, 'seg_' + str(i), file_basename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            audiowrite(save_path, audio_seg, fs, norm=False, )
    else:
        for i, audio_seg in enumerate(audio_segments):
            save_path = os.path.join(dest_dir, basename + '_seg_' + str(i) + ext)
            audiowrite(save_path, audio_seg, fs, norm=False, )


def audio_segmenter_4_numpy(audio, segment_len, stepsize, fs=16000, window='hann', padding=False, pow_2=False):
    '''
    将numpy格式的单通道语音划分为 clips
    :param audio: numpy格式声音
    :param fs: 声音的采样率
    :param segment_len: 声音片段的长度(单位 s)
    :param stepsize: 声音片段每次向前移动的时间长度(单位 s)
    :param window:
    :param padding: 是否补全最后一个声音片段，若为True，则从开头截取一段声音进行补全
    :param pow_2:
    :return:
    '''
    
    seg_len = next_greater_power_of_2(segment_len * fs) if pow_2 else int(segment_len * fs)
    step_size = int(stepsize * fs)
    
    # complement the audio
    if padding:
        if len(audio) > seg_len and (len(audio) - seg_len) % step_size != 0:
            audio = np.append(audio, audio[0: step_size - (len(audio) - seg_len) % step_size])
        elif len(audio) < seg_len:
            while len(audio) < seg_len:
                audio = np.append(audio, audio)
            audio = audio[:seg_len]
    else:
        if len(audio) > seg_len and (len(audio) - seg_len) % step_size != 0:
            audio = audio[: - ((len(audio) - seg_len) % step_size)]
        elif len(audio) < seg_len:
            raise ValueError('audio is too short to be segmented')
    
    # split the audio
    num_segments = (len(audio) - seg_len) // step_size + 1
    audio_segments = np.asarray([audio[i * step_size:seg_len + i * step_size] for i in range(num_segments)])
    
    if (window is not None) and (len(audio_segments) > 0):
        win = get_window(window, seg_len)
        audio_segments = audio_segments * win
    
    return np.asarray(audio_segments)


def rolling_window(a, window, step=1):
    # http://ellisvalentiner.com/post/2017-03-21-np-strides-trick
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::step]


def framesig(sig, frame_len, frame_step, winfunc=lambda x: np.ones((x,)), stride_trick=True):
    """Frame a signal into overlapping frames.

    :param sig: the audio signal to frame.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
    :param stride_trick: use stride trick to compute the rolling window and window multiplication faster
    :returns: an array of frames. Size is NUMFRAMES by frame_len.
    """
    slen = len(sig)
    frame_len = int(frame_len)
    frame_step = int(frame_step)
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((1.0 * slen - frame_len) / frame_step))
    
    padlen = int((numframes - 1) * frame_step + frame_len)
    
    zeros = np.zeros((padlen - slen,))
    padsignal = np.concatenate((sig, zeros))
    
    if stride_trick:
        win = winfunc(frame_len)
        frames = rolling_window(padsignal, window=frame_len, step=frame_step)
    else:
        indices = np.tile(np.arange(0, frame_len), (numframes, 1)) + np.tile(
            np.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
        indices = np.array(indices, dtype=np.int32)
        frames = padsignal[indices]
        win = np.tile(winfunc(frame_len), (numframes, 1))
    
    return frames * win


def deframesig(frames, siglen, frame_len, frame_step, winfunc=lambda x: np.ones((x,))):
    """Does overlap-add procedure to undo the action of framesig.

    :param frames: the array of frames.
    :param siglen: the length of the desired signal, use 0 if unknown. Output will be truncated to siglen samples.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
    :returns: a 1-D signal.
    """
    frame_len = int(frame_len)
    frame_step = int(frame_step)
    numframes = np.shape(frames)[0]
    assert np.shape(frames)[1] == frame_len, '"frames" matrix is wrong size, 2nd dim is not equal to frame_len'
    
    indices = np.tile(np.arange(0, frame_len), (numframes, 1)) + np.tile(
        np.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
    indices = np.array(indices, dtype=np.int32)
    padlen = (numframes - 1) * frame_step + frame_len
    
    if siglen <= 0:
        siglen = padlen
    
    rec_signal = np.zeros((padlen,))
    window_correction = np.zeros((padlen,))
    win = winfunc(frame_len)
    
    for i in range(0, numframes):
        window_correction[indices[i, :]] = window_correction[
                                               indices[i, :]] + win + 1e-15  # add a little bit so it is never zero
        rec_signal[indices[i, :]] = rec_signal[indices[i, :]] + frames[i, :]
    
    rec_signal = rec_signal / window_correction
    return rec_signal[0:siglen]


def magspec(frames, NFFT):
    """Compute the magnitude spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the magnitude spectrum of the corresponding frame.
    """
    if np.shape(frames)[1] > NFFT:
        logging.warn(
            'frame length (%d) is greater than FFT size (%d), frame will be truncated. Increase NFFT to avoid.',
            np.shape(frames)[1], NFFT)
    complex_spec = np.fft.rfft(frames, NFFT)
    return np.absolute(complex_spec)


def powspec(frames, NFFT):
    """Compute the power spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the power spectrum of the corresponding frame.
    """
    return 1.0 / NFFT * np.square(magspec(frames, NFFT))


def logpowspec(frames, NFFT, norm=1):
    """Compute the log power spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :param norm: If norm=1, the log power spectrum is normalised so that the max value (across all frames) is 0.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the log power spectrum of the corresponding frame.
    """
    ps = powspec(frames, NFFT)
    ps[ps <= 1e-30] = 1e-30
    lps = 10 * np.log10(ps)
    if norm:
        return lps - np.max(lps)
    else:
        return lps


def preemphasis(signal, coeff=0.95):
    """perform preemphasis on the input signal.

    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.
    :returns: the filtered signal.
    """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


if __name__ == '__main__':
    audio = np.array(range(500))
    # audio = np.tile(audio, (5, 1))
    # res = rolling_window(audio, 100, 50)
    fileter = get_window('hann', 1024)
    print("F")
    curve_name = ['fileter', ]
    curve_data = [fileter, ]
    color = ['r', ]
    plot_curve(data=list(zip(curve_name, curve_data, color)), )
