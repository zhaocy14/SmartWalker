import os, sys
import librosa
import numpy as np
from mylib.audiolib import audio_segmenter_4_numpy

EPS = np.finfo(np.float32).eps
REF_POWER = 1e-12


class audioFeatureExtractor(object):
    def __init__(self, num_channel=4, fs=16000, num_gcc_bin=128, num_mel_bin=128, fft_len=None, fft_seg_len=None,
                 fft_stepsize_ratio=None, datatype='mic', ):
        '''
        extract features for audio
        :param fs: sample frequency
        :param num_gcc_bin: number of gcc-phat features
        :param num_mel_bin: number of log-mel features
        :param datatype: type of audio( 'mic' or 'foa')
        :param num_channel: number of channels
        '''
        super(audioFeatureExtractor, self).__init__()
        
        self.fs = fs
        self.num_channel = num_channel
        self.fft_len = fft_len
        self.num_mel_bin = num_mel_bin
        self.num_gcc_bin = num_gcc_bin
        if self.fft_len is not None:
            self.mel_weight = librosa.filters.mel(sr=self.fs, n_fft=self.fft_len, n_mels=self.num_mel_bin).T
        self.fft_seg_len = fft_seg_len
        self.fft_stepsize_ratio = fft_stepsize_ratio
        if (self.fft_seg_len is not None) and (self.fft_stepsize_ratio is not None):
            self.fft_stepsize = self.fft_seg_len * self.fft_stepsize_ratio
        
        self.datatype = datatype
        assert self.datatype == 'mic', f'{self.datatype} is not supported yet'
    
    def get_rfft_spectrogram(self, audio, ):
        '''
        calculate rfft_spectrogram of audio
        :param audio: [channel * sample_point]
        :return:
        '''
        audio = np.array(audio)
        fft_spectra = [np.fft.rfft(i, ) for i in audio]
        return np.asarray(fft_spectra)
    
    def _get_foa_intensity_vectors(self, linear_spectra):
        raise AssertionError('TODO don\'t support foa_intensity_vectors anymore')
        
        IVx = np.real(np.conj(linear_spectra[:, :, 0]) * linear_spectra[:, :, 3])
        IVy = np.real(np.conj(linear_spectra[:, :, 0]) * linear_spectra[:, :, 1])
        IVz = np.real(np.conj(linear_spectra[:, :, 0]) * linear_spectra[:, :, 2])
        
        normal = EPS + (np.abs(linear_spectra[:, :, 0]) ** 2 + np.abs(linear_spectra[:, :, 1]) ** 2 + np.abs(
            linear_spectra[:, :, 2]) ** 2 + np.abs(linear_spectra[:, :, 3]) ** 2) / 2.
        # normal = np.sqrt(IVx**2 + IVy**2 + IVz**2) + EPS
        IVx = np.dot(IVx / normal, self.mel_weight)
        IVy = np.dot(IVy / normal, self.mel_weight)
        IVz = np.dot(IVz / normal, self.mel_weight)
        
        # we are doing the following instead of simply concatenating to keep the processing similar to mel_spec and gcc
        foa_iv = np.dstack((IVx, IVy, IVz))
        foa_iv = foa_iv.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], -1))
        if np.isnan(foa_iv).any():
            print('Feature extraction is generating nan outputs')
            exit()
        return foa_iv
    
    # ------------------------------- EXTRACT FEATURE AND PREPROCESS IT -------------------------------
    def get_gcc_phat(self, audio, rfft_spectra=None):
        '''
        calculate gcc_phat feature of audio
        :param audio: [channel * sample_point]
        :param rfft_spectra:
        :return:
        '''
        
        audio = np.array(audio)
        assert audio.shape[0] == self.num_channel
        rfft_spectra = self.get_rfft_spectrogram(audio) if (rfft_spectra is None) else rfft_spectra
        
        gcc_ls = []
        for i in range(self.num_channel):
            for j in range(i + 1, self.num_channel):
                R = np.conj(rfft_spectra[i]) * rfft_spectra[j]
                cc = np.fft.irfft(np.exp(1.j * np.angle(R)))
                # cc = np.fft.irfft(R / (np.abs(R) + EPS))
                cc = np.concatenate((cc[-self.num_gcc_bin // 2:], cc[:self.num_gcc_bin // 2]))
                gcc_ls.append(cc)
        return np.asarray(gcc_ls)
    
    def get_log_mel(self, audio, rfft_spectra=None):  # TODO don't support log_mel anymore
        raise AssertionError('TODO don\'t support log_mel anymore')
        audio = np.array(audio)
        if audio.ndim == 3:
            audio = audio[0]
        if rfft_spectra is None:
            rfft_spectra = self.get_rfft_spectrogram(audio)
        
        mag_spectra = np.abs(rfft_spectra) ** 2
        mel_spectra = np.matmul(mag_spectra, self.mel_weight)
        log_mel_spectra = librosa.power_to_db(mel_spectra)
        
        return log_mel_spectra
    
    def get_stft(self, audio, ):
        '''
        calculate STFT of audio
        :param audio: [channel * sample_point]
        :param seg_len: length of an audio clip (in seconds)
        :param stepsize_ratio: length ratio of each forward movement between two adjacent segments (in seconds)
        :return: (num_channel * 2) * num_audio_clips * frequency_bins
                for the first dimension: real imag; real imag; real imag; real imag;
        '''
        assert (self.fft_seg_len is not None) and (self.fft_stepsize is not None)
        audio = np.array(audio)
        feature_ls = []
        for channel in audio:
            audio_seg = audio_segmenter_4_numpy(channel, fs=self.fs, segment_len=self.fft_seg_len,
                                                stepsize=self.fft_stepsize, window='hann', padding=False, pow_2=False)
            feature = self.get_rfft_spectrogram(audio_seg, )
            feature = np.asarray(feature)
            feature_ls.extend([feature.real, feature.imag, ])
        return np.asarray(feature_ls)  # .transpose((1, 0, 2))


if __name__ == '__main__':
    
    def set_global_seeds(seed):
        import random
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
    
    
    set_global_seeds(0)
    test_audio = np.random.rand(4, 4096)
    fe = audioFeatureExtractor(fs=16000)
    # feature = fe.get_gcc_phat(audio=test_audio, )
    feature = fe.get_stft(audio=test_audio, )
    print(feature)
    print(feature.shape)
