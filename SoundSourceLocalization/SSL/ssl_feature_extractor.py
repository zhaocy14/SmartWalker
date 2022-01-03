"""
    GCC Processor Part
"""

from numpy.fft import fft, ifft, rfft, irfft
import wave
from lib.utils import plot_curve, plot_hist
import os
import numpy as np
import scipy.io.wavfile as wav
from sklearn import preprocessing
import joblib
import matplotlib.pyplot as plot
import librosa
import shutil
import math

EPS = np.finfo(float).eps
REF_POWER = 1e-12


class FeatureExtractor:
    def __init__(self, fs, fft_len, num_gcc_bin, num_mel_bin=128, datatype='mic', num_channel=4):
        super(FeatureExtractor, self).__init__()
        
        self.fs = fs
        self.fft_len = fft_len
        self.num_mel_bin = num_mel_bin
        self.num_gcc_bin = num_gcc_bin
        self.mel_weight = librosa.filters.mel(sr=self.fs, n_fft=self.fft_len, n_mels=self.num_mel_bin).T
        
        self.eps = np.finfo(float).eps
        self.num_channel = num_channel
        self.datatype = datatype
    
    # INPUT FEATURES
    @staticmethod
    def _next_greater_power_of_2(x):
        return 2 ** (int(x) - 1).bit_length()
    
    @staticmethod
    def _next_lower_power_of_2(x):
        return 2 ** ((int(x)).bit_length() - 1)
    
    def get_rfft_spectrogram(self, audio):
        audio = np.array(audio)
        if audio.ndim == 3:
            audio = audio[0]
        
        fft_spectra = []
        for i in range(self.num_channel):
            temp_rfft = rfft(audio[i], n=self.fft_len)
            fft_spectra.append(temp_rfft)
        
        return np.array(fft_spectra)
    
    def get_log_mel(self, audio=None, rfft_spectra=None):
        if rfft_spectra is None:
            audio = np.array(audio)
            if audio.ndim == 3:
                audio = audio[0]
            rfft_spectra = self.get_rfft_spectrogram(audio)
        
        mag_spectra = np.abs(rfft_spectra) ** 2
        mel_spectra = np.matmul(mag_spectra, self.mel_weight)
        log_mel_spectra = librosa.power_to_db(mel_spectra)
        
        return log_mel_spectra
    
    def _get_foa_intensity_vectors(self, linear_spectra):
        pass
        # IVx = np.real(np.conj(linear_spectra[:, :, 0]) * linear_spectra[:, :, 3])
        # IVy = np.real(np.conj(linear_spectra[:, :, 0]) * linear_spectra[:, :, 1])
        # IVz = np.real(np.conj(linear_spectra[:, :, 0]) * linear_spectra[:, :, 2])
        #
        # normal = self.eps + (np.abs(linear_spectra[:, :, 0]) ** 2 + np.abs(linear_spectra[:, :, 1]) ** 2 + np.abs(
        #     linear_spectra[:, :, 2]) ** 2 + np.abs(linear_spectra[:, :, 3]) ** 2) / 2.
        # # normal = np.sqrt(IVx**2 + IVy**2 + IVz**2) + self.eps
        # IVx = np.dot(IVx / normal, self.mel_weight)
        # IVy = np.dot(IVy / normal, self.mel_weight)
        # IVz = np.dot(IVz / normal, self.mel_weight)
        #
        # # we are doing the following instead of simply concatenating to keep the processing similar to mel_spec and gcc
        # foa_iv = np.dstack((IVx, IVy, IVz))
        # foa_iv = foa_iv.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], -1))
        # if np.isnan(foa_iv).any():
        #     print('Feature extraction is generating nan outputs')
        #     exit()
        # return foa_iv
    
    def get_gcc_phat(self, audio=None, rfft_spectra=None):
        if rfft_spectra is None:
            audio = np.array(audio)
            if audio.ndim == 3:
                audio = audio[0]
            rfft_spectra = self.get_rfft_spectrogram(audio)
        
        # gcc_channels = self.nCr(self.num_channel, 2)
        gcc_ls = []
        for i in range(self.num_channel):
            for j in range(i + 1, self.num_channel):
                R = np.conj(rfft_spectra[i]) * rfft_spectra[j]
                cc = irfft(np.exp(1.j * np.angle(R)))
                # cc = irfft(R / (np.abs(R) + self.eps))
                cc = np.concatenate((cc[-self.num_gcc_bin // 2:], cc[:self.num_gcc_bin // 2]))
                gcc_ls.append(cc)
        return np.array(gcc_ls)
    
    # ------------------------------- EXTRACT FEATURE AND PREPROCESS IT -------------------------------
    def extract_all_feature(self, audio):
        rfft_spectra = self.get_rfft_spectrogram(audio)
        
        if self.datatype is 'foa':
            pass
            # extract intensity vectors
            # foa_iv = self._get_foa_intensity_vectors(spect)
            # feat = np.concatenate((mel_spect, foa_iv), axis=-1)
        elif self.datatype is 'mic':
            gcc_phat = self.get_gcc_phat(rfft_spectra=rfft_spectra)
            log_mel = self.get_log_mel(rfft_spectra=rfft_spectra)
            
            return np.array([gcc_phat, log_mel])
        
        else:
            raise ValueError('Unknown dataset format {}'.format(self.datatype))
    
    def preprocess_features(self):
        pass
        # Setting up folders and filenames
        # spec_scaler = None
        #
        # # pre-processing starts
        #
        # spec_scaler = preprocessing.StandardScaler()
        # for file_cnt, file_name in enumerate(os.listdir(self._feat_dir)):
        #     print('{}: {}'.format(file_cnt, file_name))
        #     feat_file = np.load(os.path.join(self._feat_dir, file_name))
        #     spec_scaler.partial_fit(feat_file)
        #     del feat_file
        # joblib.dump(
        #     spec_scaler,
        #     normalized_features_wts_file
        # )
        #
        # print('Normalizing feature files:')
        # print('\t\tfeat_dir_norm {}'.format(self._feat_dir_norm))
        # for file_cnt, file_name in enumerate(os.listdir(self._feat_dir)):
        #     print('{}: {}'.format(file_cnt, file_name))
        #     feat_file = np.load(os.path.join(self._feat_dir, file_name))
        #     feat_file = spec_scaler.transform(feat_file)
        #     np.save(
        #         os.path.join(self._feat_dir_norm, file_name),
        #         feat_file
        #     )
        #     del feat_file
        #
        # print('normalized files written to {}'.format(self._feat_dir_norm))

#
# class GccGenerator:
#     def __init__(self, gcc_width_half=30, gcc_width_half_bias=50):
#         self.gcc_width_half = gcc_width_half
#         self.gcc_width_half_bias = gcc_width_half_bias
#
#     def gcc_phat(self, sig, refsig, fs=1, max_tau=None, ):
#         if isinstance(sig, list):
#             sig = np.array(sig)
#
#         if isinstance(refsig, list):
#             refsig = np.array(refsig)
#
#         # Generalized Cross Correlation Phase Transform
#         SIG = rfft(sig, )
#         REFSIG = rfft(refsig, )
#         R = SIG * np.conj(REFSIG)
#
#         cc = irfft(R / (np.abs(R) + EPS), )
#         center_shift = len(cc) // 2
#
#         cc = np.roll(cc, center_shift)
#         cc_feature = cc[center_shift - self.gcc_width_half:center_shift + self.gcc_width_half + 1]
#         # find max cross correlation index
#         shift = np.argmax(np.abs(cc)) - center_shift
#         tau = shift  # / float(interp * fs) * 340
#
#         # curve_name = ['R', 'cc', 'SIG', 'REFSIG', 'sig', 'refsig', ]
#         # curve_data = [R / np.abs(R), cc, np.abs(SIG) / 4, np.abs(REFSIG) / 4, sig, refsig, ]
#         # color = ['r', 'g', 'black', 'purple', 'b', 'cyan']
#         # plot_curve(data=list(zip(curve_name, curve_data, color)))
#
#         return tau, cc, cc_feature
#
#     def cal_gcc_online(self, input_dir, save_count, type='Vector', debug=False, denoise=False, special_wav='u'):
#         for i in range(1, 5):
#             if debug:
#                 if i == 1:
#                     p = 2
#                 elif i == 2:
#                     p = 4
#                 elif i == 3:
#                     p = 1
#                 elif i == 4:
#                     p = 3
#             else:
#                 p = i
#
#             if denoise is True:
#                 mic_name = str(save_count) + "_de_" + "mic%d" % p + ".wav"
#             else:
#                 mic_name = str(save_count) + "_" + "mic%d" % p + ".wav"
#
#             if special_wav != 'u':
#                 mic_name = special_wav[:len(special_wav) - 4] + "_" + "mic%d" % p + ".wav"
#
#             wav = wave.open(os.path.join(input_dir, mic_name), 'rb')
#
#             n_frame = wav.getnframes()
#             fs = wav.getframerate()
#             data = np.frombuffer(wav.readframes(n_frame), dtype=np.short)
#
#             locals()['data%d' % i] = data
#
#         gcc_vector = []
#
#         center = int(len(locals()['data%d' % 1]) / 2)
#
#         gcc_bias = []
#         for i in range(1, 5):
#             for j in range(i + 1, 5):
#                 tau, cc = self.gcc_phat(locals()['data%d' % i], locals()['data%d' % j], fs)
#                 for k in range(center - self.gcc_width_half, center + self.gcc_width_half + 1):
#                     gcc_vector.append(cc[k])
#                 gcc_bias.append(cc)
#
#         # add bias
#         pair1 = gcc_bias[0]
#         pair2 = gcc_bias[1]
#         pair3 = gcc_bias[2]
#         pair4 = gcc_bias[3]
#         pair5 = gcc_bias[4]
#         pair6 = gcc_bias[5]
#
#         center = int(len(pair1) / 2)
#
#         p1 = pair1[center - self.gcc_width_half_bias:center + self.gcc_width_half_bias]
#         p2 = pair2[center - self.gcc_width_half_bias:center + self.gcc_width_half_bias]
#         p3 = pair3[center - self.gcc_width_half_bias:center + self.gcc_width_half_bias]
#         p4 = pair4[center - self.gcc_width_half_bias:center + self.gcc_width_half_bias]
#         p5 = pair5[center - self.gcc_width_half_bias:center + self.gcc_width_half_bias]
#         p6 = pair6[center - self.gcc_width_half_bias:center + self.gcc_width_half_bias]
#
#         bias1 = list(p1).index(np.max(p1)) - self.gcc_width_half_bias
#         bias2 = list(p2).index(np.max(p2)) - self.gcc_width_half_bias
#         bias3 = list(p3).index(np.max(p3)) - self.gcc_width_half_bias
#         bias4 = list(p4).index(np.max(p4)) - self.gcc_width_half_bias
#         bias5 = list(p5).index(np.max(p5)) - self.gcc_width_half_bias
#         bias6 = list(p6).index(np.max(p6)) - self.gcc_width_half_bias
#
#         bias = [bias1, bias2, bias3, bias4, bias5, bias6]
#
#         if type == 'Bias':
#             return bias
#
#         return gcc_vector
