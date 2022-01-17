"""
    GCC Processor Part
"""

import numpy as np
from numpy.fft import fft, ifft, rfft, irfft
import wave
import os
from lib.utils import plot_curve, plot_hist

EPS = np.finfo(float).eps
REF_POWER = 1e-12


class GccGenerator:
    def __init__(self, gcc_width_half=30, gcc_width_half_bias=50):
        self.gcc_width_half = gcc_width_half
        self.gcc_width_half_bias = gcc_width_half_bias
    
    def gcc_phat(self, sig, refsig, fs=1, max_tau=None, ):
        if isinstance(sig, list):
            sig = np.array(sig)
        
        if isinstance(refsig, list):
            refsig = np.array(refsig)
        
        # Generalized Cross Correlation Phase Transform
        SIG = rfft(sig, )
        REFSIG = rfft(refsig, )
        R = SIG * np.conj(REFSIG)
        
        cc = irfft(R / (np.abs(R) + EPS), )
        center_shift = len(cc) // 2
        
        cc = np.roll(cc, center_shift)
        cc_feature = cc[center_shift - self.gcc_width_half:center_shift + self.gcc_width_half + 1]
        # find max cross correlation index
        shift = np.argmax(np.abs(cc)) - center_shift
        tau = shift  # / float(interp * fs) * 340
        
        # curve_name = ['R', 'cc', 'SIG', 'REFSIG', 'sig', 'refsig', ]
        # curve_data = [R / np.abs(R), cc, np.abs(SIG) / 4, np.abs(REFSIG) / 4, sig, refsig, ]
        # color = ['r', 'g', 'black', 'purple', 'b', 'cyan']
        # plot_curve(data=list(zip(curve_name, curve_data, color)))
        
        return tau, cc, cc_feature
    
    def cal_gcc_online(self, input_dir, save_count, type='Vector', debug=False, denoise=False, special_wav='u'):
        for i in range(1, 5):
            if debug:
                if i == 1:
                    p = 2
                elif i == 2:
                    p = 4
                elif i == 3:
                    p = 1
                elif i == 4:
                    p = 3
            else:
                p = i
            
            if denoise is True:
                mic_name = str(save_count) + "_de_" + "mic%d" % p + ".wav"
            else:
                mic_name = str(save_count) + "_" + "mic%d" % p + ".wav"
            
            if special_wav != 'u':
                mic_name = special_wav[:len(special_wav) - 4] + "_" + "mic%d" % p + ".wav"
            
            wav = wave.open(os.path.join(input_dir, mic_name), 'rb')
            
            n_frame = wav.getnframes()
            fs = wav.getframerate()
            data = np.frombuffer(wav.readframes(n_frame), dtype=np.short)
            
            locals()['data%d' % i] = data
        
        gcc_vector = []
        
        center = int(len(locals()['data%d' % 1]) / 2)
        
        gcc_bias = []
        for i in range(1, 5):
            for j in range(i + 1, 5):
                tau, cc = self.gcc_phat(locals()['data%d' % i], locals()['data%d' % j], fs)
                for k in range(center - self.gcc_width_half, center + self.gcc_width_half + 1):
                    gcc_vector.append(cc[k])
                gcc_bias.append(cc)
        
        # add bias
        pair1 = gcc_bias[0]
        pair2 = gcc_bias[1]
        pair3 = gcc_bias[2]
        pair4 = gcc_bias[3]
        pair5 = gcc_bias[4]
        pair6 = gcc_bias[5]
        
        center = int(len(pair1) / 2)
        
        p1 = pair1[center - self.gcc_width_half_bias:center + self.gcc_width_half_bias]
        p2 = pair2[center - self.gcc_width_half_bias:center + self.gcc_width_half_bias]
        p3 = pair3[center - self.gcc_width_half_bias:center + self.gcc_width_half_bias]
        p4 = pair4[center - self.gcc_width_half_bias:center + self.gcc_width_half_bias]
        p5 = pair5[center - self.gcc_width_half_bias:center + self.gcc_width_half_bias]
        p6 = pair6[center - self.gcc_width_half_bias:center + self.gcc_width_half_bias]
        
        bias1 = list(p1).index(np.max(p1)) - self.gcc_width_half_bias
        bias2 = list(p2).index(np.max(p2)) - self.gcc_width_half_bias
        bias3 = list(p3).index(np.max(p3)) - self.gcc_width_half_bias
        bias4 = list(p4).index(np.max(p4)) - self.gcc_width_half_bias
        bias5 = list(p5).index(np.max(p5)) - self.gcc_width_half_bias
        bias6 = list(p6).index(np.max(p6)) - self.gcc_width_half_bias
        
        bias = [bias1, bias2, bias3, bias4, bias5, bias6]
        
        if type == 'Bias':
            return bias
        
        return gcc_vector
