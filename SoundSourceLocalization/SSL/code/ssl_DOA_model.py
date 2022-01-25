import os, sys

crt_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(crt_dir)
# print('sys.path:', sys.path)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from SoundSourceLocalization.SSL.code.ssl_feature_extractor import audioFeatureExtractor
from SoundSourceLocalization.mylib.utils import wise_standard_normalizaion


class DOA(object):
    def __init__(self, fft_len, num_gcc_bin=128, num_mel_bin=128, fs=16000, num_channel=4, fft_seg_len=None,
                 fft_stepsize_ratio=None, gcc_norm=None, model_dir='./model/EEGNet/ckpt', ):
        super(DOA, self).__init__()
        self.fs = fs
        self.gcc_norm = gcc_norm
        self.md_dir = model_dir
        self.model = self.__load_model__(md_dir=self.md_dir)
        self.feature_extractor = audioFeatureExtractor(fs=fs, num_gcc_bin=num_gcc_bin, num_mel_bin=num_mel_bin,
                                                       fft_len=fft_len, fft_seg_len=fft_seg_len,
                                                       fft_stepsize_ratio=fft_stepsize_ratio,
                                                       num_channel=num_channel, datatype='mic', )
    
    def __load_model__(self, md_dir=None):
        md_dir = self.md_dir if (md_dir is None) else md_dir
        return keras.models.load_model(md_dir)
    
    def extract_stft(self, audio):
        '''
        extract stft feature for a multi-channel signal clip
        Args:
            audio:
        Returns:
        '''
        stft_feature = self.feature_extractor.get_stft(audio)
        stft_feature = stft_feature[:, :, 5:].transpose((1, 2, 0))
        
        return stft_feature
    
    def extract_gcc_phat_4_pair(self, audio_pair):
        gcc_feature = self.feature_extractor.get_gcc_phat(audio_pair)
        if self.gcc_norm is not None:
            gcc_feature = wise_standard_normalizaion(data=gcc_feature, normalization=self.gcc_norm)
        
        return gcc_feature
    
    def extract_gcc_phat_4_batch(self, x_batch, ):
        x_batch = np.array(x_batch)
        
        # if x_batch.ndim == 2:
        #     x_batch = x_batch[np.newaxis, np.newaxis, :, :]
        # elif x_batch.ndim == 3:
        #     x_batch = x_batch[:, np.newaxis, :, :]
        # elif x_batch.ndim == 4:
        #     pass
        # else:
        #     raise TypeError('The ndim of the input for DOA prediction model is not right')
        
        gcc_features = []
        for audio_pair in x_batch:
            gcc_features.append([self.extract_gcc_phat_4_pair(audio_pair[0]), ])
        return np.array(gcc_features)
    
    def predict(self, gcc_batch, invalid_classes=None):
        x_batch = np.array(gcc_batch)
        class_prob = self.model.predict(x_batch, )
        
        if invalid_classes is not None:
            class_prob[:, invalid_classes] = 0
            class_prob = class_prob / class_prob.mean(axis=1)
        class_cate = np.argmax(class_prob, axis=1)
        
        return class_prob, class_cate


if __name__ == '__main__':
    model = keras.models.load_model('./model\\EEGNet\\ckpt')
