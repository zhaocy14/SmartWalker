import os, sys

crt_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(crt_dir)
# print('sys.path:', sys.path)

import numpy as np
from scipy.special import softmax
import tensorflow as tf
from tensorflow import keras
from SoundSourceLocalization.SSL.code.ssl_feature_extractor import audioFeatureExtractor
from SoundSourceLocalization.mylib.utils import wise_standard_normalizaion


class DOA(object):
    def __init__(self, fft_len, num_gcc_bin=128, num_mel_bin=128, fs=16000, num_channel=4, fft_seg_len=None,
                 fft_stepsize_ratio=None, gcc_norm=None, model_name='ResCNN', model_dir='./model/ResCNN/base_model', ):
        super(DOA, self).__init__()
        assert model_name == 'ResCNN'
        self.fs = fs
        self.gcc_norm = gcc_norm
        self.md_dir = model_dir
        self.audio_feature_extractor = audioFeatureExtractor(fs=fs, num_gcc_bin=num_gcc_bin, num_mel_bin=num_mel_bin,
                                                             fft_len=fft_len, fft_seg_len=fft_seg_len,
                                                             fft_stepsize_ratio=fft_stepsize_ratio,
                                                             num_channel=num_channel, datatype='mic', )
        self.feature_extractor, self.classifier = self.__load_model__(md_dir=self.md_dir)
    
    def __load_model__(self, md_dir=None):
        md_dir = self.md_dir if (md_dir is None) else md_dir
        fe_dir = os.path.join(md_dir, 'feature_extractor', 'ckpt')
        cls_dir = os.path.join(md_dir, 'classifier', 'ckpt')
        feature_extractor = tf.keras.models.load_model(fe_dir)
        classifier = tf.keras.models.load_model(cls_dir)
        feature_extractor.compile()
        classifier.compile()
        return feature_extractor, classifier
    
    def extract_gcc_phat_4_pair(self, audio_pair):
        gcc_feature = self.audio_feature_extractor.get_gcc_phat(audio_pair)
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
    
    def preprocess_stft(self, feature):
        '''
        preprocess stft feature for a multi-channel signal clip
        Args:
            stft:
        Returns:
        '''
        
        x = feature[:, :, 5:].transpose((1, 2, 0))
        
        # log
        x_sign = np.sign(x)
        x_abs = np.abs(x)
        x_log = np.log(x_abs + 1)
        x = x_sign * x_log
        # norm
        # x = (x - np.mean(x)) / np.std(x)
        # x = x / np.std(x)
        
        return x
    
    def predict_sample(self, audio, invalid_classes=None):
        stft_feature = self.audio_feature_extractor.get_stft(audio)
        stft_feature = self.preprocess_stft(feature=stft_feature)
        x_batch = np.array([stft_feature])
        
        abstract_feature = self.feature_extractor.predict(x_batch, )
        class_logits = self.classifier.predict(abstract_feature, )
        class_prob = softmax(class_logits)
        if invalid_classes is not None:
            class_prob[:, invalid_classes] = 0
            class_prob = class_prob / class_prob.mean(axis=1)
        class_cate = np.argmax(class_prob[0], )
        
        return class_cate, class_prob


if __name__ == '__main__':
    model = keras.models.load_model('./model\\EEGNet\\ckpt')
