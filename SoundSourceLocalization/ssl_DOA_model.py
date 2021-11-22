import os
import argparse
import platform
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.python.keras.losses import Loss
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, TensorBoard
from lib import utils, models_tf
from lib.mi_data import load_hole_dataset, one_hot_encoder
import shutil
from ssl_feature_extractor import FeatureExtractor
from lib.utils import wise_standard_normalizaion


class DOA:
    def __init__(self, model_dir, fft_len, num_gcc_bin=128, num_mel_bin=128, fs=16000, gcc_norm='sample-wise'):
        super(DOA, self).__init__()
        # setting keras
        sysstr = platform.system()
        if (sysstr == "Windows"):
            gpus = tf.config.experimental.list_physical_devices('GPU')
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            K.set_image_data_format('channels_first')
        elif (sysstr == "Linux"):
            pass
        else:
            pass
        
        self.md_dir = model_dir
        self.model = self.__load_model__()
        self.feature_extractor = FeatureExtractor(fs=fs, fft_len=fft_len, num_gcc_bin=num_gcc_bin,
                                                  num_mel_bin=num_mel_bin, )
        self.fs = fs
        self.gcc_norm = 'sample-wise'
    
    def __load_model__(self, ):
        return keras.models.load_model(self.md_dir)
    
    def extract_gcc_phat_4_pair(self, audio_pair):
        # gcc_feature_ls = []
        # for i in range(len(audio_pair)):
        #     for j in range(i + 1, len(audio_pair)):
        #         tau, _, gcc_feature = self.gcc_generator.gcc_phat(audio_pair[i], audio_pair[j], fs=self.fs)
        #         gcc_feature_ls.append(gcc_feature)
        # gcc_feature = np.array([gcc_feature_ls])
        #
        # # normalization
        # if self.normalization is not None:
        #     gcc_feature = wise_standard_normalizaion(gcc_feature, self.normalization)
        #
        # return gcc_feature[0]
        gcc_feature = self.feature_extractor.get_gcc_phat(audio_pair)
        if self.gcc_norm is not None:
            gcc_feature = wise_standard_normalizaion(data=gcc_feature, normalization=self.gcc_norm)
        
        return gcc_feature
    
    def extract_gcc_phat_4_batch(self, x_batch, ):
        x_batch = np.array(x_batch)
        
        if x_batch.ndim == 2:
            x_batch = x_batch[np.newaxis, np.newaxis, :, :]
        elif x_batch.ndim == 3:
            x_batch = x_batch[:, np.newaxis, :, :]
        elif x_batch.ndim == 4:
            pass
        else:
            raise TypeError('The ndim of the input for DOA prediction model is not right')
        
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
