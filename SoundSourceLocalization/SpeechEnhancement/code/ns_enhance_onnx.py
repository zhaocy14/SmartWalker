# !/usr/bin/env python3
import os, sys

CRT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CRT_DIR)
# print('sys.path:', sys.path)

# print('sys.path:', sys.path)

import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
import onnxruntime as ort
import SoundSourceLocalization.SpeechEnhancement.code.ns_featurelib as ns_featurelib


class NSnet2Enhancer(object):
    """NSnet2 enhancer class."""
    
    def __init__(self, modelfile, cfg=None):
        """Instantiate NSnet2 given a trained model path."""
        self.cfg = {
            'winlen'  : 0.02,
            'hopfrac' : 0.5,
            'fs'      : 16000,
            'mingain' : -80,
            'feattype': 'LogPow'
        }
        self.frameShift = float(self.cfg['winlen']) * float(self.cfg["hopfrac"])
        self.fs = int(self.cfg['fs'])
        self.Nfft = int(float(self.cfg['winlen']) * self.fs)  # 320
        self.mingain = 10 ** (self.cfg['mingain'] / 20)  # 0.0001
        """load onnx model"""
        self.ort = ort.InferenceSession(modelfile)
        self.dtype = np.float32
    
    def enhance(self, x):
        """Obtain the estimated filter"""
        onnx_inputs = {
            self.ort.get_inputs()[0].name: x.astype(self.dtype)
        }
        out = self.ort.run(None, onnx_inputs)[0][0]
        return out
    
    def __call__(self, sigIn, inFs):
        """Enhance a single Audio signal."""
        assert inFs == self.fs, "Inconsistent sampling rate!"
        
        inputSpec = ns_featurelib.calcSpec(sigIn, self.cfg)  # complex spectral
        inputFeature = ns_featurelib.calcFeat(inputSpec, self.cfg)
        # shape: [batch x time x freq]
        inputFeature = np.expand_dims(np.transpose(inputFeature), axis=0)
        
        # Obtain network output
        out = self.enhance(inputFeature)
        
        # limit suppression gain
        Gain = np.transpose(out)
        Gain = np.clip(Gain, a_min=self.mingain, a_max=1.0)
        outSpec = inputSpec * Gain
        
        # go back to time domain
        sigOut = ns_featurelib.spec2sig(outSpec, self.cfg)
        
        return sigOut


def load_onnx_model(model_path='../model/ns_nsnet2-20ms-baseline.onnx'):
    model_path = os.path.abspath(os.path.join(CRT_DIR, model_path))
    # check model path
    assert os.path.exists(model_path)
    
    # Create the enhancer
    enhancer = NSnet2Enhancer(modelfile=model_path)
    # get modelname
    model_name = os.path.split(os.path.basename(model_path))
    
    return enhancer, model_name


def denoise_nsnet2(audio=None, fs=None, audio_ipath=None, audio_opath=None, model=None, model_name=None, ):
    '''
    denoise audio with model
    And audio_ipath enjoys higher priority than (audio, fs)
    :param audio:
    :param fs:
    :param audio_ipath:
    :param audio_opath:
    :param model:
    :param model_name:
    :return:
    '''
    if audio_ipath is not None:
        audio, fs, = sf.read(audio_ipath)
        if audio is not None:
            print('Warning: audio and fs will be ignored due to the existence of audio_ipath')
    
    if len(audio.shape) > 1:  # if >= one channel, only select the first channel
        audio = audio[:, 0]
    # audio_len = len(audio)
    de_audio = model(audio, fs)
    # de_audio = np.array(de_audio)[:audio_len]
    if audio_opath is not None:
        os.makedirs(os.path.dirname(audio_opath), exist_ok=True)
        # print('Writing output to:', str(audio_opath))
        sf.write(audio_opath, de_audio, fs)
    return de_audio


if __name__ == '__main__':
    model, _ = load_onnx_model()
    audio = np.zeros((16000,))
    
    y = denoise_nsnet2(audio=audio, fs=16000, model=model, )
    
    print(len(y))
