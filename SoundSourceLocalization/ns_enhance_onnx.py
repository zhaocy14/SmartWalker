# !/usr/bin/env python3
import os
import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
import onnxruntime as ort
import ns_featurelib


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
        self.Nfft = int(float(self.cfg['winlen']) * self.fs)
        self.mingain = 10 ** (self.cfg['mingain'] / 20)
        
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
        
        inputSpec = ns_featurelib.calcSpec(sigIn, self.cfg)
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


"""
    Inference script for NSnet2 baseline.
"""


def load_onnx_model(model_path='./model/ns_nsnet2-20ms-baseline.onnx'):
    # check model path
    assert os.path.exists(model_path)
    
    # Create the enhancer
    model = NSnet2Enhancer(modelfile=model_path)
    # get modelname
    model_name = os.path.split(os.path.basename(model_path))
    
    return model, model_name


def denoise_nsnet2(audio=None, fs=None, audio_ipath=None, audio_opath=None, model=None, model_name=None, ):
    if audio_ipath is not None:
        audio, fs, = sf.read(audio_ipath)
    
    if len(audio.shape) > 1:  # if >= one channel, only select the first channel
        audio = audio[:, 0]
    
    audio = model(audio, fs)
    
    if audio_opath is not None:
        if not os.path.exists(os.path.dirname(audio_opath)):
            os.makedirs(os.path.dirname(audio_opath))
        # print('Writing output to:', str(audio_opath))
        sf.write(audio_opath, audio, fs)
    return audio


if __name__ == '__main__':
    model, _ = load_onnx_model()
    
    audio_ipath = '../dataset/hole/1s_16000/ini/src_-2_-4_rl/walker_-1.0_1_-2.0_315/walker_-1.0_1_-2.0_315_mic1_seg20.wav'
    audio_opath = '../dataset/hole/1s_16000/test/walker_-1.0_1_-2.0_315_mic1_seg20.wav'
    
    denoise_nsnet2(audio_ipath=audio_ipath, audio_opath=audio_opath, model=model, )
