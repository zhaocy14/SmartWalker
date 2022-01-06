import os, sys
import pyaudio

# sample audio
RECORD_DEVICE_NAME = "USB Camera-B4.09.24.1"
SAMPLE_RATE = 16000
CHANNELS = 4
RECORD_WIDTH = 2
CHUNK = 1024
CHUNK_SIZE = 16  # 1ms的采样点数，此参数可以使得语音队列中每一个值对应1ms的音频

# KeyWord Spotting
MAX_COMMAND_SECONDS = 3
CLIP_MS = 1000
KWS_WINDOW_STRIDE_MS = 100

# Noise Suppression
RECORD_SECONDS = 1.1  # 1

# Reinforcement Learning
GCC_LENG = 366
GCC_BIAS = 6
ACTION_SPACE = 8

FORMAT = pyaudio.paInt16

FORWARD_SECONDS = 3
STEP_SIZE = 0  # 1

pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
print(father_path)
sys.path.append(father_path)

MODEL_PATH = father_path + "/resource/model/save20.ckpt"
WAV_PATH = father_path + "/resource/wav/online"
ONLINE_MODEL_PATH = father_path + "/resource/model/online.ckpt"

# KWS parameters
KWS_WAVE_PATH = father_path + "/resource/stream_tmp"
KWS_MODEL_PATH = father_path + "/resource/Pretrained_models/DNN/follow.pb"
KWS_LABEL_PATH = father_path + "/resource/Pretrained_models/follow_labels.txt"

# sliding window size can be seen in KWS detector
