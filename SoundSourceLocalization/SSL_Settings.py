import os, sys
import pyaudio
import global_variables

# sample audio
RECORD_DEVICE_NAME = "USB Camera-B4.09.24.1"
SAMPLE_RATE = 16000
CHANNELS = 4
RECORD_WIDTH = 2
CHUNK = 1024
CHUNK_SIZE = 16  # 1ms的采样点数，此参数可以使得语音队列中每一个值对应1ms的音频
AUDIO_COMMUNICATION_TOPIC = 'audio'
ENERGY_THRESHOLD = 0.5

# KeyWord Spotting
MAX_COMMAND_SECONDS = 3
CLIP_MS = 1000
KWS_WINDOW_STRIDE_MS = 200
KWS_COMMUNICATION_TOPIC = 'keyword'
WORD_QUEUE_CLEAR_COMMUNICATION_TOPIC = 'WORD_QUEUE_CLEAR'

# Noise Suppression
RECORD_SECONDS = 1.1  # 1

# SSL
KWS_TIMEOUT_SECONDS = 0.5
SSL_WAIT_COMMUNICATION_TOPIC = 'WAIT'
SSL_POSE_COMMUNICATION_TOPIC = global_variables.CommTopic.POSE.value
SSL_NAV_COMMUNICATION_TOPIC = global_variables.CommTopic.SL.value

# 在SSL模块接收KWS识别的关键词时，由于会在一个（可能）连续的时间内，传来多段语音。此参数集用来表征用户说一次关键词，SSL收集持续多长时间内的关键词语音

# Reinforcement Learning
GCC_LENG = 366
GCC_BIAS = 6
ACTION_SPACE = 8

FORMAT = pyaudio.paInt16

FORWARD_SECONDS = 3
STEP_SIZE = 0.5  # 1

pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
print(father_path)
sys.path.append(father_path)
# KWS parameters
KWS_WAVE_PATH = father_path + "/resource/stream_tmp"
KWS_MODEL_PATH = father_path + "/resource/Pretrained_models/DNN/follow.pb"
KWS_LABEL_PATH = father_path + "/resource/Pretrained_models/follow_labels.txt"

MODEL_PATH = father_path + "/resource/model/save20.ckpt"
WAV_PATH = father_path + "/resource/wav/online"
ONLINE_MODEL_PATH = father_path + "/resource/model/online.ckpt"

# sliding window size can be seen in KWS detector


if __name__ == '__main__':
    print(SSL_NAV_COMMUNICATION_TOPIC, SSL_POSE_COMMUNICATION_TOPIC)
