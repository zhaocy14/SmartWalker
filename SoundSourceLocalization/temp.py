# !/usr/bin/env python
# -*- coding:utf-8 _*-
# @Author: swang
# @Contact: wang00sheng@gmail.com
# @Project Name: SoundSourceLocalization
# @File: temp.py
# @Time: 2022/02/13/15:54
# @Software: PyCharm
import os, sys

pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)

from Communication.Modules.Variables import *
from Communication.Modules.Receive import ReceiveZMQ
from Communication.Modules.Transmit import TransmitZMQ

if __name__ == '__main__':
    
    rzo = ReceiveZMQ.get_instance()
    
    for topic, message in rzo.start(topic=pose_topic):
        print(message)
    
    # """Transmit demo"""
    # transmitterObj = TransmitZMQ.get_instance()
    # transmitterObj.single_send(sl_topic, "content here")
