# !/usr/bin/env python
# -*- coding:utf-8 _*-
# @Author: swang
# @Contact: wang00sheng@gmail.com
# @Project Name: SmartWalker-master
# @File: test_ws.py
# @Time: 2021/11/13/22:28
# @Software: PyCharm
import os
import sys
import time
import random
import warnings
import numpy as np
from copy import deepcopy

if __name__ == '__main__':
    
    
    from Communication.Transmitter import Transmitter
    from Communication.Modules.Variables import *
    
    """Transmit demo"""
    transmitterObj = Transmitter()
    transmitterObj.single_send(sl_topic, "content here")
    

    from Communication.Modules.Variables import *
    from Communication.Modules.Receive import ReceiveZMQ

    rzo = ReceiveZMQ.get_instance()

    for topic, message in rzo.start(topic=pose_topic):
        print(message)
    
    print('Hello World!')
