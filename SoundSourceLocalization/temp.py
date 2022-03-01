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

from threading import Thread
from Communication.Modules.Receive import ReceiveZMQ
from Communication.Modules.Transmit import TransmitZMQ
from Communication.Cpp_command import CppCommand

from global_variables import WalkerPort, CommTopic


def start_CppCommand():
    with CppCommand.get_instance() as cco:
        # cco.start_sensors(stdout=True)
        cco.start_navigation(stdout=True, driver_ctrl=False, ir_sensor=False, map_file="latest")


if __name__ == '__main__':
    p1 = Thread(target=start_CppCommand)
    p1.start()
    
    rzo = ReceiveZMQ.get_instance()
    
    for topic, message in rzo.start(topic=CommTopic.POSE):
        print(message)
    
    # """Transmit demo"""
    transmitterObj = TransmitZMQ.get_instance()
    transmitterObj.single_send(CommTopic.SL, (330, 80))
    
    p1.join()
