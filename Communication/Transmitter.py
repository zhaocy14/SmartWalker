import os, sys
pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)

import threading
import time

from Communication.Modules.Transmit import TransmitZMQ
tzo = TransmitZMQ.get_instance()
# tzo = TransmitZMQ.get_instance(port="5456")
from Communication.Modules.Infrared_transmit import InfraredTransmit
from Communication.Modules.Time_transmit import TimeTransmit
from Communication.Modules.Variables import *


class Transmitter(object):
    def __init__(self, mode="online"):
        self.ir_trans = InfraredTransmit(mode=mode)
        self.time_trans = TimeTransmit(topic=pose_topic)
        
    
    def start_IR(self):
        t1 = threading.Thread(target=self.ir_trans.start)
        t1.start()
    

    def start_Timer(self):
        # t1 = threading.Thread(target=self.time_trans.start)
        # t1.start()
        self.time_trans.start(use_thread=True)

    
    def single_send(self, topic, msg):
        tzo.send(topic, msg)


if __name__ == "__main__":
    transmitterObj = Transmitter(mode="offline")
    transmitterObj.start_Timer()
    # while True:
    #     transmitterObj.single_send(sl_topic, "this is a sound location")
    #     time.sleep(1)
