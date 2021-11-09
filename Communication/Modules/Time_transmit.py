#
# Created on Sat Oct 09 2021
# Author: Owen Yip
# Mail: me@owenyip.com
#
import os,sys
pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)

import numpy as np
import time
from datetime import datetime

from Communication.Modules.Variables import *
from Communication.Modules.Transmit import TransmitZMQ
tzo = TransmitZMQ.get_instance()


class TimeTransmit(object):
    def __init__(self, topic=None):
        if topic is None:
            self.topic = time_topic
        else:
            self.topic = topic
    

    def start(self):
        while True:
          msg = datetime.now().strftime("%H:%M:%S")
          tzo.send(self.topic, msg)
          time.sleep(1)


if __name__ == "__main__":
    # ir_sock = InfraredTransmit(topic=driver_topic)
    ir_sock = TimeTransmit(ir_topic)
    ir_sock.start()