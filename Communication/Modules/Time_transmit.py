#
# Created on Sat Oct 09 2021
# Author: Owen Yip
# Mail: me@owenyip.com
#
import os,sys
pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)
grandpa_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".." + os.path.sep + "..")
sys.path.append(grandpa_path)

import time
from datetime import datetime
import threading

from global_variables import CommTopic
from Communication.Modules.Transmit import TransmitZMQ
tzo = TransmitZMQ.get_instance()


class TimeTransmit(object):
    def __init__(self, topic=None):
        self.topic = CommTopic.TIME.value
    
    
    def start(self, use_thread=False):
        def _start():
            while True:
              msg = datetime.now().strftime("%H:%M:%S")
              tzo.send(self.topic, msg)
              time.sleep(1)
        
        if use_thread:
            th1 = threading.Thread(target=_start)
            th1.start()
        else:
            _start()


if __name__ == "__main__":
    # ir_sock = InfraredTransmit(topic=driver_topic)
    time_sock = TimeTransmit("Time")
    time_sock.start()