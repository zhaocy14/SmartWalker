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
import threading
import json
from datetime import datetime

from Sensors import Infrared_Sensor
# from Communication.Modules.Variables import *
from global_variables import CommTopic
from Communication.Modules.Transmit import TransmitZMQ
tzo = TransmitZMQ.get_instance()


class InfraredTransmit(object):
    def __init__(self, mode="online"):
        # if topic is None:
        #     self.topic = ir_topic
        # else:
        #     self.topic = topic
        self.topic = CommTopic.IR.value
        self.mode = mode
    

    def start(self, use_thread=False):
        self.start_IR()
        def _start():
            while True:
                if self.mode == "online" and self.IRSensor.distance_data.size > 0:
                    msg = json.dumps(self.IRSensor.distance_data.tolist())
                    tzo.send(self.topic, msg)
                    time.sleep(0.2)

                else:
                    # msg = "[150.0, 105.4296875, 90.98307291666666, 150.0, 133.90625, 150.0, 150]"
                    msg = datetime.now().strftime("%H:%M:%S")
                    tzo.send(self.topic, msg)
                    time.sleep(1)
                    
        if use_thread:
            th1 = threading.Thread(target=_start)
            th1.start()
        else:
            _start()
            
    def start_IR(self):
        if self.mode == "online":
            self.IRSensor = Infrared_Sensor.Infrared_Sensor(sensor_num=5,baud_rate=115200, is_windows=False)
            # infrared = Infrared_Sensor(sensor_num=7,baud_rate=115200, is_windows=False)
            thread_infrared = threading.Thread(target=self.IRSensor.read_data,args=(False, False, True))
            thread_infrared.start()


if __name__ == "__main__":
    # ir_sock = InfraredTransmit(topic=driver_topic, mode="local")
    ir_sock = InfraredTransmit()
    ir_sock.start()
