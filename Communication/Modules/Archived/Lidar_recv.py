#
# Created on Wed Sep 22 2021
# Author: Owen Yip
# Mail: me@owenyip.com
#
# Socket number: 
# RpLidar is 5450, IMU is 5451, Driver is 5452
#
import os,sys
pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)

import threading

# from Network import FrontFollowingNetwork as FFL
from global_variables import CommTopic
from Communication.Modules.Receive import ReceiveZMQ
rzo = ReceiveZMQ.get_instance()

class LidarRecv(object):
    def __init__(self):
        self.topic = CommTopic.LIDAR
    

    def start(self, use_thread=False):
        def _start():
            for topic, msg in rzo.start(topics=[self.topic]):
                """Handle the pose data here"""
                print("Received request - {}::{}".format(topic, msg))
        
        if use_thread:
            th1 = threading.Thread(target=_start)
            th1.start()
        else:
            _start()


if __name__ == "__main__":
    # drvSockObj = DriverRecv(mode="local")
    poseSockObj = LidarRecv()
    poseSockObj.start()