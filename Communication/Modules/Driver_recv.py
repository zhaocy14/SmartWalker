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
grandpa_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".." + os.path.sep + "..")
sys.path.append(grandpa_path)
import threading
import json
import time

# from Driver import ControlOdometryDriver as cd
# from Network import FrontFollowingNetwork as FFL
# from Communication.Modules.Variables import *
from Sensors.STM32 import STM32Sensors
from global_variables import CommTopic
from Communication.Modules.Receive import ReceiveZMQ
rzo = ReceiveZMQ.get_instance()

class DriverRecv(object):
    def __init__(self, mode="online", freq_hz=10):
        self.mode = mode
        # if topic is None:
        #     self.topic = driver_topic
        # else:
        #     self.topic = topic
        self.topic = CommTopic.DRIVER.value
        self.STM32 = None
        self.last_ts = time.time()
        self.freq_hz = freq_hz
    

    def start(self, use_thread=False):
        self.start_driver()
        def _start():
            for topic, msg in rzo.start(topics=[self.topic]):
                if self.mode == "online":
                    control = json.loads(msg)
                    if len(control) > 0:
                        self.send_control(control)
                print("Received request - {}::{}".format(topic, msg))

        if use_thread:
            th1 = threading.Thread(target=_start)
            th1.start()
        else:
            _start()

    
    def send_control(self, control = None):
        if control is not None:
            # self.CD.speed = control['speed']
            # self.CD.radius = control['radius']
            # self.CD.omega = control['omega']
            ''''Convert to new STM32 params'''
            _linearVelocity = control['speed'] * 100
            _angularVelocity = -control['omega']
            _distanceToCenter = control['radius']
            if self.STM32:
                current_ts = time.time()
                if current_ts - self.last_ts > (1.0 / self.freq_hz):
                    self.STM32.UpdateDriver(linearVelocity=_linearVelocity,angularVelocity=_angularVelocity,distanceToCenter=_distanceToCenter)
                    self.last_ts = current_ts
                else:
                    pass
            else:
                print('Current mode is not "online", control received', control)
    
    def start_driver(self):
        if self.mode == "online":
            self.STM32 = STM32Sensors()
            thread_cd = threading.Thread(target=self.STM32.STM_loop, args=())
            self.STM32.UpdateDriver(linearVelocity=0,angularVelocity=0,distanceToCenter=0)
            thread_cd.start()


if __name__ == "__main__":
    # drvSockObj = DriverRecv(mode="local")
    drvSockObj = DriverRecv()
    drvSockObj.start()


# CD = cd.ControlDriver(record_mode=False, left_right=0)
# thread_cd = threading.Thread(target=CD.control_part, args=())

# def send_control(control = None):
#     CD.speed = control['speed']
#     CD.radius = control['radius']
#     CD.omega = control['omega']

# init_control = {
#   "speed": 0,
#   "radius": 0,
#   "omega": 0
# }
# send_control(init_control)
# thread_cd.start()

# context = zmq.Context()
# socket = context.socket(zmq.SUB)
# socket.connect("tcp://127.0.0.1:5454")
# topicfilter = "DRIVER_RECV"
# socket.setsockopt_string(zmq.SUBSCRIBE, topicfilter)

# while True:
#     #  Wait for next request from client
#     message = socket.recv_string()
#     if message:
#       message = message.replace(topicfilter, "")
#       control = json.loads(message)
#       if len(control) > 0:
#         send_control(control)
#       print("Received request: %s" % control)

#     #  Do some 'work'
#     time.sleep(0)

#     #  Send reply back to client
#     # socket.send(b"World")

