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
import json

from Driver import ControlOdometryDriver as cd
# from Network import FrontFollowingNetwork as FFL
from Communication.Modules.Receive import ReceiveZMQ
rzo = ReceiveZMQ.get_instance()
from Communication.Modules.Variables import *

class DriverRecv(object):
    def __init__(self, topic=None, mode="online"):
        self.mode = mode
        if topic is None:
            self.topic = driver_topic
        else:
            self.topic = topic

        if mode == "online":
            self.CD = cd.ControlDriver(record_mode=False, left_right=0)
            thread_cd = threading.Thread(target=self.CD.control_part, args=())
            init_control = {
              "speed": 0,
              "radius": 0,
              "omega": 0
            }
            self.send_control(init_control)
            thread_cd.start()
    

    def start(self, use_thread=False):
        def _start():
            for topic, msg in rzo.start(self.topic):
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
            self.CD.speed = control['speed']
            self.CD.radius = control['radius']
            self.CD.omega = control['omega']


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

