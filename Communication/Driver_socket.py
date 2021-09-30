#
# Created on Wed Sep 22 2021
# Author: Owen Yip
# Mail: me@owenyip.com
#
# Socket number: 
# RpLidar is 5450, IMU is 5451, Driver is 5452
#

import numpy as np
import os,sys
pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)
import time
import threading
import zmq
import json

from Driver import ControlOdometryDriver as cd
# from Network import FrontFollowingNetwork as FFL

CD = cd.ControlDriver(record_mode=False, left_right=0)
thread_cd = threading.Thread(target=CD.control_part, args=())

def send_control(control = None):
    CD.speed = control['speed']
    CD.radius = control['radius']
    CD.omega = control['omega']

init_control = {
  "speed": 0,
  "radius": 0,
  "omega": 0
}
send_control(init_control)
thread_cd.start()

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://127.0.0.1:5452")
topicfilter = ""
socket.setsockopt_string(zmq.SUBSCRIBE, topicfilter)

while True:
    #  Wait for next request from client
    message = socket.recv()
    if message:
      control = json.loads(message)
      if len(control) > 0:
        send_control(control)
      print("Received request: %s" % control)

    #  Do some 'work'
    time.sleep(0)

    #  Send reply back to client
    # socket.send(b"World")