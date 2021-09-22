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

from Driver import ControlOdometryDriver as cd
from Network import FrontFollowingNetwork as FFL


CD = cd.ControlDriver(record_mode=True, left_right=0)

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5452")

while True:
    #  Wait for next request from client
    message = socket.recv_json()
    print("Received request: %s" % message)

    #  Do some 'work'
    time.sleep(0)

    #  Send reply back to client
    socket.send(b"World")