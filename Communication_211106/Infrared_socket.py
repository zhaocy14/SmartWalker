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
import threading
import zmq
import json

from Sensors import Infrared_Sensor
IRSensor = Infrared_Sensor.Infrared_Sensor(sensor_num=7,baud_rate=115200, is_windows=False)
# infrared = Infrared_Sensor(sensor_num=7,baud_rate=115200, is_windows=False)
thread_infrared = threading.Thread(target=IRSensor.read_data,args=(False, False, True))
thread_infrared.start()

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5453")

while True:
    #  Wait for next request from client
    if IRSensor.distance_data.size > 0:
      # message = socket.recv()
      # if message:
      #   control = json.loads(message)
      #   print("Received request: %s" % control)
#      IRSensor.distance_data[0] = 20.0
#      IRSensor.distance_data[6] = 20.0
      msg = json.dumps(IRSensor.distance_data.tolist())
      socket.send_string(msg)
      print("Sending data: %s" % msg)

    # msg = "[79.84375000000001, 105.4296875, 90.98307291666666, 150.0, 133.90625, 107.28515625, 85.32924107142858]"
    # msg = "[150.0, 105.4296875, 90.98307291666666, 150.0, 133.90625, 150.0, 150]"
    #  Do some 'work'
    time.sleep(0.2)
