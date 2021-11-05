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
import zmq
import json

context = zmq.Context()

def transmit():
  socket = context.socket(zmq.PUB)
  socket.bind("tcp://*:5454")

  while True:
      msg = json.dumps('''position data here''')
      socket.send_string(msg)
      print("Sending data: %s" % msg)


def receive():
  socket = context.socket(zmq.SUB)
  socket.connect("tcp://127.0.0.1:5455")
  topicfilter = ""
  socket.setsockopt_string(zmq.SUBSCRIBE, topicfilter)

  while True:
       #  Wait for next request from client
        message = socket.recv()
        if message:
          control = json.loads(message)
          print("Received request: %s" % control)

if __name__ == "__main__":
    transmit()