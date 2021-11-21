#
# Created on Sat Oct 09 2021
# Author: Owen Yip
# Mail: me@owenyip.com
#
from datetime import datetime
import os, sys

pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)

import numpy as np
import time
import zmq
import json
import threading


class CLIENT:
    def __init__(self):
        context = zmq.Context()
        sl_port = "5454"
        self.transmit_topic = "NAV_SL_LOCATION::"
        self.receive_topic = "NAV_WALKER_POSE::"
        
        self.transmit_socket = context.socket(zmq.PUB)
        self.transmit_socket.bind("tcp://*:5455")
        
        self.receive_socket = context.socket(zmq.SUB)
        self.receive_socket.connect("tcp://127.0.0.1:%s" % sl_port)
        self.receive_socket.setsockopt_string(zmq.SUBSCRIBE, self.receive_topic)
    
    def transmit(self, message):
        msg = "%s%s" % (self.transmit_topic, message)
        self.transmit_socket.send_string(msg)
        print("Sending data: %s" % msg)
    
    def transmit_forever(self, message):
        while True:
            self.transmit(message)
            time.sleep(1)
    
    def receive(self):
        message = self.receive_socket.recv_string()
        return message.replace(self.receive_topic, "")
    
    def receive_forever(self):
        while True:
            message = self.receive()
            control = json.loads(message)
            print("Received request: %s" % control)


if __name__ == "__main__":
    client = CLIENT()
    p1 = threading.Thread(target=client.receive_forever, args=())
    
    p1.start()
    while True:
        msg = input('Please input the aim location:')
        msg = list(map(int, msg.split(',')))
        msg = json.dumps(msg)
        p2 = threading.Thread(target=client.transmit, args=((msg,)))
        p2.start()
        p2.join()
