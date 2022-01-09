#
# Created on Sat Oct 09 2021
# Author: Owen Yip
# Mail: me@owenyip.com
#
from datetime import datetime
import os, sys
import numpy as np
import time
import zmq
import json
import threading
from Communication.Transmitter import Transmitter
from Communication.Receiver import ReceiveZMQ
from Communication.Modules.Variables import sl_topic, pose_topic


class CLIENT():
    def __init__(self):
        super(CLIENT, self).__init__()
        self.transmiter = Transmitter()
        self.receiver = ReceiveZMQ.get_instance()
        self.sl_topic = sl_topic
        self.pose_topic = pose_topic
    
    def transmit(self, message):
        message = json.dumps(message)
        self.transmiter.single_send(self.sl_topic, message)
        print("Sending data: %s" % message)
    
    def transmit_forever(self, message):
        while True:
            self.transmit(message)
            time.sleep(1)
    
    def receive(self):
        for topic, message in self.receiver.start(topic=self.pose_topic):
            print("Receiving data: %s" % message)
            return message
    
    def receive_forever(self):
        while True:
            message = self.receive()
            # control = json.loads(message)
            # print("Received data: %s" % control)
            time.sleep(5)
    
    def receive_one(self):
        while True:
            message = self.receive()
            control = json.loads(message)
            print("Received data: %s" % control)
            if message != '':
                return message


def convert_map_location_2_owen(location):
    if np.allclose(location, [60, 425]):  # 1
        location = [120, 440]
    elif np.allclose(location, [160, 320]):  # 2
        location = [196, 326]
    elif np.allclose(location, [220, 15]):  # 9
        location = [246, 30]
    elif np.allclose(location, [530, 220]):  # 18
        location = [560, 232]
    else:
        location = [location[0] + 40, location[1] + 12]
    return location


if __name__ == "__main__":
    
    coordinates = [[None, None],  # 0
                   [60, 425],  # 1
                   [160, 320],  # 2
                   [340, 425],  # 3
                   [530, 320],  # 4
                   [215, 220],  # 5
                   [170, 160],  # 6
                   [220, 100],  # 7
                   [280, 160],  # 8
                   [220, 15],  # 9
                   [460, 15],  # 10
                   [420, 220],  # 11
                   [160, 425],  # 12
                   [530, 425],  # 13
                   [280, 220],  # 14
                   [280, 100],  # 15
                   [280, 15],  # 16
                   [160, 220],  # 17
                   [530, 220],  # 18
                   [170, 100],  # 19
                   [550, 15]]  # 20
    
    # msg = 'server to client'
    client = CLIENT()
    # p1 = threading.Thread(target=client.receive_forever, args=())
    # p1.start()
    #
    # p2 = threading.Thread(target=client.transmit_forever, args=((msg,)))
    # p2.start()
    
    while True:
        try:
            # msg = input('Please input the aim location:')
            # msg = list(map(int, msg.split(',')))
            node = int(input('Please input the aim node:'))
            msg = coordinates[node]
            msg = convert_map_location_2_owen(msg)
            p2 = threading.Thread(target=client.transmit, args=((msg,)))
            p2.start()
            p2.join()
        except:
            pass
