#
# Created on Tue Nov 09 2021
# Author: Owen Yip
# Mail: me@owenyip.com
#
import time

import numpy as np
import re
import os,sys
import zmq
pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)
grandpa_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".." + os.path.sep + "..")
sys.path.append(grandpa_path)
from global_variables import WalkerPort, CommTopic

class ReceiveZMQ(object):
    _instance = None

    @staticmethod
    def get_instance(address="127.0.0.1", port=WalkerPort.RECV.value, lidarPort=WalkerPort.LIDAR.value, imuPort=WalkerPort.IMU.value):
        if ReceiveZMQ._instance is None:
            ReceiveZMQ(address=address, port=port, lidarPort=lidarPort, imuPort=imuPort)
        return ReceiveZMQ._instance

    def get_id(self):
        return self._id

    def __init__(self, address="127.0.0.1", port=WalkerPort.RECV.value, lidarPort=WalkerPort.LIDAR.value, imuPort=WalkerPort.IMU.value):
        if ReceiveZMQ._instance is not None:
            raise Exception('only one instance can exist')
        else:
            self._id = id(self)
            ReceiveZMQ._instance = self
        #Create the subscription socket
        self.address = address
        self.port = port
        self.lidarPort = lidarPort
        self.imuPort = imuPort
        self.context = zmq.Context()
    
    
        
    def start(self, topics=[CommTopic.DRIVER.value, CommTopic.POSE.value]):
        poller = zmq.Poller()
        subs = []

        for _topic in topics:
            sub = self.context.socket(zmq.SUB)
            sub.connect("tcp://%s:%s" % (self.address, self.port))
            sub.setsockopt_string(zmq.SUBSCRIBE, _topic)
            subs.append(sub)
            poller.register(sub, zmq.POLLIN)
        
        while True:
            socks = dict(poller.poll())
            for _sub in subs:
                if _sub in socks and socks[_sub] == zmq.POLLIN:
                    string = _sub.recv_string()
                    yield string.split("::")
                    
                    

    def start_old(self, topic=""):
        poller = zmq.Poller()
        # Create driver socket to receive command
        drvSocket_sub = self.context.socket(zmq.SUB)
        drvSocket_sub.connect("tcp://%s:%s" % (self.address, self.port))
        drvSocket_sub.setsockopt_string(zmq.SUBSCRIBE, CommTopic.DRIVER.value)

        # Create driver socket to receive command
        poseSocket_sub = self.context.socket(zmq.SUB)
        poseSocket_sub.connect("tcp://%s:%s" % (self.address, self.port))
        poseSocket_sub.setsockopt_string(zmq.SUBSCRIBE, CommTopic.POSE.value)

        poller.register(drvSocket_sub, zmq.POLLIN)
        poller.register(poseSocket_sub, zmq.POLLIN)
        # Work on requests from both server and publisher
        should_continue = True
        while should_continue:
            socks = dict(poller.poll())
            print(socks)
            if drvSocket_sub in socks and socks[drvSocket_sub] == zmq.POLLIN:
                string = drvSocket_sub.recv_string()
                if topic == CommTopic.DRIVER.value:
                    yield string.split("::")
            
            if poseSocket_sub in socks and socks[poseSocket_sub] == zmq.POLLIN:
                string = poseSocket_sub.recv_string()
                if topic == CommTopic.POSE.value:
                    yield string.split("::")


    def startLidar(self):
        poller = zmq.Poller()
        # Create driver socket to receive command
        lidarSocket_sub = self.context.socket(zmq.SUB)
        lidarSocket_sub.connect("tcp://%s:%s" % (self.address, self.lidarPort))
        lidarSocket_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        poller.register(lidarSocket_sub, zmq.POLLIN)
        should_continue = True
        while should_continue:
            socks = dict(poller.poll())
            if lidarSocket_sub in socks and socks[lidarSocket_sub] == zmq.POLLIN:
                string = lidarSocket_sub.recv_string()
                parsed_data = re.match(r"(?P<time>\d+) (?P<theta>\d+\.\d+) (?P<dist>\d+\.\d+) (?P<q>\d+)", string)
                yield parsed_data.groupdict()


if __name__ == "__main__":
    ReceiveZMQObject = ReceiveZMQ(address="127.0.0.1", port="5455")
    for topic, messagedata in ReceiveZMQObject.start():
        print("Processing ... %s: %s" % (topic, messagedata))