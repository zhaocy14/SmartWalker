#
# Created on Tue Nov 09 2021
# Author: Owen Yip
# Mail: me@owenyip.com
#
import numpy as np
import re
import os,sys
pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)
import zmq

from Communication.Modules.Variables import *

class ReceiveZMQ(object):
    _instance = None

    @staticmethod
    def get_instance(address="127.0.0.1", port="5454", lidarPort="5450", imuPort="5451"):
        if ReceiveZMQ._instance is None:
            ReceiveZMQ(address=address, port=port, lidarPort=lidarPort, imuPort=imuPort)
        return ReceiveZMQ._instance

    def get_id(self):
        return self._id

    def __init__(self, address="127.0.0.1", port="5454", lidarPort="5450", imuPort="5451"):
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
    

    def start(self, topic=""):
        poller = zmq.Poller()
        # Create driver socket to receive command
        drvSocket_sub = self.context.socket(zmq.SUB)
        drvSocket_sub.connect("tcp://%s:%s" % (self.address, self.port))
        drvSocket_sub.setsockopt_string(zmq.SUBSCRIBE, driver_topic)

        # Create driver socket to receive command
        poseSocket_sub = self.context.socket(zmq.SUB)
        poseSocket_sub.connect("tcp://%s:%s" % (self.address, self.port))
        poseSocket_sub.setsockopt_string(zmq.SUBSCRIBE, pose_topic)

        poller.register(drvSocket_sub, zmq.POLLIN)
        poller.register(poseSocket_sub, zmq.POLLIN)
        # Work on requests from both server and publisher
        should_continue = True
        while should_continue:
            socks = dict(poller.poll())
            if drvSocket_sub in socks and socks[drvSocket_sub] == zmq.POLLIN:
                string = drvSocket_sub.recv_string()
                if topic == driver_topic:
                    yield string.split("::")
            
            if poseSocket_sub in socks and socks[poseSocket_sub] == zmq.POLLIN:
                string = poseSocket_sub.recv_string()
                if topic == pose_topic:
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