#
# Created on Sat Oct 09 2021
# Author: Owen Yip
# Mail: me@owenyip.com
#
import os, sys
import threading
import numpy as np
import time
import zmq
import json

pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)


class Client(object):
    def __init__(self):
        super(Client, self).__init__()
        context = zmq.Context()
        
        self.send_port = 5455
        self.send_topic = "Client Sends"
        self.send_socket = context.socket(zmq.PUB)
        self.send_socket.bind("tcp://*:%d" % self.send_port)
        
        self.recv_port = 5454
        self.recv_topic = "Server Sends"
        self.recv_socket = context.socket(zmq.SUB)
        self.recv_socket.connect("tcp://localhost:%d" % self.recv_port)
        self.recv_socket.setsockopt_string(zmq.SUBSCRIBE, self.recv_topic)
    
    def send(self, message):
        message = json.dumps(message)
        msg = "%s%s" % (self.send_topic, message)
        self.send_socket.send_string(msg)
        print("Sending data:", message)
    
    def send_forever(self, message=''):
        i = 0
        while True:
            self.send(message + str(i))
            i += 1
            i %= 100000
            time.sleep(1)
    
    def receive(self):
        message = self.recv_socket.recv_string()
        return message[len(self.recv_topic):]
    
    def recv_forever(self):
        while True:
            message = self.receive()
            control = json.loads(message)
            print("Received request:", control)


if __name__ == "__main__":
    msg = 'client to server'
    client = Client()
    p1 = threading.Thread(target=client.send_forever, args=((msg,)))
    p2 = threading.Thread(target=client.recv_forever, args=())
    
    p1.start()
    p2.start()
