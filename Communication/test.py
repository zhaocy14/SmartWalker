import zmq
import random
import sys
import time


context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5452")

while True:
    topic = 10001
    messagedata = random.randrange(1,215) - 80
    socket.send_string("%d %d" % (topic, messagedata))
    time.sleep(1)