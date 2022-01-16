#
# Created on Sat Oct 09 2021
# Author: Owen Yip
# Mail: me@owenyip.com
#
import os, sys
import time
import zmq
import threading
import numpy as np
from CommunicationPeer import CommunicationPeer

pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)


class WalkerServer(CommunicationPeer):
    def __init__(self, ):
        context = zmq.Context()
        
        self.send_port = 8008
        self.send_topic = 'WalkerServer Sends...'
        self.send_socket = context.socket(zmq.PUB)
        self.send_socket.bind('tcp://*:%d' % self.send_port)
        
        self.recv_port = 8080
        self.recv_topic = 'WalkerClient Sends...'
        self.recv_socket = context.socket(zmq.SUB)
        self.recv_socket.bind('tcp://*:%d' % self.recv_port)
        self.recv_socket.setsockopt_string(zmq.SUBSCRIBE, self.recv_topic)
        
        super(WalkerServer, self).__init__(send_port=self.send_port, send_topic=self.send_topic,
                                           send_socket=self.send_socket,
                                           recv_port=self.recv_port, recv_topic=self.recv_topic,
                                           recv_socket=self.recv_socket)
    
    def send_forever(self, message='', subtopic='', ):
        '''
        test send function
        Args:
            message:
            subtopic:
                '': no subtopic is used
                string: use the specified string as subtopic
        '''
        i = 0
        while True:
            data = np.full(shape=(2, 2), fill_value=i, dtype=int, )
            self.send(data=data, subtopic=subtopic)
            print('Send data:', data)
            i += 1
            i %= 100000
            time.sleep(1)
    
    def recv_forever(self, subtopic='', ):
        '''
        test receive function
        Args:
            message:
            subtopic:
                '': self.recv_topic will be used as the default topic
                string: the conjecture ('/') of self.recv_topic and subtopic will be used as the topic
        Returns:
            the received message
        '''
        
        while True:
            data = self.recv(subtopic=subtopic, )
            print("Received data:", data)


if __name__ == "__main__":
    send_subtopic = 'direction'
    send_message = ''
    recv_subtopic = 'audio'
    server = WalkerServer()
    p1 = threading.Thread(target=server.send_forever, args=((send_message, send_subtopic)))
    p2 = threading.Thread(target=server.recv_forever, args=(recv_subtopic,))
    
    p1.start()
    p2.start()
    print('Prepared to send data')
    print('Prepared to receive data')
    p1.join()
    p2.join()
