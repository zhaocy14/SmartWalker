# !/usr/bin/env python
# -*- coding:utf-8 _*-
# @Author: swang
# @Contact: wang00sheng@gmail.com
# @Project Name: SmartWalker-master
# @File: temp3.py
# @Time: 2022/01/17/11:47
# @Software: PyCharm

import os, sys

CRT_DIR = os.path.dirname(os.path.abspath(__file__))
F_PATH = os.path.dirname(CRT_DIR)
FF_PATH = os.path.dirname(F_PATH)
FFF_PATH = os.path.dirname(FF_PATH)
sys.path.extend([CRT_DIR, F_PATH, FF_PATH, FFF_PATH, ])
# print('sys.path:', sys.path)

import time
import zmq
import json
import threading
import numpy as np
import msgpack
import msgpack_numpy as msgnp
from collections import deque  # , BlockingQueue
from CommunicationPeer import CommunicationPeer
from SoundSourceLocalization.SSL_Settings import *


class WalkerServer(CommunicationPeer):
    def __init__(self, ):
        print('-' * 20, 'init a WalkerServer class', '-' * 20, )
        context = zmq.Context()
        
        self.send_port = 8001
        self.send_topic = 'WalkerServer Sends...'
        self.send_socket = context.socket(zmq.PUB)
        self.send_socket.bind('tcp://*:%d' % self.send_port)
        
        self.recv_port = 8002
        self.recv_topic = 'WalkerClient Sends...'
        self.recv_socket = context.socket(zmq.SUB)
        self.recv_socket.bind('tcp://*:%d' % self.recv_port)
        self.recv_socket.setsockopt_string(zmq.SUBSCRIBE, self.recv_topic)
        
        super(WalkerServer, self).__init__(send_port=self.send_port, send_topic=self.send_topic,
                                           send_socket=self.send_socket,
                                           recv_port=self.recv_port, recv_topic=self.recv_topic,
                                           recv_socket=self.recv_socket)
        self.subtopic_buffer_dict = {
            AUDIO_COMMUNICATION_TOPIC   : None,
            SSL_WAIT_COMMUNICATION_TOPIC: None,
            # KWS_COMMUNICATION_TOPIC             : None,
            # WORD_QUEUE_CLEAR_COMMUNICATION_TOPIC: False,
        }
    
    def recv(self, subtopic='', ):
        '''
        overload the corresponding function of its parent.
        take different recv actions based on the subtopics.
        AUDIO_COMMUNICATION_TOPIC: only audio_receiver will block until an audio frame is received and all the other messages will be stored in the class.
        KWS_COMMUNICATION_TOPIC: only check this class's buffer for messages
        
        Warning: will rewrite the data in {self.subtopic_buffer_dict} even if the last data is not used.
        Args:
            subtopic:
                '': self.recv_topic will be used as the default topic
                string: the conjecture ('/') of self.recv_topic and subtopic will be used as the topic
        Returns:
            the received data
        '''
        if subtopic == AUDIO_COMMUNICATION_TOPIC:
            while True:
                by_message = self.recv_bytes(subtopic='', )
                subtopic_key = ''
                for subtopic_key in self.subtopic_buffer_dict.keys():
                    subtopic_prefix = ('/' + subtopic_key).encode('utf-8')
                    if by_message.startswith(subtopic_prefix):
                        by_message = by_message[len(subtopic_prefix):]
                        data = msgpack.loads(by_message, object_hook=msgnp.decode, use_list=False, raw=True)
                        self.subtopic_buffer_dict[subtopic_key] = \
                            data  # will rewrite the data even if the last data is not used.
                        break
                    else:
                        continue
                if subtopic_key == AUDIO_COMMUNICATION_TOPIC:
                    # print("audio frame is receiced~")  # TODO: for debugging
                    return self.subtopic_buffer_dict[subtopic_key]
        elif subtopic == SSL_WAIT_COMMUNICATION_TOPIC:
            data = self.subtopic_buffer_dict[SSL_WAIT_COMMUNICATION_TOPIC]
            self.subtopic_buffer_dict[SSL_WAIT_COMMUNICATION_TOPIC] = None
            return data
        
        # elif subtopic == WORD_QUEUE_CLEAR_COMMUNICATION_TOPIC:
        #     data = self.subtopic_buffer_dict[WORD_QUEUE_CLEAR_COMMUNICATION_TOPIC]
        #     self.subtopic_buffer_dict[WORD_QUEUE_CLEAR_COMMUNICATION_TOPIC] = False
        #     return data
        
        else:
            raise ValueError(
                'Warning: Unknown subtopic is found to receive message. And audio message might be dropped by it.')
    
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
            data = i
            # data = np.full(shape=(2, 2), fill_value=i, dtype=int, )
            self.send(data=data, subtopic=subtopic)
            # print('Send data:', data)
            i += 1
            # i %= 100000
            # time.sleep(1)
    
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
        # last_data = None
        # count = 0
        while True:
            data = self.recv(subtopic=subtopic, )
            # if (last_data is not None) and (data - last_data != 1):
            #     print(f'data is lost. last_data: {last_data} ---- data: {data}')
            #     count += 1
            # last_data = data
            # print("ratio of lost data:{}%".format(count / data * 100))
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
