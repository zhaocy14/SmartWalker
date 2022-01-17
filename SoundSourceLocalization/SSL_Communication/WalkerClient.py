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


class WalkerClient(CommunicationPeer):
    def __init__(self, ):
        print('-' * 20, 'init a WalkerClient class', '-' * 20, )
        context = zmq.Context()
        self.send_port = 6016
        self.send_topic = 'WalkerClient Sends...'
        self.send_socket = context.socket(zmq.PUB)
        self.send_socket.connect("tcp://127.0.0.1:%d" % self.send_port)
        
        self.recv_port = 6015
        self.recv_topic = 'WalkerServer Sends...'
        self.recv_socket = context.socket(zmq.SUB)
        self.recv_socket.connect("tcp://127.0.0.1:%d" % self.recv_port)
        self.recv_socket.setsockopt_string(zmq.SUBSCRIBE, self.recv_topic)
        
        super(WalkerClient, self).__init__(send_port=self.send_port, send_topic=self.send_topic,
                                           send_socket=self.send_socket,
                                           recv_port=self.recv_port, recv_topic=self.recv_topic,
                                           recv_socket=self.recv_socket)
        
        self.subtopic_buffer_dict = {
            # AUDIO_COMMUNICATION_TOPIC           : None,
            KWS_COMMUNICATION_TOPIC             : None,
            WORD_QUEUE_CLEAR_COMMUNICATION_TOPIC: False,
            SSL_COMMUNICATION_TOPIC             : None,
        }
    
    def recv(self, subtopic='', ):
        '''
        overload the corresponding function of its parent.
        take different recv actions based on the subtopics.
        KWS_COMMUNICATION_TOPIC: only audio_receiver will block until an audio frame is received and all the other messages will be stored in the class.
        WORD_QUEUE_CLEAR_COMMUNICATION_TOPIC: only check this class's buffer for messages
        Args:
            subtopic:
                '': self.recv_topic will be used as the default topic
                string: the conjecture ('/') of self.recv_topic and subtopic will be used as the topic
        Returns:
            the received data
        '''
        if subtopic == KWS_COMMUNICATION_TOPIC:
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
                if subtopic_key == KWS_COMMUNICATION_TOPIC:
                    return self.subtopic_buffer_dict[subtopic_key]
        
        elif subtopic == WORD_QUEUE_CLEAR_COMMUNICATION_TOPIC:
            data = self.subtopic_buffer_dict[WORD_QUEUE_CLEAR_COMMUNICATION_TOPIC]
            self.subtopic_buffer_dict[WORD_QUEUE_CLEAR_COMMUNICATION_TOPIC] = False
            return data
        
        elif subtopic == SSL_COMMUNICATION_TOPIC:
            data = self.subtopic_buffer_dict[SSL_COMMUNICATION_TOPIC]
            self.subtopic_buffer_dict[SSL_COMMUNICATION_TOPIC] = None
            return data
        
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
        
        last_data = None
        count = 0
        while True:
            data = self.recv(subtopic=subtopic, )
            # if (last_data is not None) and (data - last_data != 1):
            #     print(f'data is lost. last_data: {last_data} ---- data: {data}')
            #     count += 1
            # last_data = data
            # print("ratio of lost data:{}%".format(count / data * 100))
            print("Received data:", data)


if __name__ == "__main__":
    # import paramiko
    #
    # ssh = paramiko.SSHClient()
    # ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # ssh.connect('127.0.0.1', 6015, 'swang', '7keepo', timeout=-1)
    # stdin, stdout, stderr = ssh.exec_command('ls')
    #
    # for i in stdout.readlines():
    #     print(i)
    
    # # 建立一个SSHClient对象以后，除了执行命令，还可以开启一个sftp的session，用于传输文件、创建文件夹等等。
    # # 新建 sftp session
    # sftp = ssh.open_sftp()
    # # 创建目录
    # sftp.mkdir('abc')
    # # 从远程主机下载文件，如果失败， 这个可能会抛出异常。
    # sftp.get('test.sh', '/home/testl.sh')
    # # 上传文件到远程主机，也可能会抛出异常
    # sftp.put('/home/test.sh', 'test.sh')
    # import subprocess
    #
    # subprocess.call('ssh -L 6015:net-g14:8008 swang@gatekeeper.cs.hku.hk', shell=True)
    # os.system('plink -L 6015:net-g14:8008 swang@gatekeeper.cs.hku.hk -pw ZpBrwNaX')
    # os.system('sshpass -p ZpBrwNaX ssh -L 6015:net-g14:8008 swang@gatekeeper.cs.hku.hk')
    # print('Tunneling has been built.')
    # sshpass -p ZpBrwNaX ssh -L 6015:net-g14:8008 swang@gatekeeper.cs.hku.hk
    
    send_subtopic = 'audio'
    send_message = ''
    recv_subtopic = 'direction'
    client = WalkerClient()
    p1 = threading.Thread(target=client.send_forever, args=((send_message, send_subtopic)))
    p2 = threading.Thread(target=client.recv_forever, args=(recv_subtopic,))
    
    p1.start()
    p2.start()
    print('Prepared to send data')
    print('Prepared to receive data')
    p1.join()
    p2.join()
