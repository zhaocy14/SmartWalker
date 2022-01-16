#
# Created on Sat Oct 09 2021
# Author: Owen Yip
# Mail: me@owenyip.com
#
import os, sys
import time
import zmq
import json
import threading
import numpy as np
import msgpack
import msgpack_numpy as msgnp
from CommunicationPeer import CommunicationPeer

pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)


class WalkerClient(CommunicationPeer):
    def __init__(self, ):
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
