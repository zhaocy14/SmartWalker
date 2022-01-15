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
        
        self.send_port = 6016
        self.send_topic = "Client Sends"
        self.send_socket = context.socket(zmq.PUB)
        self.send_socket.connect("tcp://127.0.0.1:%d" % self.send_port)
        
        self.recv_port = 6015
        self.recv_topic = "Server Sends"
        self.recv_socket = context.socket(zmq.SUB)
        self.recv_socket.connect("tcp://127.0.0.1:%d" % self.recv_port)
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
    
    msg = 'client to server'
    client = Client()
    p1 = threading.Thread(target=client.send_forever, args=((msg,)))
    p2 = threading.Thread(target=client.recv_forever, args=())
    
    p1.start()
    p2.start()
    print('Prepared to send data')
    print('Prepared to receive data')
    p1.join()
    p2.join()
