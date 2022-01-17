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

pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)


class CommunicationPeer(object):
    def __init__(self, send_port, send_topic, send_socket, recv_port, recv_topic, recv_socket):
        super(CommunicationPeer, self).__init__()
        
        self.send_port = send_port
        self.send_topic = send_topic
        self.send_socket = send_socket
        
        self.recv_port = recv_port
        self.recv_topic = recv_topic
        self.recv_socket = recv_socket
    
    def send_bytes(self, by_message, subtopic='', ):
        '''
        send a message in byte format with the specified topic
        Args:
            by_message: message bytes
            subtopic:
                '': no subtopic is used
                string: use the specified string as subtopic
        '''
        topic = self.send_topic if subtopic == '' else self.send_topic + '/' + subtopic
        by_topic = topic.encode('utf-8')
        socket_message = by_topic + by_message
        self.send_socket.send(socket_message, copy=False)
    
    def send_json(self, data, subtopic='', ):
        '''
        send a message in json format with the specified topic
        Args:
            data: data in json format
            subtopic:
                '': no subtopic is used
                string: use the specified string as subtopic
        '''
        data = json.dumps(data)
        by_data = data.encode('utf-8')
        self.send_bytes(by_message=by_data, subtopic=subtopic)
        # print("Sending data:", message)
    
    # def send_np(self, array, subtopic='', ):
    #     '''
    #     send a numpy array with the specified topic
    #     Args:
    #         array:
    #         subtopic:
    #             '': no subtopic is used
    #             string: use the specified string as subtopic
    #     '''
    #     by_array = msgpack.dumps(array, default=msgnp.encode, use_bin_type=False)
    #     self.send_bytes(by_message=by_array, subtopic=subtopic)
    #     # print("Sending array:", array)
    
    def send(self, data, subtopic='', ):
        '''
        send anything with the specified topic
        Args:
            data:
            subtopic:
                '': no subtopic is used
                string: use the specified string as subtopic
        '''
        by_data = msgpack.dumps(data, default=msgnp.encode, use_bin_type=False)
        self.send_bytes(by_message=by_data, subtopic=subtopic)
        # print("Sending data:", data)
    
    def recv_bytes(self, subtopic='', ):
        '''
        receive message bytes with the specified topic
        Args:
            subtopic:
                '': self.recv_topic will be used as the default topic
                string: the conjecture ('/') of self.recv_topic and subtopic will be used as the topic
        Returns:
            the received message bytes
        '''
        
        topic = self.recv_topic if subtopic == '' else self.recv_topic + '/' + subtopic
        by_topic = topic.encode('utf-8')
        message = self.recv_socket.recv()
        return message[len(by_topic):]
    
    def recv_json(self, subtopic='', ):
        '''
        receive a message in json format with the specified topic
        Args:
            subtopic:
                '': self.recv_topic will be used as the default topic
                string: the conjecture ('/') of self.recv_topic and subtopic will be used as the topic
        Returns:
            the received message in json format
        '''
        
        by_message = self.recv_bytes(subtopic=subtopic, )
        message = by_message.decode('utf-8')
        data = json.loads(message)
        
        return data
    
    def recv(self, subtopic='', ):
        '''
        receive anything with the specified topic
        Args:
            subtopic:
                '': self.recv_topic will be used as the default topic
                string: the conjecture ('/') of self.recv_topic and subtopic will be used as the topic
        Returns:
            the received data
        '''
        
        by_message = self.recv_bytes(subtopic=subtopic, )
        data = msgpack.loads(by_message, object_hook=msgnp.decode, use_list=False, raw=True)
        
        return data


class testPeer(CommunicationPeer):
    def __init__(self, ):
        # Client
        context = zmq.Context()
        self.send_port = 6016
        self.send_topic = 'Client Sends'
        self.send_socket = context.socket(zmq.PUB)
        self.send_socket.connect("tcp://127.0.0.1:%d" % self.send_port)
        self.recv_port = 6015
        self.recv_topic = 'Server Sends'
        self.recv_socket = context.socket(zmq.SUB)
        self.recv_socket.connect("tcp://127.0.0.1:%d" % self.recv_port)
        self.recv_socket.setsockopt_string(zmq.SUBSCRIBE, self.recv_topic)
        
        # # Server
        # context = zmq.Context()
        # self.send_port = 8008
        # self.send_topic = 'Server Sends'
        # self.send_socket = context.socket(zmq.PUB)
        # self.send_socket.bind('tcp://*:%d' % self.send_port)
        # self.recv_port = 8080
        # self.recv_topic = 'Client Sends'
        # self.recv_socket = context.socket(zmq.SUB)
        # self.recv_socket.bind('tcp://*:%d' % self.recv_port)
        # self.recv_socket.setsockopt_string(zmq.SUBSCRIBE, self.recv_topic)
        
        super(testPeer, self).__init__(send_port=self.send_port, send_topic=self.send_topic,
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
    send_subtopic = 'audio'
    send_message = ''
    recv_subtopic = 'direction'
    peer = testPeer()
    p1 = threading.Thread(target=peer.send_forever, args=((send_message, send_subtopic)))
    p2 = threading.Thread(target=peer.recv_forever, args=(recv_subtopic,))
    
    p1.start()
    p2.start()
    print('Prepared to send data')
    print('Prepared to receive data')
    p1.join()
    p2.join()
