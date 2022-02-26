#
# Created on Sun Feb 27 2022
# Author: Owen Yip
# Mail: me@owenyip.com
#

import zmq
import os,sys
pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)
grandpa_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".." + os.path.sep + "..")
sys.path.append(grandpa_path)
from global_variables import WalkerPort

class TransmitZMQ(object):
    _instance = None

    @staticmethod
    def get_instance(address="*", port=WalkerPort.TRANSMIT.value):
        if TransmitZMQ._instance is None:
            TransmitZMQ(address=address, port=port)
        return TransmitZMQ._instance

    def get_id(self):
        return self._id

    def __init__(self, address="*", port=WalkerPort.TRANSMIT.value):
        if TransmitZMQ._instance is not None:
            raise Exception('only one instance can exist')
        else:
            self._id = id(self)
            TransmitZMQ._instance = self
        context = zmq.Context()
        self.socket = context.socket(zmq.PUB)
        self.socket.bind("tcp://{}:{}".format(address, port))
        print('Transmit initialized')

    
    def send(self, topic=None, msg=None):
        if topic is not None and msg is not None:
            message = "{}::{}".format(topic, msg)
            print("{}::{}".format(topic, msg))
            self.socket.send_string(message)