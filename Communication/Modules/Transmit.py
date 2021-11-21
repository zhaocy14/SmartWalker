import zmq

class TransmitZMQ(object):
    _instance = None

    @staticmethod
    def get_instance(address="*", port="5455"):
        if TransmitZMQ._instance is None:
            TransmitZMQ(address=address, port=port)
        return TransmitZMQ._instance

    def get_id(self):
        return self._id

    def __init__(self, address="*", port="5455"):
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