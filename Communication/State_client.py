#
# Created on Thu Dec 30 2021
# Author: Owen Yip
# Mail: me@owenyip.com
#
import zmq
from enum import Enum
from global_variables import WalkerPort, WalkerState

class StateClient():
    _instance = None


    @staticmethod
    def get_instance(address="127.0.0.1", port=WalkerPort.WALKER_STATE.value):
        if StateClient._instance is None:
            StateClient(address, port)
        return StateClient._instance


    def get_id(self):
        return self._id


    def __init__(self, address="127.0.0.1", port=WalkerPort.WALKER_STATE.value):
        if StateClient._instance is not None:
            raise Exception('only one instance can exist')
        else:
            self._id = id(self)
            StateClient._instance = self
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://{}:{}".format(address, port))
        
    
    def get_walker_state(self):
        self.socket.send_string("state_control.get_walker_state")
        result = self.socket.recv_string()
        return WalkerState.getByValue(result)
    
    
    def change_walker_state(self, state):
        state_string = ''
        if isinstance(state, Enum):
            state_string = state.value
        elif isinstance(state, Enum):
            state_string = state
        else:
            print('only Enum or str allowed')
            return False
        self.socket.send_string("state_control.update_walker_state::{}".format(state_string))
        result = self.socket.recv_string()
        if result == 'success':
            return True
        else:
            return False
        
        
    def get_power_level(self):
        self.socket.send_string("state_control.get_power_level")
        result = self.socket.recv_string()
        return int(result)
        
    
    def update_power_level(self, power_level):
        self.socket.send_string("state_control.update_power_level::{}".format(power_level))
        result = self.socket.recv_string()
        if result == 'success':
            return True
        else:
            return False