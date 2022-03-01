#
# Created on Thu Dec 30 2021
# Author: Owen Yip
# Mail: me@owenyip.com
#

"""Command to build the binary
pyinstaller --additional-hooks-dir PyInstallerHook xxxx.py
"""

import os,sys
pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)
from global_variables import WalkerState, WalkerPort
import zmq
import psutil
from distutils import util


class StateControl():
    _instance = None
    current_state = WalkerState.IDLE.NOT_CHARGING
    power_level = 0
    is_charging = False

    @staticmethod
    def get_instance():
        if StateControl._instance is None:
            StateControl()
        return StateControl._instance

    def get_id(self):
        return self._id

    def __init__(self, address="*", port=WalkerPort.WALKER_STATE.value):
        if StateControl._instance is not None:
            raise Exception("only one instance can exist")
        else:
            self._id = id(self)
            StateControl._instance = self

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://{}:{}".format(address, port))
        print("State Control initialized, at tcp://{}:{}".format(address, port))

    def start(self):
        while True:
            recv_msg = self.socket.recv_string()
            print('receive', recv_msg)
            if 'state_control.get_walker_state' in recv_msg:
                msg = self.get_walker_state().value
                self.socket.send_string(msg)
            elif 'state_control.set_walker_state' in recv_msg:
                args = recv_msg.split("::")
                msg = 'fail'
                if len(args) > 1:
                    if WalkerState.getByValue(args[1]):
                        self.set_walker_state(
                            WalkerState.getByValue(args[1]))
                        msg = 'success'
                self.socket.send_string(msg)
            elif 'state_control.get_power_level' in recv_msg:
                msg = self.get_power_level()
                self.socket.send_string(msg)
            elif 'state_control.set_power_level' in recv_msg:
                args = recv_msg.split("::")
                msg = 'fail'
                if len(args) > 1:
                    if args[1]:
                        self.set_power_level(int(args[1]))
                        msg = 'success'
                self.socket.send_string(msg)
            elif 'state_control.get_charging' in recv_msg:
                msg = str(self.get_charging())
                self.socket.send_string(msg)
            elif 'state_control.set_charging' in recv_msg:
                args = recv_msg.split("::")
                msg = 'fail'
                if len(args) > 1:
                    if args[1]:
                        self.set_charging(util.strtobool(args[1]))
                        msg = 'success'
                self.socket.send_string(msg)
            elif 'state_control.can_i_run' in recv_msg:
                args = recv_msg.split("::")
                msg = 'fail'
                if len(args) > 1:
                    if WalkerState.getByValue(args[1]) and self.can_i_run(args[1]):
                        self.set_walker_state(WalkerState.getByValue(args[1]))
                        msg = 'success'
                self.socket.send_string(msg)

    def get_walker_state(self):
        """Speical handling for IDLE state: check charging status"""
        if self.current_state in WalkerState.IDLE:
            if self.get_charging():
                self.current_state = WalkerState.IDLE.CHARGING
            else:
                self.current_state = WalkerState.IDLE.NOT_CHARGING
        return self.current_state

    def set_walker_state(self, state):
        """Speical handling for IDLE state: check charging status"""
        if state in WalkerState.IDLE:
            if self.get_charging():
                state = WalkerState.IDLE.CHARGING
            else:
                state = WalkerState.IDLE.NOT_CHARGING
        self.current_state = state

    def get_power_level(self):
        return int(self.power_level)
    
    def set_power_level(self, power_level):
        self.power_level = int(power_level)
    
    def get_charging(self):
        return self.is_charging
    
    def set_charging(self, charging):
        self.is_charging = charging

    # Todo: monitor the main process health status
    def monitor_main_process(self):
        for process in psutil.process_iter():
            # get all process info in one shot
            with process.oneshot():
                # get the process id
                pid = process.pid
                name = process.name()
                if 'SmartWalker' in name:
                    print(name, process.cpu_percent(), process.status())
                    parent = psutil.Process(pid)
                    children = parent.children(recursive=True)
                    for child_process in children:
                        print(child_process.name())
                    return
                    # print(name, process.cpu_percent(), process.status()process.chil)
                if pid == 0:
                    # System Idle Process for Windows NT, useless to see anyways
                    continue
        return True

    # Todo (Owen): Start the walker program
    def start_walker_program(self):
        return True

    # Todo (Owen): Check walker in power station
    def stop_walker_program(self):
        return True
    
    # Todo (Owen): Determine whether the process can run
    def can_i_run(self, state):
        return True


if __name__ == "__main__":
    state_control = StateControl().get_instance()
    # state_control.monitor_main_process()
    state_control.start()
