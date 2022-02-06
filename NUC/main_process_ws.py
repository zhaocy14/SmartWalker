# !/usr/bin/env python
# -*- coding:utf-8 _*-
# @Author: swang
# @Contact: wang00sheng@gmail.com
# @Project Name: SmartWalker-master
# @File: test.py
# @Time: 2022/01/17/18:32
# @Software: PyCharm

import os, sys

module_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(module_dir)
sys.path.append(project_dir)
data_path = os.path.join(project_dir, 'data')
# print('sys.path:', sys.path)
# print('data_path:', data_path)


import time
import numpy as np
import threading
from threading import Thread
import multiprocessing  # Event
from multiprocessing import Process, Value, Pipe, Queue, Event

# independent systems
from Sensors import IRCamera, IMU, softskin, Infrared_Sensor
from Sensors import GPS_Module, heartrate
from Following.Preprocessing import Leg_detector
from Following.Network import FrontFollowingNetwork as FFLNetwork
from Following import FrontFollow
from Driver import ControlOdometryDriver as cd
# from Communication.Cpp_command import CppCommand
from SoundSourceLocalization.SSL_Settings import *
from SoundSourceLocalization.client_ssl_loop_multiThread import Voice_Process


class MainProgramme_ws(object):
    def __init__(self, ):
        super(MainProgramme_ws, self).__init__()
        
        self.camera = IRCamera.IRCamera()
        self.Softskin = softskin.SoftSkin()
        self.infrared_sensor = Infrared_Sensor.Infrared_Sensor()
        self.leg_detector = Leg_detector.Leg_detector(is_zmq=True)
        self.driver = cd.ControlDriver()
        self.FFL = FrontFollow.FFL(self.camera, self.leg_detector, self.driver, self.infrared_sensor, self.Softskin)
        
        self.health_state = True
        # self.IMU = IMU.IMU()
        # self.GPS = GPS_Module.GPS()
        # self.HeartRate = heartrate()
        
        # threading
        self.thread_Leg = threading.Thread(target=self.leg_detector.scan_procedure, args=(False, True))
        self.thread_CD = threading.Thread(target=self.driver.control_part, args=())
        self.thread_Infrared = threading.Thread(target=self.infrared_sensor.read_data, args=())
        self.thread_Softskin = threading.Thread(target=self.Softskin.read_and_record, args=())
        
        # threading flags and event control
        self.mainEvent = threading.Event()
        self.mainRestartEvent = threading.Event()
        self.is_SSL_pass = False
        
        # Process  event control and shared variables
        self.SSL_Event = multiprocessing.Event()
        self.VoiceMenu_Command_Queue = multiprocessing.Queue()  # TODO: Warning: maxlen is not set. And it may raise Error (out of memory)
        self.Voice = Voice_Process(VoiceMenu_Command_Queue=self.VoiceMenu_Command_Queue, SSL_Event=self.SSL_Event, )
    
    def start_sensor(self):
        self.thread_CD.start()
        self.thread_Infrared.start()
        self.thread_Softskin.start()
        self.thread_Leg.start()
    
    def start_manual_mode(self):
        self.driver.stopMotor()
        self.FFL.FFLevent.clear()
        self.SSL_Event.clear()
        self.mainEvent.clear()
        self.mainRestartEvent.set()
        self.is_SSL_pass = True
    
    def start_auto_mode(self):
        self.driver.startMotor()
        self.restart(is_SSL_bypass=True)
    
    def stop_main_procedure(self):
        # set all the thread event as False
        self.mainEvent.clear()
        self.FFL.FFLevent.clear()
        self.SSL_Event.clear()
    
    def restart(self, is_SSL_bypass: bool = False):
        self.mainRestartEvent.set()
        if is_SSL_bypass:
            self.is_SSL_pass = False
            self.SSL_Event.set()
        else:
            self.is_SSL_pass = True
            self.SSL_Event.clear()
        self.mainEvent.set()
        self.SSL_Event.clear()
        self.is_SSL_pass = True
    
    def main_procedure(self):
        """this is the main procedure of different threads"""
        # start the threads
        self.start_sensor()
        # start the FFL but with event clear
        self.FFL.FFLevent.clear()
        self.FFL.start_FFL()
        # start the SSL but with event clear
        self.SSL_Event.clear()  # TODO: clear all the buffers
        self.Voice.start()
        # start the manual mode
        while True:
            self.mainEvent.wait()
            try:
                if not self.is_SSL_pass:
                    # start the SSL first
                    self.SSL_Event.set()
                    # if some one press the softskin stop the SSL
                    while True:
                        if self.Softskin.max_pressure > 30:
                            self.SSL_Event.clear()
                            break
                        time.sleep(0.1)
                self.Softskin.skin_unlock_event.wait()
                # start the FFL
                self.FFL.TurnOnDriver()
                self.FFL.FFLevent.set()
                # restart the main procedure
                self.mainRestartEvent.wait()
                self.FFL.TurnOffDriver()
                self.mainRestartEvent.clear()
            except:
                pass
    
    def state_manager(self, ):
        '''
        manage the walker state based on the command sent by Voice_Process (VoiceMenu)
        '''
        self.command_ls = ['voice menu', 'redraw map', 'charge', 'start',
                           'sleep', 'voice menu off', 'hand operation', 'help', ]
        while True:
            if self.VoiceMenu_Command_Queue.empty():
                time.sleep(0.1)
                continue
            else:
                cmd = self.VoiceMenu_Command_Queue.get(block=True, timeout=1)
            
            if cmd == 'voice menu':
                pass
            elif cmd == 'voice menu off':
                pass
            elif cmd == 'redraw map':
                pass
            elif cmd == 'charge':
                pass
            elif cmd == 'start':
                pass
            elif cmd == 'sleep':
                pass
            elif cmd == 'voice menu':
                pass
            elif cmd == 'hand operation':
                pass
            elif cmd == 'help':
                pass
            else:
                raise ValueError(
                    'Unknown command is sent to MainProgramme (state_manager) by Voice_Process (VoiceMenu)')
    
    def run(self, ):
        '''
        run state_manager and main_procedure in parallel.
        main_procedure tries to run the whole normal process ( SSL -> FFL (Softskin) )
        '''
        p1 = Thread(target=self.state_manager, args=())
        p2 = Thread(target=self.main_procedure, args=())
        p1.start()
        p2.start()
        p1.join()


if __name__ == '__main__':
    print('-' * 20, 'Hello World!', '-' * 20)
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    
    MappingMicro = True
    useCD = False
    left_right = 0
    SSL_Event = multiprocessing.Event()
    SSL_Event.set()  # TODO: for debugging
    VoiceMenu_Command_Queue = multiprocessing.Queue()  # TODO: Warning: maxlen is not set. And it may raise Error (out of memory)
    vp = Voice_Process(VoiceMenu_Command_Queue=VoiceMenu_Command_Queue, SSL_Event=SSL_Event, MappingMicro=MappingMicro,
                       useCD=useCD, left_right=left_right, )
    p1 = Process(target=vp.start, args=())
    p1.start()
    
    while True:
        if VoiceMenu_Command_Queue.empty():
            time.sleep(0.1)
            continue
        else:
            cmd = VoiceMenu_Command_Queue.get(block=True, timeout=1)
        print('Received command:', cmd)
        if cmd == 'start':
            SSL_Event.set()
        elif cmd == 'sleep':
            SSL_Event.clear()
    
    p1.join()
    
    print('-' * 20, 'Brand-new World!', '-' * 20)
