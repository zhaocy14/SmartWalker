import os, sys
# pwd = os.path.abspath(os.path.abspath(__file__))
# father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
# sys.path.append(father_path)
# data_path = os.path.abspath(
#     os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".." +
#     os.path.sep + "data")
pwd = os.path.dirname(os.path.abspath(__file__))
father_path = os.path.dirname(pwd)
sys.path.append(father_path)
data_path = os.path.join(father_path, 'data')
import time
import threading
import numpy as np
from Sensors import IRCamera, IMU, softskin, Infrared_Sensor
from Sensors import GPS_Module, heartrate
from Following.Preprocessing import Leg_detector
from Following.Network import FrontFollowingNetwork as FFLNetwork
from Following import FrontFollow
from Driver import ControlOdometryDriver as cd
from Communication.Cpp_command import CppCommand

import multiprocessing  # Event
from multiprocessing import Process, Value, Pipe, Queue, Event
from Communication.Cpp_command import CppCommand
from SoundSourceLocalization.SSL_Settings import *
from SoundSourceLocalization.client_ssl_loop_multiThread import Voice_Process

class VoiceMenu(object):
    def __init__(self):
        super().__init__()
        self.SSL = []
        self.Top = True
        self.Second = False
        self.SSLFlag = False
        self.FFLFlag = False

    def Voice_procedure(self):
        while True:
            time.sleep(3)


class SSL(object):
    def __init__(self):
        super().__init__()
        self.SSL = []
        self.Top = True
        self.Second = False
        self.SSLEvent = threading.Event()
        self.SSLEvent.clear()
        self.SSLthread = threading.Thread(target=self.SSLMain, args=())

    def SSLMain(self):
        while True:
            self.SSLEvent.wait()
            time.sleep(5)
            break

    def startSSL(self):
        self.SSLthread.start()


class MainProgramme(object):
    def __init__(self):
        super().__init__()

        self.camera = IRCamera.IRCamera()
        self.Softskin = softskin.SoftSkin()
        self.infrared_sensor = Infrared_Sensor.Infrared_Sensor()
        # self.IMU = IMU.IMU()
        # self.GPS = GPS_Module.GPS()
        # self.HeartRate = heartrate()
        self.leg_detector = Leg_detector.Leg_detector(is_zmq=True)
        self.driver = cd.ControlDriver()
        self.FFL = FrontFollow.FFL(self.camera,self.leg_detector,self.driver,self.infrared_sensor,self.Softskin)
        self.SSL = SSL()

        self.health_state = True


        self.VoiceMenu = VoiceMenu()
        self.VoiceMenuEvent = threading.Event()

        # threading
        self.thread_Leg = threading.Thread(target=self.leg_detector.zmq_scan, args=(False, True))
        self.thread_CD = threading.Thread(target=self.driver.control_part, args=())
        self.thread_Infrared = threading.Thread(target=self.infrared_sensor.read_data, args=())
        self.thread_Softskin = threading.Thread(target=self.Softskin.read_and_record, args=())

        # threading flags and event control
        self.mainEvent = threading.Event()
        self.mainRestartEvent = threading.Event()
        self.is_SSL_pass = False

        # state information


    def start_sensor(self):
        # self.thread_CD.start()
        self.thread_Infrared.start()
        self.thread_Softskin.start()
        self.thread_Leg.start()

    def start_manual_mode(self):
        self.driver.stopMotor()
        self.FFL.FFLevent.clear()
        self.SSL.SSLEvent.clear()
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
        self.SSL.SSLEvent.clear()

    def restart(self,is_SSL_bypass:bool=False):
        self.mainRestartEvent.set()
        if is_SSL_bypass:
            self.is_SSL_pass = False
            self.SSL.SSLEvent.set()
        else:
            self.is_SSL_pass = True
            self.SSL.SSLEvent.clear()
        self.mainEvent.set()
        self.SSL.SSLEvent.clear()
        # self.is_SSL_pass = True

    def main_procedure(self):
        """this is the main procedure of different threads"""
        # start the threads
        self.start_sensor()
        # start the FFL but with event clear
        self.FFL.FFLevent.clear()
        self.FFL.start_FFL()
        # start the SSL but with event clear
        self.SSL.SSLEvent.clear()
        self.SSL.startSSL()
        # start the manual mode
        while True:
            print("0")
            self.mainEvent.wait()
            print("1")
            try:
                if not self.is_SSL_pass:
                    print("2")
                    # start the SSL first
                    self.SSL.SSLEvent.set()
                    # if some one press the softskin stop the SSL
                    while True:
                        if self.Softskin.max_pressure > 30:
                            self.SSL.SSLEvent.clear()
                            break
                        time.sleep(0.1)
                    print("3")
                # self.Softskin.skin_unlock_event.wait()
                # start the FFL
                self.FFL.TurnOnDriver()
                self.FFL.FFLevent.set()
                print("4")
                # restart the main procedure
                self.mainRestartEvent.wait()
                self.FFL.TurnOffDriver()
                self.mainRestartEvent.clear()
            except:
                pass



if __name__ == "__main__":
    mp = MainProgramme()
    thread_main = threading.Thread(target=mp.main_procedure, args=())
    thread_main.start()
    time.sleep(1)
    mp.mainEvent.set()
    time.sleep(10)
    print("stop everything!")
    mp.stop_main_procedure()
    print("finished!")
    time.sleep(3)
    print("restart!")
    mp.restart(True)