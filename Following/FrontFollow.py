import os, sys

pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)
data_path = os.path.abspath(
    os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".." +
    os.path.sep + "data")
import time
import threading
import numpy as np

from Sensors import IRCamera, IMU, Infrared_Sensor, softskin
from Following.Preprocessing import Leg_detector
from Following.Network import FrontFollowingNetwork
from Driver import ControlOdometryDriver


class network_data(object):
    def __init__(self, buffer_len: int = 10, ir_data_width: int = 768, leg_data_width: int = 4):
        super().__init__()
        self.buffer = np.zeros((1, buffer_len * (ir_data_width + leg_data_width), 1))
        self.buffer_len = buffer_len
        self.ir_data_width = ir_data_width
        self.leg_data_width = leg_data_width
        # for program efficiency and readity
        self.leg_start_index = self.buffer_len * self.ir_data_width
        self.total_length = self.buffer_len * (self.ir_data_width + self.leg_data_width)

    def update(self, ir_data: np.ndarray, leg_data: np.ndarray):
        # add the new frame to the rear part, the data are queue
        self.buffer[0,0:(self.buffer_len - 1) * self.ir_data_width, 0] = self.buffer[0,
                                                                       self.ir_data_width:self.buffer_len * self.ir_data_width,
                                                                       0]
        self.buffer[0, (self.buffer_len - 1) * self.ir_data_width:self.buffer_len * self.ir_data_width] = ir_data
        self.buffer[0, self.leg_start_index:self.leg_start_index + (self.buffer_len - 1) * self.leg_data_width,
        0] = self.buffer[0, self.leg_start_index + self.leg_data_width:self.total_length, 0]
        self.buffer[0, self.leg_start_index + (self.buffer_len - 1) * self.leg_data_width: self.total_length, 0] = leg_data



class FFL(object):

    def __init__(self, camera: IRCamera.IRCamera, leg_detector: Leg_detector.Leg_detector,
                 control_driver: ControlOdometryDriver.ControlDriver, infrared_sensor: Infrared_Sensor.Infrared_Sensor,
                 soft_skin: softskin.SoftSkin,
                 win_width: int = 10):

        super().__init__()
        self.Camera = camera
        self.LD = leg_detector
        self.CD = control_driver
        self.Infrared = infrared_sensor
        self.Softskin = soft_skin

        # initialize the Front-Following
        self.FFLNet = FrontFollowingNetwork.FrontFollowing_Model(win_width=win_width)
        self.load_weight()

        # this is the serial input buffer for the FFLNet
        self.data_buffer = network_data(buffer_len=win_width)

        # store the latest frame of img data and leg data
        self.leg_data = np.zeros((1, 4))  # left_leg(x,y) right_leg(x,y)
        self.leg_center_data = np.zeros((1, 6))  # left_leg(x,y) right_leg(x,y) center(x,y)
        self.ir_data_width = 768
        self.ir_data = np.zeros((1, self.ir_data_width))

        # this is the buffer for storing the leg positional information
        self.position_buffer_length = 3
        self.position_buffer = np.zeros((6, self.position_buffer_length))

        # threshold settings
        self.max_ir = 40
        self.min_ir = 10
        self.leg_threshold = 40
        self.leg_bias = 0.4

        # lidar_detection boundary
        self.forward_boundary = 8
        self.backward_boundary = -5
        self.center_left_boundary = 1
        self.center_right_boundary = 0.3
        self.left_boundary = 8.5
        self.right_boundary = -7
        self.left_max_boundary = 14.5  # left max value
        self.right_max_boundary = -14  # right max value

        # speed parameter setting
        self.fspeed = 0.1  # forward speed
        self.bspeed = -0.1  # backward speed
        self.lomega = 0.1  # left omega
        self.romega = -0.1  # right omega

        # start_thread
        # self.thread_Leg = threading.Thread(target=self.LD.scan_procedure, args=(False, True))
        self.thread_Leg = threading.Thread(target=self.LD.zmq_scan,args=(False,False))
        self.thread_CD = threading.Thread(target=self.CD.control_part, args=())
        self.thread_Infrared = threading.Thread(target=self.Infrared.read_data, args=())
        self.thread_Softskin = threading.Thread(target=self.Softskin.read_and_record, args=())
        self.thread_main = threading.Thread(target=self.main_FFL, args=(False,False))

        # thread event
        self.FFLevent = threading.Event()
        self.FFLevent.clear()

        # control driver manager
        self.isDriverChangeAllow = True

    def load_weight(self, weight_path: str = "../NUC/checkpoints_combine/Combine"):
        self.FFLNet.combine_net.load_weights(weight_path)

    def update_position_buffer(self, is_average: bool = True):
        """just use a average buffer"""
        human_position = (self.leg_data[0:2] + self.leg_data[2:4]) / 2
        self.position_buffer[:, 0:-1] = self.position_buffer[:, 1:self.position_buffer_length]
        self.position_buffer[0:4, -1] = self.leg_data
        self.position_buffer[4:6, -1] = human_position
        if is_average:
            current_position = np.mean(self.position_buffer, axis=1)
        else:
            current_position = self.position_buffer[:, -1]
        return current_position

    def TurnOnDriver(self):
        self.isDriverChangeAllow = True

    def TurnOffDriver(self):
        self.isDriverChangeAllow = False

    def updateDriver(self, Speed:float = 0, Omega:float = 0, Radius:float = 0):
        if self.isDriverChangeAllow:
            self.CD.speed = Speed
            self.CD.radius = Radius
            self.CD.omega = Omega

    def go_backward(self, show: bool = False):
        self.updateDriver(Speed=self.bspeed,Omega=0,Radius=0)
        if show:
            print("go backward!")

    def go_forward(self, show: bool = False):
        if self.LD.obstacle_array[0, 1] > 1 or self.Infrared.distance_data[1:4].min() < 25:
            self.stop(show=show)
            if show:
                print("forward but obstacle")
        else:
            self.updateDriver(Speed=self.fspeed,Omega=0,Radius=0)
        if show:
            print("go forward!")

    def turn_left(self, left_foot_position: float, is_in_place: bool = False, show: bool = False):
        if is_in_place:
            # if there is an obstacle
            if self.LD.obstacle_array[0, 0] > 1 or self.LD.obstacle_array[0, 3] > 1 or self.Infrared.distance_data[
                0] < 25:
                self.stop(show=show)
                if show:
                    print("left in space but obstacle")
            else:
                self.updateDriver(Speed=0, Omega=self.lomega, Radius=0)
                if show:
                    print("turn left in place!")
        else:
            # if there is an obstacle
            if self.LD.obstacle_array[0, 0] > 1 or self.LD.obstacle_array[0, 3] > 1 or self.Infrared.distance_data[
                0] < 25:
                self.stop(show=show)
                if show:
                    print("left but obstacle")
            else:
                radius = max(50, 30 + abs(
                    50 * (self.left_max_boundary - left_foot_position) / (self.left_max_boundary - self.left_boundary)))
                omega = 10 / radius
                self.updateDriver(Speed=0, Omega=omega, Radius=radius)
                if show:
                    print("turn left!")

    def turn_right(self, right_foot_position: float, is_in_place: bool = False, show: bool = False):
        if is_in_place:
            if self.LD.obstacle_array[0, 2] > 1 or self.LD.obstacle_array[0, 4] > 1 or self.Infrared.distance_data[
                4] < 25:
                # if there is an obstacle
                self.stop(show=show)
                if show:
                    print("right in place but obstacle")
            else:
                self.updateDriver(Speed=0, Omega=self.romega, Radius=0)
                if show:
                    print("turn right in place!")
        else:
            if self.LD.obstacle_array[0, 2] > 1 or self.LD.obstacle_array[0, 4] > 1 or self.Infrared.distance_data[
                4] < 25:
                # if there is an obstacle
                self.stop(show=show)
                if show:
                    print("right but obstacle")
            else:
                radius = max(50, 30 + abs(50 * (right_foot_position - self.right_max_boundary) / (
                            self.right_boundary - self.right_max_boundary)))
                omega = -10 / radius
                self.updateDriver(Speed=0, Omega=omega, Radius=radius)
                if show:
                    print("turn right!")

    def stop(self, show: bool = False):
        self.updateDriver(Speed=0, Omega=0, Radius=0)
        if show:
            print("stop!")



    def main_FFL(self, show: bool = True, demo:bool = False):
        # # first make sure the CD is stopped
        # self.updateDriver(Speed=0,Omega=0,Radius=0)
        while True:
            # try:
                self.FFLevent.wait()
                self.Camera.get_irdata_once(demo=demo)
                if self.Softskin.max_pressure > 120:
                    # Abnormal pressure detected
                    # run the unlock pattern
                    print("EMERGENCY!!!Runing unlock programe")
                    self.Softskin.unlock()
                    if show:
                        print("Abnormal Pressure:", self.Softskin.max_pressure)
                    continue
                if len(self.Camera.temperature) == 768:

                    # normalize data
                    self.ir_data = np.array(self.Camera.temperature).reshape((self.ir_data_width, 1))
                    self.ir_data = (self.ir_data - self.min_ir) / (self.max_ir - self.min_ir)
                    self.leg_data = np.array([self.LD.left_leg[0], self.LD.left_leg[1], self.LD.right_leg[0], self.LD.right_leg[1]]).reshape((4))
                    self.leg_data = self.leg_data / self.leg_threshold + self.leg_bias
                    # update the network input buffer
                    self.data_buffer.update(self.ir_data, self.leg_data)
                    # network_prediction
                    result = self.FFLNet.combine_net.predict(self.data_buffer.buffer)
                    max_possibility = result.max()
                    action_label = np.unravel_index(np.argmax(result), result.shape)[1]
                    # lidar_position
                    current_position = self.update_position_buffer(is_average=True)
                    # first detect backward
                    if -40 < current_position[4] < self.backward_boundary:
                        self.go_backward(show=show)
                    # then detect going forward/ turn left/right
                    elif max_possibility >= 0.3:
                        if action_label == 0:
                            self.stop(show=show)
                        elif action_label == 1:
                            self.go_forward(show=show)
                        elif action_label == 2:
                            self.turn_left(left_foot_position=current_position[1], is_in_place=False, show=show)
                        elif action_label == 3:
                            self.turn_right(right_foot_position=current_position[3], is_in_place=False, show=show)
                        elif action_label == 4:
                            self.turn_left(left_foot_position=current_position[1], is_in_place=True, show=show)
                        elif action_label == 5:
                            self.turn_right(right_foot_position=current_position[3], is_in_place=True, show=show)

                    elif current_position[4] > self.forward_boundary:
                        if current_position[5] > self.center_left_boundary \
                                and current_position[0] > current_position[2] \
                                and current_position[1] > self.left_boundary:
                            self.turn_left(left_foot_position=current_position[1], is_in_place=False, show=show)
                        elif current_position[5] < self.center_right_boundary \
                                and current_position[2] > current_position[0] \
                                and current_position[3] < self.right_boundary:
                            self.turn_right(right_foot_position=current_position[3], is_in_place=False, show=show)
                        elif action_label == 4:
                            self.turn_left(left_foot_position=current_position[1], is_in_place=True, show=show)
                        elif action_label == 5:
                            self.turn_right(right_foot_position=current_position[3], is_in_place=True, show=show)
                        else:
                            self.go_forward(show=show)
            # except Exception as error:
            #     print("Error:", error)
            #     pass

    def start_sensor(self):
        # self.thread_CD.start()
        self.thread_Infrared.start()
        self.thread_Softskin.start()
        self.thread_Leg.start()

    def start_FFL(self):
        # self.start_sensor()
        # time.sleep(1)
        self.thread_main.start()


if __name__ == "__main__":
    Camera = IRCamera.IRCamera()
    Leg_Detector = Leg_detector.Leg_detector(is_zmq=True)
    Softskin = softskin.SoftSkin()
    Control = ControlOdometryDriver.ControlDriver()
    Infrared = Infrared_Sensor.Infrared_Sensor()
    FrontFollowing_Instance = FFL(camera=Camera,leg_detector=Leg_Detector,control_driver=Control,
                                  infrared_sensor=Infrared,soft_skin=Softskin)
    FrontFollowing_Instance.start_sensor()
    FrontFollowing_Instance.start_FFL()
    FrontFollowing_Instance.FFLevent.set()
