import serial
import os,sys
pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)
print(father_path)
# data_path = os.path.abspath(
#     os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".."  +
#     os.path.sep + "data")
import numpy as np
import time
import threading
import tensorflow as tf
from Sensors import IRCamera, softskin, Infrared_Sensor
from Driver import ControlOdometryDriver as CD
from Following.Network import FrontFollowingNetwork as FFL
from Following.Preprocessing import Leg_detector
# import FrontFollowing.Preprocessing.Leg_detector
# import FrontFollowing.Network.FrontFollowingNetwork as FFL

import cv2 as cv

class network_data(object):

    def __init__(self,buffer_len:int=10,ir_data_width:int=768,leg_data_width:int=4):
        self.buffer = np.zeros((buffer_len*(ir_data_width+leg_data_width),1))
        self.buffer_len = buffer_len
        self.ir_data_width = ir_data_width
        self.leg_data_width = leg_data_width

    def update(self,ir_data:np.ndarray,leg_data:np.ndarray):
        self.buffer[0:(self.buffer_len - 1) * self.ir_data_width, 0] = self.buffer[self.ir_data_width:self.buffer_len * self.ir_data_width, 0]
        self.buffer[(self.buffer_len - 1) * self.ir_data_width:self.buffer_len * self.ir_data_width] = ir_data




if __name__ == "__main__":
    # LD = Leg_detector.Leg_detector()
    # thread_leg = threading.Thread(target=LD.scan_procedure,args=())
    # thread_leg.start()
    LD = Leg_detector.Leg_detector(is_zmq=True)
    IRCamera = IRCamera.IRCamera()
    IRSensor = Infrared_Sensor.Infrared_Sensor(sensor_num=5)
    cd = CD.ControlDriver(left_right=0)
    skin = softskin.SoftSkin()

    # initialize the network
    win_width = 10
    tf_model = FFL.FrontFollowing_Model(win_width=win_width)
    weight_path = "./checkpoints_combine/Combine"
    tf_model.combine_net.load_weights(weight_path)

    # data buffer for network input
    max_ir = 40
    min_ir = 10
    ir_threshold = 27  # for binarization
    ir_data_width = 768
    additional_data_width = 4
    buffer_length = win_width
    buffer = np.zeros((buffer_length * (ir_data_width + additional_data_width), 1))

    # sensor data reading thread, network output thread and control thread

    thread_leg = threading.Thread(target=LD.zmq_scan,args=())
    thread_leg.start()
    thread_skin = threading.Thread(target=skin.read_and_record,args=())
    thread_skin.start()

    time.sleep(2) # wait for the start of the lidar

    thread_control_driver = threading.Thread(target=cd.control_part, args=())
    # thread_control_driver.start()
    # thread_infrared = threading.Thread(target=IRSensor.read_data, args=())
    # thread_infrared.start()


    while True:
        # present_time = time.time()
        if skin.max_pressure >= 120:
            CD.speed = CD.omega = CD.radius = 0
            IRCamera.get_irdata_once()
            print("Abnormal Pressure!")
            continue
        IRCamera.get_irdata_once()
        if len(IRCamera.temperature) == 768:
            normalized_temperature = np.array(IRCamera.temperature).reshape((ir_data_width, 1))
            normalized_temperature = (normalized_temperature-min_ir)/(max_ir-min_ir)
            buffer[0:(buffer_length - 1) * ir_data_width, 0] = buffer[ir_data_width:buffer_length * ir_data_width, 0]
            buffer[(buffer_length - 1) * ir_data_width:buffer_length * ir_data_width] = normalized_temperature
            """additional part start index"""
            PART2 = buffer_length * ir_data_width
            additional_data = [LD.left_leg[0], LD.left_leg[1], LD.right_leg[0], LD.right_leg[1]]
            additional_data = np.array(additional_data)/40+0.4
            additional_data = np.reshape(additional_data,(additional_data.shape[0],1))
            buffer[PART2:PART2 + (buffer_length - 1) * additional_data_width, 0] = \
                buffer[PART2 + additional_data_width:PART2 + buffer_length * additional_data_width, 0]
            buffer[PART2 + (buffer_length - 1) * additional_data_width:PART2 + buffer_length * additional_data_width] = \
                additional_data

            buffer[PART2:PART2 + buffer_length * additional_data_width, 0] = buffer[PART2:PART2 + buffer_length * additional_data_width, 0]

            predict_buffer = buffer.reshape((-1, buffer_length * (ir_data_width + additional_data_width), 1))
            result = tf_model.combine_net.predict(predict_buffer)
            max_result = result.max()
            action_label = np.unravel_index(np.argmax(result), result.shape)[1]
            # print(action_label)
            # print(max_result)
            backward_boundry = -15
            forward_security_boundry = 0
            still_security_boundry = -40

            # far_flag = 80
            close_flag = 25
            # print(LD.center_point)
            human_position = (LD.left_leg+LD.right_leg)/2
            if backward_boundry>human_position[0]>still_security_boundry:
                print("\rbackward!",end="")
                cd.speed = -0.1
                cd.omega=0
                cd.radius=0
            elif max_result == result[0, 0] or human_position[0] < still_security_boundry:
                print("\rstill!",end="")
                cd.speed = 0
                cd.omega = 0
                cd.radius = 0
            elif max_result == result[0, 1]:
                print("\rforward!",end="")
                if LD.obstacle_array[0, 1] > 1:
                    print("\r obstacle in forward!!!!",end="")
                    cd.speed = cd.omega = cd.radius = 0
                    continue
                if human_position[0] > forward_security_boundry:
                    cd.speed = 0.05
                    cd.omega = 0
                    cd.radius = 0
            elif max_result == result[0, 2]:
                print("\rturn left!",end="")
                cd.speed = 0
                cd.omega = 0.1
                cd.radius = 70
                if LD.obstacle_array[0,0] > 1 or LD.obstacle_array[0,3] > 1 or LD.obstacle_array[0,1]:
                    print("\r obstacle in turning left!!!!",end="")
                    # cd.radius = cd.radius * (200-IRSensor.distance_data[0])/100
                    cd.speed = cd.omega = cd.radius = 0
                    continue
            elif max_result == result[0, 3]:
                print("\rturn right!",end="")
                cd.speed = 0
                cd.omega = -0.1
                cd.radius = 70
                if LD.obstacle_array[0, 2] > 1 or LD.obstacle_array[0, 4] or LD.obstacle_array[0,1] > 1:
                    print("\r obstacle in turning right!!!!",end="")
                    # cd.radius = max(cd.radius-5,0)
                    cd.speed = cd.omega = cd.radius = 0
                    continue
            elif max_result == result[0, 4]:
                print("\ryuandi left",end="")
                cd.speed = 0
                cd.omega = 0.3
                cd.radius = 0
                if LD.obstacle_array[0,0] > 1 or LD.obstacle_array[0,3] > 1:
                    print("\r obstacle in turning left!!!!",end="")
                    cd.speed = cd.omega = cd.radius = 0
                    continue
            elif max_result == result[0, 5]:
                print("\ryuandi right",end="")
                cd.speed = 0
                cd.omega = -0.3
                cd.radius = 0
                if LD.obstacle_array[0, 2] > 1 or LD.obstacle_array[0, 4] > 1:
                    print("\r obstacle in turning right in space!!!!",end="")
                    cd.speed = cd.omega = cd.radius = 0
                    continue
            # print(1/(time.time()-present_time))
            # present_time = time.time()
    
# thread_leg = threading.Thread(target=LD.scan_procedure, args=(True,True,))
# thread_cd = threading.Thread(target=cd.control_part, args=())
# thread_main = threading.Thread(target=main_FFL, args=(cd, LD, Camera, FrontFollowingModel))


# thread_leg.start()
# time.sleep(3)
# thread_main.start()


