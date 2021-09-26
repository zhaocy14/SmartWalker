import serial
import os
import numpy as np
import time
import threading
import tensorflow as tf
from Sensors import IRCamera, softskin
from Network import FrontFollowingNetwork as FFL
from Driver import ControlOdometryDriver as CD
from Preprocessing import Leg_detector
import cv2 as cv


if __name__ == "__main__":
    """portal num"""
    camera_portal = '/dev/ttyUSB1'
    lidar_portal = '/dev/ttyUSB4'
    IRCamera = IRCamera.IRCamera()
    LD = Leg_detector.Leg_detector(lidar_portal)
    cd = CD.ControlDriver(left_right=0)

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
    thread_leg = threading.Thread(target=LD.scan_procedure,args=())
    thread_leg.start()
    time.sleep(2) # wait for the start of the lidar

    thread_control_driver = threading.Thread(target=cd.control_part, args=())
    thread_control_driver.start()

    while True:

        # present_time = time.time()
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
            backward_boundry = -5
            # print(LD.center_point)
            human_position = (LD.left_leg+LD.right_leg)/2
            if backward_boundry>human_position[0]>-40:
                print("\rbackward!",end="")
                cd.speed = -0.15
                cd.omega=0
                cd.radius=0
            elif max_result == result[0, 0]:
                print("\rstill!",end="")
                cd.speed = 0
                cd.omega = 0
                cd.radius = 0
            elif max_result == result[0, 1]:
                print("\rforward!",end="")
                cd.speed = 0.15
                cd.omega = 0
                cd.radius = 0
            elif max_result == result[0, 2]:
                print("\rturn left!",end="")
                cd.speed = 0
                cd.omega = 0.2
                cd.radius = 70
            elif max_result == result[0, 3]:
                print("\rturn right!",end="")
                cd.speed = 0
                cd.omega = -0.2
                cd.radius = 70
            elif max_result == result[0, 4]:
                print("\ryuandi left",end="")
                cd.speed = 0
                cd.omega = 0.3
                cd.radius = 0
            elif max_result == result[0, 5]:
                print("\ryuandi right",end="")
                cd.speed = 0
                cd.omega = -0.3
                cd.radius = 0
            # print(1/(time.time()-present_time))
            # present_time = time.time()
    
# thread_leg = threading.Thread(target=LD.scan_procedure, args=(True,True,))
# thread_cd = threading.Thread(target=cd.control_part, args=())
# thread_main = threading.Thread(target=main_FFL, args=(cd, LD, Camera, FrontFollowingModel))


# thread_leg.start()
# time.sleep(3)
# thread_main.start()


