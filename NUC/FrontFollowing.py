import numpy as np
import time
import threading

from Sensors import IRCamera
from Preprocessing import Leg_detector
from Driver import  ControlOdometryDriver as cd



"""portal num"""
camera_portal = '/dev/ttyUSB3'
lidar_portal = '/dev/ttyUSB0'
IMU_walker_portal = '/dev/ttyUSB1'


Camera = IRCamera.IRCamera()
LD = Leg_detector.Leg_detector(lidar_portal)
CD = cd.ControlDriver(record_mode=False,left_right=0)






def position_calculation(left_leg:np.ndarray, right_leg:np.ndarray,
                         position_buffer:np.ndarray,weight_array:np.ndarray):
    """buffer used to average the position information with special weight
    weight position is a 1 X buffer_length matrix to decide the weight"""
    human_position = (left_leg+right_leg)/2
    new_buffer = np.copy(position_buffer)
    new_buffer[0:new_buffer.shape[0]-1,:] = position_buffer[1:position_buffer.shape[0],:]
    # print("copy",np.r_[left_leg, right_leg, human_position].shape)
    new_buffer[-1,0] = left_leg[0]
    new_buffer[-1,1] = left_leg[1]
    new_buffer[-1,2] = right_leg[0]
    new_buffer[-1,3] = right_leg[1]
    new_buffer[-1,4] = human_position[0]
    new_buffer[-1,5] = human_position[1]
    # new_buffer[-1,:] = np.c_[left_leg,right_leg,human_position]
    # print(new_buffer.shape)
    # print(new_buffer[:,new_buffer.shape[1]-2:new_buffer.shape[1]-1].shape)
    current_position = np.matmul(weight_array,new_buffer)[0]
    return current_position, new_buffer

def main_FFL(CD:cd.ControlDriver, LD:Leg_detector.Leg_detector):
    buffer_length = 3
    position_buffer = np.zeros((buffer_length, 6))
    weight_array = np.array((range(1, buffer_length + 1))).reshape((1,3))
    weight_array = weight_array / weight_array.sum()
    CD.speed = 0
    CD.omega = 0
    CD.radius = 0
    while True:
        time.sleep(0.1)
        current_left_leg = LD.left_leg
        current_right_leg = LD.right_leg
        current_position, position_buffer = position_calculation(current_left_leg,current_right_leg,
                                                                 position_buffer,weight_array)
        np.set_printoptions(precision=3,suppress=False)
        print(current_position)
        forward_boundry = 2
        backward_boundry = -8
        center_boundry = 1.2
        left_boundry = 9
        right_boundry = -9
        if backward_boundry > current_position[4] > -40:
            CD.speed = -0.1
            CD.omega = 0
            CD.radius = 0
        elif current_position[4] > forward_boundry:
            if current_position[5] > center_boundry and current_position[1] > left_boundry:
                CD.speed = 0
                CD.omega = 0.3
                CD.radius = 4
            elif current_position[5] <-center_boundry and current_position[3] < right_boundry:
                CD.speed = 0
                CD.omega = -0.3
                CD.radius = 4
            else:
                CD.speed = 0.1
                CD.omega = 0
                CD.radius = 0
        else:
            CD.speed = 0
            CD.omega = 0
            CD.radius = 0


thread_leg = threading.Thread(target=LD.scan_procedure, args=())
thread_cd = threading.Thread(target=CD.control_part,args=())
thread_main = threading.Thread(target=main_FFL,args=(CD,LD))

thread_leg.start()
time.sleep(1)
thread_cd.start()
thread_main.start()


