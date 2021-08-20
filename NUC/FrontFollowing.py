import numpy as np
import time
import threading

from Sensors import IRCamera
from Preprocessing import Leg_detector
from Driver import  ControlOdometryDriver as cd



"""portal num"""
# camera_portal = '/dev/ttyUSB2'
lidar_portal = '/dev/ttyUSB0'
IMU_walker_portal = '/dev/ttyUSB3'


Camera = IRCamera.IRCamera()
LD = Leg_detector.Leg_detector(lidar_portal)
CD = cd.ControlDriver()


thread_leg = threading.Thread(target=LD.scan_procedure, args=())
thread_cd = threading.Thread(target=CD.control_part,args=())



def position_calculation(left_leg:np.ndarray, right_leg:np.ndarray,
                         position_buffer:np.ndarray,weight_array:np.ndarray):
    """buffer used to average the position information with special weight
    weight position is a 1 X buffer_length matrix to decide the weight"""
    human_position = (left_leg+right_leg)/2
    new_buffer = np.copy(position_buffer)
    new_buffer[0:new_buffer.shape[0]-1,:] = position_buffer[1:position_buffer.shape[0],:]
    new_buffer[-1,:] = np.r_[left_leg,right_leg,human_position]

    current_position = np.matmul(weight_array,new_buffer[:,new_buffer.shape[1]-2:new_buffer.shape[1]-1])
    return current_position, new_buffer

def main_FFL(CD:cd.ControlDriver, LD:Leg_detector.Leg_detector):
    buffer_length = 3
    position_buffer = np.zeros((buffer_length, 6))
    weight_array = np.array((range(1, buffer_length + 1)))
    weight_array = weight_array / weight_array.sum()
    CD.speed = 0
    CD.omega = 0
    CD.radius = 0
    while True:
        time.sleep(0.2)
        current_left_leg = LD.left_leg
        current_right_leg = LD.right_leg
        current_position, position_buffer = position_calculation(current_left_leg,current_right_leg,
                                                                 position_buffer,weight_array)

        forward_boundry = 5
        backward_boundry = -5
        left_boundry = -5
        right_boundry = 5
        if current_position[0] < backward_boundry:
            CD.speed = -0.5
            CD.omega = 0
            CD.radius = 0
        elif current_position[0] > forward_boundry:
            if current_position[1] < left_boundry:
                CD.speed = 0
                CD.omega = 0.5
                CD.radius = 1
            elif current_position[1] > right_boundry:
                CD.speed = 0
                CD.omega = -0.5
                CD.radius = 1
            else:
                CD.speed = 0.5
                CD.omega = 0
                CD.radius = 0
        else:
            CD.speed = 0
            CD.omega = 0
            CD.radius = 0





