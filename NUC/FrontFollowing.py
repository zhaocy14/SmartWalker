import numpy as np
import os,sys
pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)
data_path = os.path.abspath(
    os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".."  +
    os.path.sep + "data")
import time
import threading

from Sensors import IRCamera, IMU
from Preprocessing import Leg_detector
from Driver import ControlOdometryDriver as cd
from Network import FrontFollowingNetwork as FFL

"""portal num"""

camera_portal = '/dev/ttyUSB0'
lidar_portal = '/dev/ttyUSB3'
# IMU_walker_portal = '/dev/ttyUSB0'
IMU_human_portal = '/dev/ttyUSB1'
# IMU_left_leg_portal = '/dev/ttyUSB6'
# IMU_right_leg_portal = '/dev/ttyUSB3'
# IMU_human_portal = '/dev/ttyUSB5'
# IMU_left_leg_portal = '/dev/ttyUSB6'
# IMU_right_leg_portal = '/dev/ttyUSB7'


Camera = IRCamera.IRCamera()
LD = Leg_detector.Leg_detector(lidar_portal)
CD = cd.ControlDriver(record_mode=True, left_right=0)
win_width = 10
FrontFollowingModel = FFL.FrontFollowing_Model(win_width=win_width)
weight_path = "./checkpoints_combine/Combine"
FrontFollowingModel.combine_net.load_weights(weight_path)

# IMU_walker = IMU.IMU(name="walker")
# IMU_walker.open_serial(IMU_walker_portal)
# IMU_right_leg = IMU.IMU(name="right_leg")
# IMU_right_leg.open_serial(IMU_right_leg_portal)
# IMU_left_leg = IMU.IMU(name="left_leg")
# IMU_left_leg.open_serial(IMU_left_leg_portal)

IMU_human = IMU.IMU(name="human")
IMU_human.open_serial(IMU_human_portal)

# IMU_human = IMU.IMU(name="human")
# IMU_human.open_serial(IMU_human_portal)

"""recording output"""
file_path = os.path.abspath(data_path+os.path.sep+"output.txt")

def position_calculation(left_leg: np.ndarray, right_leg: np.ndarray,
                         position_buffer: np.ndarray, weight_array: np.ndarray):
    """buffer used to average the position information with special weight
    weight position is a 1 X buffer_length matrix to decide the weight"""
    human_position = (left_leg + right_leg) / 2
    new_buffer = np.copy(position_buffer)
    new_buffer[0:new_buffer.shape[0] - 1, :] = position_buffer[1:position_buffer.shape[0], :]
    new_buffer[-1, 0] = left_leg[0]
    new_buffer[-1, 1] = left_leg[1]
    new_buffer[-1, 2] = right_leg[0]
    new_buffer[-1, 3] = right_leg[1]
    new_buffer[-1, 4] = human_position[0]
    new_buffer[-1, 5] = human_position[1]
    current_position = np.matmul(weight_array, new_buffer)[0]
    return current_position, new_buffer



def main_FFL(CD: cd.ControlDriver, LD: Leg_detector.Leg_detector, IR: IRCamera.IRCamera, FFL_Model:FFL.FrontFollowing_Model, file_path, IMU:IMU.IMU):
    # weight buffer for lidar detection
    position_buffer_length = 3
    position_buffer = np.zeros((position_buffer_length, 6))
    weight_array = np.array((range(1, position_buffer_length + 1))).reshape((1, 3))
    weight_array = weight_array / weight_array.sum()
    CD.speed = 0
    CD.omega = 0
    CD.radius = 0
    # walker rear wheel distance = 56

    # data buffer for neural network
    max_ir = 40
    min_ir = 10
    ir_data_width = 768
    additional_data_width = 4
    buffer_length = win_width
    buffer = np.zeros((buffer_length * (ir_data_width + additional_data_width), 1))

    file_record = open(file_path,'w')

    while True:
        IR.get_irdata_once()
        if len(IR.temperature) == 768:
            # update buffer and predict
            normalized_temperature = np.array(IR.temperature).reshape((ir_data_width, 1))
            normalized_temperature = (normalized_temperature - min_ir) / (max_ir - min_ir)
            buffer[0:(buffer_length - 1) * ir_data_width, 0] = buffer[ir_data_width:buffer_length * ir_data_width, 0]
            buffer[(buffer_length - 1) * ir_data_width:buffer_length * ir_data_width] = normalized_temperature
            """additional part start index"""
            PART2 = buffer_length * ir_data_width
            additional_data = [LD.left_leg[0], LD.left_leg[1], LD.right_leg[0], LD.right_leg[1]]
            additional_data = np.array(additional_data) / 40 + 0.4
            additional_data = np.reshape(additional_data, (additional_data.shape[0], 1))
            buffer[PART2:PART2 + (buffer_length - 1) * additional_data_width, 0] = \
                buffer[PART2 + additional_data_width:PART2 + buffer_length * additional_data_width, 0]
            buffer[PART2 + (buffer_length - 1) * additional_data_width:PART2 + buffer_length * additional_data_width] = \
                additional_data

            buffer[PART2:PART2 + buffer_length * additional_data_width, 0] = buffer[
                                                                             PART2:PART2 + buffer_length * additional_data_width,
                                                                             0]
            predict_buffer = buffer.reshape((-1, buffer_length * (ir_data_width + additional_data_width), 1))
            result = FFL_Model.combine_net.predict(predict_buffer)
            max_possibility = result.max()
            action_label = np.unravel_index(np.argmax(result), result.shape)[1]
            current_left_leg = LD.left_leg
            current_right_leg = LD.right_leg
            current_position, position_buffer = position_calculation(current_left_leg, current_right_leg,
                                                                     position_buffer, weight_array)
            max_boundary=14.5   #left max value
            min_boundary=-14   #right max value
            forward_boundry = 8
            backward_boundry = -5
            center_left_boundry = 1   #change gwz
            center_right_boundry = 0.3
            left_boundry = 8.5   #change gwz
            right_boundry = -7
            if backward_boundry > current_position[4] > -40:
                CD.speed = -0.1
                CD.omega = 0
                CD.radius = 0
                str1 = "backward"
            elif current_position[4] > forward_boundry:
                if current_position[5] > center_left_boundry \
                        and current_position[0] > current_position[2] \
                        and current_position[1] > left_boundry :
                          # and action_label==2 :
                    CD.speed = 0
                    radius = 30+abs(50*(max_boundary-current_position[1])/(max_boundary-left_boundry))
                    if radius < 50 :
                        radius = 50
                    CD.radius = radius
                    CD.omega = 10/CD.radius
                    str1 = "left"
                    time.sleep(0.1)
                elif current_position[5] < center_right_boundry \
                        and current_position[2] > current_position[0] \
                        and current_position[3] < right_boundry :
                        # and action_label== 3 :
                    CD.speed = 0
                    radius = 30+abs(50*(current_position[3]-min_boundary)/(right_boundry-min_boundary))
                    if radius < 50 :
                      radius = 50
                    CD.radius = radius
                    CD.omega = -10/CD.radius
                    str1 = "right"
                    time.sleep(0.1)
                else:
                    CD.speed = 0.1
                    CD.omega = 0
                    CD.radius = 0
                    str1 = "forward"
            elif  action_label== 4 :
                CD.speed = 0
                radius = abs(20*(center_left_boundry-current_position[1])/(max_boundary-center_left_boundry))
                if radius < 10:
                    radius = 10
                CD.radius = 0
                CD.omega = 0.2
                str1 = "left in space"
                time.sleep(0.1)
            elif  action_label== 5:
                CD.speed = 0
                radius = abs(20*(current_position[3]-min_boundary)/(center_left_boundry-min_boundary))
                if radius < 10 :
                    radius = 10
                CD.radius = 0
                CD.omega = -0.2
                str1 = "right in space"
                time.sleep(0.1)
            else:
                CD.speed=0
                CD.omega=0
                CD.radius = 0
                str1 = "stop"
            print("\rleft leg:%.2f,%.2f  right:%.2f,%.2f  human:%.2f,%.2f choice:%s,%.2f,%.2f,%.2f "
                 %(current_position[0], current_position[1], current_position[2],
                   current_position[3], current_position[4], current_position[5],str1,CD.speed,CD.omega,CD.radius),end="")

            # record.append(str1)

            record = [action_label] + current_position.tolist() + list(IMU.a) + list(IMU.w) + list(IMU.Angle)
            file_record.write(str1+" "+str(record)+"\n")
            file_record.flush()


thread_leg = threading.Thread(target=LD.scan_procedure, args=(False,True,))
thread_cd = threading.Thread(target=CD.control_part, args=())
thread_main = threading.Thread(target=main_FFL, args=(CD, LD, Camera, FrontFollowingModel,file_path,IMU_human))
# thread_IMU_walker = threading.Thread(target=IMU_walker.read_record,args=())
thread_IMU_human = threading.Thread(target=IMU_human.read_record,args=())

thread_IMU_human = threading.Thread(target=IMU_human.read_record,args=())

# thread_IMU_left = threading.Thread(target=IMU_left_leg.read_record,args=())
# thread_IMU_right = threading.Thread(target=IMU_right_leg.read_record,args=())



thread_leg.start()
time.sleep(3)
thread_cd.start()
thread_main.start()
# thread_IMU_human.start()
# thread_IMU_walker.start()
thread_IMU_human.start()
# thread_IMU_walker.start()
# thread_IMU_left.start()
# thread_IMU_right.start()