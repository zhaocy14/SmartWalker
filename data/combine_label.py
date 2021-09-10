import sys, os
# import time
import numpy as np
import math
import matplotlib.pyplot as pyplot
import cv2 as cv

pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd))
# sys.path.append(father_path)
# print(father_path)

def get_data(direction):
    # 原始信息获取
    file = open(direction)
    list_ir_data = file.readlines()
    lists = []
    for lines in list_ir_data:
        lines = lines.strip("\n")
        lines = lines.strip('[')
        lines = lines.strip(']')
        lines = lines.split(", ")
        lists.append(lines)
    file.close()
    array_data = np.array(lists)
    rows_data = array_data.shape[0]
    columns_data = array_data.shape[1]
    data = np.zeros((rows_data, columns_data))
    for i in range(rows_data):
        for j in range(columns_data):
            data[i][j] = float(array_data[i][j])
    return data

"""load the data"""

direction_ir_data = os.path.abspath(father_path + os.path.sep + "ir_data.txt")
ir_data = get_data(direction_ir_data)
print("ir",ir_data.shape)

# direction_softskin = "./Record_data/data/softskin.txt"
# softskin_data = get_data(direction_softskin)
# print(softskin_data.shape)


direction_IMU_walker = os.path.abspath(father_path + os.path.sep + "IMU.txt")
walker_IMU_data = get_data(direction_IMU_walker)
print("IMU",walker_IMU_data.shape)

direction_driver = os.path.abspath(father_path + os.path.sep + "driver.txt")
driver_data = get_data(direction_driver)
print("driver",driver_data.shape)

direction_leg  = os.path.abspath(father_path + os.path.sep + "leg.txt")
leg_data = get_data(direction_leg)
print("leg",leg_data.shape)

"""以最低更新频率的irdata为基准，用时间帧对准其它数据"""
def select_data(ir_data, target_data):
    remain_data = np.zeros((ir_data.shape[0],target_data.shape[1]-1))
    j = 0
    for i in range(ir_data.shape[0]):
        time_error = 100
        while abs(ir_data[i,0] - target_data[j,0]) < time_error:
            time_error = abs(ir_data[i,0] - target_data[j,0])
            j += 1
            if j >= target_data.shape[0]:
                j -= 1
                break
            # print(time_error,i,j)
        remain_data[i, :] = target_data[j,1:target_data.shape[1]]
    return remain_data

"""合并两个数据集"""
def combine_data_from_two_dataset(data1,data2):
    row_1, col_1 = data1.shape
    row_2, col_2 = data2.shape
    data_combine = np.zeros((row_1,col_1+col_2-1))
    data_combine[:,0:col_1] = data1
    data_combine[:,col_1:col_1+col_2-1] = select_data(data1,data2)
    return data_combine

data_combine = combine_data_from_two_dataset(ir_data,walker_IMU_data)
data_combine = combine_data_from_two_dataset(data_combine,leg_data)
data_combine = combine_data_from_two_dataset(data_combine,driver_data)
# print(data_combine.shape)

"""Labeling"""
"""Load the combined data"""
data_combine_cp = np.copy(data_combine)
ir_data = data_combine_cp[:, 1:769]
walker_IMU = data_combine_cp[:, 769:778]
leg = data_combine_cp[:, 778:782]
# leg[:,1] = leg[:,0] / 40
# leg[:,3] = leg[:,2] / 40
# leg[:,0] = (leg[:,1] + 20) / 65
# leg[:,2] = (leg[:,2] + 20) / 65
driver = data_combine[:,782:791]
left_wheel = driver[:,3]
right_wheel = driver[:,4]
walker_orientation = walker_IMU[:,-1]


def normalization(img,binarizaion:bool=False):
    """according to an average value of the image to decide the threshold"""
    new_img = np.copy(img)
    if len(new_img.shape) != 0:
        if binarizaion:
            threshold = max(new_img.mean() + 1.4, 23)
            new_img[new_img < threshold] = 0
            new_img[new_img >= threshold] = 1
        else:
            new_img = (new_img-10)/(40-10)
    return new_img

def filter(img,binarization:bool=False):
    img_new = np.copy(img)
    if binarization:
        img_new = img_new.reshape((24,32))
        filter_kernel = np.ones((2, 2)) / 4
        """other filters"""
        # filter_kernel = np.array([[1,1,1],[1,1,1],[1,1,1]])/10
        # filter_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        for j in range(1):
            img_new = cv.filter2D(img_new, -1, filter_kernel)
        img_new = img_new.flatten()
    else:
        pass
    return img_new

sum_max = 0
sum_min = 0
for i in range(ir_data.shape[0]):
    ir_data[i,0:ir_data.shape[1]] = filter(normalization(ir_data[i,0:ir_data.shape[1]]))


"""Adjust the orientation: 所有角度减去最初的角度，这样后面所有的角度都是相对起点的角度而不是地磁角度"""
def adjust_orientation(orientation):
    average_len = 5
    average_orientation = sum(orientation[0:average_len])/average_len
    # print(average_orientation)
    orientation = orientation - average_orientation
    for i in range(orientation.shape[0]):
        if orientation[i] < -180:
            orientation[i] += 360
        if orientation[i] > 180:
            orientation[i] -= 360
    return orientation

walker_orientation = adjust_orientation(walker_orientation)


"""计算每个采样点前后角度变化，而不是同时刻两者的角度差"""
def get_orientation_gradient(orientation):
    gradient = np.zeros((orientation.shape))
    for i in range(gradient.shape[0]):
        if i == 0:
            continue
        elif i < gradient.shape[0] - 1:
            gradient[i] = orientation[i] - orientation[i-1]
            if gradient[i] <= -180:
                gradient[i] += 360
            if gradient[i] >= 180:
                gradient[i] -= 360
        else:
            gradient[i] = gradient[i-1]

    return gradient


walker_gradient = get_orientation_gradient(walker_orientation)


def create_label(gradient, walker_omega, omega_flag, positive_gradient_flag, negative_gradient_flag, left_wheel, right_wheel):
    label = np.copy(gradient)
    for i in range(gradient.shape[0]):
        # 原地
        if abs(max(walker_omega[i,:])) <= omega_flag:
            label[i] = 0
        # 左转
        elif gradient[i] > positive_gradient_flag:
            if left_wheel[i] * right_wheel[i] >= 0:
                # 前行左转
                label[i] = 2
            else:
                # 原地左转
                label[i] = 4
        # 右转
        elif gradient[i] < negative_gradient_flag:
            if left_wheel[i] * right_wheel[i] >= 0:
                # 前行右转
                label[i] = 3
            else:
                # 原地右转
                label[i] = 5
        else:
            if left_wheel[i] < 0 and right_wheel[i] < 0:
                # 前行
                label[i] = 1
            else:
                # 后退 新的数据没有收集后退，如果录到相关的，改为原地
                label[i] = 0
    return label

walker_omega = walker_IMU[:,3:6]

label_gradient = create_label(walker_gradient, walker_omega, omega_flag=0.2, positive_gradient_flag=1.8,
                              negative_gradient_flag=-1.8, left_wheel=left_wheel, right_wheel=right_wheel)

"""去除某些突变值"""
def filter_label(label,width):
    filtered_label = np.copy(label)
    for i in range(width,filtered_label.shape[0]-width):
        if label[i-width] == label[i+width]:
            filtered_label[i] = label[i-width]
    return filtered_label


label_gradient = filter_label(label_gradient,1)
label_gradient = filter_label(label_gradient,2)

# print(ir_data.shape,label_gradient.shape)
o_train_data_path = os.path.abspath(father_path + os.path.sep + "o" + str(ir_data.shape[0])+"_data.txt")
np.savetxt(o_train_data_path,ir_data,fmt="%.3f")
o_train_label_path = os.path.abspath(father_path + os.path.sep + "o" + str(label_gradient.shape[0])+"_label.txt")
np.savetxt(o_train_label_path,label_gradient,fmt="%d")

"""Calculate the number/proportion of different movements"""
number = np.zeros((2,7))
for i in range(label_gradient.shape[0]):
    number[0,int(label_gradient[i])] += 1
data_num = sum(number[0,:])
for i in range(7):
    number[1,i] = number[0,i]/data_num*100

print("Still: %d,  Forward: %d, Left: %d, Right: %d, Left_Still: %d, Right_Still: %d, Backward: %d"
      % (number[0,0],number[0,1], number[0,2],  number[0,3], number[0,4], number[0,5], number[0,6]))
print("Still: %.2f,  Forward: %.2f, Left: %.2f, Right: %.2f, Left_Still: %.2f, Right_Still: %.2f, Backward: %.2f"
      % (number[1,0],number[1,1], number[1,2],  number[1,3], number[1,4], number[1,5], number[1,6]))

"""Generate the training data with label"""
data_with_label_gradient = np.c_[label_gradient, ir_data, leg]


"""拼接相邻数据生成时间序列数据"""
train_data = np.copy(data_with_label_gradient)

original_label = train_data[:, 0]
original_data = train_data[:, 1:train_data.shape[1]]

ir_data_width = 768
leg_data_width = 4
# softskin_width = 32
softskin_width = leg_data_width
# max_ir = original_data[:,0:768].max()
# max_sk = original_data[:,768:800].max()

"""计算合并后的尺寸，用作确定LSTM的数据量"""
"""win_width确定模型的帧数"""
win_width = 10
win_width = win_width
step_length = 1
data_num = int((train_data.shape[0] - win_width-1) / step_length + 1)
concatenate_data = np.zeros((data_num, original_data.shape[1] * win_width))

for i in range(data_num):
    for j in range(win_width):
        concatenate_data[i, j * ir_data_width:(j + 1) * ir_data_width] = original_data[i + j, 0:ir_data_width]
        softskin_start_position = win_width * ir_data_width
        concatenate_data[i, softskin_start_position + j * softskin_width: softskin_start_position + (
                j + 1) * softskin_width] = original_data[i + j, ir_data_width:ir_data_width + softskin_width]

# max_ir = concatenate_data[:,0:win_width*ir_data_width].max()
max_ir = 55
min_ir = 10
# max_sk = concatenate_data[:, win_width * ir_data_width:concatenate_data.shape[1]].max()
# print(max_sk)
max_sk = 100

# concatenate_data[:, 0:win_width * ir_data_width] = (concatenate_data[:, 0:win_width * ir_data_width]-min_ir) / (max_ir-min_ir)

"""normalize with skin max pressure"""
# concatenate_data[:, win_width * ir_data_width:concatenate_data.shape[1]] = concatenate_data[:,
#                                                                            win_width * ir_data_width:
#                                                                            concatenate_data.shape[1]] / max_sk

"""把softskin的数据置0，表示手离开"""
concatenate_data[:, win_width * ir_data_width:concatenate_data.shape[1]] = (concatenate_data[:, win_width * ir_data_width:concatenate_data.shape[1]])/40+0.4
# print(concatenate_data[:, win_width * ir_data_width:concatenate_data.shape[1]].max())
# print(concatenate_data[:, win_width * ir_data_width:concatenate_data.shape[1]].min())

"""打上label"""
concatenate_label = np.zeros((concatenate_data.shape[0], 1))
for i in range(concatenate_label.shape[0]):
    concatenate_label[i, 0] = original_label[i + win_width-1]


concatenate_data_path = os.path.abspath(father_path + os.path.sep + "t" + str(concatenate_data.shape[0])+"_data.txt")
np.savetxt(concatenate_data_path,concatenate_data,fmt="%.3f")

concatenate_label_path = os.path.abspath(father_path + os.path.sep + "t" + str(concatenate_data.shape[0])+"_label.txt")
np.savetxt(concatenate_label_path,concatenate_label,fmt="%d")
print("data shape:",concatenate_data.shape)
print("label shape:",concatenate_label.shape)