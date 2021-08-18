import sys, os
# import time
import numpy as np
import math
import matplotlib.pyplot as pyplot
from PIL import Image
import cv2
from Preprocessing import PositionalProcessing as cc
import numpy as np
from scipy import signal

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
direction_ir_data = "./Record_data/data/ir_data.txt"
ir_data = get_data(direction_ir_data)
print(ir_data.shape)

direction_driver = "./Record_data/data/driver.txt"
driver_data = get_data(direction_driver)
print(driver_data.shape)

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

data_combine = combine_data_from_two_dataset(ir_data,driver_data)
print(data_combine.shape)


ir_data = data_combine[:,1:769]
X = data_combine[:,769]
Y = data_combine[:,770]
theta = data_combine[:,771]

dx = np.array(X)
for i in range(1,X.shape[0]):
    dx[i] = X[i]-X[i-1]

dy = np.array(Y)
for i in range(1,Y.shape[0]):
    dy[i] = Y[i]-Y[i-1]

dz = np.array(dx)
dz = dz*0

dtheta = np.array(theta)
for i in range(1,theta.shape[0]):
    dtheta[i] = theta[i]-theta[i-1]
    if dtheta[i] <= -math.pi:
        dtheta[i] += math.pi
    elif dtheta[i] >= math.pi:
        dtheta[i] -= math.pi

Calibration = cc.CC_matrix()
def my_binarization(img, threshold):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] >= threshold:
                img[i][j] = 1
            else:
                img[i][j] = 0
    return img


# def my_binarization(img, win, beta=0.9):
#     if win % 2 == 0: win = win - 1
#     # 边界的均值有点麻烦
#     # 这里分别计算和和邻居数再相除
#     kern = np.ones([win, win])
#     sums = signal.correlate2d(img, kern, 'same')
#     cnts = signal.correlate2d(np.ones_like(img), kern, 'same')
#     means = sums // cnts
#     # 如果直接采用均值作为阈值，背景会变花
#     # 但是相邻背景颜色相差不大
#     # 所以乘个系数把它们过滤掉
#     img = np.where(img < means * beta, 0, 255)
#     return img

def my_chazhi(img):
    img_new = np.copy(img)
    for i in range(1,img.shape[0]-1):
        for j in range(1,img.shape[1]-1):
            img_new[i][j] = img[i-1][j-1] + img[i-1][j] + img[i-1][j+1] + img[i][j-1] + img[i][j+1] + img[i+1][j-1] + img[i+1][j] + img[i+1][j+1] + img[i][j]
            img_new[i][j] = img_new[i][j]/9

    """four lines"""
    for i in range(1, img.shape[0]-1):
        img_new[i][0] = img[i-1][0] + img[i][0] + img[i+1][0] + img[i-1][1] + img[i][1] + img[i+1][1]
        img_new[i][0] = img_new[i][0] / 6
        img_new[i][31] = img[i-1][30] + img[i][30] + img[i+1][30] + img[i-1][31] + img[i][31] + img[i+1][31]
        img_new[i][31] = img_new[i][31] / 6

    for j in range(1, img.shape[1]-1):
        img_new[0][j] = img[0][j-1] + img[0][j] + img[0][j+1] + img[1][j-1] + img[1][j] + img[1][j+1]
        img_new[0][j] = img_new[0][j] / 6
        img_new[23][j] = img[22][j-1] + img[22][j] + img[22][j+1] + img[23][j-1] + img[23][j] + img[23][j+1]
        img_new[23][j] = img_new[23][j] / 6

    img_new[0][0] = (img[0][0] + img[0][1] + img[1][0] + img[1][1])/4
    img_new[0][31] = (img[0][30] + img[0][31] + img[1][30] + img[1][31]) / 4
    img_new[23][0] = (img[23][0] + img[23][1] + img[22][0] + img[22][1]) / 4
    img_new[23][31] = (img[23][30] + img[23][31] + img[22][30] + img[22][31]) / 4

    return img_new


scope = 26
print(ir_data.shape)
for i in range(400):
    temperature = np.array(ir_data[i], np.float32).reshape(24, 32)
    """original"""
    # temperature = (temperature - temperature.min()) / (temperature.max() - temperature.min()) * 256
    """new binarize"""
    # temperature = my_binarization(temperature,5,beta=0.98)
    """binarize"""
    filter_kernel = np.ones((2, 2)) / 4
    # filter_kernel = np.array([[1,1,1],[1,1,1],[1,1,1]])/10
    # filter_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    for j in range(1):
        temperature = cv2.filter2D(temperature,-1,filter_kernel)
    threshold = max([np.mean(temperature)+1.6,23])
    temperature = my_binarization(temperature,threshold=threshold)*256

    im = Image.fromarray(temperature)
    im = im.resize((32 * scope, 24 * scope), Image.BILINEAR)
    if im.mode != "RGB":
        im = im.convert('RGB')
    path = './Record_data/img/img'+str(i+1)+'.jpg'
    im.save(path)
    # im = np.array(im)
    # cv2.imshow("Foot",im)
    # cv2.waitKey(1)


# img_list = []
# position_list = []

# for i in range(0,56):
#     img_list.append(ir_data[i])
#     dx_sum = dx[0:i].sum()
#     dy_sum = dy[0:i].sum()
#     position_list.append([dx[i],dy[i],0,dtheta[i]])

# for i in range(0,56):
#     temperature = np.array(ir_data[i], np.float32).reshape(24, 32)
#     temperature = Calibration.get_img_one_flame(temperature, position_list[i][0]*10, (position_list[i][1]*10), (position_list[i][2]),(position_list[i][3]))
#     temperature = (temperature-temperature.min())/(temperature.max()-temperature.min()) * 256
#     im = Image.fromarray(temperature)
#     im = im.resize((32 * scope, 24 * scope), Image.BILINEAR)
#     if im.mode == "F":
#         im = im.convert('RGB')
#     path = './Record_data/img/c_img'+str(i+1)+'.jpg'
#     im.save(path)
#     # im = np.array(im)
#     # cv2.imshow("Foot",im)
#     # cv2.waitKey(1)
#
# combine_img = np.array(ir_data[i], np.float32).reshape(24, 32)
# combine_img = combine_img*0
# for i in range(11,17):
#     temperature = np.array(ir_data[i], np.float32).reshape(24, 32)
#     temperature = Calibration.get_img_one_flame(temperature, position_list[i][0] * 10, (position_list[i][1] * 10),
#                                                 (position_list[i][2]), (position_list[i][3]))
#     combine_img = combine_img+temperature*(17-i)/(17-11)
# combine_img = combine_img/6
#
# threshold = 15.3
# idx = combine_img<threshold
# combine_img[idx] = 0
# idx = combine_img>=threshold
# combine_img[idx] = 1
# max_combine = combine_img.max()
# min_combine = combine_img.min()
# combine_img = (combine_img-min_combine)/(max_combine-min_combine)*256
#
#
#
#
# im = Image.fromarray(combine_img)
# im = im.resize((32 * scope, 24 * scope), Image.BILINEAR)
# if im.mode == "F":
#     im = im.convert('RGB')
# path = './Record_data/img/combine.jpg'
# im.save(path)