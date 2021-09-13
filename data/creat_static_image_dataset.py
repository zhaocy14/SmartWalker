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
ir_data = get_data(direction_ir_data)[:,1:769]
print("ir",ir_data.shape)


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

for i in range(ir_data.shape[0]):
    ir_data[i,0:ir_data.shape[1]] = filter(normalization(ir_data[i,0:ir_data.shape[1]]))

label = np.ones((ir_data.shape[0],1))


"""0:still, 1:forward, 2:left turn, 3:right turn, 4:yuandi left 5:yuandi right"""
intention_class = 5
label = label*intention_class


s_train_data_path = os.path.abspath(father_path + os.path.sep + "s"+str(ir_data.shape[0])+"_data.txt")
np.savetxt(s_train_data_path,ir_data,fmt="%.3f")
s_train_label_path = os.path.abspath(father_path + os.path.sep + "s"+str(ir_data.shape[0])+"_label.txt")
np.savetxt(s_train_label_path,label,fmt="%d")
