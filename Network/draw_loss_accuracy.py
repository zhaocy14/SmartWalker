import numpy as np
import matplotlib as plt
import os,sys

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


file_name = "combine_curve.txt" #   tendency_curve.txt    current_curve.txt
pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd))
curve_path = os.path.abspath(father_path + os.path.sep + file_name)

curve = get_data(curve_path)
print(curve.shape)