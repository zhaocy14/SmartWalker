import numpy as np
import matplotlib.pyplot as plt
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


combine_file = "combine_curve.txt" #   tendency_curve.txt    current_curve.txt
tendency_file = "tendency_curve.txt" #   tendency_curve.txt    current_curve.txt
current_file = "current_curve.txt" #   tendency_curve.txt    current_curve.txt
pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd))
combine_path = os.path.abspath(father_path + os.path.sep + combine_file)
tendency_path = os.path.abspath(father_path + os.path.sep + tendency_file)
current_path = os.path.abspath(father_path + os.path.sep + current_file)

combine_curve = get_data(combine_path)
tendency_curve = get_data(tendency_path)
current_curve = get_data(current_path)

length = 200

combine_val_loss = combine_curve[0:length,2].tolist()
tendency_val_loss = tendency_curve[0:length,2].tolist()
current_val_loss = current_curve[0:length,2].tolist()
# length = range(length)
# plt.figure(figsize=(11,8.2))
# plt.plot(length,combine_val_loss,color='#2878B5',label="FFLNet Loss")
# plt.plot(length,tendency_val_loss,color='#C82423',label="Tendency Net Loss")
# plt.plot(length,current_val_loss,color='#F8AC8C',label="Current Net Loss")
# plt.legend(prop={'family' : 'Times New Roman', 'size': 24})
# plt.xlabel('Iteration',fontdict={'family' : 'Times New Roman', 'size': 30})
# plt.ylabel('Loss',fontdict={'family' : 'Times New Roman', 'size': 30})
# plt.xticks(fontproperties = 'Times New Roman', size = 24)
# plt.yticks(fontproperties = 'Times New Roman', size = 24)
# plt.show()
#
combine_val_accuracy = (combine_curve[0:length,3]).tolist()
tendency_val_accuracy = (tendency_curve[0:length,3]).tolist()
current_val_accuracy = (current_curve[0:length,3]).tolist()
length = range(length)
plt.figure(figsize=(11,8.2))
plt.plot(length,combine_val_accuracy,color='#2878B5',label="FFLNet Accuracy")
plt.plot(length,tendency_val_accuracy,color='#C82423',label="Tendency Net Accuracy")
plt.plot(length,current_val_accuracy,color='#F8AC8C',label="Current Net Accuracy")
plt.legend(prop={'family' : 'Times New Roman', 'size': 24})
plt.xlabel('Iteration',fontdict={'family' : 'Times New Roman', 'size': 30})
plt.ylabel('Accuracy',fontdict={'family' : 'Times New Roman', 'size': 30})
plt.xticks(fontproperties = 'Times New Roman', size = 24)
plt.yticks(fontproperties = 'Times New Roman', size = 24)
plt.show()