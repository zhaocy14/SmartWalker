import numpy as np
import os, sys
pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)
import matplotlib.pyplot as plt

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
#
combine_curve = get_data(combine_path)
tendency_curve = get_data(tendency_path)
current_curve = get_data(current_path)
# #
length = 200
#
# combine_val_loss = combine_curve[0:length,2].tolist()
# tendency_val_loss = tendency_curve[0:length,2].tolist()
# current_val_loss = current_curve[0:length,2].tolist()
# length = range(length)
# plt.figure(figsize=(12,8.7))
# plt.plot(length,combine_val_loss,color='#2878B5',label="FFLNet")
# plt.plot(length,tendency_val_loss,color='#C82423',label="Tendency$^+$")
# plt.plot(length,current_val_loss,color='#F8AC8C',label="Current$^+$")
# plt.legend(prop={'family' : 'Times New Roman', 'size': 40})
# plt.xlabel('Iteration',fontdict={'family' : 'Times New Roman', 'size': 38})
# plt.ylabel('Loss',fontdict={'family' : 'Times New Roman', 'size': 38})
# plt.xticks(fontproperties = 'Times New Roman', size = 32)
# plt.yticks(fontproperties = 'Times New Roman', size = 32)
# plt.show()

combine_val_accuracy = (combine_curve[0:length,3]).tolist()
tendency_val_accuracy = (tendency_curve[0:length,3]).tolist()
current_val_accuracy = (current_curve[0:length,3]).tolist()
length = range(length)
plt.figure(figsize=(12,8.7))
plt.plot(length,combine_val_accuracy,color='#2878B5',label="FFLNet")
plt.plot(length,tendency_val_accuracy,color='#C82423',label="Tendency$^+$")
plt.plot(length,current_val_accuracy,color='#F8AC8C',label="Current$^+$")
plt.legend(prop={'family' : 'Times New Roman', 'size': 40})
plt.xlabel('Iteration',fontdict={'family' : 'Times New Roman', 'size': 38})
plt.ylabel('Accuracy',fontdict={'family' : 'Times New Roman', 'size': 38})
plt.xticks(fontproperties = 'Times New Roman', size = 32)
plt.yticks(fontproperties = 'Times New Roman', size = 32)
plt.show()

# combine_path = os.path.abspath(
#     father_path + os.path.sep + "data" + os.path.sep + "classification.txt")
# classification = np.loadtxt(combine_path)
# print(classification)
# print(classification.sum())
# classes = ['Stop', 'Forward', 'Left Forward ', 'Right Forward', 'Spot Left', 'Spot Right']
#
# confusion_matrix = classification
# confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
# plt.figure(figsize=(11.8,8))
# plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)  # 按照像素显示出矩阵
# # plt.title('confusion_matrix')
# # plt.colorbar()
# tick_marks = np.arange(len(classes))
# plt.xticks(tick_marks, classes,rotation="25",size=20)
# plt.yticks(tick_marks, classes,size=20)
# thresh = confusion_matrix.max() / 2
# # iters = [[i,j] for i in range(len(classes)) for j in range((classes))]
# # ij配对，遍历矩阵迭代器
# iters = np.reshape([[[i, j] for j in range(6)] for i in range(6)], (confusion_matrix.size, 2))
# for i, j in iters:
#     plt.text(j, i, "%.3f"%(confusion_matrix[i, j]), horizontalalignment="center",
#              color='white' if confusion_matrix[i, j] > thresh else 'black',size=24)  # 显示对应的数字

# plt.ylabel('Real',size=20)
# plt.xlabel('Prediction',size=25)
# plt.tight_layout()
# plt.show()


