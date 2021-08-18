import numpy as np
import os


if __name__ == "__main__":
    """data part"""
    data0_path = "./Record_data/data/concatenate_data902.txt"
    data0 = np.loadtxt(data0_path)

    data1_path = "./Record_data/data/concatenate_data906.txt"
    data1 = np.loadtxt(data1_path)

    data2_path = "./Record_data/data/concatenate_data1468.txt"
    data2 = np.loadtxt(data2_path)

    data3_path = "./Record_data/data/concatenate_data1543.txt"
    data3 = np.loadtxt(data3_path)

    data4_path = "./Record_data/data/concatenate_data1947.txt"
    data4 = np.loadtxt(data4_path)

    data5_path = "./Record_data/data/concatenate_data1959.txt"
    data5 = np.loadtxt(data5_path)

    data6_path = "./Record_data/data/concatenate_data2193.txt"
    data6 = np.loadtxt(data6_path)


    """concatenate the data"""
    concatenate_data = np.concatenate([data0, data1, data2, data3, data4,data5,data6], 0)

    concatenate_data_path = "./Record_data/server/data.txt"
    np.savetxt(concatenate_data_path, concatenate_data, fmt="%.3f")


    """label part"""
    label0_path = "./Record_data/data/concatenate_label902.txt"
    label0 = np.loadtxt(label0_path)

    label1_path = "./Record_data/data/concatenate_label906.txt"
    label1 = np.loadtxt(label1_path)

    label2_path = "./Record_data/data/concatenate_label1468.txt"
    label2 = np.loadtxt(label2_path)

    label3_path = "./Record_data/data/concatenate_label1543.txt"
    label3 = np.loadtxt(label3_path)

    label4_path = "./Record_data/data/concatenate_label1947.txt"
    label4 = np.loadtxt(label4_path)

    label5_path = "./Record_data/data/concatenate_label1959.txt"
    label5 = np.loadtxt(label5_path)

    label6_path = "./Record_data/data/concatenate_label2193.txt"
    label6 = np.loadtxt(label6_path)

    """concatenate the label"""
    concatenate_label = np.concatenate(
        [label0, label1, label2, label3, label4,label5,label6], 0)

    concatenate_label_path = "./Record_data/server/label.txt"
    np.savetxt(concatenate_label_path, concatenate_label, fmt="%d")


    """Calculate the number/proportion of different movements"""
    number = np.zeros((2, 7))
    for i in range(concatenate_label.shape[0]):
        number[0, int(concatenate_label[i])] += 1
    data_num = sum(number[0, :])
    for i in range(7):
        number[1, i] = number[0, i] / data_num * 100

    print("Still: %d,  Forward: %d, Left: %d, Right: %d, Left_Still: %d, Right_Still: %d, Backward: %d"
          % (number[0, 0], number[0, 1], number[0, 2], number[0, 3], number[0, 4], number[0, 5], number[0, 6]))
    print("Still: %.2f,  Forward: %.2f, Left: %.2f, Right: %.2f, Left_Still: %.2f, Right_Still: %.2f, Backward: %.2f"
          % (number[1, 0], number[1, 1], number[1, 2], number[1, 3], number[1, 4], number[1, 5], number[1, 6]))

