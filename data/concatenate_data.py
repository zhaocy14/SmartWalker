import numpy as np
import os

pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd))



def concatenate_data(num_list: list = [], is_o:bool=False, is_s:bool=False):
    if len(num_list) != 0:
        if not is_o and not is_s:
            for i, number in enumerate(num_list):
                data_path = os.path.abspath(father_path + os.path.sep + str(number) + "t_data.txt")
                label_path = os.path.abspath(father_path + os.path.sep + str(number) + "t_label.txt")
                data = np.loadtxt(data_path)
                label = np.loadtxt(label_path)
                if i == 0:
                    final_data = data
                    final_label = label
                else:
                    final_data = np.concatenate([data, final_data], 0)
                    final_label = np.concatenate([label, final_label], 0)
            print(final_data.shape)

            final_data_path = os.path.abspath(father_path + os.path.sep + "t_data.txt")
            final_label_path = os.path.abspath(father_path + os.path.sep + "t_label.txt")
            """Calculate the number/proportion of different movements"""
            number = np.zeros((2, 7))
            for i in range(final_label.shape[0]):
                number[0, int(final_label[i])] += 1
            data_num = sum(number[0, :])
            for i in range(7):
                number[1, i] = number[0, i] / data_num * 100

            print("Still: %d,  Forward: %d, Left: %d, Right: %d, Left_Still: %d, Right_Still: %d, Backward: %d"
                  % (number[0, 0], number[0, 1], number[0, 2], number[0, 3], number[0, 4], number[0, 5], number[0, 6]))
            print(
                "Still: %.2f,  Forward: %.2f, Left: %.2f, Right: %.2f, Left_Still: %.2f, Right_Still: %.2f, Backward: %.2f"
                % (number[1, 0], number[1, 1], number[1, 2], number[1, 3], number[1, 4], number[1, 5], number[1, 6]))
        elif is_o:
            for i, number in enumerate(num_list):
                data_path = os.path.abspath(father_path + os.path.sep + str(number) + "o_data.txt")
                label_path = os.path.abspath(father_path + os.path.sep + str(number) + "o_label.txt")
                data = np.loadtxt(data_path)
                label = np.loadtxt(label_path)
                if i == 0:
                    final_data = data
                    final_label = label
                else:
                    final_data = np.concatenate([data, final_data], 0)
                    final_label = np.concatenate([label, final_label], 0)
            print(final_data.shape)
            final_data_path = os.path.abspath(father_path + os.path.sep + "o_data.txt")
            final_label_path = os.path.abspath(father_path + os.path.sep + "o_label.txt")
        elif is_s:
            for i, number in enumerate(num_list):
                data_path = os.path.abspath(father_path + os.path.sep + str(number) + "s_data.txt")
                label_path = os.path.abspath(father_path + os.path.sep + str(number) + "s_label.txt")
                data = np.loadtxt(data_path)
                label = np.loadtxt(label_path)
                if i == 0:
                    final_data = data
                    final_label = label
                else:
                    final_data = np.concatenate([data, final_data], 0)
                    final_label = np.concatenate([label, final_label], 0)
            print(final_data.shape)
            final_data_path = os.path.abspath(father_path + os.path.sep + "s_data.txt")
            final_label_path = os.path.abspath(father_path + os.path.sep + "s_label.txt")


        # print(final_data)
        np.savetxt(final_data_path, final_data, fmt="%.3f")
        np.savetxt(final_label_path, final_label, fmt="%d")



if __name__ == "__main__":
    num_list = [875, 1027, 1043, 1190, 1514, 1516, 1909]
    concatenate_data(num_list)

    num_list = [884, 1036, 1052, 1199, 1523, 1525, 1918]
    concatenate_data(num_list,is_o=True)


    num_list = [11, 14, 17, 24, 32, 53,65,68,80,93,95,99,102,122,128]
    concatenate_data(num_list,is_s=True)

    # num_list = [1754]
    # concatenate_data(num_list)
    #
    # num_list = [1763]
    # concatenate_data(num_list,is_o=True)

    # num_list = [1456]
    # concatenate_data(num_list,istestdata=True)