import numpy as np
import os

pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd))


def concatenate_data(num_list: list = [],istestdata:bool=False):
    if len(num_list) != 0:
        for i, number in enumerate(num_list):
            data_path = os.path.abspath(father_path + os.path.sep + str(number) + "data.txt")
            label_path = os.path.abspath(father_path + os.path.sep + str(number) + "label.txt")
            data = np.loadtxt(data_path)
            label = np.loadtxt(label_path)
            if i == 0:
                final_data = data
                final_label = label
            else:
                final_data = np.concatenate([data, final_data], 0)
                final_label = np.concatenate([label, final_label], 0)
        if not istestdata:
            final_data_path = os.path.abspath(father_path + os.path.sep + "data.txt")
            final_label_path = os.path.abspath(father_path + os.path.sep + "label.txt")
            # print(final_data)
            np.savetxt(final_data_path, final_data, fmt="%.3f")
            np.savetxt(final_label_path, final_label, fmt="%d")

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


        else:
            final_data_path = os.path.abspath(father_path + os.path.sep + "test_data.txt")
            final_label_path = os.path.abspath(father_path + os.path.sep + "test_label.txt")
            np.savetxt(final_data_path, final_data, fmt="%.3f")
            np.savetxt(final_label_path, final_label, fmt="%d")

            """Calculate the number/proportion of different movements"""
            number = np.zeros((2, 7))
            for i in range(final_label.shape[0]):
                number[0, int(final_label[i])] += 1
            data_num = sum(number[0, :])
            for i in range(7):
                number[1, i] = number[0, i] / data_num * 100

            print("Still: %d,  Forward: %d, Left: %d, Right: %d, Left_Still: %d, Right_Still: %d, Backward: %d"
                  % (
                  number[0, 0], number[0, 1], number[0, 2], number[0, 3], number[0, 4], number[0, 5], number[0, 6]))
            print(
                "Still: %.2f,  Forward: %.2f, Left: %.2f, Right: %.2f, Left_Still: %.2f, Right_Still: %.2f, Backward: %.2f"
                % (
                number[1, 0], number[1, 1], number[1, 2], number[1, 3], number[1, 4], number[1, 5], number[1, 6]))


if __name__ == "__main__":
    num_list = [902, 906, 1468, 1543, 1947, 1959, 2193]
    concatenate_data(num_list)

    num_list = [1456]
    concatenate_data(num_list,istestdata=True)