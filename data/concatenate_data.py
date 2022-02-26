import numpy as np
import os

pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd))



def concatenate_data(num_list: list = [], is_o:bool=False, is_s:bool=False,istest:bool=False):
    if len(num_list) != 0:
        if not is_o and not is_s:
            for i, number in enumerate(num_list):
                data_path = os.path.abspath(father_path + os.path.sep + "t"+str(number) + "_data.txt")
                label_path = os.path.abspath(father_path + os.path.sep + "t"+str(number) + "_label.txt")
                data = np.loadtxt(data_path)
                label = np.loadtxt(label_path)
                if i == 0:
                    final_data = data
                    final_label = label
                else:
                    final_data = np.concatenate([data, final_data], 0)
                    final_label = np.concatenate([label, final_label], 0)
            print(final_data.shape)
            if istest:
                final_data_path = os.path.abspath(father_path + os.path.sep + "test_t_data.txt")
                final_label_path = os.path.abspath(father_path + os.path.sep + "test_t_label.txt")
            else:
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
                data_path = os.path.abspath(father_path + os.path.sep + "o"+str(number) + "_data.txt")
                label_path = os.path.abspath(father_path + os.path.sep + "o"+str(number) + "_label.txt")
                data = np.loadtxt(data_path)
                label = np.loadtxt(label_path)
                if i == 0:
                    final_data = data
                    final_label = label
                else:
                    final_data = np.concatenate([data, final_data], 0)
                    final_label = np.concatenate([label, final_label], 0)
            print(final_data.shape)
            if istest:
                final_data_path = os.path.abspath(father_path + os.path.sep + "test_o_data.txt")
                final_label_path = os.path.abspath(father_path + os.path.sep + "test_o_label.txt")
            else:
                final_data_path = os.path.abspath(father_path + os.path.sep + "o_data.txt")
                final_label_path = os.path.abspath(father_path + os.path.sep + "o_label.txt")


        elif is_s:
            for i, number in enumerate(num_list):
                data_path = os.path.abspath(father_path + os.path.sep + "s"+str(number) + "_data.txt")
                label_path = os.path.abspath(father_path + os.path.sep + "s"+str(number) + "_label.txt")
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
        return final_data,final_label

def concatenate_o_s(o_data,o_label,s_data,s_label):
    final_data = np.concatenate([o_data, s_data], 0)
    final_label = np.concatenate([o_label, s_label], 0)
    final_data_path = os.path.abspath(father_path + os.path.sep + "os_data.txt")
    final_label_path = os.path.abspath(father_path + os.path.sep + "os_label.txt")
    np.savetxt(final_data_path, final_data, fmt="%.3f")
    np.savetxt(final_label_path, final_label, fmt="%d")
    return final_data,final_label

if __name__ == "__main__":
    print("start concatenate t data")
    num_list = [77, 394, 506, 617, 807, 1026, 1513, 1515,1753, 1899, 1908, 1932, 2256, 2796, 3019]
    t_data,t_label = concatenate_data(num_list)

    print("start concatenate o data")
    num_list = [87, 404, 516, 627, 817, 1036, 1523, 1525, 1763,1909, 1918, 1942, 2266, 2806, 3029]
    o_data,o_label=concatenate_data(num_list,is_o=True)

    print("start concatenate s data")
    num_list = [11, 14, 17, 24, 28, 29, 32, 36, 44, 53, 65,68,80,83,95,99,102,107,122,128,209,237,267,278,327,404]
    s_data,s_label=concatenate_data(num_list,is_s=True)

    print("start concatenate os data")
    concatenate_o_s(o_data,o_label,s_data,s_label)

    print("start concatenate test_t data")
    num_list = [874,1042,1189,1206]
    concatenate_data(num_list,istest=True)

    print("start concatenate test_o data")
    num_list = [884,1052,1199,1216]
    concatenate_data(num_list,is_o=True,istest=True)

