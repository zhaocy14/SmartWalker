import tensorflow as tf
import numpy as np
import os, sys
import time
pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)
from Network import FrontFollowingNetwork as FFL

combine_path = os.path.abspath(father_path + os.path.sep + "NUC" + os.path.sep + "checkpoints_combine" + os.path.sep +
                               "Combine")
tendency_path = os.path.abspath(father_path + os.path.sep + "NUC" + os.path.sep + "checkpoints_tendency" + os.path.sep +
                                "Tendency")
current_path = os.path.abspath(father_path + os.path.sep + "NUC" + os.path.sep + "checkpoints_os_current" + os.path.sep +
                               "Current")
# combine_path = "/data/cyzhao/checkpoints_combine/Combine"
# tendency_path = "/data/cyzhao/checkpoints_tendency/Tendency"
# current_path = "/data/cyzhao/checkpoints_os_current/Current"
win_width = 10
FFLNet = FFL.FrontFollowing_Model(win_width=win_width)
FFLNet.combine_net.load_weights(combine_path)
# FFLNet.tendency_net.load_weights(tendency_path)
# FFLNet.current_net.load_weights(current_path)


test_data_path = os.path.abspath(father_path + os.path.sep + "data" + os.path.sep + "cyzhao" + os.path.sep + "test_t_data.txt")
test_label_path = os.path.abspath(father_path + os.path.sep + "data" + os.path.sep + "cyzhao" + os.path.sep + "test_t_label.txt")
# test_data_path = "/data/cyzhao/test_t_data.txt"
# test_label_path = "/data/cyzhao/test_t_label.txt"
test_label = np.loadtxt(test_label_path)
test_data = np.loadtxt(test_data_path)

print(test_data.shape)
#
# ir_data = test_data[:, 0:int((win_width - 1) * 768)]
# leg_data = test_data[:, int(win_width * 768):int(win_width * 768 + (win_width - 1) * 4)]
# test_tendency_data = np.concatenate([ir_data, leg_data], axis=1)
#
# ir_data = test_data[:, int((win_width - 1) * 768):int(win_width*768)]
# # leg_data = test_data[:, int(frames * 768 + (frames - 1) * 4):int(frames*(768+4))]
# test_current_data = ir_data
#
#
test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1], 1))
# test_tendency_data = np.reshape(test_tendency_data, (test_tendency_data.shape[0], test_tendency_data.shape[1], 1))
# test_current_data = np.reshape(test_current_data, (test_current_data.shape[0], test_current_data.shape[1], 1))


# def get_evaluation(lambda_network:float,FFL:FFL.FrontFollowing_Model,tendency_data:np.ndarray,current_data:np.ndarray,label:np.ndarray):
#     data_num = label.shape[0]
#     final_correct_num = 0
#     for i in range(data_num):
#         print("\r testing iteration %d for lambda %.1f"%(i,lambda_network),end="")
#         tendency_test = tendency_data[i, :, :].reshape((-1, 9 * (768 + 4), 1))
#         current_test = current_data[i,:,:].reshape((-1,768,1))
#
#         result_tendency = FFL.tendency_net.predict(tendency_test)[0]
#         result_current = FFL.current_net.predict(current_test)[0]
#
#         final_test = lambda_network*result_tendency + (1-lambda_network)*result_current
#
#         action_label = np.unravel_index(np.argmax(final_test), final_test.shape)[0]
#
#         if action_label == label[i]:
#             final_correct_num += 1
#     return final_correct_num/data_num
#
# # record = np.zeros((21,2))
# # start_time = time.time()
# # for i in range(21):
# #     lambda_network = i*5/100
# #     print(lambda_network)
# #     # accuracy = get_evaluation(lambda_network, FFLNet, test_tendency_data, test_current_data, test_label)
# #     record[i,0] = lambda_network
# #     # record[i,1] = accuracy
# #     print("iteration time:%f"%(time.time()-start_time))
# #     start_time = time.time()
# # record_path = "/data/cyzhao/lambda_record.txt"
# # np.savetxt(record_path,record)

classification_matrix = np.zeros((6,6))
for i in range(test_data.shape[0]):
    # print("\r%d/%d"%(i,test_data.shape[0]),end="")
    result = FFLNet.combine_net.predict(test_data[i, :, :].reshape((-1, 10 * (768 + 4), 1)))[0]
    action_label = np.unravel_index(np.argmax(result), result.shape)[0]
    leg_data = test_data[i,-4:test_data.shape[1],:]
    # if action_label == 0:
    #     print(leg_data)
    #     if (leg_data[0,0] + leg_data[2,0])>0.65:
    #         action_label = 1
    # elif action_label == 1:
    #     print(leg_data)
    #     if (leg_data[0,0] + leg_data[2,0])<=0.7:
    #         action_label = 0
    classification_matrix[int(test_label[i]),action_label] += 1
combine_path = os.path.abspath(
    father_path + os.path.sep + "data" + os.path.sep + "classification.txt")
np.savetxt(combine_path,classification_matrix)

classification = np.loadtxt(combine_path)
print(classification)
