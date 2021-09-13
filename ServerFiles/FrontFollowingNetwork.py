#-*- coding: UTF-8 -*-
import sys,os
pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from typing import Tuple
import random
from Network import resnet

class Conv_part(keras.Model):

    def __init__(self, filter_unit: int = 100):
        super().__init__()
        self.layer1 = keras.layers.Conv2D(filters=filter_unit, kernel_size=3, strides=1, activation="relu",
                                          padding="SAME")
        self.layer1_bn = keras.layers.BatchNormalization()
        self.layer2 = keras.layers.Conv2D(filters=filter_unit, kernel_size=3, strides=1, activation="relu",
                                          padding="SAME")
        self.layer2_bn = keras.layers.BatchNormalization()
        self.layer3 = keras.layers.Conv2D(filters=filter_unit, kernel_size=3, strides=1, activation="relu",
                                          padding="SAME")
        self.layer3_bn = keras.layers.Conv2D(filters=filter_unit, kernel_size=3, strides=1, activation="relu",
                                             padding="SAME")
        self.layer4 = keras.layers.MaxPool2D(pool_size=3, strides=2)

    def call(self, inputs):
        y = self.layer1(inputs)
        y = self.layer1_bn(y)
        y = self.layer2(y)
        y = self.layer2_bn(y)
        y = self.layer3(y)
        y = self.layer3_bn(y)
        y = self.layer4(y)
        return y


class Skin_part(keras.Model):

    def __init__(self, input_shape):
        super().__init__()
        self.softskin_width = 128
        self.layer_r = keras.layers.Reshape(input_shape=(input_shape), target_shape=(1, self.softskin_width))
        self.layer_1 = keras.layers.Dense(self.dense_unit, activation="relu")
        self.layer_2 = keras.layers.Dense(self.dense_unit, activation="relu")

    def call(self, input):
        y = self.layer_r(input)
        y = self.layer_1(y)
        y = self.layer_2(y)
        return y


class FrontFollowing_Model(object):

    def __init__(self, win_width: int = 10, tendency_CNN_unit:int = 10, current_CNN_unit:int = 10,
                 is_skin_input: bool = False, is_multiple_output: bool = False,show:bool=False):
        super().__init__()
        """data shape part"""
        self.win_width = win_width
        self.ir_data_width = 768
        self.softskin_width = 32
        self.leg_width = 4
        """network parameter"""
        self.dense_unit = 10
        self.CNN_filter_unit_tendency = 20
        self.CNN_filter_unit_current = 20
        self.show_summary = show
        self.is_multiple_output = is_multiple_output
        self.is_skin_input = is_skin_input

        """network building"""
        self.tendency_ir_part = Conv_part(self.CNN_filter_unit_tendency)
        self.current_ir_part = Conv_part(self.CNN_filter_unit_current)
        # self.tendency_ir_part = resnet.get_model("resnet34")
        # self.current_ir_part = resnet.get_model("resnet34")
        # self.skin_part = Skin_part()
        self.tendency_net = self.create_tendency_net()
        self.current_net = self.create_current_net()
        self.combine_net = self.creat_combine_net()

    def call(self, inputs: np.ndarray) -> tf.Tensor:
        return self.model(inputs)

    def skin_dense_layers(self, inputs: tf.Tensor, input_shape: Tuple) -> tf.Tensor:
        y = keras.layers.Reshape(input_shape=(input_shape), target_shape=(1, self.softskin_width))(inputs)
        y = keras.layers.Dense(self.dense_unit, activation="relu")(y)
        y = keras.layers.Dense(self.dense_unit, activation="relu")(y)
        return y

    def feature_abstraction(self, ir_data: tf.Tensor, skin_data: tf.Tensor, leg_data: tf.Tensor) -> tf.Tensor:
        if self.is_skin_input:
            """skin data as part of input"""
            data_num = int(ir_data.shape[1]/self.ir_data_width)
            for i in range(data_num):
                # split the tensor buffer into frames
                [ir_one_frame, ir_data] = tf.split(ir_data,
                                                   [self.ir_data_width, self.ir_data_width * (data_num - 1 - i)],
                                                   axis=1)
                [skin_one_frame, skin_data] = tf.split(skin_data, [self.softskin_width,
                                                                   self.softskin_width * (data_num - 1 - i)],
                                                       axis=1)

                # ir image feature abstraction
                output_ir = keras.layers.Reshape(input_shape=(self.ir_data_width, 1), target_shape=(32, 24, 1))(
                    ir_one_frame)
                output_ir = self.tendency_ir_part(output_ir)
                output_ir = keras.layers.Flatten()(output_ir)

                # soft skin feature abstraction
                skin_shape = skin_one_frame.shape
                output_skin = keras.layers.Flatten()(self.skin_dense_layers(skin_one_frame, skin_shape))
                output_leg = keras.layers.Flatten()(leg_data)

                # feature vector concatenate
                if i == 0:
                    output_feature = keras.layers.concatenate([output_ir, output_skin, output_leg])
                else:
                    output_feature = keras.layers.concatenate([output_feature, output_ir, output_skin, output_leg])
            return output_feature

        else:
            """skin data is not included in the input"""
            ir_data_num = int(ir_data.shape[1]/self.ir_data_width)
            for i in range(ir_data_num):
                # split the tensor buffer into frames
                [ir_one_frame, ir_data] = tf.split(ir_data,
                                                   [self.ir_data_width, self.ir_data_width * (ir_data_num - 1 - i)],
                                                   axis=1)

                # ir image feature abstraction
                output_ir = keras.layers.Reshape(input_shape=(self.ir_data_width, 1), target_shape=(32, 24, 1))(
                    ir_one_frame)
                output_ir = self.tendency_ir_part(output_ir)
                output_ir = keras.layers.Flatten()(output_ir)

                # leg feature just shortcut
                output_leg = keras.layers.Flatten()(leg_data)
                if i == 0:
                    output_feature = keras.layers.concatenate([output_ir, output_leg])
                else:
                    output_feature = keras.layers.concatenate([output_feature, output_ir, output_leg])
            return output_feature

    def create_tendency_net(self) -> tf.keras.Model:
        win_width = self.win_width - 1
        if self.is_skin_input:
            input_all = keras.Input(shape=((self.ir_data_width + self.softskin_width + self.leg_width) * win_width, 1))

            """Split the input data into two parts:ir data and softskin data"""
            [input_ir, input_softskin, input_leg] = tf.split(input_all, [self.ir_data_width * win_width,
                                                                         self.softskin_width * win_width,
                                                                         self.leg_width * win_width],
                                                             axis=1)

            output_combine = self.feature_abstraction(ir_data=input_ir, skin_data=input_softskin, leg_data=input_leg)
        else:
            input_all = keras.Input(shape=((self.ir_data_width + self.leg_width) * win_width, 1))
            [input_ir, input_leg] = tf.split(input_all, [self.ir_data_width * win_width,
                                                         self.leg_width * win_width],
                                             axis=1)
            output_combine = self.feature_abstraction(ir_data=input_ir, leg_data=input_leg, skin_data=input_leg)
        output_reshape = keras.layers.Reshape(input_shape=(output_combine.shape),
                                              target_shape=(win_width, int(output_combine.shape[1] / win_width)))(
            output_combine)

        # LSTM part
        output_tendency = keras.layers.LSTM(64, activation='tanh',kernel_regularizer=keras.regularizers.l2(0.001))(output_reshape)
        output_tendency = keras.layers.Dense(128, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001))(output_tendency)
        output_tendency = keras.layers.Dropout(0.5)(output_tendency)
        output_tendency = keras.layers.Dense(256, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001))(output_tendency)
        output_tendency = keras.layers.Dropout(0.5)(output_tendency)
        output_tendency = keras.layers.Dense(64, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001))(output_tendency)
        output_final = keras.layers.Dropout(0.5)(output_tendency)
        if not self.is_multiple_output:
            output_final = keras.layers.Dense(6, activation='softmax')(output_final)
            model = keras.Model(inputs=input_all, outputs=output_final)
            if self.show_summary:
                model.summary()
            return model
        else:
            actor = keras.layers.Dense(6, activation='relu')(output_final)
            critic = keras.layers.Dense(1)(output_final)
            model = keras.Model(inputs=input_all, outputs=[actor, critic])
            if self.show_summary:
                model.summary()
            model.compile(optimizer='RMSprop',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])
            return model

    def create_current_net(self) -> tf.keras.Model:
        input_figure = keras.Input(shape=(self.ir_data_width, 1))
        output_ir = keras.layers.Reshape(input_shape=(self.ir_data_width, 1), target_shape=(32, 24, 1))(input_figure)
        output_ir = self.current_ir_part(output_ir)

        output_ir = keras.layers.Flatten()(output_ir)
        output_ir = keras.layers.Dense(128, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001))(output_ir)
        output_ir = keras.layers.Dropout(0.5)(output_ir)
        output_ir = keras.layers.Dense(256, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001))(output_ir)
        output_ir = keras.layers.Dropout(0.5)(output_ir)
        output_ir = keras.layers.Dense(64, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001))(output_ir)
        output_ir = keras.layers.Dropout(0.5)(output_ir)
        if not self.is_multiple_output:
            output_final = keras.layers.Dense(6, activation='softmax')(output_ir)
            model = keras.Model(inputs=input_figure, outputs=output_final)
            if self.show_summary:
                model.summary()
            return model
        else:
            actor = keras.layers.Dense(6, activation='relu')(output_ir)
            critic = keras.layers.Dense(1)(output_ir)
            model = keras.Model(inputs=input_figure, outputs=[actor, critic])
            if self.show_summary:
                model.summary()
            model.compile(optimizer='RMSprop',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])
            return model

    def creat_combine_net(self) -> tf.keras.Model:
        input_all = keras.Input(shape=((self.ir_data_width + self.leg_width) * self.win_width, 1))
        [input_ir, input_leg] = tf.split(input_all, [self.ir_data_width * self.win_width,
                                                     self.leg_width * self.win_width],
                                         axis=1)
        [input_ir_tendency, input_ir_current] = tf.split(input_ir,
                                                         [self.ir_data_width * (self.win_width - 1),
                                                          self.ir_data_width],
                                                         axis=1)
        [input_leg_tendency, input_leg_current] = tf.split(input_leg,
                                                           [self.leg_width * (self.win_width - 1),
                                                            self.leg_width],
                                                           axis=1)
        # tendency part
        tendency_feture_combine = self.feature_abstraction(ir_data=input_ir_tendency, leg_data=input_leg_tendency,
                                                           skin_data=input_leg_tendency)
        tendency_feature = keras.layers.Reshape(input_shape=(tendency_feture_combine.shape),
                                                target_shape=(self.win_width-1,
                                                              int(tendency_feture_combine.shape[1] / (self.win_width-1))))(
            tendency_feture_combine)
        output_tendency = keras.layers.LSTM(64, activation='tanh',kernel_regularizer=keras.regularizers.l2(0.001))(tendency_feature)
        output_tendency = keras.layers.Dense(128, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001))(output_tendency)
        output_tendency = keras.layers.Dropout(0.5)(output_tendency)
        output_tendency = keras.layers.Dense(256, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001))(output_tendency)
        output_tendency = keras.layers.Dropout(0.5)(output_tendency)
        output_tendency = keras.layers.Dense(64, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001))(output_tendency)
        output_tendency = keras.layers.Dropout(0.5)(output_tendency)

        # current part
        output_current = keras.layers.Reshape(input_shape=(self.ir_data_width, 1), target_shape=(32, 24, 1))(
            input_ir_current)
        output_current = self.current_ir_part(output_current)

        output_current = keras.layers.Flatten()(output_current)
        output_current = keras.layers.Dense(128, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001))(output_current)
        output_current = keras.layers.Dropout(0.5)(output_current)
        output_current = keras.layers.Dense(256, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001))(output_current)
        output_current = keras.layers.Dropout(0.5)(output_current)
        output_current = keras.layers.Dense(64, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001))(output_current)
        output_current = keras.layers.Dropout(0.5)(output_current)

        # print(output_tendency.shape,output_current.shape)
        Lambda = 0.1
        output_current = tf.math.multiply(output_current,Lambda)
        output_tendency = tf.math.multiply(output_tendency,1-Lambda)
        output_final = tf.add(output_current, output_tendency)
        output_final = keras.layers.Dense(6, activation='softmax')(output_final)
        model = keras.Model(inputs=input_all, outputs=output_final)
        if self.show_summary:
            model.summary()
        return model

def setup_seed(seed):
    tf.random.set_seed(seed)
    random.seed(seed)  # 为python设置随机种子
    np.random.seed(seed)  # 为numpy设置随机种子
    tf.random.set_seed(seed)  # tf cpu fix seed

if __name__ == "__main__":

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    FFL_Model = FrontFollowing_Model(win_width=10)
    train_current = False
    setup_seed(20)


    def training(train_net_name:str="t", max_epochs:int=1000):
        if train_net_name == "c":
            """data loading"""
            current_os_data_path = "/data/cyzhao/os_data.txt"
            current_os_data = np.loadtxt(current_os_data_path)
            current_os_label_path = "/data/cyzhao/os_label.txt"
            current_os_label = np.loadtxt(current_os_label_path)
            current_os_label = current_os_label.reshape((current_os_label.shape[0],1))

            current_s_data_path = "/data/cyzhao/s_data.txt"
            current_s_data = np.loadtxt(current_s_data_path)
            current_s_label_path = "/data/cyzhao/s_label.txt"
            current_s_label = np.loadtxt(current_s_label_path)
            current_s_label = current_s_label.reshape((current_s_label.shape[0],1))

            test_data_path = "/data/cyzhao/test_o_data.txt"
            test_data = np.loadtxt(test_data_path)
            test_label_path = "/data/cyzhao/test_o_label.txt"
            test_label = np.loadtxt(test_label_path)
            test_label = test_label.reshape((test_label.shape[0],1))

            optimizer = tf.keras.optimizers.Adam(learning_rate=0.000001)
            FFL_Model.current_net.compile(optimizer=optimizer,
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])

            FFL_Model.current_net.fit(current_s_data,current_s_label,batch_size=128,epochs=20,verbose=1)
            FFL_Model.current_net.save_weights('./checkpoints_s_current/Current')

            FFL_Model.current_net.evaluate(test_data,test_label,verbose=1)

            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0000001)
            FFL_Model.current_net.compile(optimizer=optimizer,
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])

            epochs_num = 0
            max_test_acc = 0
            max_acc_epoch = 0
            file_curve_path = "./current_curve.txt"
            file_curve = open(file_curve_path,'w')
            while True:
                current_os_dataset = np.concatenate([current_os_label, current_os_data], axis=1)
                np.random.shuffle(current_os_dataset)
                current_os_label = current_os_dataset[:, 0]
                current_os_data = current_os_dataset[:, 1:current_os_dataset.shape[1]]
                current_os_label = current_os_label.reshape((current_os_label.shape[0], 1))
                print("epoch now: %d" % epochs_num)
                if epochs_num >= max_epochs:
                    break
                else:
                    history = FFL_Model.current_net.fit(current_os_data, current_os_label, batch_size=64, epochs=1, validation_data=(test_data,test_label), verbose=1)
                    epochs_num += 1
                    test_loss = history.history['val_loss'][0]
                    test_acc = history.history['val_accuracy'][0]
                    train_loss = history.history['loss'][0]
                    train_acc = history.history['accuracy'][0]
                    file_curve.write(str([train_loss, train_acc, test_loss, test_acc]) + "\n")
                    file_curve.flush()
                    if test_acc >= max_test_acc:
                        max_test_acc = test_acc
                        max_acc_epoch = epochs_num
                        FFL_Model.current_net.save_weights('./checkpoints_os_current/Current')
                    if test_acc > 0.8:
                        break
                    print("The maximum test accuracy is:%.3f, at epochs:%d"%(max_test_acc,max_acc_epoch))
            file_curve.close()
            
        elif train_net_name == "t":
            tendency_data_path = "/data/cyzhao/t_data.txt"
            tendency_data = np.loadtxt(tendency_data_path)

            frames = int(tendency_data.shape[1]/(768+4))
            ir_data = tendency_data[:,0:int((frames-1)*768)]
            leg_data = tendency_data[:,int(frames*768):int(frames*768+(frames-1)*4)]
            tendency_data = np.concatenate([ir_data,leg_data],axis=1)
            tendency_label_path = "/data/cyzhao/t_label.txt"
            tendency_label = np.loadtxt(tendency_label_path)
            tendency_label = tendency_label.reshape((tendency_label.shape[0],1))

            """train data and test data are from different dataset"""
            test_data_path = "/data/cyzhao/test_t_data.txt"
            test_data = np.loadtxt(test_data_path)
            test_label_path = "/data/cyzhao/test_t_label.txt"
            test_label = np.loadtxt(test_label_path)
            ir_data = test_data[:,0:int((frames-1)*768)]
            leg_data = test_data[:,int(frames*768):int(frames*768+(frames-1)*4)]
            test_data = np.concatenate([ir_data, leg_data], axis=1)
            test_label = test_label.reshape((test_label.shape[0], 1))
            test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1], 1))

            optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
            FFL_Model.tendency_net.compile(optimizer=optimizer,
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])

            epochs_num = 0
            max_test_acc = 0
            max_acc_epoch = 0
            file_curve_path = "./tendency_curve.txt"
            file_curve = open(file_curve_path,'w')
            while True:
                tendency_dataset = np.concatenate([tendency_label, tendency_data], axis=1)
                np.random.shuffle(tendency_dataset)
                tendency_label = tendency_dataset[:, 0]
                tendency_data = tendency_dataset[:, 1:tendency_dataset.shape[1]]
                tendency_label = tendency_label.reshape((tendency_label.shape[0], 1))
                if epochs_num >= max_epochs:
                    break
                print("epoch now: %d" % epochs_num)
                # FFL_Model.tendency_net.fit(train_data, train_label, batch_size=128, epochs=1,validation_data=(validation_data,validation_label),verbose=1)
                history = FFL_Model.tendency_net.fit(tendency_data, tendency_label, validation_data=(test_data,test_label), batch_size=64, epochs=1, verbose=1)
                test_loss = history.history['val_loss'][0]
                test_acc = history.history['val_accuracy'][0]
                train_loss = history.history['loss'][0]
                train_acc = history.history['accuracy'][0]
                file_curve.write(str([train_loss,train_acc,test_loss,test_acc])+"\n")
                file_curve.flush()
                epochs_num += 1
                if test_acc >= max_test_acc:
                    FFL_Model.tendency_net.save_weights('./checkpoints_tendency/Tendency')
                    max_test_acc = test_acc
                    max_acc_epoch = epochs_num
                if test_acc >= 0.88:
                    break
                print("The maximum test accuracy is:%.3f, at epochs:%d" % (max_test_acc, max_acc_epoch))
            file_curve.close()
            
        elif train_net_name == "a":
            all_data_path = "/data/cyzhao/t_data.txt"
            all_data = np.loadtxt(all_data_path)
            all_label_path = "/data/cyzhao/t_label.txt"
            all_label = np.loadtxt(all_label_path)
            all_label = all_label.reshape((all_label.shape[0], 1))
            print(all_data.shape)
            """train data and test data are from different dataset"""
            test_data_path = "/data/cyzhao/test_t_data.txt"
            test_data = np.loadtxt(test_data_path)
            test_label_path = "/data/cyzhao/test_t_label.txt"
            test_label = np.loadtxt(test_label_path)
            test_label = test_label.reshape((test_label.shape[0], 1))
            test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1], 1))
            print(test_data.shape)
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)
            FFL_Model.combine_net.compile(optimizer=optimizer,
                                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                           metrics=['accuracy'])

            epochs_num = 0
            max_test_acc = 0
            max_acc_epoch = 0
            file_curve_path = "./combine_curve.txt"
            file_curve = open(file_curve_path, 'w')

            while True:
                all_dataset = np.concatenate([all_label, all_data], axis=1)
                np.random.shuffle(all_dataset)
                all_label = all_dataset[:, 0]
                all_data = all_dataset[:, 1:all_dataset.shape[1]]
                all_label = all_label.reshape((all_label.shape[0], 1))
                if epochs_num >= max_epochs:
                    break
                print("epoch now: %d" % epochs_num)
                # FFL_Model.tendency_net.fit(train_data, train_label, batch_size=128, epochs=1,validation_data=(validation_data,validation_label),verbose=1)
                history = FFL_Model.combine_net.fit(all_data, all_label,
                                                     validation_data=(test_data, test_label), batch_size=100, epochs=1,
                                                     verbose=1)
                test_loss = history.history['val_loss'][0]
                test_acc = history.history['val_accuracy'][0]
                train_loss = history.history['loss'][0]
                train_acc = history.history['accuracy'][0]
                file_curve.write(str([train_loss, train_acc, test_loss, test_acc]) + "\n")
                file_curve.flush()
                epochs_num += 1
                if test_acc >= max_test_acc:
                    FFL_Model.combine_net.save_weights('./checkpoints_combine/Combine')
                    max_test_acc = test_acc
                    max_acc_epoch = epochs_num
                if test_acc >= 0.88:
                    break
                print("The maximum test accuracy is:%.3f, at epochs:%d" % (max_test_acc, max_acc_epoch))
                if epochs_num == 30:
                    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
                    FFL_Model.combine_net.compile(optimizer=optimizer,
                                                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                                  metrics=['accuracy'])
            file_curve.close()
            
        elif train_net_name == "three":
            all_data_path = "/data/cyzhao/t_data.txt"
            all_data = np.loadtxt(all_data_path)
            all_label_path = "/data/cyzhao/t_label.txt"
            all_label = np.loadtxt(all_label_path)
            all_label = all_label.reshape((all_label.shape[0], 1))
            # tendency part
            frames = int(all_data.shape[1] / (768 + 4))
            ir_data = all_data[:, 0:int((frames - 1) * 768)]
            leg_data = all_data[:, int(frames * 768):int(frames * 768 + (frames - 1) * 4)]
            tendency_data = np.concatenate([ir_data, leg_data], axis=1)
            # current part
            ir_data = all_data[:, int((frames - 1) * 768):int(frames*768)]
            # leg_data = all_data[:, int(frames * 768 + (frames - 1) * 4):int(frames*(768+4))]
            current_data = ir_data
            # label the same
            all_label_path = "/data/cyzhao/t_label.txt"
            all_label = np.loadtxt(all_label_path)
            all_label = all_label.reshape((all_label.shape[0], 1))
            
            """train data and test data are from different dataset"""
            test_data_path = "/data/cyzhao/test_t_data.txt"
            test_data = np.loadtxt(test_data_path)
            ir_data = test_data[:, 0:int((frames - 1) * 768)]
            leg_data = test_data[:, int(frames * 768):int(frames * 768 + (frames - 1) * 4)]
            test_tendency_data = np.concatenate([ir_data, leg_data], axis=1)
            ir_data = test_data[:, int((frames - 1) * 768):int(frames*768)]
            # leg_data = test_data[:, int(frames * 768 + (frames - 1) * 4):int(frames*(768+4))]
            test_current_data = ir_data

            test_label_path = "/data/cyzhao/test_t_label.txt"
            test_label = np.loadtxt(test_label_path)
            test_label = test_label.reshape((test_label.shape[0], 1))
            test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1], 1))
            test_tendency_data = np.reshape(test_tendency_data, (test_tendency_data.shape[0], test_tendency_data.shape[1], 1))
            test_current_data = np.reshape(test_current_data, (test_current_data.shape[0], test_current_data.shape[1], 1))
            print(test_data.shape)
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
            FFL_Model.combine_net.compile(optimizer=optimizer,
                                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                          metrics=['accuracy'])
            FFL_Model.current_net.compile(optimizer=optimizer,
                                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                          metrics=['accuracy'])
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.00002)
            FFL_Model.tendency_net.compile(optimizer=optimizer,
                                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                          metrics=['accuracy'])

            epochs_num = 0
            max_all_acc = 0
            max_all_epoch = max_tendency_epoch = max_current_epoch = 0
            max_tendency_acc = 0
            max_current_acc = 0
            file_curve_path = "./combine_curve.txt"
            file_curve = open(file_curve_path, 'w')
            file_tendency_path = "./tendency_curve.txt"
            file_tendency= open(file_tendency_path, 'w')
            file_current_path = "./current_curve.txt"
            file_current = open(file_current_path, 'w')
            while True:
                if epochs_num >= max_epochs:
                    break
                print("epoch now: %d" % epochs_num)
                # FFL_Model.tendency_net.fit(train_data, train_label, batch_size=128, epochs=1,validation_data=(validation_data,validation_label),verbose=1)
                history = FFL_Model.combine_net.fit(all_data, all_label,
                                                    validation_data=(test_data, test_label), batch_size=100, epochs=1,
                                                    verbose=1)
                history_t = FFL_Model.tendency_net.fit(tendency_data, all_label,
                                                    validation_data=(test_tendency_data, test_label), batch_size=100, epochs=1,
                                                    verbose=1)
                history_c = FFL_Model.current_net.fit(current_data, all_label,
                                                    validation_data=(test_current_data, test_label), batch_size=100, epochs=1,
                                                    verbose=1)
                test_loss = history.history['val_loss'][0]
                test_acc = history.history['val_accuracy'][0]
                train_loss = history.history['loss'][0]
                train_acc = history.history['accuracy'][0]
                file_curve.write(str([train_loss, train_acc, test_loss, test_acc]) + "\n")
                file_curve.flush()
                if test_acc >= max_all_acc:
                    FFL_Model.combine_net.save_weights('./checkpoints_combine/Combine')
                    max_all_acc = test_acc
                    max_all_epoch = epochs_num

                test_loss = history_t.history['val_loss'][0]
                test_acc = history_t.history['val_accuracy'][0]
                train_loss = history_t.history['loss'][0]
                train_acc = history_t.history['accuracy'][0]
                file_tendency.write(str([train_loss, train_acc, test_loss, test_acc]) + "\n")
                file_tendency.flush()
                if test_acc >= max_tendency_acc:
                    FFL_Model.tendency_net.save_weights('./checkpoints_tendency/Tendency')
                    max_tendency_acc = test_acc
                    max_tendency_epoch = epochs_num

                test_loss = history_c.history['val_loss'][0]
                test_acc = history_c.history['val_accuracy'][0]
                train_loss = history_c.history['loss'][0]
                train_acc = history_c.history['accuracy'][0]
                file_current.write(str([train_loss, train_acc, test_loss, test_acc]) + "\n")
                file_current.flush()
                if test_acc >= max_current_acc:
                    FFL_Model.current_net.save_weights('./checkpoints_os_current/Current')
                    max_current_acc = test_acc
                    max_current_epoch = epochs_num

                epochs_num += 1
                print("A:%.3f acc,%d epoch, T:%.3f acc,%d epoch, C:%.3f acc,%d epoch"%(max_all_acc,max_all_epoch,max_tendency_acc,max_tendency_epoch,max_current_acc,max_current_epoch))

            file_curve.close()
            file_tendency.flush()
            file_current.flush()

    training("three",max_epochs=1000)
