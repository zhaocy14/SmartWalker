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
from Network import resnet

class Conv_part(keras.Model):

    def __init__(self,filter_unit:int=10):
        super().__init__()
        self.layer1  = keras.layers.Conv2D(filters=filter_unit, kernel_size=3, strides=1, activation="relu",
                                padding="SAME")
        self.layer1  = keras.layers.Conv2D(filters=filter_unit, kernel_size=3, strides=1, activation="relu",
                                padding="SAME")
        self.layer1  = keras.layers.Conv2D(filters=filter_unit, kernel_size=3, strides=1, activation="relu",
                                padding="SAME")
        self.layer2 = keras.layers.MaxPool2D(pool_size=3,strides=2)

    def call(self,inputs):
        y = self.layer1(inputs)
        y = self.layer2(y)
        return y

class Skin_part(keras.Model):

    def __init__(self,input_shape):
        super().__init__()
        self.softskin_width = 128
        self.layer_r = keras.layers.Reshape(input_shape=(input_shape), target_shape=(1, self.softskin_width))
        self.layer_1 = keras.layers.Dense(self.dense_unit, activation="relu")
        self.layer_2 = keras.layers.Dense(self.dense_unit, activation="relu")

    def call(self,input):
        y = self.layer_r(input)
        y = self.layer_1(y)
        y = self.layer_2(y)
        return y


class FrontFollowing_Model(object):

    def __init__(self, win_width: int = 10, is_skin_input:bool=False, is_multiple_output:bool=False):
        super().__init__()
        """data shape part"""
        self.win_width = win_width-1
        self.ir_data_width = 768
        self.softskin_width = 32
        self.leg_width = 4
        """network parameter"""
        self.dense_unit = 10
        self.CNN_filter_unit = 10
        self.show_summary = False
        self.is_multiple_output = is_multiple_output
        self.is_skin_input = is_skin_input

        """network building"""
        self.tendency_ir_part = Conv_part(self.CNN_filter_unit)
        self.current_ir_part = Conv_part(self.CNN_filter_unit)
        # self.ir_part = resnet.get_model("resnet34")
        # self.skin_part = Skin_part()
        self.tendency_net = self.create_tendency_net()
        self.current_net = self.create_current_net()

    def call(self, inputs: np.ndarray) -> tf.Tensor:
        return self.model(inputs)

    def skin_dense_layers(self,inputs:tf.Tensor,input_shape:Tuple)->tf.Tensor:
        y = keras.layers.Reshape(input_shape=(input_shape), target_shape=(1, self.softskin_width))(inputs)
        y = keras.layers.Dense(self.dense_unit, activation="relu")(y)
        y = keras.layers.Dense(self.dense_unit, activation="relu")(y)
        return y

    def feature_abstraction(self, ir_data:tf.Tensor, skin_data:tf.Tensor, leg_data:tf.Tensor) -> tf.Tensor:
        if self.is_skin_input:
            """skin data as part of input"""
            for i in range(self.win_width):
                # split the tensor buffer into frames
                [ir_one_frame, ir_data] = tf.split(ir_data, [self.ir_data_width, self.ir_data_width * (self.win_width - 1 - i)], axis=1)
                [skin_one_frame, skin_data] = tf.split(skin_data, [self.softskin_width, self.softskin_width * (self.win_width - 1 - i)],
                                                       axis=1)

                # ir image feature abstraction
                output_ir = keras.layers.Reshape(input_shape=(self.ir_data_width, 1), target_shape=(32, 24, 1))(ir_one_frame)
                output_ir = self.tendency_ir_part(output_ir)
                output_ir = keras.layers.Flatten()(output_ir)

                # soft skin feature abstraction
                skin_shape = skin_one_frame.shape
                print(skin_shape)
                output_skin = keras.layers.Flatten()(self.skin_dense_layers(skin_one_frame, skin_shape))
                output_leg = keras.layers.Flatten()(leg_data)

                # feature vector concatenate
                if i == 0:
                    output_feature = keras.layers.concatenate([output_ir, output_skin,output_leg])
                else:
                    output_feature = keras.layers.concatenate([output_feature, output_ir, output_skin,output_leg])
            return output_feature

        else:
            """skin data is not included in the input"""
            for i in range(self.win_width):
                # split the tensor buffer into frames
                [ir_one_frame, ir_data] = tf.split(ir_data,
                                                   [self.ir_data_width, self.ir_data_width * (self.win_width - 1 - i)],
                                                   axis=1)

                # ir image feature abstraction
                output_ir = keras.layers.Reshape(input_shape=(self.ir_data_width, 1), target_shape=(32, 24, 1))(ir_one_frame)
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
        if self.is_skin_input:
            input_all = keras.Input(shape=((self.ir_data_width + self.softskin_width + self.leg_width) * self.win_width, 1))

            """Split the input data into two parts:ir data and softskin data"""
            [input_ir, input_softskin, input_leg] = tf.split(input_all, [self.ir_data_width * self.win_width,
                                                              self.softskin_width * self.win_width,
                                                              self.leg_width * self.win_width],
                                                  axis=1)

            output_combine = self.feature_abstraction(ir_data=input_ir, skin_data=input_softskin, leg_data=input_leg)
        else:
            input_all = keras.Input(shape=((self.ir_data_width + self.leg_width) * self.win_width, 1))
            [input_ir, input_leg] = tf.split(input_all, [self.ir_data_width * self.win_width,
                                                              self.leg_width * self.win_width],
                                                  axis=1)
            output_combine = self.feature_abstraction(ir_data=input_ir,leg_data=input_leg,skin_data=input_leg)
        output_reshape = keras.layers.Reshape(input_shape=(output_combine.shape),
                                              target_shape=(self.win_width, int(output_combine.shape[1] / self.win_width)))(
            output_combine)

        # LSTM part
        output_final = keras.layers.LSTM(32, activation='tanh')(output_reshape)
        output_final = keras.layers.Dense(128, activation='relu')(output_final)
        output_final = keras.layers.Dense(128, activation='relu')(output_final)
        output_final = keras.layers.Dense(128, activation='relu')(output_final)
        if not self.is_multiple_output:
            output_final = keras.layers.Dense(6, activation='softmax')(output_final)
            model = keras.Model(inputs=input_all, outputs=output_final)
            if self.show_summary:
                model.summary()
            return model
        else:
            actor = keras.layers.Dense(6, activation='relu')(output_final)
            critic = keras.layers.Dense(1)(output_final)
            model = keras.Model(inputs=input_all,outputs=[actor, critic])
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
        # LSTM part
        output_ir = keras.layers.Flatten()(output_ir)
        output_ir = keras.layers.Dense(128, activation='relu')(output_ir)
        output_ir = keras.layers.Dense(128, activation='relu')(output_ir)
        output_ir = keras.layers.Dense(128, activation='relu')(output_ir)
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


if __name__ == "__main__":

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    FFL_Model = FrontFollowing_Model(win_width=9)

    train_current = True
    if train_current:
        current_o_data_path = "/data/cyzhao/o_data.txt"
        current_o_data = np.loadtxt(current_o_data_path)
        current_o_label_path = "/data/cyzhao/o_label.txt"
        current_o_label = np.loadtxt(current_o_label_path)
        current_o_label = current_o_label.reshape((current_o_label.shape[0],1))

        current_s_data_path = "/data/cyzhao/s_data.txt"
        current_s_data = np.loadtxt(current_s_data_path)
        current_s_label_path = "/data/cyzhao/s_label.txt"
        current_s_label = np.loadtxt(current_s_label_path)
        current_s_label = current_s_label.reshape((current_s_label.shape[0],1))

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        FFL_Model.current_net.compile(optimizer=optimizer,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        FFL_Model.current_net.fit(current_s_data,current_s_label,batch_size=128,epochs=100,verbose=1)
        FFL_Model.current_net.save_weights('./checkpoints_s_current/Current')

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
        FFL_Model.current_net.compile(optimizer=optimizer,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        FFL_Model.current_net.fit(current_o_data, current_o_label, batch_size=128, epochs=200, verbose=1)
        FFL_Model.current_net.save_weights('./checkpoints_o_current/Current')

    else:
        tendency_data_path = "/data/cyzhao/t_data.txt"
        tendency_data = np.loadtxt(tendency_data_path)
        tendency_label_path = "/data/cyzhao/t_label.txt"
        tendency_label = np.loadtxt(tendency_label_path)
        tendency_label = tendency_label.reshape((tendency_label.shape[0],1))

        tendency_dataset = np.concatenate([tendency_label, tendency_data], axis=1)

        # reduce the amount of still data
        still_data_idx = tendency_dataset[:,0] == 0
        still_data = tendency_dataset[still_data_idx]
        np.random.shuffle(still_data)
        other_data_idx = tendency_dataset[:,0] != 0
        other_data = tendency_dataset[other_data_idx]
        tendency_dataset = np.concatenate([still_data[0:int(still_data.shape[0]/2),:], other_data],axis=0)

        # shuffle the original data
        np.random.shuffle(tendency_dataset)
        tendency_label = tendency_dataset[:, 0]
        tendency_data = tendency_dataset[:, 1:tendency_dataset.shape[1]]

        portion_train = int(tendency_data.shape[0] * 0.8)
        portion_validation = int(tendency_data.shape[0] * 0.9)

        train_data = tendency_data[0:portion_train,:]
        train_label = tendency_label[0:portion_train]
        train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1], 1))

        validation_data = tendency_data[portion_train:portion_validation, :]
        validation_label = tendency_label[portion_train:portion_validation]
        validation_data = np.reshape(validation_data, (validation_data.shape[0], validation_data.shape[1], 1))

        test_data = tendency_data[portion_validation:tendency_data.shape[0], :]
        test_label = tendency_label[portion_validation:tendency_data.shape[0]]
        test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1], 1))

        #test_data_path = "/data/cyzhao/test_data.txt"
        #test_data = np.loadtxt(test_data_path)
        #test_label_path = "/data/cyzhao/test_label.txt"
        #test_label = np.loadtxt(test_label_path)
        #test_label = test_label.reshape((test_label.shape[0], 1))
        #test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1], 1))

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
        FFL_Model.tendency_net.compile(optimizer=optimizer,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        # FFL_Model.model.load_weights("./checkpoints34/FFL34")
        #FFL_Model.model.fit(train_data, train_label, batch_size=128, epochs=100, validation_data=(validation_data, validation_label), verbose=1)
        #FFL_Model.model.save_weights("./checkpoints34/FFL34")
        while True:
            #break
            test_loss, test_acc = FFL_Model.tendency_net.evaluate(test_data, test_label, verbose=1)
            if test_acc < 0.5:
                FFL_Model.tendency_net.fit(train_data, train_label, batch_size=128, epochs=50, validation_data=(validation_data, validation_label),verbose=1)
                FFL_Model.tendency_net.save_weights('./checkpoints_tendency/Tendency')
            elif test_acc < 0.88:
                FFL_Model.tendency_net.fit(train_data, train_label, batch_size=128, epochs=10,validation_data=(validation_data,validation_label),verbose=1)
                FFL_Model.tendency_net.save_weights('./checkpoints_tendency/Tendency')
            else:
                break




