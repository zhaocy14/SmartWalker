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

    def __init__(self):
        super().__init__()
        self.ir_data_width = 768
        self.layer1  = keras.layers.Conv2D(filters=3, kernel_size=3, strides=1, activation="relu",
                                padding="SAME")
        self.layer2 = keras.layers.MaxPool2D(pool_size=3,strides=2)

    def call(self,inputs):
        y = self.layer1(inputs)
        y = self.layer2(y)
        return y

class FrontFollowing_Model(object):

    def __init__(self, win_width: int = 10, is_skin_input:bool=False, is_multiple_output:bool=False):
        super().__init__()
        """data shape part"""
        self.win_width = win_width
        self.ir_data_width = 768
        self.softskin_width = 32
        self.leg_width = 4
        """network parameter"""
        self.filter_num = 3
        self.kernel_size = 3
        self.dense_unit = 10
        self.show_summary = True
        self.is_multiple_output = is_multiple_output
        self.is_skin_input = is_skin_input

        """network building"""
        self.reshape_layer = keras.layers.Reshape(input_shape=(self.ir_data_width, 1), target_shape=(32, 24, 1))
        self.ir_part_0 = Conv_part()
        self.ir_part = resnet.get_model("resnet34")
        self.model = self.create_model_dynamic()

    def call(self, inputs: np.ndarray) -> tf.Tensor:
        return self.model(inputs)

    def ir_conv_layers(self,inputs:tf.Tensor)->tf.Tensor:
        y = keras.layers.Reshape(input_shape=(self.ir_data_width, 1), target_shape=(32, 24, 1))(inputs)
        y = keras.layers.Conv2D(filters=self.filter_num, kernel_size=self.kernel_size, strides=1, activation="relu",
                                padding="SAME")(y)
        y = keras.layers.Dropout(0.5)(y)
        y = keras.layers.MaxPooling2D(pool_size=3, strides=2)(y)
        y = keras.layers.Dropout(0.5)(y)
        return y

    def skin_dense_layers(self,inputs:tf.Tensor,input_shape:Tuple)->tf.Tensor:
        y = keras.layers.Reshape(input_shape=(input_shape), target_shape=(1, self.softskin_width))(inputs)
        y = keras.layers.Dense(self.dense_unit, activation="relu")(y)
        y = keras.layers.Dropout(0.5)(y)
        y = keras.layers.Dense(self.dense_unit, activation="relu")(y)
        y = keras.layers.Dropout(0.5)(y)
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
                output_ir = self.ir_part(output_ir)
                output_ir = keras.layers.Flatten()(output_ir)

                # soft skin feature abstraction
                skin_shape = skin_one_frame.shape
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
                output_ir = self.ir_part(output_ir)
                output_ir = keras.layers.Flatten()(output_ir)

                # leg feature just shortcut
                output_leg = keras.layers.Flatten()(leg_data)
                if i == 0:
                    output_feature = keras.layers.concatenate([output_ir, output_leg])
                else:
                    output_feature = keras.layers.concatenate([output_feature, output_ir, output_leg])
            return output_feature


    def create_model_dynamic(self) -> tf.keras.Model:
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
        output_LSTM = keras.layers.LSTM(32, activation='tanh')(output_reshape)
        output_final = keras.layers.Dense(128, activation='relu')(output_LSTM)
        output_final = keras.layers.Dense(64, activation='relu')(output_final)
        if not self.is_multiple_output:
            output_final = keras.layers.Dense(7, activation='softmax')(output_final)
            model = keras.Model(inputs=input_all, outputs=output_final)
            if self.show_summary:
                model.summary()
            return model
        else:
            actor = keras.layers.Dense(7, activation='relu')(output_final)
            critic = keras.layers.Dense(1)(output_final)
            model = keras.Model(inputs=input_all,outputs=[actor, critic])
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

    FFL_Model = FrontFollowing_Model(win_width=10)

    concatenate_data_path = "/data/cyzhao/data.txt"
    concatenate_data = np.loadtxt(concatenate_data_path)
    concatenate_label_path = "/data/cyzhao/label.txt"
    concatenate_label = np.loadtxt(concatenate_label_path)
    concatenate_label = concatenate_label.reshape((concatenate_label.shape[0],1))


    concatenate_dataset = np.concatenate([concatenate_label, concatenate_data], axis=1)

    still_data_idx = concatenate_dataset[:,0] == 0
    still_data = concatenate_dataset[still_data_idx]
    np.random.shuffle(still_data)
    other_data_idx = concatenate_dataset[:,0] != 0
    other_data = concatenate_dataset[other_data_idx]
    concatenate_dataset = np.concatenate([still_data[0:int(still_data.shape[0]/2),:], other_data],axis=0)

    np.random.shuffle(concatenate_dataset)
    concatenate_label = concatenate_dataset[:, 0]
    concatenate_data = concatenate_dataset[:, 1:concatenate_dataset.shape[1]]

    portion_train = int(concatenate_dataset.shape[0] * 0.8)
    portion_validation = int(concatenate_dataset.shape[0] * 0.9)

    train_data = concatenate_data[0:portion_train,:]
    train_label = concatenate_label[0:portion_train]
    train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1], 1))

    validation_data = concatenate_data[portion_train:portion_validation, :]
    validation_label = concatenate_label[portion_train:portion_validation]
    validation_data = np.reshape(validation_data, (validation_data.shape[0], validation_data.shape[1], 1))

    test_data = concatenate_data[portion_validation:concatenate_dataset.shape[0], :]
    test_label = concatenate_label[portion_validation:concatenate_dataset.shape[0]]
    test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1], 1))

    #test_data_path = "/data/cyzhao/test_data.txt"
    #test_data = np.loadtxt(test_data_path)
    #test_label_path = "/data/cyzhao/test_label.txt"
    #test_label = np.loadtxt(test_label_path)
    #test_label = test_label.reshape((test_label.shape[0], 1))
    #test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1], 1))


    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    FFL_Model.model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    FFL_Model.model.load_weights("./checkpoints34/FFL34")
    #FFL_Model.model.fit(train_data, train_label, batch_size=128, epochs=100, validation_data=(validation_data, validation_label), verbose=1)
    #FFL_Model.model.save_weights("./checkpoints34/FFL34")
    while True:
        #break
        test_loss, test_acc = FFL_Model.model.evaluate(test_data, test_label, verbose=1)
        if test_acc < 0.5:
            FFL_Model.model.fit(train_data, train_label, batch_size=128, epochs=50, validation_data=(validation_data, validation_label),verbose=1)
            FFL_Model.model.save_weights('./checkpoints34/FFL34')
        elif test_acc < 0.88:
            FFL_Model.model.fit(train_data, train_label, batch_size=128, epochs=10,validation_data=(validation_data,validation_label),verbose=1)
            FFL_Model.model.save_weights('./checkpoint34/FFL34')
        else:
            break




