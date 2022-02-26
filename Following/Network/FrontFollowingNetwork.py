import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from typing import Tuple
from Following.Network import resnet
pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd)+os.path.sep+"..")

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

    def __init__(self, win_width: int = 10, is_skin_input: bool = False, is_multiple_output: bool = False,show:bool=False):
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

        """data buffer"""
        if is_skin_input:
            self.data_buffer = np.zeros((1,self.win_width*(self.ir_data_width+self.softskin_width)))
        else:
            self.data_buffer = np.zeros((1,self.win_width*(self.ir_data_width+self.leg_width)))

    def update_buffer(self, new_ir_data, new_leg_data):
        PART2 = self.win_width * self.ir_data_width
        additional_data = [LD.left_leg[0], LD.left_leg[1], LD.right_leg[0], LD.right_leg[1]]
        additional_data = np.array(additional_data) / 40 + 0.4
        additional_data = np.reshape(additional_data, (additional_data.shape[0], 1))
        buffer[PART2:PART2 + (buffer_length - 1) * additional_data_width, 0] = \
            buffer[PART2 + additional_data_width:PART2 + buffer_length * additional_data_width, 0]
        buffer[PART2 + (buffer_length - 1) * additional_data_width:PART2 + buffer_length * additional_data_width] = \
            additional_data

        buffer[PART2:PART2 + buffer_length * additional_data_width, 0] = buffer[
                                                                         PART2:PART2 + buffer_length * additional_data_width,
                                                                         0]
        predict_buffer = buffer.reshape((-1, buffer_length * (ir_data_width + additional_data_width), 1))



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
        Lambda = 0.8
        output_current = tf.math.multiply(output_current,Lambda)
        output_tendency = tf.math.multiply(output_tendency,1-Lambda)
        output_final = tf.add(output_current, output_tendency)
        output_final = keras.layers.Dense(6, activation='softmax')(output_final)
        model = keras.Model(inputs=input_all, outputs=output_final)
        if self.show_summary:
            model.summary()
        return model


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    model = FrontFollowing_Model(win_width=10,is_skin_input=False, is_multiple_output=False, show=True)
