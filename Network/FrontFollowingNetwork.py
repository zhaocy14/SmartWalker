import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from typing import Tuple
from Network import resnet
pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd)+os.path.sep+"..")

class Conv_part(keras.Model):

    def __init__(self):
        super().__init__()
        self.layer1  = keras.layers.Conv2D(filters=10, kernel_size=3, strides=1, activation="relu",
                                padding="SAME")
        self.layer1  = keras.layers.Conv2D(filters=10, kernel_size=3, strides=1, activation="relu",
                                padding="SAME")
        self.layer1  = keras.layers.Conv2D(filters=10, kernel_size=3, strides=1, activation="relu",
                                padding="SAME")
        self.layer2 = keras.layers.MaxPool2D(pool_size=3,strides=2)

    def call(self,inputs):
        y = self.layer1(inputs)
        y = self.layer2(y)
        return y

class Skin_part(keras.Model):

    def __init__(self,input_shape):
        super().__init__()
        self.softskin_width = 32
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
        self.win_width = win_width
        self.ir_data_width = 768
        self.softskin_width = 32
        self.leg_width = 4
        """network parameter"""
        self.filter_num = 3
        self.kernel_size = 3
        self.dense_unit = 10
        self.show_summary = False
        self.is_multiple_output = is_multiple_output
        self.is_skin_input = is_skin_input

        """network building"""
        self.ir_part_0 = Conv_part()
        self.ir_part = resnet.get_model("resnet34")
        # self.skin_part = Skin_part()
        self.model = self.create_model_dynamic()

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
                output_ir = self.ir_part(output_ir)
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
        keras.layers.LSTM()
        output_final = keras.layers.LSTM(32, activation='tanh')(output_reshape)
        output_final = keras.layers.Dense(128, activation='relu')(output_final)
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

    model = FrontFollowing_Model(win_width=10,is_skin_input=False,is_multiple_output=False)
    # model.model.summary()
    model.model.compile(optimizer='RMSprop',
                                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                metrics=['accuracy'])
    check_point_path = os.path.abspath(father_path+os.path.sep+"ServerFiles"+os.path.sep+"checkpoints_saved"
                                        + os.path.sep + "res34" + os.path.sep + "FFL34")
    model.model.load_weights(check_point_path)
    #
    test_data_path = os.path.abspath(father_path+os.path.sep+"data"+os.path.sep+"test_data.txt")
    test_data = np.loadtxt(test_data_path)
    test_label_path = os.path.abspath(father_path+os.path.sep+"data"+os.path.sep+"test_label.txt")
    test_label = np.loadtxt(test_label_path)
    test_label = test_label.reshape((test_label.shape[0], 1))
    test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1], 1))
    model.model.evaluate(test_data,test_label)
    # #
    # # concatenate_data_path = "./Record_data/data.txt"
    # concatenate_data = np.loadtxt(concatenate_data_path)
    # concatenate_label_path = "./Record_data/label.txt"
    # concatenate_label = np.loadtxt(concatenate_label_path)
    #
    #
    # """initialize the training process"""
    # win_width = 20
    # step_length = 1
    # ir_data_width = 768
    # softskin_width = 32




    #
    # model.compile(optimizer='RMSprop',
    #               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               metrics=['accuracy'])
    # concatenate_data = np.reshape(concatenate_data,(concatenate_data.shape[0],concatenate_data.shape[1],1))

    # """start training"""
    # while True:
    #     test_loss, test_acc = model.evaluate(concatenate_data, concatenate_label, verbose=1)
    #     if test_acc < 0.91:
    #         model.fit(concatenate_data, concatenate_label, batch_size=64, epochs=100,verbose=1)
    #         model.save_weights('./checkpoints/CNN+DNN+LSTM/CNN+DNN+LSTM_10')
    #     else:
    #         break

    """convert to STM32"""
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # tflite_model = converter.convert()
    #
    # #保存到磁盘
    # open("./Record_data/STM32/MCUNET.tflite", "wb").write(tflite_model)

    # model.save_weights('./Record_data/STM32/model.h5')


