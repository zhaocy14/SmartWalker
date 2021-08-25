import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# class decoder():
#     def __init__(self, win_width:int = 10):
#         self.win_width = win_width
#         self.ir_data_width = 768
#         self.leg_width = 4
#
#     def encoder_conv_layers(self,inputs:tf.Tensor)->tf.Tensor:
#         y = keras.layers.Reshape(input_shape=(self.ir_data_width, 1), target_shape=(32, 24, 1))(inputs)
#         y = keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, activation="relu")(y)
#         y = keras.layers.Conv2D(filters=4, kernel_size=3, strides=1, activation="relu")(y)
#         print(y.shape)
#         return y
#
#     def decoder_conv_layers(self,inputs:tf.Tensor)->tf.Tensor:
#         y = keras.layers.Conv2DTranspose(4, kernel_size=3, strides=1, activation="relu")(inputs)
#         y = keras.layers.Conv2DTranspose(16, kernel_size=3, strides=1, activation="relu")(y)
#         y = tf.keras.Conv2D(1, kernel_size=3, activation='sigmoid', padding='same')(y)
#         return y
#
#     def feature_abstraction(self, ir_data: tf.Tensor) -> tf.Tensor:
#         for i in range(self.win_width):
#             [ir_one_frame, ir_data] = tf.split(ir_data,
#                                                [self.ir_data_width, self.ir_data_width * (self.win_width - 1 - i)],
#                                                axis=1)
#             output_ir = keras.layers.Flatten()(self.encoder_conv_layers(ir_one_frame))
#             if i == 0:
#                 output_feature = output_ir
#             else:
#                 output_feature = keras.layers.concatenate([output_feature, output_ir])
#         return output_feature

    # def training_process(self, ir_data, leg_data):
    #     loss_func = tf.keras.losses.MeanSquaredError()
    #     with tf.GradientTape as tape:
    #         encoded_ir_feature = self.feature_abstraction(ir_data=ir_data)
    #         loss = loss_func(leg_data,encoded_ir_feature)

if __name__ == "__main__":
    # model = tf.keras.Sequential([keras.layers.Reshape(input_shape=(768, 1), target_shape=(32, 24, 1)),
    #     keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, activation="relu"),
    #     keras.layers.Conv2D(filters=4, kernel_size=3, strides=1, activation="relu"),
    #     keras.layers.Flatten(),
    #                              ])

    win_width = 10
    model_decoder = tf.keras.Sequential([keras.layers.InputLayer(input_shape=(1,int(4*win_width))),
                                         keras.layers.Dense(64,activation='relu'),
                                         keras.layers.Dense(128,activation='relu'),
                                         keras.layers.Dense(2240,activation='relu'),
                                         keras.layers.Reshape(input_shape=(2240,1),target_shape=(28,20,4)),
                                         keras.layers.Conv2DTranspose(filters=4,kernel_size=3,strides=1,activation='relu'),
                                         keras.layers.Conv2DTranspose(filters=16,kernel_size=3,strides=1,activation='relu'),
                                         keras.layers.Conv2D(win_width, kernel_size=3, activation='sigmoid', padding='same'),
                                         keras.layers.Flatten()])

    concatenate_data_path = "/data/cyzhao/data.txt"
    concatenate_data = np.loadtxt(concatenate_data_path)

    ir_data_width = 768
    ir_data = concatenate_data[:,0:ir_data_width*win_width]
    leg_data = concatenate_data[:,ir_data_width*win_width:concatenate_data.shape[1]]

    model_decoder.summary()
    model_decoder.compile(optimizer='adam',loss=tf.keras.losses.MeanSquaredError())
    model_decoder.fit(leg_data,ir_data,batch_size=128,epochs=100)





