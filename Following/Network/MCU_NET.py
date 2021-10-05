import os
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# # tf.config.experimental.set_memory_growth(physical_devices[1], True)

def creat_model(summary_show = False):
    model = keras.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(64, name="dense_1", activation='relu'),
        layers.Dropout(0.2),
        # layers.Dense(64,input_shape=(28,28),name="dense_1",activation='relu'),
        # layers.Dense(64,name="dense_2",activation='relu'),
        # layers.Dropout(0.2),
        layers.Dense(10,activation='softmax')
    ])
    if summary_show:
        model.summary()
    return model

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
a = x_train[23,:,:]
print(a)
np.savetxt('./STM32network/a.txt',a,fmt="%.3f")
print(y_train[23])

# model = creat_model(True)
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
#
# model.fit(x_train, y_train, epochs=35, validation_split=0.2)
#
# model.save_weights('./STM32network/checkpoints/my_checkpoint_small')
# # # 恢复权重
# model.load_weights('./STM32network/checkpoints/my_checkpoint_small')
# #
# model.save('./STM32network/MyModel.h5')
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()
#
# #保存到磁盘
# open("./STM32network/MCUNET.tflite", "wb").write(tflite_model)

# model.save_weights('./STM32network/TF_lite/my_h5_model.h5')



# imgList = os.listdir('F:/Data/pycharm/tensorflow_test/samples_training')
# img_shape = Image_size
# # 读取数据
# data = np.zeros((len(imgList),img_shape,img_shape))
# for i in range(0, len(imgList)):
#     im_name = imgList[i]
#     im_path = os.path.join('F:/Data/pycharm/tensorflow_test/samples_training/', im_name)
#     # print(im_path)
#     im = Image.open(im_path)
#     im = im.convert('L')
#     # im = im.resize((60, 60), Image.ANTIALIAS)
#     data_temp = im.getdata()
#     data_temp = np.array(data_temp)
#     # print(data_temp.shape)
#     data_temp = data_temp.reshape((64,64))
#     data_temp = data_temp[8:57,8:58]
#     data_temp = Image.fromarray(data_temp)
#     data_temp = data_temp.resize((img_shape,img_shape), Image.ANTIALIAS)
#     data_temp = data_temp.getdata()
#     data_temp = np.array(data_temp)
#     data_temp = data_temp.reshape((img_shape, img_shape))
#     data[i,:,:] = data_temp
# gan_data = data / 255
# # label_gan = np.zeros((len(imgList),1))
# # for i in range(gan_data.shape[0]):
# #     data_i = (gan_data[i, :, :]).reshape(1,-1)
# #     label_i = model.predict(data_i)
# #     label_gan[i] = np.argmax(label_i)
# # np.savetxt("label_gan.txt",label_gan,fmt="%d")
#
#
# gan_label = np.loadtxt("./label_gan.txt", dtype=int)
# gan_data = gan_data.reshape(gan_data.shape[0], -1)
# train_data = np.c_[gan_label, gan_data]
# np.random.shuffle(train_data)
# gan_data = train_data[:, 1:train_data.shape[1]]
# gan_label = train_data[:, 0]
#
# seperation = 1900
# gan_data_train = gan_data[0:seperation,:]
# gan_label_train = gan_label[0:seperation]
#
# gan_data_val = gan_data[seperation:gan_data.shape[0],:]
# gan_label_val = gan_label[seperation:gan_label.shape[0]]
#
# gan_dnn_model = creat_model(gan_data.shape[1])
# gan_dnn_model.summary()
# gan_dnn_model.compile(optimizer='sgd',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
# gan_dnn_model.fit(gan_data_train, gan_label_train,
#                   epochs=120,
#                   validation_data=(gan_data_val,gan_label_val),
#                   batch_size=8)
# # gan_dnn_model.save_weights('./checkpoints/my_checkpoint_gan_dnn')
# # gan_dnn_model.load_weights('./checkpoints/my_checkpoint_gan_dnn')
#
# gan_dnn_model.evaluate(train_x,train_label)
# label_dnn_gan = np.zeros((train_x.shape[0],1))
# print(train_x.shape)
# # print(train_x[0,:].shape)
# for i in range(train_x.shape[0]):
#     data = (train_x[i,:]).reshape(1,-1)
#     label_i = gan_dnn_model.predict(data)
#     label_dnn_gan[i] = np.argmax(label_i)
# label_difference = label_dnn_gan-train_label
# np.savetxt("label_dnn_gan.txt",label_dnn_gan,fmt="%d")
# np.savetxt("label_dnn_gan_difference.txt",label_difference,fmt="%d")