import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Flatten, Dense, Activation, Permute, Dropout, Reshape, Permute, Lambda, \
    Conv2D, MaxPooling2D, AveragePooling2D, SeparableConv2D, DepthwiseConv2D, BatchNormalization, SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2, l2
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras import backend as K
import tensorflow.keras.backend as K
K.set_image_data_format('channels_first')


# from .utils_tf import BaseClassicalModel

def FCN(nb_classes, Chans=64, SamplePoints=128, dropoutRate=None, norm_rate=0.25, ):
    input1 = Input(shape=(1, Chans, SamplePoints))
    block1 = Flatten()(input1)
    
    block1 = Dense(128, activation='relu', kernel_constraint=max_norm(norm_rate))(block1)
    block1 = BatchNormalization(axis=1)(block1)
    block1 = AveragePooling2D((1, 16))(block1)
    if dropoutRate is not None:
        block1 = Dropout(dropoutRate)(block1)
    
    block2 = Dense(128, activation='relu', kernel_constraint=max_norm(norm_rate))(block1)
    block2 = BatchNormalization(axis=1)(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    if dropoutRate is not None:
        block2 = Dropout(dropoutRate)(block2)
    
    block3 = Dense(128, activation='relu', kernel_constraint=max_norm(norm_rate))(block2)
    block3 = BatchNormalization(axis=1)(block3)
    block3 = AveragePooling2D((1, 8))(block3)
    if dropoutRate is not None:
        block3 = Dropout(dropoutRate)(block3)
    
    block4 = Dense(nb_classes, activation='relu', kernel_constraint=max_norm(norm_rate))(block3)
    softmax = Activation('softmax', name='softmax')(block4)
    
    return Model(inputs=input1, outputs=softmax)


def EEGNet(nb_classes, Chans=64, SamplePoints=128, dropoutRate=None, kernLength=64, F1=4,
           D=2, F2=8, norm_rate=0.25, dropoutType='Dropout'):
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1 = Input(shape=(1, Chans, SamplePoints))
    block1 = Conv2D(F1, (1, kernLength), padding='same',
                    input_shape=(1, Chans, SamplePoints),
                    use_bias=False)(input1)
    block1 = BatchNormalization(axis=1)(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=1)(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 2))(block1)
    # if dropoutRate is not None:
    #     block1 = dropoutType(dropoutRate)(block1)
    
    block2 = SeparableConv2D(F2, (1, 16),
                             use_bias=False, padding='same')(block1)
    block2 = BatchNormalization(axis=1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 2))(block2)
    if dropoutRate is not None:
        block2 = dropoutType(dropoutRate)(block2)
    
    flatten = Flatten(name='flatten')(block2)
    
    dense = Dense(nb_classes, name='dense',
                  kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)
    
    return Model(inputs=input1, outputs=softmax)


def DeepConvNet(nb_classes, Chans=64, SamplePoints=256, dropoutRate=0.5):
    """ Keras implementation of the Deep Convolutional Network as described in
    Schirrmeister et. al. (2017), Human Brain Mapping.

    This implementation assumes the input is a 2-second EEG signal sampled at
    128Hz, as opposed to signals sampled at 250Hz as described in the original
    paper. We also perform temporal convolutions of length (1, 5) as opposed
    to (1, 10) due to this sampling rate difference.

    Note that we use the max_norm constraint on all convolutional layers, as
    well as the classification layer. We also change the defaults for the
    BatchNormalization layer. We used this based on a personal communication
    with the original authors.

                      ours        original paper
    pool_size        1, 2        1, 3
    strides          1, 2        1, 3
    conv filters     1, 5        1, 10

    Note that this implementation has not been verified by the original
    authors.

    """
    
    # start the model
    input_main = Input((1, Chans, SamplePoints))
    block1 = Conv2D(25, (1, 5), input_shape=(1, Chans, SamplePoints),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(25, (Chans, 1), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block1)
    if dropoutRate is not None:
        block1 = Dropout(dropoutRate)(block1)
    
    block2 = Conv2D(50, (1, 5), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)
    block2 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block2)
    if dropoutRate is not None:
        block2 = Dropout(dropoutRate)(block2)
    
    block3 = Conv2D(100, (1, 5), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)
    block3 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block3)
    if dropoutRate is not None:
        block3 = Dropout(dropoutRate)(block3)
    
    block4 = Conv2D(200, (1, 4), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu', name='last_conv_out')(block4)
    # block4 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block4)
    if dropoutRate is not None:
        block4 = Dropout(dropoutRate)(block4)
    
    flatten = Flatten()(block4)
    
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)
    
    return Model(inputs=input_main, outputs=softmax)


# need these for ShallowConvNet
def square(x):
    return K.square(x)


def log(x):
    return K.log(K.clip(x, min_value=1e-7, max_value=10000))


def ShallowConvNet(nb_classes, Chans=64, SamplePoints=128, dropoutRate=0.5):
    """ Keras implementation of the Shallow Convolutional Network as described
    in Schirrmeister et. al. (2017), Human Brain Mapping.

    Assumes the input is a 2-second EEG signal sampled at 128Hz. Note that in
    the original paper, they do temporal convolutions of length 25 for EEG
    data sampled at 250Hz. We instead use length 13 since the sampling rate is
    roughly half of the 250Hz which the paper used. The pool_size and stride
    in later layers is also approximately half of what is used in the paper.

    Note that we use the max_norm constraint on all convolutional layers, as
    well as the classification layer. We also change the defaults for the
    BatchNormalization layer. We used this based on a personal communication
    with the original authors.

                     ours        original paper
    pool_size        1, 35       1, 75
    strides          1, 7        1, 15
    conv filters     1, 13       1, 25

    Note that this implementation has not been verified by the original
    authors. We do note that this implementation reproduces the results in the
    original paper with minor deviations.
    """
    
    # start the model
    input_main = Input((1, Chans, SamplePoints))
    block1 = Conv2D(40, (1, 13),
                    input_shape=(1, Chans, SamplePoints),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    # if is_denoising:
    #     block1 = Lambda(lambda t: denoising(t, name='denosing_1', embed=True, softmax=True))(block1)
    block1 = Conv2D(40, (Chans, 1), use_bias=False,
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation(square)(block1)
    block1 = AveragePooling2D(pool_size=(1, 35), strides=(1, 7))(block1)
    block1 = Activation(log)(block1)
    if dropoutRate is not None:
        block1 = Dropout(dropoutRate)(block1)
    flatten = Flatten()(block1)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)
    
    return Model(inputs=input_main, outputs=softmax)


def denoising(l, name, embed=True, softmax=True):
    """
    Feature Denoising, Fig 4 & 5.
    """
    with tf.variable_scope(name):
        f = non_local_op(l, embed=embed, softmax=softmax)
        f = Conv2D(int(l.shape[1]), (1, 1), strides=1)(f)
        f = l + f
    
    return f


def non_local_op(l, embed, softmax):
    """
    Feature Denoising, Sec 4.2 & Fig 5.
    Args:
        embed (bool): whether to use embedding on theta & phi
        softmax (bool): whether to use gaussian (softmax) version or the dot-product version.
    """
    n_in, H, W = l.shape.as_list()[1:]
    if embed:
        theta = Conv2D(n_in, (1, 1), strides=1, kernel_initializer=tf.random_normal_initializer(stddev=0.01))(l)
        phi = Conv2D(n_in, (1, 1), strides=1, kernel_initializer=tf.random_normal_initializer(stddev=0.01))(l)
        g = Conv2D(n_in, (1, 1), strides=1, kernel_initializer=tf.random_normal_initializer(stddev=0.01))(l)
    else:
        theta, phi, g = l, l, l
    
    theta = tf.transpose(theta, [0, 2, 3, 1])
    phi = tf.transpose(phi, [0, 2, 3, 1])
    g = tf.transpose(g, [0, 2, 3, 1])
    theta_flat = tf.reshape(theta,
                            [-1, theta.shape.as_list()[1] * theta.shape.as_list()[2], theta.shape.as_list()[-1]])
    phi_flat = tf.reshape(phi, [-1, phi.shape.as_list()[1] * phi.shape.as_list()[2], phi.shape.as_list()[-1]])
    g_flat = tf.reshape(g, [-1, g.shape.as_list()[1] * g.shape.as_list()[2], g.shape.as_list()[-1]])
    f = tf.matmul(theta_flat, tf.transpose(phi_flat, [0, 2, 1]))
    if softmax:
        # f = f / tf.sqrt(n_in)
        f = tf.nn.softmax(f)
        fg = tf.matmul(f, g_flat)
        fg = tf.transpose(tf.reshape(fg, [-1, *g.shape.as_list()[1:]]), [0, 3, 1, 2])
    else:
        f = f / tf.cast(H * W, tf.float32)
        fg = tf.matmul(f, g_flat)
        fg = tf.transpose(tf.reshape(fg, [-1, *g.shape.as_list()[1:]]), [0, 3, 1, 2])
    
    return fg


class TSLeNet5(tf.keras.Model):  # s model
    
    def __init__(self, num_classes, name=None, **kwargs):
        if name is None:
            name = self.__class__.__name__
        
        super(TSLeNet5, self).__init__(name=name, **kwargs)
        
        self.conv1 = tf.keras.layers.Conv1D(6, 5, activation='relu', padding='valid')
        self.pool1 = tf.keras.layers.MaxPool1D()
        self.conv2 = tf.keras.layers.Conv1D(16, 5, activation='relu', padding='valid')
        self.pool2 = tf.keras.layers.MaxPool1D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(120, activation='relu')
        self.dense2 = tf.keras.layers.Dense(84, activation='relu')
        self.out = tf.keras.layers.Dense(num_classes, activation='softmax')
    
    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        out = self.out(x)
        
        return out


class LeNet5(tf.keras.Model):
    
    def __init__(self, num_classes, name=None, **kwargs):
        if name is None:
            name = self.__class__.__name__
        
        super(LeNet5, self).__init__(name=name, **kwargs)
        
        self.conv1 = tf.keras.layers.Conv1D(6, 5, activation='relu', padding='valid')
        self.pool1 = tf.keras.layers.MaxPool1D()
        self.conv2 = tf.keras.layers.Conv1D(16, 5, activation='relu', padding='valid')
        self.pool2 = tf.keras.layers.MaxPool1D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(120, activation='relu')
        self.dense2 = tf.keras.layers.Dense(84, activation='relu')
        self.out = tf.keras.layers.Dense(num_classes, activation='linear')
    
    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        out = self.out(x)
        
        if training is False:
            out = tf.nn.softmax(out, axis=-1)
        
        return out


if __name__ == '__main__':
    
    model = EEGNet(nb_classes=8)
    model.summary()
