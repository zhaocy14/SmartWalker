# !/usr/bin/env python
# -*- coding:utf-8 _*-
# @Author: swang
# @Contact: wang00sheng@gmail.com
# @Project Name: D3QN
# @File: A2C.py
# @Time: 2021/10/29/22:54
# @Software: PyCharm
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Activation, Input, Conv2D, Flatten, Lambda, Add, BatchNormalization, Reshape
import tensorflow_probability as tfp


def FeatureExtractor(model_dir='../model/base_model/'):
    '''
    initialize a feature extractor for a d3qn RL agent.
    '''
    ckpt_dir = os.path.join(model_dir, 'feature_extractor', 'ckpt')
    fe = tf.keras.models.load_model(ckpt_dir)
    fe.trainable = False
    fe.compile()
    
    print('-' * 20, 'ResCNN_feature_extractor', '-' * 20, )
    fe.summary()
    
    return fe


def D3QN_Classifier(dueling=True, base_model_dir='../model/base_model/',
                    load_d3qn_model=False, d3qn_model_dir='../model/d3qn_model/', based_on_base_model=True):
    def __init_d3qn_model__():
        print('-' * 20, 'Initializing a new D3QN model...', '-' * 20, )
        if based_on_base_model:
            print('-' * 20, 'Based on base_model\'s classifier...', '-' * 20, )
            base_classifier = tf.keras.models.load_model(base_ckpt_dir)
            conv, bn, reshape = base_classifier.layers
            input = base_classifier.input
            base_classifier = Model(inputs=input, outputs=reshape(conv(input)))
        
        else:
            print('-' * 20, 'Even the classifier is initialized randomly...', '-' * 20, )
            base_classifier = Sequential([
                Input(shape=(30, 8, 82), name='feature_input'),
                Conv2D(1, kernel_size=(30, 1), strides=(1, 1), padding='valid', use_bias=True),
                Reshape((-1,)), ],
                name='output_conv')
        
        if dueling:
            # ------------------------------------- value network -------------------------------------------------#
            state_value = Conv2D(1, kernel_size=(30, 1), strides=(1, 1), padding='valid', use_bias=True,
                                 name='value_conv')(base_classifier.input)
            # state_value = BatchNormalization(axis=-1)(state_value)
            state_value = Reshape((-1,), name='value_reshape')(state_value)
            state_value = Dense(1, name='value_dense')(state_value)
            
            # ------------------------------------- advantage network -------------------------------------------------#
            action_advantage = base_classifier.output
            # kernel_constraint=max_norm(norm_rate))(feature_extraction.output)
            
            # ------------------------------------- add -------------------------------------------------#
            output = Add()([state_value, action_advantage])
        else:
            output = base_classifier.output
        # outputs = tf.keras.layers.Multiply()([output, [1, 1. / 2460]])
        outputs = Lambda(lambda x: x / 2460.)(output)
        return Model(inputs=base_classifier.input, outputs=outputs)
    
    def __load_d3qn_model__():
        try:
            print('-' * 20, 'Loading pre-trained D3QN classifier...', '-' * 20, )
            return tf.keras.models.load_model(d3qn_ckpt_dir)
        except Exception as e:
            print('Warning:', e)
            print('-' * 20, 'Fail to load the pre-trained D3QN model! An initialized model will be used!',
                  '-' * 20, )
            return __init_d3qn_model__()
    
    # ------------------------------------- main procedure -------------------------------------------------#
    base_ckpt_dir = os.path.join(base_model_dir, 'classifier', 'ckpt')
    d3qn_ckpt_dir = os.path.join(d3qn_model_dir, 'classifier', 'ckpt')
    
    if load_d3qn_model:
        d3qn_model = __load_d3qn_model__()
    else:
        d3qn_model = __init_d3qn_model__()
    
    print('-' * 20, 'd3qn_classifier', '-' * 20, )
    d3qn_model.summary()
    
    return d3qn_model


def SAC_actor(base_model_dir='../model/base_model/',
              load_sac_model=False, sac_model_dir='../model/sac_model/', based_on_base_model=True):
    def __init_sac_actor__():
        print('-' * 20, 'Initializing a new SAC_actor...', '-' * 20, )
        if based_on_base_model:
            print('-' * 20, 'Based on base_model\'s classifier...', '-' * 20, )
            base_classifier = tf.keras.models.load_model(base_ckpt_dir)
        else:
            print('-' * 20, 'Even the classifier is initialized randomly...', '-' * 20, )
            base_classifier = Sequential([
                Input(shape=(30, 8, 82), name='feature_input'),
                Conv2D(1, kernel_size=(30, 1), strides=(1, 1), padding='valid', use_bias=True),
                BatchNormalization(axis=-1), Reshape((-1,)), ],
                name='output_conv')
        
        output = Activation('softmax', name='softmax')(base_classifier.output)
        
        return Model(inputs=base_classifier.input, outputs=output)
    
    def __load_sac_actor__():
        try:
            print('-' * 20, 'Loading pre-trained sac_actor...', '-' * 20, )
            return tf.keras.models.load_model(sac_ckpt_dir)
        except Exception as e:
            print('Warning:', e)
            print('-' * 20, 'Fail to load the pre-trained sac_actor! An initialized model will be used!',
                  '-' * 20, )
            return __init_sac_actor__()
    
    # ------------------------------------- main procedure -------------------------------------------------#
    base_ckpt_dir = os.path.join(base_model_dir, 'classifier', 'ckpt')
    sac_ckpt_dir = os.path.join(sac_model_dir, 'actor', 'ckpt')
    
    if load_sac_model:
        sac_actor = __load_sac_actor__()
    else:
        sac_actor = __init_sac_actor__()
    
    print('-' * 20, 'sac_actor', '-' * 20, )
    sac_actor.summary()
    
    return sac_actor


def SAC_critic(base_model_dir='../model/base_model/',
               load_sac_model=False, sac_model_dir='../model/sac_model/', based_on_base_model=True):
    def __init_sac_critic__():
        print('-' * 20, 'Initializing a new SAC_critic...', '-' * 20, )
        if based_on_base_model:
            print('-' * 20, 'Based on base_model\'s classifier...', '-' * 20, )
            base_classifier = tf.keras.models.load_model(base_ckpt_dir)
        else:
            print('-' * 20, 'Even the classifier is initialized randomly...', '-' * 20, )
            base_classifier = Sequential([
                Input(shape=(30, 8, 82), name='feature_input'),
                Conv2D(1, kernel_size=(30, 1), strides=(1, 1), padding='valid', use_bias=True),
                Reshape((-1,)), ],
                name='output_conv')
        
        return Model(inputs=base_classifier.input, outputs=base_classifier.output)
    
    def __load_sac_critic__():
        try:
            print('-' * 20, 'Loading pre-trained sac_critic...', '-' * 20, )
            return tf.keras.models.load_model(sac_ckpt_dir)
        except Exception as e:
            print('Warning:', e)
            print('-' * 20, 'Fail to load the pre-trained sac_critic! An initialized model will be used!',
                  '-' * 20, )
            return __init_sac_critic__()
    
    # ------------------------------------- main procedure -------------------------------------------------#
    base_ckpt_dir = os.path.join(base_model_dir, 'classifier', 'ckpt')
    sac_ckpt_dir = os.path.join(sac_model_dir, 'critic', 'ckpt')
    
    if load_sac_model:
        sac_critic = __load_sac_critic__()
    else:
        sac_critic = __init_sac_critic__()
    
    print('-' * 20, 'sac_critic', '-' * 20, )
    sac_critic.summary()
    
    return sac_critic


if __name__ == '__main__':
    print('Hello World!')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    fe = FeatureExtractor()
    cls = D3QN_Classifier(based_on_base_model=True)
    cls = SAC_actor(based_on_base_model=True)
    cls = SAC_critic(based_on_base_model=True)
    
    print('Brand-new World!')
