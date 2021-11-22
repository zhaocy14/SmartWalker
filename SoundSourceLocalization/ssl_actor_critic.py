"""
    RL online training Part
"""
import platform

import tensorflow as tf
import math
import numpy as np
import warnings
import os
from ssl_setup import *
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

EPS = np.finfo(float).eps

sysstr = platform.system()
if (sysstr == "Windows"):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    K.set_image_data_format('channels_first')
elif (sysstr == "Linux"):
    pass
else:
    pass

    
class ActorCriticNetwork:
    def __init__(self, n_actions=ACTION_SPACE, name='actor_critic', ini_model=None, actor_lr=0.003, critic_lr=0.004,
                 actor_optimizer=None, critic_optimizer=None, gamma=0.99,
                 ini_model_dir='./model/EEGNet/ckpt', save_model_dir='./actor_critic_model/ckpt'):
        super(ActorCriticNetwork, self).__init__()
        
        self.n_actions = n_actions
        self.model_name = name
        self.ini_model = ini_model
        self.ini_model_dir = ini_model_dir
        self.save_model_dir = save_model_dir
        self.model = self.__load_model__()
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_optimizer = actor_optimizer if (actor_optimizer is not None) else tf.keras.optimizers.Adam(
            learning_rate=actor_lr)
        self.critic_optimizer = critic_optimizer if (critic_optimizer is not None) else tf.keras.optimizers.Adam(
            learning_rate=actor_lr)
        
        self.gamma = gamma
        self.value = None
        self.value_ = None
        self.act_prob = None
        self.act_prob_ = None
        self.state = None
        self.state_ = None
        self.action = None  # store actual action (delete invalid directions)
        self.action_ = None
        self.reward = None
        self.reward_ = None
        self.reward_sum = 0
        self.td_error = None
    
    def __load_model__(self):
        if self.ini_model is None:
            model = keras.models.load_model(self.ini_model_dir)
        else:
            model = self.ini_model
        model.summary()
        
        inputs = model.inputs
        outputs = model.get_layer(index=(-3)).output
        feature_exaction_block = keras.Model(inputs=inputs, outputs=outputs, name='feature_exaction')
        features = feature_exaction_block(inputs, training=False)
        for i in range(len(feature_exaction_block.layers)):
            feature_exaction_block.layers[i].trainable = False
        
        value = Dense(1, activation=None, kernel_initializer='random_uniform',
                      bias_initializer='zeros', name='value_dense', )(features)
        act_prob = model.output
        
        ac_model = keras.Model(inputs=inputs, outputs=(value, act_prob))
        ac_model.summary()
        
        return ac_model
    
    def cal_actor_loss(self, act_prob):
        log_prob = tf.math.log(act_prob[:, self.action] + EPS)
        exp_v = tf.reduce_mean(log_prob * self.td_error)
        return -exp_v
    
    def cal_critic_loss(self, value, value_, reward):
        self.td_error = reward + self.gamma * value_ - value
        
        return tf.math.square(self.td_error)
    
    def learn(self, state, state_, reward):
        if state is None:
            state = state_
        state = np.array([state])
        state_ = np.array([state_])
        with tf.GradientTape() as tape:
            (value, act_prob) = self.model(state, training=True)  # Logits for this minibatch
            (value_, act_prob_) = self.model(state_, training=False)  # Logits for this minibatch
            critic_ls = self.cal_critic_loss(value, value_, reward)
        critic_grads = tape.gradient(critic_ls, self.model.trainable_weights)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.model.trainable_weights))
        
        with tf.GradientTape() as tape:
            (_, act_prob) = self.model(state, training=True)  # Logits for this minibatch
            actor_ls = self.cal_actor_loss(act_prob)
        actor_grads = tape.gradient(actor_ls, self.model.trainable_weights)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.model.trainable_weights))
    
    def predict(self, state, invalid_classes=None):
        state = np.array([state])
        (value, class_prob) = self.model.predict(state)
        
        if invalid_classes is not None:
            class_prob[:, invalid_classes] = 0
            class_prob = class_prob / class_prob.mean(axis=1)
        class_cate = np.argmax(class_prob, axis=1)
        
        return class_prob, class_cate


#
# class Agent:
#     def __init__(self, alpha=0.0003, gamma=0.99, n_actions=2):
#         self.gamma = gamma
#         self.n_actions = n_actions
#         self.action = None
#         self.action_space = [i for i in range(self.n_actions)]
#
#         self.actor_critic = ActorCriticNetwork(n_actions=n_actions)
#
#         self.actor_critic.compile(optimizer=Adam(learning_rate=alpha))
#
#     def choose_action(self, observation):
#         state = tf.convert_to_tensor([observation])
#         _, probs = self.actor_critic(state)
#
#         action_probabilities = tfp.distributions.Categorical(probs=probs)
#         action = action_probabilities.sample()
#         log_prob = action_probabilities.log_prob(action)
#         self.action = action
#
#         return action.numpy()[0]
#
#     def save_models(self):
#         print('... saving models ...')
#         self.actor_critic.save_weights(self.actor_critic.checkpoint_file)
#
#     def load_models(self):
#         print('... loading models ...')
#         self.actor_critic.load_weights(self.actor_critic.checkpoint_file)
#
#     def learn(self, state, reward, state_, done):
#         state = tf.convert_to_tensor([state], dtype=tf.float32)
#         state_ = tf.convert_to_tensor([state_], dtype=tf.float32)
#         reward = tf.convert_to_tensor(reward, dtype=tf.float32)  # not fed to NN
#         with tf.GradientTape(persistent=True) as tape:
#             state_value, probs = self.actor_critic(state)
#             state_value_, _ = self.actor_critic(state_)
#             state_value = tf.squeeze(state_value)
#             state_value_ = tf.squeeze(state_value_)
#
#             action_probs = tfp.distributions.Categorical(probs=probs)
#             log_prob = action_probs.log_prob(self.action)
#
#             delta = reward + self.gamma * state_value_ * (1 - int(done)) - state_value
#             actor_loss = -log_prob * delta
#             critic_loss = delta ** 2
#             total_loss = actor_loss + critic_loss
#
#         gradient = tape.gradient(total_loss, self.actor_critic.trainable_variables)
#         self.actor_critic.optimizer.apply_gradients(zip(
#             gradient, self.actor_critic.trainable_variables))
#

if __name__ == '__main__':
    n_features, n_actions, lr, gamma = 10, 3, 0.001, 0.2
    gcc_feature = np.zeros((1, 6, 61))
    AC = ActorCriticNetwork()
    print(AC.predict(gcc_feature))

# Actor(n_features, n_actions, lr)
# Critic(n_features, n_actions, lr, gamma)
