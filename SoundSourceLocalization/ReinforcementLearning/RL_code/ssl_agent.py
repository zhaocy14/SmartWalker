# !/usr/bin/env python
# -*- coding:utf-8 _*-
# @Author: swang
# @Contact: wang00sheng@gmail.com
# @Project Name: RL_Simulation
# @File: agent.py
# @Time: 2021/10/31/10:19
# @Software: PyCharm

import os
import sys
import time
import random
import numpy as np
from copy import deepcopy
import collections
from collections import defaultdict
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.utils import plot_model
import tensorflow_probability as tfp
import warnings
from model_tf.actor_critic import ActorCriticNetwork


class Agent:
    def __init__(self, alpha=1., gamma=0.99, num_action=8, ac_model_dir='../model/ac_model', load_ac_model=False):
        super(Agent, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.action = None
        self.num_action = num_action
        self.action_space = list(range(num_action))
        self.ac_model_dir = ac_model_dir
        self.load_ac_model = load_ac_model
        
        lr_schedule = tf.keras.experimental.CosineDecay(initial_learning_rate=0.005, decay_steps=1e4)
        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.01, decay_steps=decay_steps, decay_rate=0.99, staircase=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
        self.actor_critic = ActorCriticNetwork(num_action=num_action, load_ac_model=load_ac_model,
                                               ac_model_dir=ac_model_dir, optimizer=self.optimizer)
        self.replay_buffer = ReplayMemory(100)
    
    def choose_action(self, state):
        state = tf.convert_to_tensor([state])
        _, probs = self.actor_critic(state, training=False)
        action_prob = tfp.distributions.Categorical(probs=probs)
        action = action_prob.sample()
        # log_prob = action_prob.log_prob(action)
        self.action = action
        
        return action.numpy()[0]
    
    def learn(self, state, action, reward, state_, done):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        action = tf.convert_to_tensor(action, dtype=tf.float32)
        if not done:
            state_ = tf.convert_to_tensor([state_], dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)  # not fed to NN
        with tf.GradientTape() as tape:
            # with tf.GradientTape(persistent=True) as tape:
            state_value, probs = self.actor_critic(state, training=True)
            state_value = tf.squeeze(state_value)
            action_probs = tfp.distributions.Categorical(probs=probs)
            log_prob = action_probs.log_prob(action)
            if not done:
                state_value_, _ = self.actor_critic(state_, training=False)
                state_value_ = tf.squeeze(state_value_)
                delta = reward + self.gamma * state_value_ - state_value
            else:
                delta = reward - state_value
            
            actor_loss = -log_prob * delta
            critic_loss = delta ** 2
            total_loss = self.alpha * actor_loss + critic_loss
        
        gradient = tape.gradient(total_loss, self.actor_critic.ac_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient, self.actor_critic.ac_model.trainable_variables))


class ReplayMemory():  # .定义经验池（队列）
    def __init__(self, max_size):
        super(ReplayMemory, self).__init__()
        self.max_size = max_size
        self.buffer = collections.deque(maxlen=self.max_size)
    
    def len(self):
        return len(self.buffer)
    
    def append(self, state, log_prob, reward, state_, done):
        exp = (state, log_prob, reward, state_, done)
        self.buffer.append(exp)  # 增加一条经验(obs，action，reward，next_obs，done)
    
    def sample(self, batch_size):
        buffer_len = self.len()
        batch_size = buffer_len if buffer_len < batch_size else batch_size
        
        mini_batch = random.sample(self.buffer, batch_size)
        state_batch, action_batch, reward_batch, state_batch_, done_batch = list(zip(*mini_batch))
        
        return np.array(state_batch).astype('float32'), \
               np.array(action_batch).astype('float32'), np.array(reward_batch).astype('float32'), \
               np.array(state_batch_).astype('float32'), np.array(done_batch).astype('float32')


if __name__ == '__main__':
    
    agent = Agent()
    print('Hello World!')
