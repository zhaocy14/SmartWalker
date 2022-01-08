# !/usr/bin/env python
# -*- coding:utf-8 _*-
# @Author: swang
# @Contact: wang00sheng@gmail.com
# @Project Name: RL_Simulation
# @File: main.py
# @Time: 2021/10/28/14:36
# @Software: PyCharm
import os
import sys
import time
import math
import random
import cv2
import numpy as np
from copy import deepcopy
from model_tf.actor_critic import ActorCriticNetwork
from agent import Agent
from env import MAP_ENV
import tensorflow as tf
import tensorflow.keras.backend as K
from lib.utils import plot_curve


class RL_game():
    def __init__(self, num_action=8):
        super(RL_game, self).__init__()
        self.env = MAP_ENV()
        self.agent = Agent(alpha=1., num_action=num_action)
        self.epochs = 1000
        self.print_interval = 10
    
    def play(self):
        episode_reward = []
        for episode_idx in range(self.epochs):
            reward_history = []
            state, done = self.env.reset()
            num_step = 0
            while not done:
                num_step += 1
                action = self.agent.choose_action(state)
                state_, reward, done, info = self.env.step(action)
                # self.agent.learn(state, action, reward, state_, done)
                state = state_
                reward_history.append(reward)
                
                # if num_step % self.print_interval == 0:
                #     print('episode: ', episode_idx, 'num_step: ', num_step, '\trecent_avg_reward: ',
                #           np.mean(reward_history[-self.print_interval:]))
                if num_step > 100:
                    break
            # print('episode: ', episode_idx, '\ttotal_reward: ', np.sum(reward_history))
            episode_reward.append(np.sum(reward_history))
            if episode_idx % self.print_interval == 0:
                print('episode: ', episode_idx, '\tavg_total_reward: ', np.mean(episode_reward))
        curve_name = ['Training reward', ]
        curve_data = [episode_reward, ]
        color = ['r', ]
        # name = 'Pre-trained Actor Classifier'
        name = 'Pre-trained Actor Classifier - without learning'
        # name = 'Randomly initialized Actor Classifier - 1'
        # name = 'Randomly initialized Actor Classifier - 2'
        # name = 'Randomly initialized Actor Classifier - 3'
        title = 'Episode-wise training reward - ' + name
        img_path = './' + name + '.jpg'
        plot_curve(data=list(zip(curve_name, curve_data, color)), title=title, img_path=img_path, linewidth=0.5)
        print('Name: ', name)
        np.savez('./' + name + '.npz', data=episode_reward)


if __name__ == '__main__':
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 6)
    tf.config.experimental.set_memory_growth(gpus[0], True)
    K.set_image_data_format('channels_first')
    
    game = RL_game()
    game.play()
    print('Hello World!')
