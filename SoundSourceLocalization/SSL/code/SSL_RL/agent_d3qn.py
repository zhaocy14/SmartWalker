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
import tensorflow as tf
import tensorflow_probability as tfp
# from model_tf.PER import Memory
from .agent_models import D3QN_Classifier, FeatureExtractor
from collections import deque


class DQNAgent(object):
    def __init__(self, num_action=8, reward_discount_rate=0.95, lr=0.001,
                 ddqn=True, dueling=True, softUpdate=True, softUpdate_tau=0.1, learnTimes=1,
                 usePER=False, batch_size=8, memory_size=64,
                 eps_decay=False, ini_eps=1.0, min_eps=0.01, eps_decay_rate=0.999,
                 base_model_dir='../model/base_model', d3qn_model_dir='../model/d3qn_model',
                 load_d3qn_model=True, based_on_base_model=True, d3qn_model_name=None, **kwargs):
        super(DQNAgent, self).__init__()
        assert usePER == False, 'PER has not been checked'
        if d3qn_model_name is None:
            self.name = self.__class__.__name__
            print('Warning:',
                  'A DQNAgent class is initialized with default name (\'DQNAgent\'). And it may load unexpected models.')
        else:
            self.name = d3qn_model_name
        
        # space
        self.num_action = num_action
        
        # defining model parameters
        self.ddqn = ddqn  # use double deep q network
        self.softUpdate = softUpdate  # use soft parameter update
        self.dueling = dueling  # use dealing network
        self.eps_decay = eps_decay  # use epsilon greedy strategy if False, min_eps will be used.
        self.usePER = usePER  # use priority experienced replay
        self.tau = softUpdate_tau  # target network soft update hyperparameter
        self.learnTimes = learnTimes  # how many times to update the model when self.learn() is called
        self.discount_rate = reward_discount_rate  # discount rate
        self.lr = lr  # learning rate
        
        # exploration hyperparameters for epsilon and epsilon greedy strategy
        self.ini_eps = ini_eps  # exploration probability at start
        self.min_eps = min_eps  # minimum exploration probability
        self.eps_decay_rate = eps_decay_rate  # exponential decay rate for exploration prob
        
        # Instantiate memory
        self.batch_size = batch_size
        self.memory_size = memory_size
        if self.usePER:
            self.MEMORY = Memory(memory_size)
        else:
            self.memory = deque(maxlen=memory_size)
        
        self.load_d3qn_model = load_d3qn_model
        self.based_on_base_model = based_on_base_model
        self.base_model_dir = base_model_dir
        self.d3qn_model_dir = os.path.join(d3qn_model_dir, self.name, )
        os.makedirs(self.d3qn_model_dir, exist_ok=True)
        
        # create main model and target model
        self.feature_extractor = FeatureExtractor(model_dir=self.base_model_dir, )
        self.model = D3QN_Classifier(dueling=self.dueling, base_model_dir=self.base_model_dir,
                                     load_d3qn_model=self.load_d3qn_model,
                                     based_on_base_model=self.based_on_base_model,
                                     d3qn_model_dir=self.d3qn_model_dir, loadTarget=False)
        if self.ddqn:
            self.target_model = D3QN_Classifier(dueling=self.dueling, base_model_dir=self.base_model_dir,
                                                load_d3qn_model=self.load_d3qn_model,
                                                based_on_base_model=self.based_on_base_model,
                                                d3qn_model_dir=self.d3qn_model_dir, loadTarget=True)
            # self.update_target_model(tau=1.0)
        
        self.compile()
    
    def save_model(self, model_dir=None, ):
        '''
        save the RL model. If ddqn, save target_model, else save model.
        :param model_dir:
        :return:
        '''
        
        model_dir = self.d3qn_model_dir if (model_dir is None) else model_dir
        
        fe_dir = os.path.join(model_dir, 'feature_extractor', 'ckpt')
        tf.keras.models.save_model(model=self.feature_extractor, filepath=fe_dir, )
        c_dir = os.path.join(model_dir, 'classifier', 'ckpt')
        tf.keras.models.save_model(model=self.model, filepath=c_dir, )
        if self.ddqn:
            ct_dir = os.path.join(model_dir, 'classifier', 'target_ckpt')
            tf.keras.models.save_model(model=self.target_model, filepath=ct_dir, )
    
    def compile(self, **kwargs):
        '''
        compile the RL model.
        :param kwargs:
        :return:
        '''
        # lr_schedule = tf.keras.experimental.CosineDecay(initial_learning_rate=lr, decay_steps=1e4)
        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.01, decay_steps=decay_steps, decay_rate=0.99, staircase=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.model.compile(optimizer=self.optimizer, loss="mean_squared_error", metrics=['accuracy'], **kwargs)
    
    def update_target_model(self, tau=None):
        '''
        bases on softUpdate, update target_model
        :param tau: the weight of model (not target_model)
        :return:
        '''
        tau = self.tau if tau is None else tau
        if self.ddqn and (not self.softUpdate):
            self.target_model.set_weights(self.model.get_weights())
        elif self.ddqn and self.softUpdate:
            model_theta = self.model.get_weights()
            target_model_theta = self.target_model.get_weights()
            for idx, (weight, target_weight) in enumerate(zip(model_theta, target_model_theta)):
                target_weight = weight * tau + target_weight * (1 - tau)
                target_model_theta[idx] = target_weight
            self.target_model.set_weights(target_model_theta)
    
    def remember(self, state, action, reward, state_, done, feature_extractor=None, **kwargs):
        '''
        save the experience to memory buffer.
        '''
        
        # extract feature to remember
        if feature_extractor is not None:
            state = feature_extractor.get_stft_feature(audio=state)
        state = self.feature_extractor.predict(np.array([state]))[0]
        if state_ is not None:
            if feature_extractor is not None:
                state_ = feature_extractor.get_stft_feature(audio=state_)
            state_ = self.feature_extractor.predict(np.array([state_]))[0]
        experience = state, action, reward, state_, done
        if self.usePER:
            self.MEMORY.append(experience)
        else:
            self.memory.append(experience)
    
    def remember_batch(self, batch_experience, useDiscount=True, feature_extractor=None, **kwargs):
        '''
        save a batch of experience to memory buffer.
        if discount: apply discount to the reward.
        '''
        if useDiscount:
            for i in range(len(batch_experience) - 2, -1, -1):
                batch_experience[i][2] += self.discount_rate * batch_experience[i + 1][2]
        for experience in batch_experience:
            self.remember(*experience, feature_extractor=feature_extractor, **kwargs)
    
    def learn_sample(self, state, action, reward, state_, done, ):
        self.learn_per_batch([state], [action], [reward], [state_], [done], )
    
    def learn_per_batch(self, state, action, reward, state_, done, ):
        '''
        optimize the model for one batch
        :param state:
        :param action:
        :param reward:
        :param state_:
        :param done:
        :return:
        '''
        state = np.array(state)
        action = np.array(action, dtype=np.int32)
        reward = np.array(reward, dtype=np.float32)
        state_ = np.array(state_)
        done = np.array(done, dtype=np.bool)
        
        target = self.model.predict(state)  # predict Q for starting state with the main network
        target_old = np.array(target)
        target_next = self.model.predict(state_)  # predict best action in ending state with the main network
        if self.ddqn:
            target_value = self.target_model.predict(state_)  # predict Q for ending state with the target network
        
        for i in range(len(done)):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                if self.ddqn:  # Double - DQN
                    a = np.argmax(target_next[i])  # current Q Network selects the action
                    target[i][action[i]] = reward[i] + self.discount_rate * (
                        target_value[i][a])  # target Q Network to evaluate
                else:  # Standard - DQN ---- DQN chooses the max Q value among next actions
                    target[i][action[i]] = reward[i] + self.discount_rate * (np.amax(target_next[i]))
        
        self.model.fit(state, target, batch_size=min(len(done), self.batch_size), verbose=0)
        
        return target_old, target
    
    def replay(self, ):
        '''
        experience replay and learn on a batch
        '''
        if self.usePER:
            tree_idx, minibatch = self.MEMORY.sample(self.batch_size)
        else:
            minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        
        # minibatch = np.array(minibatch, dtype=object)
        # state, action, reward, next_state, done = minibatch.T
        # state, next_state = np.concatenate(state), np.concatenate(next_state)
        # action = np.asarray(action, dtype=np.int32)
        # reward = np.asarray(reward, dtype=np.float64)
        # done = np.asarray(done, dtype=np.bool)
        # state, action, reward, state_, done = list(zip(*minibatch))
        # for i, i_state_ in enumerate(state_):
        #     if i_state_ is None:
        #         state_[i] = state[i]
        state, action, reward, state_, done = [], [], [], [], []
        for i_state, i_action, i_reward, i_state_, i_done in minibatch:
            state.append(i_state), action.append(i_action), reward.append(i_reward), done.append(i_done)
            state_.append(i_state_ if (i_state_ is not None) else i_state)
        
        target_old, target = self.learn_per_batch(state, action, reward, state_, done)
        
        if self.usePER:
            indices = np.arange(self.batch_size, dtype=np.int32)
            absolute_errors = np.abs(target_old[indices, action] - target[indices, action])
            # Update priority
            self.MEMORY.batch_update(tree_idx, absolute_errors)
    
    def learn(self, learnTimes=None):
        learnTimes = self.learnTimes if learnTimes is None else learnTimes
        for _ in range(learnTimes):
            self.replay()
    
    def act(self, state, decay_step, ):
        '''
        return the action and explore_prob based on eps_decay
        :param state:
        :param decay_step:
        :return:
        '''
        
        # EPSILON GREEDY STRATEGY
        if self.eps_decay:  # Improved version of epsilon greedy strategy for Q-learning
            explore_prob = self.min_eps + (self.ini_eps - self.min_eps) * self.eps_decay_rate ** decay_step
        else:
            explore_prob = self.min_eps
        
        if explore_prob > random.random():  # Make a random action (exploration)
            return random.randrange(self.num_action), explore_prob
        else:  # Get action from Q-network (exploitation)
            y_pred = self.predict(state)
            print('y_pred:', y_pred[0])
            return np.argmax(y_pred[0]), explore_prob
    
    def predict(self, state, ):
        '''
        produce the output of the given state with model.
        '''
        state = np.array([state])
        state = self.feature_extractor.predict(state)
        
        return self.model.predict(state)


if __name__ == '__main__':
    print('Hello World!')
    
    agent = DQNAgent()
    
    print('Brand-new World!')
