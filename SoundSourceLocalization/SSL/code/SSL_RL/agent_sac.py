# !/usr/bin/env python
# -*- coding:utf-8 _*-
# @Author: swang
# @Contact: wang00sheng@gmail.com
# @Project Name: RL_Simulation_D3QN
# @File: temp.py
# @Time: 2022/02/01/19:58
# @Software: PyCharm
import os
import sys
import time
import random
import warnings
import numpy as np
from copy import deepcopy
from collections import deque

import tensorflow as tf
import tensorflow_probability as tfp

from .agent_models import FeatureExtractor, SAC_actor, SAC_critic

EPS = np.finfo(float).eps


class SACAgent(object):
    """Soft Actor-Critic (SAC)

    References
    ----------
    [1] Tuomas Haarnoja*, Aurick Zhou*, Kristian Hartikainen*, George Tucker,
        Sehoon Ha, Jie Tan, Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter
        Abbeel, and Sergey Levine. Soft Actor-Critic Algorithms and
        Applications. arXiv preprint arXiv:1812.05905. 2018.
    """
    
    def __init__(self, num_Q=2, num_action=8, reward_scale=1.0, reward_discount_rate=0.75,
                 policy_lr=4e-4, Q_lr=4e-4, alpha_lr=4e-4,
                 usePER=False, batch_size=8, memory_size=64, learnTimes=1,
                 softUpdate=True, softUpdate_tau=5e-3,
                 base_model_dir='../model/base_model', sac_model_dir='../model/d3qn_model',
                 load_sac_model=True, sac_model_name=None, based_on_base_model=False,
                 **kwargs):
        super(SACAgent, self).__init__()
        assert softUpdate == True, 'softUpdate must be True for SAC'
        assert usePER == False, 'PER has not been created'
        if sac_model_name is None:
            self.name = self.__class__.__name__
            print('Warning:',
                  'A SACAgent class is initialized with default name (\'SACAgent\'). And it may load unexpected models.')
        else:
            self.name = sac_model_name
        
        # space
        self.num_action = num_action
        # reward
        self.reward_scale = reward_scale
        self.discount_rate = reward_discount_rate
        # optimize
        self.learnTimes = learnTimes
        self.softUpdate = softUpdate  # use soft parameter update
        self.tau = softUpdate_tau  # target network soft update hyperparameter
        self.policy_lr = policy_lr
        self.Q_lr = Q_lr
        self.alpha_lr = alpha_lr
        self.target_entropy = self.__heuristic_target_entropy__(self.num_action)
        # memory
        self.usePER = usePER
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.memory = deque(maxlen=memory_size)
        # model
        self.num_Q = num_Q
        self.load_sac_model = load_sac_model
        self.base_model_dir = base_model_dir
        self.based_on_base_model = based_on_base_model
        self.sac_model_dir = os.path.join(sac_model_dir, self.name)
        os.makedirs(self.sac_model_dir, exist_ok=True)
        
        self.feature_extractor = FeatureExtractor(model_dir=self.base_model_dir, )
        self.policy = SAC_actor(base_model_dir=self.base_model_dir,
                                load_sac_model=self.load_sac_model, sac_model_dir=self.sac_model_dir,
                                based_on_base_model=self.based_on_base_model)
        self.Qs = self.load_Q_model(loadTarget=False)
        self.Q_targets = self.load_Q_model(loadTarget=True)
        # self.update_target_model(tau=tf.constant(1.0))
        self.log_alpha, self.alpha = self.load_alpha()
        
        # optimizer
        self.Q_optimizers = tuple(tf.optimizers.Adam(learning_rate=self.Q_lr, name=f'Q_{i}_optimizer')
                                  for i in range(self.num_Q))
        self.policy_optimizer = tf.optimizers.Adam(learning_rate=self.policy_lr, name="policy_optimizer")
        self.alpha_optimizer = tf.optimizers.Adam(learning_rate=self.alpha_lr, name='alpha_optimizer')
    
    def load_alpha(self, ):
        if self.load_sac_model:
            print(f'Loading pre-trained alpha ---- ', end='')
            log_alpha = float(np.load(os.path.join(self.sac_model_dir, 'alpha.npz'))['log_alpha'])
            log_alpha = tf.Variable(log_alpha)
            alpha = tfp.util.DeferredTensor(log_alpha, tf.exp)
            print(float(alpha.numpy()))
        else:
            print('Initializing the alpha ---- 0.')
            log_alpha = tf.Variable(0.0)
            alpha = tfp.util.DeferredTensor(log_alpha, tf.exp)
        
        return log_alpha, alpha
    
    def load_Q_model(self, loadTarget=False):
        Qs = []
        for i in range(self.num_Q):
            Q = SAC_critic(base_model_dir=self.base_model_dir,
                           load_sac_model=self.load_sac_model, sac_model_dir=self.sac_model_dir,
                           based_on_base_model=self.based_on_base_model, index=i, loadTarget=loadTarget)
            Qs.append(Q)
        return Qs
    
    def __heuristic_target_entropy__(self, action_space_size):
        ''' return target_entropy for discrete action space '''
        return -np.log(1.0 / action_space_size) * 0.5
    
    # @tf.function(experimental_relax_shapes=True)
    def __compute_Q_targets__(self, rewards, next_states, dones):
        
        alpha = tf.convert_to_tensor(self.alpha)
        reward_scale = tf.convert_to_tensor(self.reward_scale)
        discount = tf.convert_to_tensor(self.discount_rate)
        
        next_actions, (next_probs, next_log_probs), _ = self.actions_and_log_probs(next_states)
        next_Qs_values = tuple(Q(next_states) for Q in self.Q_targets)
        next_Q_values = tf.reduce_min(next_Qs_values, axis=0)
        next_values = next_probs * (next_Q_values - alpha * next_log_probs)
        next_values = tf.reduce_sum(next_values, axis=1, )
        
        dones = tf.cast(dones, next_values.dtype)
        Q_targets = reward_scale * rewards + discount * (1.0 - dones) * next_values
        
        return tf.stop_gradient(Q_targets)
    
    # @tf.function(experimental_relax_shapes=True)
    def __update_critic__(self, states, actions, rewards, next_states, dones):
        '''
        Update the Q-function.
        See Equations (5, 6) in [1], for further information of the Q-function update rule.
        '''
        
        Q_targets = self.__compute_Q_targets__(rewards, next_states, dones)
        Q_targets = tf.expand_dims(Q_targets, axis=1)
        Qs_values = []
        Qs_losses = []
        for Q, optimizer in zip(self.Qs, self.Q_optimizers):
            with tf.GradientTape() as tape:
                Q_values = Q(states)
                Q_values = tf.gather_nd(Q_values, np.array((np.arange(len(actions)), actions)).T)
                Q_values = tf.expand_dims(Q_values, axis=1)
                Q_losses = 0.5 * (tf.losses.MSE(y_true=Q_targets, y_pred=Q_values))
                Q_loss = tf.nn.compute_average_loss(Q_losses)
            
            gradients = tape.gradient(Q_loss, Q.trainable_variables)
            optimizer.apply_gradients(zip(gradients, Q.trainable_variables))
            Qs_losses.append(Q_losses)
            Qs_values.append(Q_values)
        
        return Qs_values, Qs_losses
    
    # @tf.function(experimental_relax_shapes=True)
    def __update_actor__(self, states):
        '''
        Update the policy.

        See Section 4.2 in [1], for further information of the policy update,
        and Section 5 in [1] for further information of the entropy update.
        '''
        
        alpha = tf.convert_to_tensor(self.alpha)
        states = tf.convert_to_tensor(states)
        
        with tf.GradientTape() as tape:
            actions, (probs, log_probs), _ = self.actions_and_log_probs(states)
            Qs_targets = tuple(Q(states) for Q in self.Qs)
            Q_targets = tf.reduce_min(Qs_targets, axis=0)
            
            policy_losses = alpha * log_probs - Q_targets
            policy_losses = tf.reduce_sum(probs * policy_losses, axis=1)
            policy_loss = tf.nn.compute_average_loss(policy_losses)
        
        policy_gradients = tape.gradient(policy_loss, self.policy.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_gradients, self.policy.trainable_variables))
        
        return policy_losses
    
    # @tf.function(experimental_relax_shapes=True)
    def __update_alpha__(self, states):
        actions, (probs, log_probs), _ = self.actions_and_log_probs(states)
        actions, probs, log_probs = tf.stop_gradient(actions), tf.stop_gradient(probs), tf.stop_gradient(log_probs)
        
        with tf.GradientTape() as tape:
            alpha_losses = -self.alpha * tf.stop_gradient(
                tf.reduce_sum(probs * log_probs, axis=1) + self.target_entropy)  # TODO log_alpha?
            alpha_loss = tf.nn.compute_average_loss(alpha_losses)
            # NOTE(hartikainen): It's important that we take the average here, \
            # otherwise we end up effectively having `batch_size` times too large learning rate.
        
        alpha_gradients = tape.gradient(alpha_loss, [self.log_alpha])
        self.alpha_optimizer.apply_gradients(zip(alpha_gradients, [self.log_alpha]))
        
        return alpha_losses
    
    # @tf.function(experimental_relax_shapes=True)
    def update_target_model(self, tau=None):
        if self.softUpdate:
            tau = self.tau if tau is None else tau
        else:
            tau = 1.0
        for Q, Q_target in zip(self.Qs, self.Q_targets):
            # for source_weight, target_weight in zip(Q.trainable_variables, Q_target.trainable_variables):
            #     target_weight.assign(tau * source_weight + (1.0 - tau) * target_weight)
            model_theta = Q.get_weights()
            target_model_theta = Q_target.get_weights()
            for idx, (weight, target_weight) in enumerate(zip(model_theta, target_model_theta)):
                target_weight = weight * tau + target_weight * (1 - tau)
                target_model_theta[idx] = target_weight
            Q_target.set_weights(target_model_theta)
    
    def save_model(self, model_dir=None, ):
        '''
        save the RL model. If ddqn, save target_model, else save model.
        :param model_dir:
        :return:
        '''
        model_dir = self.sac_model_dir if (model_dir is None) else model_dir
        fe_dir = os.path.join(model_dir, 'feature_extractor', 'ckpt')
        tf.keras.models.save_model(model=self.feature_extractor, filepath=fe_dir, )
        p_dir = os.path.join(model_dir, 'actor', 'ckpt')
        tf.keras.models.save_model(model=self.policy, filepath=p_dir, )
        for i, (Q, Q_target) in enumerate(zip(self.Qs, self.Q_targets)):
            c_dir = os.path.join(model_dir, 'critic', f'ckpt_{i}')
            ct_dir = os.path.join(model_dir, 'critic', f'target_ckpt_{i}')
            tf.keras.models.save_model(model=Q, filepath=c_dir, )
            tf.keras.models.save_model(model=Q_target, filepath=ct_dir, )
        a_dir = os.path.join(model_dir, 'alpha.npz', )
        np.savez(file=a_dir, alpha=np.array(self.alpha), log_alpha=np.array(self.log_alpha))
    
    def remember(self, state, action, reward, state_, done, feature_extractor=None):
        '''
        save the experience to memory buffer.
        
        feature_extractor: extract raw feature

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
        self.memory.append(experience)
    
    def remember_batch(self, batch_experience, useDiscount=True, feature_extractor=None):
        '''
        save a batch of experience to memory buffer.
        if discount: apply discount to the reward.
        '''
        if useDiscount:
            for i in range(len(batch_experience) - 2, -1, -1):
                batch_experience[i][2] += self.discount_rate * batch_experience[i + 1][2]
        for experience in batch_experience:
            self.remember(*experience, feature_extractor=feature_extractor, )
    
    def learn_sample(self, state, action, reward, state_, done, ):
        self.learn_per_batch([state], [action], [reward], [state_], [done], )
    
    def learn_per_batch(self, states, actions, rewards, next_states, dones, ):
        '''
        optimize the model for one batch
        :param state:
        :param action:
        :param reward:
        :param state_:
        :param done:
        :return:
        '''
        states, next_states = np.array(states), np.array(next_states)
        actions = np.array(actions, dtype=np.int32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.bool)
        
        Qs_values, Qs_losses = self.__update_critic__(states, actions, rewards, next_states, dones)
        policy_losses = self.__update_actor__(states)
        alpha_losses = self.__update_alpha__(states)
        
        return (Qs_values, Qs_losses), policy_losses, alpha_losses
    
    def replay(self, ):
        '''
        experience replay and learn on a batch
        '''
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        
        state, action, reward, state_, done = [], [], [], [], []
        for i_state, i_action, i_reward, i_state_, i_done in minibatch:
            state.append(i_state), action.append(i_action), reward.append(i_reward), done.append(i_done)
            state_.append(i_state_ if (i_state_ is not None) else i_state)
        state, state_ = np.array(state), np.array(state_)
        action = np.array(action, dtype=np.int32)
        reward = np.array(reward, dtype=np.float32)
        done = np.array(done, dtype=np.bool)
        
        return state, action, reward, state_, done
    
    def learn(self, learnTimes=None):
        learnTimes = self.learnTimes if learnTimes is None else learnTimes
        for _ in range(learnTimes):
            state, action, reward, state_, done = self.replay()
            self.learn_per_batch(state, action, reward, state_, done)
    
    def actions_and_log_probs(self, states):
        """Given the state, produces an action, the probability of the action, the log probability of the action, and
        the argmax action"""
        
        # states = self.feature_extractor.predict(states)
        action_probs = self.policy(states)
        log_action_probs = tf.math.log(action_probs + EPS)
        max_prob_actions = tf.math.argmax(action_probs, axis=-1)
        actions = tfp.distributions.Categorical(probs=action_probs, dtype=tf.int32).sample()  # .cpu()
        
        return actions, (action_probs, log_action_probs), max_prob_actions
    
    def act(self, state, **kwargs):
        '''
        return the action and explore_prob based on eps_decay
        :param state:
        :return:
        '''
        action_prob = self.predict(state)
        # log_action_prob = np.log(action_prob + EPS)
        # max_prob_action = np.argmax(action_prob, dim=-1)
        action = tfp.distributions.Categorical(probs=action_prob, dtype=tf.int32).sample()[0]  # .cpu()
        return np.asarray(action), \
               (np.round(action_prob[0][action], 3), np.round(self.alpha.numpy(), 3), np.round(action_prob[0], 3))
    
    def predict(self, state, ):
        '''
        produce the output of the given state with model.
        '''
        state = np.array([state])
        state = self.feature_extractor.predict(state)
        
        return self.policy.predict(state)
    
    # @tf.function(experimental_relax_shapes=True)
    # def td_targets(self, rewards, discounts, next_values):
    #     return rewards + discounts * next_values


if __name__ == '__main__':
    print('Hello World!')
    
    agent = SACAgent(based_on_base_model=False)
    
    print('Brand-new World!')
