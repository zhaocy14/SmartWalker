# !/usr/bin/env python
# -*- coding:utf-8 _*-
# @Author: swang
# @Contact: wang00sheng@gmail.com
# @Project Name: RL_Simulation
# @File: env.py
# @Time: 2021/10/31/10:43
# @Software: PyCharm
import os
import sys
import time
import random
import warnings
import numpy as np
from copy import deepcopy
from map import Map_graph
import pickle


class MAP_ENV():
    def __init__(self, ):
        super(MAP_ENV, self).__init__()
        self.ds_path = '../data/4F_CYC/256ms_0.13_400_16000/norm_drop_denoised_norm_ini_hann_dict_gcc_phat_128.pkl'
        self.map = Map_graph(ds_path=self.ds_path)
        self.epochs = 1000
        self.num_actions = 8
        self.src_id = None
        self.wk_id = None
        self.abs_doa = None
        self.state = None
        self.done = None
        
        with open(self.ds_path, 'rb') as fo:
            ds = pickle.load(fo)
        self.dataset = ds['dataset']
        del ds
    
    def get_state(self, src_id, wk_id, abs_doa):
        rela_doa = self.map.cal_relative_doa(src_id, wk_id, abs_doa)
        src_coord = self.map.get_coordinate(src_id)
        src_key = '_'.join(list(map(str, src_coord)))
        wk_coord = self.map.get_coordinate(wk_id)
        wk_coord = np.insert(wk_coord, 1, [1], axis=0)
        wk_key = '_'.join(list(map(str, wk_coord)))
        
        state_ls = None
        if wk_key in self.dataset[src_key].keys():
            # print('src_key', src_key, 'wk_key', wk_key)
            str_doa = str(rela_doa * 45)
            if str_doa in self.dataset[src_key][wk_key].keys():
                state_ls = self.dataset[src_key][wk_key][str_doa]
            else:
                for i in range(self.num_actions // 2 + 1):
                    doa_ls = list({(rela_doa + i + self.num_actions) % self.num_actions,
                                   (rela_doa - i + self.num_actions) % self.num_actions, })
                    # print('src_key', src_key, 'wk_key', wk_key)
                    # print('doa: ', self.dataset[src_key][wk_key].keys())
                    # print('doa_ls: ', doa_ls)
                    str_doa_ls = [str(j * 45) for j in doa_ls if str(j * 45) in self.dataset[src_key][wk_key].keys()]
                    if len(str_doa_ls) > 0:
                        str_doa = np.random.choice(str_doa_ls, 1)[0]
                        state_ls = self.dataset[src_key][wk_key][str_doa]
                        break
        else:
            path = self.map.find_shortest_map_path(src_id, wk_id, )
            rela_dir = self.map.find_relative_direction(path[-1], path[-2])
            sub_wk_id = self.map.find_id_by_direction(wk_id, rela_dir)
            sub_wk_coord = self.map.get_coordinate(sub_wk_id)
            sub_wk_coord = np.insert(sub_wk_coord, 1, [1], axis=0)
            sub_wk_key = '_'.join(list(map(str, sub_wk_coord)))
            # print('src_key', src_key, 'wk_key', sub_wk_key)
            
            str_doa = str(rela_doa * 45)
            if str_doa in self.dataset[src_key][sub_wk_key].keys():
                state_ls = self.dataset[src_key][sub_wk_key][str_doa]
            else:
                for i in range(self.num_actions // 2 + 1):
                    doa_ls = list({(rela_doa + i + self.num_actions) % self.num_actions,
                                   (rela_doa - i + self.num_actions) % self.num_actions, })
                    str_doa_ls = [str(j * 45) for j in doa_ls if
                                  str(j * 45) in self.dataset[src_key][sub_wk_key].keys()]
                    if len(str_doa_ls) > 0:
                        str_doa = np.random.choice(str_doa_ls, 1)[0]
                        state_ls = self.dataset[src_key][sub_wk_key][str_doa]
                        break
        if state_ls is None:
            print('src_key 550_15 wk_key 280_1_15  doa:', self.dataset['550_15']['280_1_15'].keys())
            print('src_key', src_key, 'wk_key', wk_key)
            print('src_key', src_key, 'wk_key', sub_wk_key)
            print('str_doa', str(rela_doa * 45))
            print('Wait')
        # state = np.random.choice(state_ls.values(), 1)[0]  # TODO
        state = random.sample(list(state_ls.values()), 1)
        return state
    
    def reset(self):
        self.src_id = [self.map.random_src_id()]
        self.wk_id = self.map.random_wk_id(src_id=self.src_id[-1])
        self.abs_doa = self.map.random_doa()
        self.state = self.get_state(self.src_id[-1], self.wk_id, self.abs_doa)
        self.done = (len(self.src_id) == 0)
        
        return self.state, self.done
    
    def next_position(self, id, abs_action):
        next_id = None
        for i in range(self.num_actions // 2 + 1):
            doa_ls = list({(abs_action + i + self.num_actions) % self.num_actions,
                           (abs_action - i + self.num_actions) % self.num_actions, })
            id_ls = self.map.nodes[id].get_neighbor()[doa_ls]
            id_ls = np.array(id_ls)
            id_ls = id_ls[id_ls != None]
            if len(id_ls) > 0:
                next_id = np.random.choice(id_ls, 1)[0]
                break
        
        return next_id
    
    def cal_abs_action(self, action):
        return (action + self.abs_doa - 2 + self.num_actions) % self.num_actions
    
    def cal_right_action(self, src_id, wk_id, ):
        abs_action = self.map.find_relative_direction(src_id, wk_id, )  # TODO
        return (abs_action - self.abs_doa + 2 + self.num_actions) % self.num_actions
    
    def step(self, action):
        crt_src_id = self.src_id[-1]
        abs_action = self.cal_abs_action(action)
        next_id = self.next_position(self.wk_id, abs_action)
        true_abs_action = self.map.find_relative_direction(next_id, self.wk_id, )
        right_action = self.cal_right_action(crt_src_id, self.wk_id)
        if next_id == crt_src_id:
            self.src_id.pop()
        elif not self.map.is_data_neighbor(crt_src_id, next_id):
            intermediary_src = self.map.find_intermediary_src(crt_src_id, next_id)
            for i in intermediary_src:
                self.src_id.append(i)
                if self.map.is_data_neighbor(i, next_id):
                    break
        self.wk_id = next_id
        self.abs_doa = true_abs_action
        
        # setting rewards
        reward = -1
        if right_action == action:
            reward += 2
        elif abs((right_action - action + self.num_actions) % self.num_actions) <= 1:
            reward += 1
        else:
            reward += -1
        
        if len(self.src_id) == 0:
            done = True
            reward += 100
            state_ = None
        else:
            done = False
            crt_src_id = self.src_id[-1]
            state_ = self.get_state(crt_src_id, self.wk_id, self.abs_doa)
        
        self.state = state_
        info = None
        return state_, reward, done, info
    
    def render(self):
        pass


if __name__ == '__main__':
    map_env = MAP_ENV()
    print('Hello World!')
