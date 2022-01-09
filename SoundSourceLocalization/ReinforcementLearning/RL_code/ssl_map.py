# !/usr/bin/env python
# -*- coding:utf-8 _*-
# @Author: swang
# @Contact: wang00sheng@gmail.com
# @Project Name: RL_Simulation
# @File: Env_MAP.py
# @Time: 2021/10/31/10:19
# @Software: PyCharm


import os
import sys
import time
import random
import numpy as np
from copy import deepcopy
import networkx as nx
import pickle


class Node(object):
    def __init__(self, ):
        super(Node, self).__init__()
    
    def set_coordinate(self, coordinate):
        self.coordinate = np.array(coordinate)
    
    def set_neighbors(self, neighbors):
        self.neighbors = np.array(neighbors)
    
    def set_id(self, id):
        self.id = id
    
    def get_neighbor(self):
        return self.neighbors
    
    def get_coordinate(self):
        return self.coordinate


class Map_graph(object):
    def __init__(self, ds_path=None):
        super(Map_graph, self).__init__()
        self.ds_path = ds_path
        coordinates = [[None, None],
                       [60, 425],
                       [160, 320],
                       [340, 425],
                       [530, 320],
                       [215, 220],
                       [170, 160],
                       [220, 100],
                       [280, 160],
                       [220, 15],
                       [460, 15],
                       [420, 220],
                       [160, 425],
                       [530, 425],
                       [280, 220],
                       [280, 100],
                       [280, 15],
                       [160, 220],
                       [530, 220],
                       [170, 100],
                       [550, 15]]
        map_adj_ls = np.load('../data/adjacency_list.npz')['adjacency_list']
        self.num_node = len(coordinates)
        self.coordinates = np.array(coordinates)
        self.map_adj_ls = np.array(map_adj_ls)
        self.map_graph = self.construct_map_graph()
        self.data_adj_ls = self.cal_data_adjacency_list()
        self.data_graph = self.construct_data_graph()
        self.src_ids = np.where(np.any(self.data_adj_ls, axis=-1))[0]
        print('Number of src_ids: ', len(self.src_ids), '\n', 'src_ids: ', self.src_ids)
        
        self.nodes = np.array([Node() for _ in range(self.num_node)])
        self.__init_nodes__()
        # self.print_map_graph()
    
    def __init_nodes__(self):
        node_idx = np.arange(self.num_node)
        for i, adj_ls in enumerate(self.map_adj_ls):
            if np.all(adj_ls == np.inf):
                self.nodes[i] = None
            else:
                neighbors = np.full((8,), None)
                neighbors[np.array(adj_ls[adj_ls != np.inf], dtype=int)] = node_idx[adj_ls != np.inf]
                # neighbors[np.array(adj_ls[adj_ls != np.inf], dtype=int)] = self.nodes[adj_ls != np.inf]
                self.nodes[i].set_neighbors(neighbors)
                self.nodes[i].set_coordinate(self.coordinates[i])
                self.nodes[i].set_id(i)
    
    def print_map_graph(self):
        print('-' * 20, 'Graph of nodes', '-' * 20, )
        for i, node in enumerate(self.nodes):
            if node is None:
                continue
            print('id: ', node.id)
            print('directions: ', list(range(8)))
            # ids = []
            # for j, neighbor in enumerate(node.neighbors):
            #     if neighbor is not None:
            #         ids.append(neighbor.id)
            #     else:
            #         ids.append('')
            # print('neighbors: ', ids)
            print('neighbors:  ', node.neighbors)
        print('-' * 20, 'Finish printing graph', '-' * 20, )
    
    def cal_data_adjacency_list(self, ):
        def load_dataset():
            # Statistical information of the dataset
            if self.ds_path is None:
                ds_path = '../data/4F_CYC/256ms_0.13_400_16000/norm_drop_denoised_norm_ini_hann_dict_gcc_phat_128.pkl'
            else:
                ds_path = self.ds_path
            with open(ds_path, 'rb') as fo:
                ds = pickle.load(fo)
            return ds['dataset']
        
        def dataset_info(dataset):
            info = dict()
            for src_key in dataset.keys():
                info[src_key] = list(dataset[src_key].keys())
            return info
        
        dataset = load_dataset()
        data_info = dataset_info(dataset)
        del dataset
        # cal_data_adjacency_list
        self.data_adj_ls = np.full((self.num_node, self.num_node), False)
        for src_key in data_info.keys():
            src_coord = list(map(int, src_key.split('_')))
            src_idx = np.where(np.all(self.coordinates == [src_coord], axis=-1))[0][0]
            for wk_key in data_info[src_key]:
                wk_coord = list(map(int, wk_key.split('_')))
                wk_coord = [wk_coord[0], wk_coord[-1]]
                wk_idx = np.where(np.all(self.coordinates == [wk_coord], axis=-1))[0][0]
                # self.data_adj_ls[src_idx][wk_idx] = True
                path = nx.dijkstra_path(self.map_graph, source=src_idx, target=wk_idx)
                for i in path:
                    self.data_adj_ls[src_idx][i] = True
        self.data_adj_ls[np.diag_indices_from(self.data_adj_ls)] = False
        
        print('-' * 20, 'Data graph', '-' * 20, )
        for src_id, adj_ls in enumerate(self.data_adj_ls):
            wk_ids = np.where(adj_ls)[0]
            if len(wk_ids) > 0:
                print('src:', src_id, '-' * 4, 'wk:', np.where(adj_ls)[0])
        print('-' * 20, 'Finish printing graph', '-' * 20, )
        
        return self.data_adj_ls
    
    def construct_map_graph(self, ):
        map_adj_ls = np.array(self.map_adj_ls)
        row, col = np.array(np.where(map_adj_ls != np.inf))
        
        G = nx.Graph()
        for i in {*row, *col}:
            G.add_node(i)
        for i, j in list(zip(row, col)):
            distance = np.linalg.norm(self.coordinates[i] - self.coordinates[j], ord=2)
            G.add_weighted_edges_from([(i, j, distance)])
        
        return G
    
    def construct_data_graph(self, ):  # src -> wk
        data_adj_ls = np.array(self.data_adj_ls)
        row, col = np.array(np.where(data_adj_ls))
        
        G = nx.DiGraph()
        for i in {*row, *col}:
            G.add_node(i)
        for i, j in list(zip(row, col)):
            distance = nx.dijkstra_path_length(self.map_graph, source=i, target=j)
            G.add_weighted_edges_from([(i, j, distance)])
        
        return G
    
    def random_src_id(self):
        return np.random.choice(self.src_ids, 1)[0]
    
    def random_wk_id(self, src_id):  # randomly select one node from src_id's data_children
        children = np.where(self.data_adj_ls[src_id], )[0]
        return np.random.choice(children, 1)[0]
    
    def random_doa(self):
        return np.random.choice(range(8), 1)[0]
    
    def find_id_by_direction(self, base_id, direction):  # find the id in the specific direction of base_id
        return self.nodes[base_id].get_neighbor()[direction]
    
    def find_relative_direction(self, src_id, wk_id, ):  # find where src_id is relative to wk_id
        # TODO 仅支持当前地图
        path = self.find_shortest_map_path(src_id, wk_id, )
        wk_neighbors = self.nodes[wk_id].get_neighbor()
        return np.where(wk_neighbors == path[-2])[0][0]
    
    def cal_relative_doa(self, src_id, wk_id, abs_doa):
        src_2_wk_doa = self.find_relative_direction(src_id, wk_id)
        rela_doa = (src_2_wk_doa - abs_doa + 2 + 16) % 8
        return rela_doa
    
    def is_data_neighbor(self, src_id, wk_id, ):
        return self.data_adj_ls[src_id][wk_id]
    
    def find_shortest_map_path(self, src_id, wk_id):
        path = nx.dijkstra_path(self.map_graph, source=src_id, target=wk_id)
        # distance = nx.dijkstra_path_length(self.map_graph, source=src_id, target=wk_id)
        return path
    
    def find_shortest_data_path(self, src_id, wk_id):
        path = nx.dijkstra_path(self.data_graph, source=src_id, target=wk_id)
        # distance = nx.dijkstra_path_length(self.data_graph, source=src_id, target=wk_id)
        return path
    
    def find_intermediary_src(self, src_id, wk_id):
        path = self.find_shortest_data_path(src_id, wk_id)
        return path[1:-1]
    
    def get_coordinate(self, id):
        return self.nodes[id].get_coordinate()


if __name__ == '__main__':
    # adjacency_list = np.full((21, 21), np.inf)
    # for i in range(21):
    #     for j in range(21):
    #         ipt = input('{:.1f}相对于{:.1f}的位置关系：'.format(j, i))
    #         try:
    #             ipt = int(ipt)
    #             print(ipt)
    #             adjacency_list[i][j] = ipt
    #         except:
    #             pass
    # np.savez('./adjacency_list.npz', adjacency_list=adjacency_list)
    # print(adjacency_list)
    
    map_graph = Map_graph()
