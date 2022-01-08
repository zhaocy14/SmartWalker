import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from lib.utils import plot_curve

if __name__ == '__main__':
    ### plot curve
    
    # name = 'Pre-trained Actor Classifier'
    # name = 'Pre-trained Actor Classifier - without learning'
    # name = 'Randomly initialized Actor Classifier - 1'
    # name = 'Randomly initialized Actor Classifier - 2'
    name = 'Randomly initialized Actor Classifier - 3'
    data_path = './' + name + '.npz'
    data = np.load(data_path)['data'][:200]
    ave_n = 5
    data = np.convolve(data, np.ones((ave_n,)) / ave_n, mode='valid')
    curve_name = ['Training reward', ]
    curve_data = [data, ]
    color = ['r', ]
    
    title = 'Episode-wise training reward - ' + name
    img_path = './ave_' + name + '.jpg'
    plot_curve(data=list(zip(curve_name, curve_data, color)), title=title, img_path=img_path, linewidth=0.5)
    print('Name: ', name)
