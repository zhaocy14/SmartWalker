import scipy.io as scio
import h5py
import numpy as np
import time
from SoundSourceLocalization.mylib.utils import standard_normalizaion, wise_standard_normalizaion, shuffle_data, \
    split_data
from scipy.signal import resample


def mi_load(data_path, s_id, is_processed=None):
    """ load MI dataset lubin processed """
    data_temp = scio.loadmat(data_path + '/A0' + str(s_id + 1) + '.mat')
    data = np.transpose(data_temp['x'], (2, 0, 1))
    labels = np.asarray(data_temp['y']).squeeze()
    
    if is_processed == 'cov':
        data = cov_process(data)
        data = np.reshape(data, [len(data), -1])
    elif is_processed == 'csp':
        data = csp_process([data, labels], filter)
        data = np.reshape(data, [len(data), -1])
    else:
        data = np.reshape(data, [data.shape[0], 1, data.shape[1], data.shape[2]])
    
    data = standard_normalizaion(data)
    s = s_id * np.ones(shape=[len(labels)])
    
    return data, labels, s


def one_hot_encoder(y, num_classes=None, dtype='float32'):
    """  copied from  tf.keras.utils.to_categorical"""
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def load_hole_dataset(sbj_idx, ds_path, shuffle=True, normalization=None, split=None, one_hot=False):
    ds = np.load(ds_path, allow_pickle=True)
    x = ds['x']
    y = ds['y']
    del ds
    
    x = np.concatenate(x[sbj_idx], axis=0)
    x = np.expand_dims(x, axis=1)
    y = np.concatenate(y[sbj_idx], axis=0)[:, -1] // 45
    if one_hot:
        y = one_hot_encoder(y)
    if normalization is not None:
        for i in range(len(x)):
            x[i] = wise_standard_normalizaion(x[i], normalization)
    if shuffle:
        x, y = shuffle_data([x, y])
    if split is not None:
        split_idx = int(len(y) * split)
        return x[:split_idx], y[:split_idx], x[split_idx:], y[split_idx:]
    return x, y


def cov_process(data):
    """ Covariance matrix """
    cov_data = []
    data_size = len(data)
    for i in range(data_size):
        data_temp = np.dot(data[i], np.transpose(data[i]))  # / np.trace(np.dot(data[i], np.transpose(data[i])))
        data_temp = np.reshape(data_temp, [-1])
        cov_data.append(data_temp)
    
    return np.asarray(cov_data)


def csp_process(data, filter):
    """ Common Spatial Pattern """
    csp_data = []
    data_size = len(data[0])
    for i in range(data_size):
        data_temp = np.dot(filter, data[0][i])
        data_temp = np.dot(data_temp, np.transpose(data_temp)) / np.trace(
            np.dot(data_temp, np.transpose(data_temp)))
        csp_data.append(data_temp)
    
    return np.asarray(csp_data)
