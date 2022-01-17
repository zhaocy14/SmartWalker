import os
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from PIL import Image
import cv2


def next_greater_power_of_2(x):
    return 2 ** (int(x) - 1).bit_length()


def next_lower_power_of_2(x):
    return 2 ** ((int(x) - 1).bit_length() - 1)


def add_prefix_and_suffix_4_basename(path, prefix=None, suffix=None):
    dir_path, basename = os.path.split(path)
    filename, ext = os.path.splitext(basename)
    filename = str(prefix if prefix is not None else '') + filename + str(suffix if suffix is not None else '') + ext
    
    return os.path.join(dir_path, filename)


def standard_normalizaion(x):
    return (x - np.mean(x)) / np.std(x)


def wise_standard_normalizaion(data, normalization=None):
    data = np.array(data)
    if normalization is None:
        return data
    assert normalization in ['sample-wise', 'channel-wise', 'samplepoint-wise']
    data_ndim = data.ndim
    if data_ndim == 2:
        data = data[np.newaxis,]
    for i in range(len(data)):
        if normalization == 'sample-wise':
            data[i, :, :] = standard_normalizaion(data[i, :, :])
        elif normalization == 'channel-wise':
            data[i, :, :] = [standard_normalizaion(data[i, j, :]) for j in range(data.shape[-2])]
        elif normalization == 'samplepoint-wise':
            data[i, :, :] = np.array([standard_normalizaion(data[i, :, j]) for j in range(data.shape[-1])]).T
        else:
            print('-' * 20, 'normalization is incorrectly assigned', '-' * 20)
            exit(1)
    if data_ndim == 2:
        return np.array(data)[0]
    return np.array(data)


def split_data(data, split=0.8, shuffle=True):
    x = data[0]
    y = data[1]
    data_size = len(x)
    split_index = int(data_size * split)
    indices = np.arange(data_size)
    if shuffle:
        indices = np.random.permutation(indices)
    x_train = x[indices[:split_index]]
    y_train = y[indices[:split_index]]
    x_test = x[indices[split_index:]]
    y_test = y[indices[split_index:]]
    return x_train, y_train, x_test, y_test


def split_data_wid(data, split=0.8, shuffle=True):
    x = data[0]
    y = data[1]
    s = data[2]
    data_size = len(x)
    split_index = int(data_size * split)
    indices = np.arange(data_size)
    if shuffle:
        indices = np.random.permutation(indices)
    x_train = x[indices[:split_index]]
    y_train = y[indices[:split_index]]
    s_train = s[indices[:split_index]]
    x_test = x[indices[split_index:]]
    y_test = y[indices[split_index:]]
    return x_train, y_train, s_train, x_test, y_test


def split_data_both(data, split=0.8, shuffle=True):
    x = data[0]
    x_poison = data[1]
    y = data[2]
    s = data[3]
    data_size = len(x)
    split_index = int(data_size * split)
    indices = np.arange(data_size)
    if shuffle:
        indices = np.random.permutation(indices)
    x_train = x[indices[:split_index]]
    x_train_poison = x_poison[indices[:split_index]]
    y_train = y[indices[:split_index]]
    s_train = s[indices[:split_index]]
    x_test = x[indices[split_index:]]
    y_test = y[indices[split_index:]]
    return x_train, x_train_poison, y_train, s_train, x_test, y_test


def shuffle_data(data, random_seed=None):
    '''
    data: [x, y]   type: numpy
    '''
    x, y = data
    data_size = x.shape[0]
    shuffle_index = get_shuffle_index(data_size, random_seed=random_seed)
    
    return x[shuffle_index], y[shuffle_index]


def get_shuffle_index(data_size, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    return np.random.permutation(np.arange(data_size))


def bca(y_true, y_pred):
    m = confusion_matrix(y_true, y_pred)
    numb = m.shape[0]
    acc_each_label = 0
    for i in range(numb):
        acc = m[i, i] / np.sum(m[i, :], keepdims=False).astype(np.float32)
        acc_each_label += acc
    return acc_each_label / numb


def get_split_indices(data_size, split=[9, 1], shuffle=True):
    if len(split) < 2:
        raise TypeError(
            'The length of split should be larger than 2 while the length of your split is {}!'.format(len(split)))
    split = np.array(split)
    split = split / np.sum(split)
    if shuffle:
        indices = get_shuffle_index(data_size)
    else:
        indices = np.arange(data_size)
    split_indices_list = []
    start = 0
    for i in range(len(split) - 1):
        end = start + int(np.floor(split[i] * data_size))
        split_indices_list.append(indices[start:end])
        start = end
    split_indices_list.append(indices[start:])
    return split_indices_list


def batch_iter(data, batchsize, shuffle=True, random_seed=None):
    # Example: batches = list(utils.batch_iter([x_train, y_train], batchsize=batchsize, shuffle=True, random_seed=None))
    
    '''split dataset into batches'''
    if shuffle:
        x, y = shuffle_data(data, random_seed=random_seed)
    else:
        x, y = data
    data_size = x.shape[0]
    nb_batches = np.ceil(data_size / batchsize).astype(np.int)
    
    for batch_id in range(nb_batches):
        start_index = batch_id * batchsize
        end_index = min((batch_id + 1) * batchsize, data_size)
        yield x[start_index:end_index], y[start_index:end_index]


# def batch_iter( data, batchsize, shuffle=True, random_seed=None  ):
#     data = np.array(list(data))
#     data_size = data.shape[0]
#     num_batches = np.ceil(data_size/batchsize).astype(np.int)
#     # Shuffle the data
#     if shuffle:
#         shuffle_indices = get_shuffle_index(data_size)
#         shuffled_data = data[shuffle_indices]
#     else:
#         shuffled_data = data
#     for batch_num in range(num_batches):
#         start_index = batch_num*batchsize
#         end_index = min((batch_num+1)*batchsize, data_size)
#         yield shuffled_data[start_index:end_index]
#


def calculate_accuracy(y, y_pred, target_id=None):
    """
    Computes the accuracy as well as num_adv of attack of the target class.

    Args:
        y: ground truth labels. Accepts one hot encodings or labels.
        y_pred: predicted labels. Accepts probabilities or labels.
        target_id: target class

    Returns:
        accuracy
        accuracy_nb: number of samples which are classified correctly
        target_rate:
        target_total: number of samples which changed their labels from others to target_id
    """
    y = checked_argmax(y, to_numpy=True)  # tf.argmax(y, axis=-1).numpy()
    y_pred = checked_argmax(y_pred, to_numpy=True)  # tf.argmax(y_pred, axis=-1).numpy()
    accuracy = np.mean(np.equal(y, y_pred))
    accuracy_nb = np.sum(np.equal(y, y_pred))
    if target_id is not None:
        non_target_idx = (y != target_id)
        target_total = np.sum((y_pred[non_target_idx] == target_id))
        target_rate = target_total / np.sum(non_target_idx)
        
        # Cases where non_target_idx is 0, so target_rate becomes nan
        if np.isnan(target_rate):
            target_rate = 1.  # 100% target num_adv for this batch
        
        return accuracy, accuracy_nb, target_rate, target_total
    else:
        return accuracy, accuracy_nb


def checked_argmax(y, to_numpy=False):
    """
    Performs an argmax after checking if the input is either a tensor
    or a numpy matrix of rank 2 at least.

    Should be used in most cases for conformity throughout the
    codebase.

    Args:
        y: an numpy array or tensorflow tensor
        to_numpy: bool, flag to convert a tensor to a numpy array.

    Returns:
        an argmaxed array if possible, otherwise original array.
    """
    if y.ndim > 1:
        y = np.argmax(y, axis=-1)
    if to_numpy:
        return np.array(y)
    else:
        return y


def extract_nb_from_str(str):
    pattern = re.compile(r'\d+')
    res = re.findall(pattern, str)
    return list(map(int, res))


def get_files_by_suffix(root, suffix):
    if isinstance(suffix, str):
        suffix = (suffix,)
    else:
        suffix = tuple(suffix)
    file_list = []
    for parent, dirs, files in os.walk(root):
        for f in files:
            path = os.path.normpath(os.path.join(parent, f))
            if path.endswith(suffix):
                # img: (('.jpg', '.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                file_list.append(path)
    return file_list


def get_files_by_prefix(root, prefix):
    if isinstance(prefix, str):
        prefix = (prefix,)
    else:
        prefix = tuple(prefix)
    
    file_list = []
    for parent, dirs, files in os.walk(root):
        for f in files:
            if f.startswith(prefix):
                path = os.path.normpath(os.path.join(parent, f))
                file_list.append(path)
    return file_list


def get_dirs_by_suffix(root, suffix):
    if isinstance(suffix, str):
        suffix = (suffix,)
    else:
        suffix = tuple(suffix)
    
    dir_list = []
    for parent, dirs, files in os.walk(root):
        for d in dirs:
            path = os.path.normpath(os.path.join(parent, d))
            if path.endswith(suffix):
                dir_list.append(path)
    return dir_list


def get_dirs_by_prefix(root, prefix):
    if isinstance(prefix, str):
        prefix = (prefix,)
    else:
        prefix = tuple(prefix)
    
    dir_list = []
    for parent, dirs, files in os.walk(root):
        for d in dirs:
            if d.startswith(prefix):
                path = os.path.normpath(os.path.join(parent, d))
                dir_list.append(path)
    return dir_list


def plot_curve(data, title=None, img_path=None, show=True, y_lim=None, linestyle='-', linewidth=1):
    '''
    data: tuple of every curve's label, data and color
    for example:
        curve_name = ['Training acc_t', 'Validation acc_t', 'Test acc_t']
        curve_data = [train_acc, val_acc, test_acc]
        color = ['r', 'y', 'cyan']
        utils.plot_curve(data=list(zip(curve_name, curve_data, color)), title=title, img_path=img_path)

    '''
    plt.figure()
    for i in data:
        x_len = len(i[1])
        x = list(range(0, x_len))
        plt.plot(x, i[1], i[2], label=i[0], linestyle=linestyle, linewidth=linewidth)
    if y_lim is not None:
        plt.ylim(y_lim)
    plt.title(title)
    plt.legend()
    if img_path is not None:
        if not os.path.exists(os.path.dirname(img_path)):
            os.mkdir(os.path.dirname(img_path))
        plt.savefig(img_path)
    if show:
        plt.show()
    else:
        plt.close()



def plot_hist(data, title=None, img_path=None, bins=100, show=True):
    '''
    data: tuple of every curve's label, data and color
    for example:
        curve_name = ['Training acc_t', 'Validation acc_t', 'Test acc_t']
        curve_data = [train_acc, val_acc, test_acc]
        color = ['r', 'y', 'cyan']
        utils.plot_curve(data=list(zip(curve_name, curve_data, color)), title=title, img_path=img_path)

    '''
    plt.figure()
    for i in data:
        plt.hist(i[1], bins, color=i[2], label=i[0])
    
    # plt.ylim(0, 1.1)
    plt.title(title)
    plt.legend()
    if img_path is not None:
        plt.savefig(img_path)
    if show:
        plt.show()
    else:
        plt.close()


def img_splice(img_paths, save_path, sgl_img_size):
    '''
    img_paths: 2-D list storing the paths of images
    sgl_img_size: size of single image
    '''
    
    width, height = sgl_img_size
    nb_column = max([len(i) for i in img_paths])
    nb_row = len(img_paths)
    res_img = Image.new(mode='RGB', size=(width * nb_column, height * nb_row), color=(255, 255, 255))
    for i in range(len(img_paths)):
        for j in range(len(img_paths[i])):
            # load imgs
            img = Image.open(img_paths[i][j])
            
            res_img.paste(img, (width * j, height * (i),
                                width * (j + 1), height * (i + 1)))
    res_img.save(save_path)
    return res_img


def otsu_threshold(data):
    data = np.array([data], dtype=np.uint8)
    threshold, res_data, = cv2.threshold(data, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return res_data[0], threshold


if __name__ == '__main__':
    dir = 'G:\SmartWalker\SmartWalker-master\SoundSourceLocalization'
    file = 'G:\SmartWalker\SmartWalker-master\SoundSourceLocalization\lib//audiolib.py'
    
    print(add_prefix_and_suffix_4_basename(dir, 13, 14))
    print(add_prefix_and_suffix_4_basename(file, 13, 14))
