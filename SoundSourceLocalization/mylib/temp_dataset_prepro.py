import os
import numpy as np
import soundfile as sf
import subprocess
import glob
import librosa
import random
import tempfile
import shutil
from audiolib import audioread, audiowrite, audio_segmenter, normalize
from lib.utils import get_files_by_suffix, get_dirs_by_prefix, plot_curve
from ns_enhance_onnx import load_onnx_model, denoise_nsnet2
from ssl_gcc_generator import GccGenerator


def decode_dir(dir_path):
    dir_name = os.path.basename(dir_path)
    dir_split = dir_name.split('_')[1:3]
    
    return list(map(float, dir_split))


def decode_file_basename(file_path):
    file_name = os.path.basename(os.path.normpath(file_path))
    file_split = file_name.split('_')[1:5]
    
    return list(map(float, file_split))


def print_info_of_walker_and_sound_source(dataset_path):
    files = get_files_by_suffix(dataset_path, '.wav')
    dirs = get_dirs_by_prefix(dataset_path, 'src')
    
    file_info = []
    x, y, z, d, = set(), set(), set(), set(),
    for i in files:
        temp = decode_file_basename(i)
        file_info.append(temp)
        x.add(temp[0])
        y.add(temp[1])
        z.add(temp[2])
        d.add(temp[3])
    print('-' * 20 + 'info of smart walker' + '-' * 20, '\n', 'x:', sorted(x), '\n', 'y:', sorted(y), '\n', 'z:',
          sorted(z),
          '\n', 'd:', sorted(d), )
    
    dir_info = []
    s_x, s_y, = set(), set(),
    for i in dirs:
        temp = decode_dir(i)
        dir_info.append(temp)
        s_x.add(temp[0])
        s_y.add(temp[1])
    print('-' * 20 + 'info of sound source' + '-' * 20, '\n', 'x:', sorted(s_x), '\n', 'y:', sorted(s_y), )


def plot_map(dataset_path):
    dirs = get_dirs_by_prefix(dataset_path, 'src')
    
    for dir in dirs:
        print('\n', '-' * 20 + 'room_map' + '-' * 20, )
        
        arrows = ['->', '-°', '|^', '°-', '<-', '.|', '!!', '|.']
        files = get_files_by_suffix(dir, '.wav')
        [s_x, s_y] = list(map(float, os.path.basename(dir).split('_')[1:3]))
        print(s_x, s_y)
        
        room_map = np.ndarray((15, 19), dtype=object, )
        for i in files:
            temp = decode_file_basename(i)
            w_x = int(float(temp[0]) * 2) + 7
            w_z = int(float(temp[2]) * 2) + 9
            w_doa = int(temp[3]) // 45
            room_map[w_x, w_z] = arrows[w_doa]
        s_x = int(s_x * 2) + 7
        s_y = int(s_y * 2) + 9
        room_map[s_x, s_y] = 'oo'
        room_map = np.flip(room_map, axis=0)
        room_map = np.flip(room_map, axis=1)
        
        for i in range(len(room_map)):
            print(list(room_map[i]))


def clean_audio_clips(ds_path):
    'delete the audio clips which cannot be paired (four microphones)'
    for src_name in os.listdir(ds_path):
        for walker_name in os.listdir(os.path.join(ds_path, src_name)):
            dir_path = os.path.join(ds_path, src_name, walker_name)
            files = get_files_by_suffix(dir_path, '.wav')
            seg_num = [int(os.path.basename(i)[:-4].split('seg')[-1]) for i in files]
            for i in range(min(seg_num), max(seg_num) + 1):
                seg_files = get_files_by_suffix(dir_path, 'seg' + str(i) + '.wav')
                # print(len(seg_files))
                if len(seg_files) < 4:
                    print(seg_files)
                    for seg_file in seg_files:
                        os.remove(seg_file)


def segment_audio(dataset_root, dir_name, seg_len, overlap_per=0., fs=16000, threshold=None):
    '''Segment the audio clips to segment_len in secs and save them into dir_name'''
    'dataset -> ds  ;  data -> dt  ;    file -> f  ;   dir -> d  '
    
    ini_dspath = os.path.join(dataset_root, 'initial')
    save_dspath = os.path.join(dataset_root, dir_name + '_' + str(fs), 'ini')
    
    files = get_files_by_suffix(ini_dspath, '.wav')
    for file in files:
        # calculate the save_dpath
        file_dir = os.path.dirname(file)
        file_name, _ = os.path.splitext(os.path.basename(file))
        file_name = '_'.join(file_name.split('_')[:-1])
        rel_path = os.path.relpath(file_dir, start=ini_dspath, )
        save_dpath = os.path.join(save_dspath, rel_path, file_name)
        
        # segment the audio
        audio_segmenter(file, save_dpath, segment_len=seg_len, overlap_per=overlap_per, fs=fs, dropout=True,
                        threshold=threshold)
    clean_audio_clips(save_dspath)


def denoise_audio_clips(dataset_root, dir_name, ):
    '''Denoise the audio clips and save them into dir_name'''
    'dataset -> ds  ;  data -> dt  ;    file -> f  ;   dir -> d  '
    
    ini_dspath = os.path.join(dataset_root, dir_name, 'ini')
    save_dspath = os.path.join(dataset_root, dir_name, 'denoise_nsnet2')
    
    files = get_files_by_suffix(ini_dspath, '.wav')
    
    model, _ = load_onnx_model(model_path='./ns_nsnet2-20ms-baseline.onnx')
    
    for file in files:
        # calculate the save_dpath
        file_name, _ = os.path.splitext(os.path.basename(file))
        rel_path = os.path.relpath(file, start=ini_dspath, )
        save_dpath = os.path.join(save_dspath, rel_path, )
        
        # denoise the audio
        denoise_nsnet2(audio_ipath=file, audio_opath=save_dpath, model=model, )


def normalize_audio_clips(dataset_root, dir_name, ):
    '''Denoise the audio clips and save them into dir_name'''
    'dataset -> ds  ;  data -> dt  ;    file -> f  ;   dir -> d  '
    
    ini_dspath = os.path.join(dataset_root, dir_name, 'denoise_nsnet2')
    save_dspath = os.path.join(dataset_root, dir_name, 'normalized_denoise_nsnet2')
    
    files = get_files_by_suffix(ini_dspath, '.wav')
    
    for file in files:
        # calculate the save_dpath
        file_name, _ = os.path.splitext(os.path.basename(file))
        rel_path = os.path.relpath(file, start=ini_dspath, )
        save_dpath = os.path.join(save_dspath, rel_path, )
        
        # normalize the audio
        # denoise_nsnet2(audio_ipath=file, audio_opath=save_dpath, model=model, )
        audio, fs = audioread(file)
        audiowrite(save_dpath, audio, sample_rate=fs, norm=True, target_level=-25, clipping_threshold=0.99)


def decode_audio_path(path):
    '''decode the info of the path of one audio
    ss: sound source
    wk: walker
    '''
    path = os.path.normpath(path)
    [wk_x, wk_z, wk_y, doa] = decode_file_basename(path)
    
    file_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
    file_split = file_name.split('_')[1:3]
    [ss_x, ss_y] = list(map(float, file_split))
    
    return [ss_x, ss_y, wk_x, wk_y, wk_z, doa]


def pack_data_into_dataset(dataset_root, dir_name, data_proprecess_type='ini'):
    '''Denoise the audio clips and save them into dir_name'''
    'dataset -> ds  ;  data -> dt  ;    file -> f  ;   dir -> d  '
    
    ini_dspath = os.path.join(dataset_root, dir_name, data_proprecess_type)
    save_dspath = os.path.join(dataset_root, dir_name + '_' + data_proprecess_type + '.npz')
    if not os.path.exists(os.path.dirname(save_dspath)):
        os.makedirs(os.path.dirname(save_dspath))
    
    ds_audio_list, ds_label_list = [], []
    for src_name in os.listdir(ini_dspath):
        print('-' * 20, 'Processing ', src_name, '-' * 20, )
        src_audio_list, src_label_list = [], []
        for walker_name in os.listdir(os.path.join(ini_dspath, src_name)):
            dir_path = os.path.join(ini_dspath, src_name, walker_name)
            files = get_files_by_suffix(dir_path, '.wav')
            seg_set = set([int(os.path.basename(i)[:-4].split('seg')[-1]) for i in files])
            # print(dir_path, '\n', seg_set)
            for i in seg_set:
                seg_files = sorted(get_files_by_suffix(dir_path, 'seg' + str(i) + '.wav'))
                # print(len(seg_files))
                seg_audio_list, seg_label_list = [], []
                for file in seg_files:
                    # print(os.path.basename(file))
                    audio, _ = audioread(file)
                    seg_audio_list.append(audio)
                    seg_label_list.append(decode_audio_path(file))  # decode the info of the audio
                
                src_audio_list.append(seg_audio_list)
                src_label_list.append(seg_label_list[0])
        ds_audio_list.append(np.array(src_audio_list))
        ds_label_list.append(np.array(src_label_list))
    ds_audio_array = np.array(ds_audio_list, dtype=object)
    ds_label_array = np.array(ds_label_list, dtype=object)
    
    time_len = dir_name.split('_')[0]
    fs = int(dir_name.split('_')[1])
    np.savez(file=save_dspath, x=ds_audio_array, y=ds_label_array, fs=fs, time_len=time_len,
             data_proprecess_type=data_proprecess_type,
             description='The first dimension of this dataset is organized based on the location of different sound sources. \nAnd the following dimensions are sample_number * channels (the number of microphones, which is 4 in this dataset) * sample_points. \nThe label is [ss_x, ss_y, wk_x, wk_y, wk_z, doa] \ni.e. ss: sound source | wk: walker | doa: direction of arrival | \nx , y , z:  vertical coordinate! , horizontal coordinate! , hight of the walker (always be 1 in this dataset)\nfs: sample rate\ntime_len: the time length of one clip\ndata_proprecess_type: the different preprocessing stage of this dataset')


def extract_gcc_phat(dataset_path, ):
    '''Denoise the audio clips and save them into dir_name'''
    'dataset -> ds  ;  data -> dt  ;    file -> f  ;   dir -> d  '
    
    ini_dspath = dataset_path
    save_dspath = os.path.join(ini_dspath[:-4] + '_' + 'gcc_phat' + '.npz')
    if not os.path.exists(os.path.dirname(save_dspath)):
        os.makedirs(os.path.dirname(save_dspath))
    
    dataset = np.load(ini_dspath, allow_pickle=True)
    x = dataset['x']
    gcc_phat = GccGenerator(gcc_width_half=30, gcc_width_half_bias=50)
    gcc_ds_ls = []
    for i in range(len(x)):
        gcc_src_ls = []
        for j in range(len(x[i])):
            audio_ls = x[i][j]
            gcc_seg_ls = []
            for k in range(len(audio_ls)):
                for l in range(k + 1, len(audio_ls)):
                    tau, _, gcc_feature = gcc_phat.gcc_phat(audio_ls[k], audio_ls[l], fs=fs)
                    gcc_seg_ls.append(gcc_feature)
                    
                    # curve_name = ['gcc_feature', ]
                    # curve_data = [gcc_feature, ]
                    # color = ['r', ]
                    # plot_curve(data=list(zip(curve_name, curve_data, color)),
                    #            title=str(tau) + '_' + str(k) + '_' + str(l))
                    #
            gcc_src_ls.append(gcc_seg_ls)
        gcc_seg_array = np.array(gcc_src_ls)
        gcc_ds_ls.append(gcc_seg_array)
    gcc_ds_array = np.array(gcc_ds_ls, dtype=object)
    np.savez(file=save_dspath, x=gcc_ds_array, y=dataset['y'], fs=dataset['fs'], time_len=dataset['time_len'],
             description='The first dimension of this dataset is organized based on the location of different sound sources. \nAnd the following dimensions are sample_number * combinations (the different combinations of two microphones, which is C*2_4 = 6  in this dataset) * gcc_features. \nThe label is [ss_x, ss_y, wk_x, wk_y, wk_z, doa] \ni.e. ss: sound source | wk: walker | doa: direction of arrival | \nx , y , z:  vertical coordinate! , horizontal coordinate! , hight of the walker (always be 1 in this dataset)\nfs: sample rate\ntime_len: the time length of one clip')


if __name__ == '__main__':
    print('-' * 20 + 'Preprocessing the dateset' + '-' * 20)
    dataset_root = '../dataset/hole'
    dataset_ini = os.path.join(dataset_root, 'initial')
    
    # print_info_of_walker_and_sound_source(dataset_ini)
    
    # plot_map(dataset_ini)
    
    segment_para_set = {
        '32ms' : {
            'name'       : '32ms',
            'time_len'   : 32 / 1000,
            'threshold'  : 100,
            'overlap_per': 0.5
        },
        '50ms' : {
            'name'       : '50ms',
            'time_len'   : 50 / 1000,
            'threshold'  : 100,
            'overlap_per': 0.5
        },
        '64ms' : {
            'name'       : '64ms',
            'time_len'   : 64 / 1000,
            'threshold'  : 100,
            'overlap_per': 0.5
        },
        '128ms': {
            'name'       : '128ms',
            'time_len'   : 128 / 1000,
            'threshold'  : 200,  # 100?
            'overlap_per': 0.5
        },
        '256ms': {
            'name'       : '256ms',
            'time_len'   : 256 / 1000,
            'threshold'  : 400,
            'overlap_per': 0.67
        },
        '1s'   : {
            'name'       : '1s',
            'time_len'   : 1024 / 1000,
            'threshold'  : 800,
            'overlap_per': 0.9
        },
    }
    seg_para = segment_para_set['256ms']
    data_proprecess_type = 'ini'
    fs = 16000
    print('-' * 20 + 'parameters' + '-' * 20, '\n', seg_para)
    # segment_audio(dataset_root, seg_para['name'], seg_para['time_len'], seg_para['overlap_per'], fs,
    #               seg_para['threshold'])
    # denoise_audio_clips(dataset_root, dir_name=seg_para['name'] + '_' + str(fs))
    # normalize_audio_clips(dataset_root, dir_name=seg_para['name'] + '_' + str(fs))
    pack_data_into_dataset(dataset_root, dir_name=seg_para['name'] + '_' + str(fs),
                           data_proprecess_type=data_proprecess_type)
    dataset_path = os.path.join(dataset_root, seg_para['name'] + '_' + str(fs) + '_' + data_proprecess_type + '.npz')
    extract_gcc_phat(dataset_path=dataset_path)
