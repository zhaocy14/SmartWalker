# !/usr/bin/env python
# -*- coding:utf-8 _*-
# @Author: swang
# @Contact: wang00sheng@gmail.com
# @Project Name: SoundSourceLocalization
# @File: temp.py
# @Time: 2022/02/13/15:54
# @Software: PyCharm
import os
import sys
import time
import random
import warnings
import numpy as np
from copy import deepcopy

if __name__ == '__main__':
    
    print('Hello World!')
    while True:
        print('Done?', end='\t')
        while True:
            with open('/home/swadmin/project/SmartWalker-master/SoundSourceLocalization/temp_input.txt', 'r+') as f:
                lines = f.readlines()
            if len(lines) > 0:
                input_str = lines[0].strip('\n')
                print('input_str:', input_str)
                f.truncate(0)
                break
        time.sleep(0.5)

print('Brand-new World!')
