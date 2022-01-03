# !/usr/bin/env python
# -*- coding:utf-8 _*-
# @Author: swang
# @Contact: wang00sheng@gmail.com
# @Project Name: SmartWalker-master
# @File: temp_test.py
# @Time: 2022/01/02/21:23
# @Software: PyCharm
import os
import sys
import time
import random
import warnings
import numpy as np
from copy import deepcopy

from multiprocessing.sharedctypes import Array
from multiprocessing import Process, Lock


def add_one(lock, arr):
    lock.acquire()
    for i in range(len(arr)):
        arr[i] += 1
        print(arr[:])
    # lock.release()
    # print(arr[:])


if __name__ == '__main__':
    lock = Lock()
    arr = Array('i', range(10))
    print(arr[:])
    p1 = Process(target=add_one, args=(lock, arr))
    p2 = Process(target=add_one, args=(lock, arr))
    p1.start()
    p2.start()
    
    print('Hello World!')
