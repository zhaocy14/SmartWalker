# !/usr/bin/env python
# -*- coding:utf-8 _*-
# @Author: swang
# @Contact: wang00sheng@gmail.com
# @Project Name: SmartWalker-master
# @File: temp1.py
# @Time: 2022/03/01/22:13
# @Software: PyCharm
import os, sys
import time
import random
import warnings
import numpy as np
from copy import deepcopy
from threading import Thread


class Number(object):
    _instance = None
    
    @staticmethod
    def get_instance():
        if Number._instance is None:
            Number()
        return Number._instance
    
    def __init__(self, ):
        if Number._instance is not None:
            raise Exception('only one instance can exist')
        else:
            Number._instance = self
    
    def start(self, i):
        i = 0
        while True:
            i += 1
            time.sleep(5)
            
            yield i


def print_number(i):
    NumberObject = Number.get_instance()
    
    for number in NumberObject.start(i=0):
        print(i, end='')


if __name__ == '__main__':
    print('Hello World!')
    p1 = Thread(target=print_number, args=('-',))
    p2 = Thread(target=print_number, args=('*',))
    p1.start()
    p2.start()
    
    print('Brand-new World!')
