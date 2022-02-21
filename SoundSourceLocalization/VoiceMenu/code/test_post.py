# !/usr/bin/env python
# -*- coding:utf-8 _*-
# @Author: swang
# @Contact: wang00sheng@gmail.com
# @Project Name: SoundSourceLocalization
# @File: test_post.py
# @Time: 2022/02/08/14:25
# @Software: PyCharm
import os
import sys
import time
import random
import warnings
import numpy as np
from copy import deepcopy

import requests
import json

if __name__ == '__main__':
    
    def send_help_request():
        url = 'http://smartwalker.cs.hku.hk/smartwalker-backend/api/v1/notification/help'
        s = json.dumps({
            'from'       : 'SW000001',
            'to'         : 'smartwalker-demo-user',
            'probability': 0.2,
        })
        requests.post(url, data=s)
    
    
    print('Hello World!')
    send_help_request()
    print('Brand-new World!')
