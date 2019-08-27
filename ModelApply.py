# -*- coding: utf-8 -*-
"""
Created:    2019-08-23 15:20:07
Author:     liuyao8
Descritipn: 
"""

import pickle
from ModelTrain import get_encoding_func, get_sides_encoding_func
from Vocabulary import Vocabulary
from Config import Config
config = Config()


# 加载config
config = pickle.load(open(config.config_file, 'rb'))


# 应用数据处理


# 模型应用
