# -*- coding: utf-8 -*-
"""
Created:    2019-08-07 21:13:53
Author:     liuyao8
Descritipn: word/char选择，基于卡方统计量。
            TODO 注意！word/char是否筛选上了，在Embedding和Vocabulary时可以先不考虑，主要在向量化编码时再考虑是否过滤
"""

import numpy as np



# TODO 建议 onlyin和excluding都要有，有时excluding使用更方便！
