# -*- coding: utf-8 -*-
"""
Created:    2019-08-23 16:25:27
Author:     liuyao8
Descritipn: 样本处理：数据增强
"""

import random


# 1. 数据增强

def data_enhance_for_text(texts, categories, mode='limit'):
    """
    数据增强，打乱老样本序列顺序以生成新样本
    ARGS
        texts: iterable, 每个元素是一个token列表, token既可以是token也可以是token id
        categories: iterable, 每个元素是一个类别id，与texts各元素一一对应
        mode: 数据增强模式
            limit=基于各类别样本数量，为数量少的类别增加新样本，使各类别样本数达到 min(原样本数*2, 最大类别样本数)
            double=所有类别的样本都翻倍，不管各类别原样本数量是多少
    RETURN
        dic2: 字典，key为cate，value为该cate对应的数据增强后的样本列表
    """
    assert mode in ('limit', 'double')

    # 构建类别样本字典: <类别, (样本数, 样本列表)>
    dic1 = {}
    for text, cate in zip(texts, categories):
        if cate not in dic1:
            dic1[cate] = (1, [text, ])
        else:
            dic1[cate][0] += 1
            dic1[cate][1].append(text)
    num_max = max([val[0] for val in dic1.values()])    # 最大类别样本数

    # 数据增强
    dic2 = {}
    for cate, (num, texts) in dic1.items():
        if mode == 'limit':
            num_extra = min(num, num_max - num)             # 数据增强后样本数为 min(原样本数*2, 最大类别样本数)
            texts_extra = random.sample(texts, num_extra)   # 从原样本中随机挑选若干样本用于生成新样本
        else:
            texts_extra = texts.copy()
        for text in texts_extra:
            random.shuffle(text)    # 打乱原序列顺序
            texts.append(text)
        dic2[cate] = texts
    return dic2



if __name__ == '__main__':
    # 项目暂未使用
    pass