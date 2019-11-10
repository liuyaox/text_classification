# -*- coding: utf-8 -*-
"""
Created:    2019-08-07 21:13:53
Author:     liuyao8
Descritipn: word/char选择，基于卡方统计量。
            TODO 注意！word/char是否筛选上了，在Embedding和Vocabulary时可以先不考虑，主要在向量化编码时再考虑是否过滤
"""

import numpy as np
from collections import Counter


# TODO 建议 onlyin和excluding都要有，有时excluding使用更方便！

# 基于卡方统计量，进行特征选择

def occurrence_matrix(texts, categories):
    """
    基于texts和category原始数据，计算token与category的共现矩阵
    ARGS
        texts: iterable, 每个元素是一个token列表, token既可以是token也可以是token id
        categories: iterable, 每个元素是一个类别id，与texts各元素一一对应
    RETURN
        tokens: tokens列表
        matrix: 列表，元素与tokens一一对应，相当于token与category共现矩阵，可用于计算两者卡方统计量，从而进行特征选择(token选择)
    NOTES
        注意，要求categories是向量化后的类别id，且要求类别id从0开始依次递增，如0,1,2,3,...
    """
    cates_num = len(set(categories))
    dic = {}
    for text, cate in zip(texts, categories):
        for token in set(text):
            if token not in dic:
                dic[token] = [0] * cates_num
                dic[token][cate] += 1
            else:
                dic[token][cate] += 1
    tokens = list(dic.keys())
    matrix = list(dic.values())
    return matrix, tokens


def chi2_value(matrix, mask=True):
    """
    基于共现矩阵计算卡方统计量
    ARGS
        matrix: 二维array或list，共现矩阵，以word，document和document category为例，行是word，列是category，某行某列取值表示：当前category下含有当前word的document数量
        mask: 当category下含有word的document数量为0时，是否不再计算category与word的卡方统计量
    RETURN
        values: 卡方统计量，等于(AD-BC)^2*N/((A+B)(A+C)(B+D)(C+D))
    """
    A = np.array(matrix, dtype=np.float)        # A: category下含有word的样本数量，注意类型为float，以便于后续各种复杂计算
    word_sum = np.sum(A, 1).reshape((-1, 1))    # 各行对应的样本数，转化为列向量
    type_sum = np.sum(A, 0)                     # 各列对应的样本数
    N = np.sum(type_sum)                        # N: 总样本数量  各行各列总和
    B = word_sum - A                            # B: 非category下含有word的样本数量
    C = type_sum - A                            # C: category下不含有word的样本数量
    D = N - A - B - C                           # D: 非category下不含有word的样本数量
    # 若针对每一列，当前列内比较各行，而确定某列后，N, A+C, B+D都是确定不变的，可省略
    # 若针对每一行，当前行内比较各列，而确定某行后，N, A+B, C+D都是确定不变的，可省略
    values = N * (A * D - B * C) ** 2 / ((A + B) * (A + C) * (B + D) * (C + D))
    if mask:
        masking = np.sign(A)       # 当A=0时，value应该为0
        values = masking * values
    return values, A, B, C, D, N


def feature_select_by_chi2(matrix, features, max_col_num=1000, mode='column', mask=True):
    """
    基于卡方统计量进行特征选择
    ARGS
        matrix,mask同chi2_value
        features: 特征列表，特征顺序务必要与matrix各行/列保持一致！用于特征索引转换为特征
        max_col_num: 每列可选择的特征数量最大值
        model: 特征选择的模式，column=各列分别选择特征然后汇总选择的特征，max=取特征各列卡方值最大值为特征卡方值从而选择特征，avg=取平均值
    RETURN
        cnter: collections.Counter，类似字典，表示选择的特征，及其被多少列选择
        selected: 列表，表示选择的特征
    """
    values, A, _, _, _, _ = chi2_value(matrix, mask)
    # 共有3种模式进行特征选择
    if mode == 'column':
        masking = np.sign(A)
        col_num = np.sum(masking, 0, dtype=np.int64)    # 各列拥有的特征数量，注意dtype为int，否则为float
        selected = []
        for i in range(A.shape[1]):                     # 遍历各列
            indices = np.argsort(values[:, i])          # 按卡方统计量排序各特征，取其排序索引
            k = min(max_col_num, col_num[i])
            topk = [features[i] for i in indices[-k:]]  # 前k个特征
            selected.extend(topk)
        cnter = Counter(selected)
        return cnter
    elif mode == 'avg':
        value = np.mean(values, axis=1)
    elif mode == 'max':
        value = np.max(values, axis=1)
    else:
        raise ValueError('mode must be column, avg or max !')
    indices = np.argsort(value)
    selected = [features[i] for i in indices[-max_col_num:]]
    return selected



if __name__ == '__main__':
    
    # 以下只是示例，项目中暂时未使用特征选择
    # 示例：基于卡方统计量进行特征选择
    texts = [['t1', 't2', 't3', 't4'], ['t2', 't3', 't5'], ['t1', 't4', 't5'], ['t2','t4'], ['t3', 't4'], ['t1', 't3', 't4']]
    categories = [1, 2, 0, 1, 0, 1]
    matrix, tokens = occurrence_matrix(texts, categories)
    cnter = feature_select_by_chi2(matrix, tokens)  # cnter即为选择的特征及其被选择的次数
