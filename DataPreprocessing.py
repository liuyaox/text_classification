# -*- coding: utf-8 -*-
"""
Created:    2019-07-03 20:53:31
Author:     liuyao8
Descritipn: 数据预处理，包括两大块
            a. 待标注数据生成：基于最原始数据，生成便于标注的数据格式
            b. 已标注数据处理：已标注数据规整、去除停用词、分词、Label规整等
                以确保在训练Word Embedding、创建Vocabulary等时不需复杂或耗时的额外处理，可直接使用！
"""

import pandas as pd
import jieba
import pickle
from functools import reduce

from Config import Config
config = Config()


# 1. 原始数据 --> 待标注数据
question_path = './data/cellphone_questions.txt'
colnames = ['question_raw', 'spu', 'follows']
data = pd.read_csv(question_path, sep='\t', header=None, names=colnames, encoding='utf8')
data2 = data.sample(frac=0.86, random_state=4321)   # 30181
data2.to_excel('./data/question_cellphone_20190715_30000.xlsx', header=True, index=False, encoding='utf8')


# 2. 已标注数据 --> 训练数据
cols_dic = {'序号': 'no', '性能&系统': 'system', '功能': 'function', '电池': 'battery', '外观': 'appearance', 
            '电话&网络': 'network', '拍照': 'photo', '附件赠品': 'accessory', '购买相关': 'purchase', 
            '品控': 'quality', '配置&硬件': 'hardware', '比较': 'contrast', '标注人': 'annotator'}
annotation = pd.read_excel(config.annotation_file, header=1, encoding='utf8').fillna(0).rename(columns=cols_dic)
annotation['question_raw'] = annotation['question_raw'].map(lambda x: ' '.join(x.split()))                  # 多个空格变1个

stopwords = [open(x, 'r', encoding='utf8').readlines() for x in config.stopwords_files]
stopwords = list(set([x.strip() for x in reduce(lambda x, y: x + y, stopwords)]))   # TODO 重要！加strip，可能会删除一个空格停用词，下面会手动添加
stopwords = stopwords + ['', ' ']   # TODO 非常重要！手动在停用词表中添加空字符串和空格！！！
pickle.dump(stopwords, open(config.public_stopwords_file, 'wb'))

get_wordsegs = lambda x: ' '.join([seg for seg in jieba.cut(x, cut_all=False) if seg not in stopwords])     # TODO 优化点：试试cut_all=True
# TODO 优化点：char-level时的停用词应该与word-level时的停用词不一样！
# 比如，'一'单独出现在word-level分词中说明没别的字可跟它组成词，它就是停用词，但出现在char-level中并不一定
get_charsegs = lambda x: ' '.join([seg for seg in x.replace(' ', '') if seg not in stopwords])              # char-level也要删除停用词

annotation['question_wordseg'] = annotation['question_raw'].map(get_wordsegs)
annotation['question_charseg'] = annotation['question_raw'].map(get_charsegs)

# TODO 重要！使用sklearn.pipeline把get_wordsegs和get_charsegs保存进pipeline！！！包括其中的stopwords!!!

cols_y = ['system', 'function', 'battery', 'appearance', 'network', 'photo', 'accessory', 'purchase', 'quality', 'hardware', 'contrast']
annotation['labels'] = annotation.apply(lambda se: se[cols_y][se[cols_y]==1].index.tolist(), axis=1)
annotation['labels'] = annotation['labels'].apply(lambda x: '&&' if len(x) == 0 else '&&'.join(x))

annotation.to_csv(config.data_file, sep='\t', index=False, encoding='utf8')
