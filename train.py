# -*- coding: utf-8 -*-
"""
Created:    2019-07-31  14:51:45
Author:     liuyao8
Descritipn: 
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MultiLabelBinarizer

from Vocabulary import seq_to_idxs, PAD_TOKEN, UNK_TOKEN
from config import Config
config = Config()


# 0. 数据准备
data = pd.read_csv(config.training_data_file, sep='\t', encoding='utf8')
data['labels'] = data['labels'].map(lambda x: [] if x == '&&' else x.split('&&'))
x_word_raw, x_char_raw, y_raw = data['question_wordseg'], data['question_charseg'], data['labels']

vocab = pickle.load(open(config.vocab_file, 'rb'))      # 词汇表，映射字典，Embedding Layer初始化权重


# 1. Token筛选



# 2. 数据和Label向量化编码
# 数据
config.WORD_MAXLEN = int(1.5 * x_word_raw.map(lambda x: len(str(x).split())).max())    # 57
config.CHAR_MAXLEN = int(1.5 * x_char_raw.map(lambda x: len(str(x).split())).max())    # 126
word_encoding = lambda x: seq_to_idxs(str(x).split(), vocab.word2idx, config.WORD_MAXLEN, UNK_TOKEN, PAD_TOKEN)
char_encoding = lambda x: seq_to_idxs(str(x).split(), vocab.char2idx, config.CHAR_MAXLEN, UNK_TOKEN, PAD_TOKEN)
x_word = x_word_raw.map(word_encoding)
x_char = x_char_raw.map(char_encoding)

# Label
mlb = MultiLabelBinarizer()
y_data = mlb.fit_transform(y_raw)       # TODO 使用训练数据还是所有数据来训练mlb？？？
config.NUM_CLASSES = len(mlb.classes_)
#y_labels = mlb.inverse_transform(y_data)    # 转化为原来的label

# 保存编码器和编码后的数据，这些数据可直接输入模型进行训练
# TODO 保存word_encoding, char_encoding, mlb这些！
pickle.dump((x_word, x_char, y_data), open(config.training_encoded_file, 'wb'))


# 3. 数据增强



# TODO max_len ??? 向量化编码时才生成，现在还不着急！

