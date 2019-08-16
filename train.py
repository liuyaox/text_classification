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
from sklearn.model_selection import train_test_split

from Vocabulary import seq_to_idxs
from config import Config
config = Config()


def data_config_prepare():
    # 0. 数据准备
    config.n_gpus = 1
    
    data = pd.read_csv(config.data_file, sep='\t', encoding='utf8')
    data['labels'] = data['labels'].map(lambda x: [] if x == '&&' else x.split('&&'))
    x_word_raw, x_char_raw, y_raw = data['question_wordseg'], data['question_charseg'], data['labels']
    
    vocab = pickle.load(open(config.vocab_file, 'rb'))      # 词汇表，映射字典，Embedding Layer初始化权重
    config.CHAR_VOCAB_SIZE = vocab.char_vocab_size
    config.WORD_VOCAB_SIZE = vocab.word_vocab_size
    config.char_embed_matrix = vocab.char_embed_matrix
    config.word_embed_matrix = vocab.word_embed_matrix
    
    
    # 结构化特征正则化 ？？  Normalize input using StandardScaler from scikit learn.
    from sklearn.preprocessing import StandardScaler
    for col in structured_cols:
        scaler = StandardScaler()
        mean_val = features[col].mean()
        features[col].fillna(mean_val, inplace=True)
        values = features[col].values
        values = values.reshape((len(values), 1))
        scaler = scaler.fit(values)
        features[col] = scaler.transform(values.tolist())
    
    
    # 1. Token筛选
    
    # 2. 数据和Label向量化编码
    # 数据
    config.WORD_MAXLEN = int(1.5 * x_word_raw.map(lambda x: len(str(x).split())).max())    # 57
    config.CHAR_MAXLEN = int(1.5 * x_char_raw.map(lambda x: len(str(x).split())).max())    # 126
    word_encoding = lambda x: seq_to_idxs(str(x).split(), vocab.word2idx, config.WORD_MAXLEN, config.UNK_IDX, config.PAD_IDX)
    char_encoding = lambda x: seq_to_idxs(str(x).split(), vocab.char2idx, config.CHAR_MAXLEN, config.UNK_IDX, config.PAD_IDX)
    x_word = np.array(x_word_raw.map(word_encoding).tolist())
    x_char = np.array(x_char_raw.map(char_encoding).tolist())
    
    # Label
    mlb = MultiLabelBinarizer()
    y_data = mlb.fit_transform(y_raw)       # TODO 使用训练数据还是所有数据来训练mlb？？？
    config.N_CLASSES = len(mlb.classes_)
    config.mlb = mlb
    
    # 保存编码后数据(可直接输入模型进行训练)
    x_word_train, x_word_test, x_char_train, x_char_test, y_train, y_test = train_test_split(
            x_word, x_char, y_data, test_size=0.3, random_state=2019)
    x_train = {'word': x_word_train, 'char': x_char_train}
    x_test = {'word': x_word_test, 'char': x_char_test}
    
    pickle.dump((x_train, y_train, x_test, y_test), open(config.data_encoded_file, 'wb'))
    pickle.dump(config, open(config.config_file, 'wb'))
    # TODO 保存word_encoding, char_encoding, mlb这些！


# 3. 数据增强
def data_augmentation():
    pass




# 4. 模型
x_train, y_train, x_test, y_test = pickle.load(open(config.data_encoded_file, 'rb'))    
config = pickle.load(open(config.config_file, 'rb'))

from model.TextCNN import TextCNN
from model.TextCNN2 import TextCNN2

textcnn = TextCNN(config)
scores, sims, vectors = textcnn.model.train_predict(x_train, y_train, x_test, y_test, epochs=(3, 10))
