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

from Vocabulary import Vocabulary, seq_to_idxs
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
    
    
    # 1. Token筛选
    
    # 2. 数据和Label向量化编码
    # 数据
    config.WORD_MAXLEN = int(1.5 * x_word_raw.map(lambda x: len(str(x).split())).max())    # 57
    config.CHAR_MAXLEN = int(1.5 * x_char_raw.map(lambda x: len(str(x).split())).max())    # 126
    word_encoding = lambda x: seq_to_idxs(str(x).split(), vocab.word2idx, config.WORD_MAXLEN, config.UNK_IDX, config.PAD_IDX)
    char_encoding = lambda x: seq_to_idxs(str(x).split(), vocab.char2idx, config.CHAR_MAXLEN, config.UNK_IDX, config.PAD_IDX)
    x_word = np.array(x_word_raw.map(word_encoding).tolist())
    x_char = np.array(x_char_raw.map(char_encoding).tolist())
    
    # left和right数据
    def get_sides(x, maxlen):
        """生成left和right原始数据(未编码) for TextRCNN  注意：只截断不补零"""
        xs = str(x).split()[: maxlen]   # 截断
        x_left = ' '.join(['UNK'] + xs[:-1])
        x_right = ' '.join(xs[1:] + ['UNK'])
        return x_left, x_right
    x_word_left = np.array(x_word_raw.map(lambda x: word_encoding(get_sides(x, config.WORD_MAXLEN)[0])).tolist())
    x_word_right = np.array(x_word_raw.map(lambda x: word_encoding(get_sides(x, config.WORD_MAXLEN)[1])).tolist())
    x_char_left = np.array(x_char_raw.map(lambda x: char_encoding(get_sides(x, config.CHAR_MAXLEN)[0])).tolist())
    x_char_right = np.array(x_char_raw.map(lambda x: char_encoding(get_sides(x, config.CHAR_MAXLEN)[1])).tolist())
    
    # 结构化特征
    word_model_tfidf, x_word_tfidf, word_model_svd, x_word_lsa = pickle.load(open(config.word_tfidf_lsa_file, 'rb'))
    #char_model_tfidf, char_tfidf, char_model_svd, char_lsa = pickle.load(open(config.char_tfidf_lsa_file, 'rb'))
    
    # Label
    mlb = MultiLabelBinarizer()
    y_data = mlb.fit_transform(y_raw)       # TODO 使用训练数据还是所有数据来训练mlb？？？
    config.N_CLASSES = len(mlb.classes_)
    config.mlb = mlb
    
    # 保存编码后数据(可直接输入模型进行训练)
    x_word_train, x_word_test, x_word_left_train, x_word_left_test, x_word_right_train, x_word_right_test, x_word_lsa_train, x_word_lsa_test, \
    x_char_train, x_char_test, x_char_left_train, x_char_left_test, x_char_right_train, x_char_right_test, \
    y_train, y_test = train_test_split(
            x_word, x_word_left, x_word_right, x_word_lsa,
            x_char, x_char_left, x_char_right, 
            y_data, test_size=0.3, random_state=2019)
    x_train = {
        'word': x_word_train,
        'word_left': x_word_left_train,
        'word_right': x_word_right_train,
        'word_structured': x_word_lsa_train,
        'char': x_char_train, 
        'char_left': x_char_left_train,
        'char_right': x_char_right_train
    }
    x_test = {
        'word': x_word_test,
        'word_left': x_word_left_test,
        'word_right': x_word_right_test,
        'word_structured': x_word_lsa_test,
        'char': x_char_test,
        'char_left': x_char_left_test,
        'char_right': x_char_right_test
    }
    
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

textcnn = TextCNN(config)
scores, sims, vectors, _, _ = textcnn.train_predict(x_train, y_train, x_test, y_test, epochs=(3, 10))
