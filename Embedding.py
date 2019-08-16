# -*- coding: utf-8 -*-
"""
Created:    2019-08-06 21:24:39
Author:     liuyao8
Descritipn: 
"""

import numpy as np
from gensim.models import Word2Vec


class CorpusGenerator(object):
    """
    使用 gensim 生成 Word2Vec 所需的语料 Generator，由文件直接生成，支持 word-level 和 char-level
    NOTES
        文件每行必须事先完成分词或分字：每一行是分隔的词或字的字符串，形如：'颜色 很 漂亮' 或 '颜 色 很 漂 亮'
    """
    def __init__(self, corpus_file, stopwords=[], sep=' '):
        self.corpus_file = corpus_file
        self.stopwords = stopwords
        self.sep = sep

    def __iter__(self):
        for line in open(self.corpus_file):
            # 输出结果：每个元素形如['颜色', '很', '漂亮'] 或 ['颜', '色', '很', '漂', '亮']，过滤指定词或字(如停用词等)
            yield [x for x in line.strip().split(self.sep) if x not in self.stopwords]


def train_w2v_model(sentences, size=100, min_count=3, window=5, sg=1, workers=8, iter=8, compute_loss=True):
    """
    训练 Word2Vec 字/词向量
    ARGS
        sentences: iterable of sentence, 其中sentence是分字/分词列表，形如：['颜色', '很', '漂亮'] 或 ['颜', '色', '很', '漂', '亮']
        其他：与Word2Vec函数参数保持一致，sg=1表示使用skip-gram算法
    RETURN
        model: 训练好的Word2Vec模型，包含(idx, token, vector)三者之间的4种映射字典：idx2token, idx2vector, token2idx, token2vector(即model.wv)
    USAGE
        待完善……
    """
    model = Word2Vec(sentences, size=size, min_count=min_count, window=window, sg=sg, workers=workers, iter=iter, compute_loss=compute_loss)
    model.idx2token = {}
    model.token2idx = {}
    model.idx2vector = {}
    for token in model.wv.vocab.keys():
        idx = model.wv.vocab[token].index    # token对应的idx
        model.idx2token[idx] = token
        model.token2idx[token] = idx
        model.idx2vector[idx] = model[token] # 可直接使用model[token]，当然也可model.wv[token]
    return model


def pretrained_embedding(embedding_file, seps=('\t', ','), header=False):
    """Public Pretrained Embedding File --> Original Full Embedding"""
    embedding = {}
    with open(embedding_file, 'r', encoding='utf-8') as fr:
        if header:
            fr.readline()                        # Drop line 1
        for line in fr:
            values = line.strip().split(seps[0])
            if len(values) >= 2:
                token = values[0]
                vector = values[1:] if seps[0] == seps[1] else values[1].split(seps[1])
                embedding[token] = np.asarray(vector, dtype='float32')
    return embedding



def example():
    """训练Word2Vec向量，并保存本地"""
    import pandas as pd
    from config import Config
    config = Config()
    
    data = pd.read_csv(config.data_file, sep='\t', encoding='utf8')
    sentences_word = data['question_wordseg'].map(lambda x: str(x).strip().split(' '))
    sentences_char = data['question_charseg'].map(lambda x: str(x).strip().split(' '))
    
    model_word2vec = train_w2v_model(sentences_word, size=config.WORD_EMBED_DIM, min_count=config.MIN_COUNT)
    model_char2vec = train_w2v_model(sentences_char, size=config.CHAR_EMBED_DIM, min_count=config.MIN_COUNT, window=10, iter=15)
    print(len(model_word2vec.wv.vocab))     # 5484
    print(len(model_char2vec.wv.vocab))     # 1595

    model_word2vec.save(config.model_word2vec_file)
    model_char2vec.save(config.model_char2vec_file)
    
    

if __name__ == '__main__':
    
    example()
    