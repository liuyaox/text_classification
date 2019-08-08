# -*- coding: utf-8 -*-
"""
Created:    2019-08-06 20:00:46
Author:     liuyao8
Descritipn: 
"""



class Config(object):

    def __init__(self):

        # 一般常量
        self.min_count = 2                  # 训练Embedding，创建Vocabulary时要求的低频下限
        self.word_embedding_dim = 100
        self.char_embedding_dim = 100
        self.public_embedding_dim = 200     # 公开训练好的Embedding向量维度
        self.BATCH_SIZE = 64
        
        
        # 暂时人为指定，根据实际情况会随时修改
        self.n_classes = 11
        self.word_maxlen = 100      # 57
        self.char_maxlen = 200      # 126


        # 常规文件和路径
        self.annotation_file = './data/商品问答_手机_已标注_30000.xlsx'              # 原始的标注数据
        self.stopwords_files = ['./data/京东商城商品评论-Stopwords.txt', 
                                './data/京东商城商品评论-Stopwords-other_github.txt']   # 公开停用词
        
        self.words_chi2_file = ''       # 基于卡方统计量筛选后的word
        self.chars_chi2_file = ''       # 基于卡方统计量筛选后的char
        
        self.training_data_file = './data/sku_qa_training_30000.csv'                # 处理好的标注数据，尚未编码
        self.training_encoded_file = './data/sku_qa_training_30000_encoded.pkl'     # 向量化编码后的训练数据
        
        self.model_word2vec_file = './model/model_word2vec.w2v'         # 训练好的Word Embedding  
        self.model_char2vec_file = './model/model_char2vec.w2v'         # 训练好的Char Embedding
        self.vocab_file = './model/vocab.pkl'       # 词汇表，包含word/char,idx,vector三者之间映射字典，Embedding Layer初始化权重
