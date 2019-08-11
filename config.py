# -*- coding: utf-8 -*-
"""
Created:    2019-08-06 20:00:46
Author:     liuyao8
Descritipn: 
"""

import argparse


class Config(object):

    def __init__(self):

        # 一般常量
        self.MIN_COUNT = 2                  # 训练Embedding，创建Vocabulary时要求的低频下限
        self.WORD_EMBEDDING_DIM = 100
        self.CHAR_EMBEDDING_DIM = 100
        self.PUBLIC_EMBEDDING_DIM = 200     # 公开训练好的Embedding向量维度
        self.BATCH_SIZE = 64
        
        
        # 暂时人为指定，根据实际情况会随时修改
        self.NUM_CLASSES = 11
        self.WORD_MAXLEN = 100      # 57
        self.CHAR_MAXLEN = 200      # 126


        # 常规文件和路径
        self.annotation_file = './data/商品问答_手机_已标注_30000.xlsx'              # 原始的标注数据
        self.stopwords_files = ['./data/京东商城商品评论-Stopwords.txt', 
                                './data/京东商城商品评论-Stopwords-other_github.txt']   # 公开停用词
        self.public_stopwords_file = './data/public_stopwords.txt'      # 合并处理好的公开停用词
        
        self.words_chi2_file = ''       # 基于卡方统计量筛选后的word
        self.chars_chi2_file = ''       # 基于卡方统计量筛选后的char
        
        self.training_data_file = './data/sku_qa_training_30000.csv'                # 处理好的标注数据，尚未编码
        self.training_encoded_file = './data/sku_qa_training_30000_encoded.pkl'     # 向量化编码后的训练数据
        
        self.model_word2vec_file = './local/model_word2vec.w2v'         # 训练好的Word Embedding  
        self.model_char2vec_file = './local/model_char2vec.w2v'         # 训练好的Char Embedding
        self.vocab_file = './local/vocab.pkl'       # 词汇表，包含word/char,idx,vector三者之间映射字典，Embedding Layer初始化权重

        self.svd_n_componets = {'word': 100, 'char': 150}
        self.word_tfidf_lsa_file = './local/word_tfidf_lsa.pkl'
        self.char_tfidf_lsa_file = './local/char_tfidf_lsa.pkl'
        


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--server',         default=None, type=int, help='[6099]')
    parser.add_argument('--phase',          default=None, help='[Train/Test]')
    parser.add_argument('--sen_len',        default=None, type=int, help='sentence length')

    parser.add_argument('--net_name',       default=None, help='[lstm]')
    parser.add_argument('--dir_date',       default=None, help='Name it with date, such as 20180102')
    parser.add_argument('--batch_size',     default=32, type=int, help='Batch size')
    parser.add_argument('--lr_base',        default=1e-3, type=float, help='Base learning rate')
    parser.add_argument('--lr_decay_rate',  default=0.1, type=float, help='Decay rate of lr')
    parser.add_argument('--epoch_lr_decay', default=1000, type=int, help='Every # epoch, lr decay lr_decay_rate')

    parser.add_argument('--layer_num',      default=2, type=int, help='Lstm layer number')
    parser.add_argument('--hidden_size',    default=64, type=int, help='Lstm hidden units')
    parser.add_argument('--gpu',            default='0', help='GPU id list')
    parser.add_argument('--workers',        default=4, type=int, help='Workers number')

    return parser.parse_args()



if __name__ == '__main__':
    
    args = get_args()
    gpu = args.gpu
    