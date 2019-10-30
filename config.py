# -*- coding: utf-8 -*-
"""
Created:    2019-08-06 20:00:46
Author:     liuyao8
Descritipn: 
"""

import argparse


class Config(object):

    def __init__(self):

        # 任务相关
        self.task = 'multilabel'
        self.token_level = 'word'       # word: word粒度  char: char粒度  both: word+char粒度
        self.N_CLASSES = 11             # 标签/类别数量
        
        
        # Embedding
        self.MIN_COUNT = 2              # 训练Embedding，创建Vocabulary时要求的低频下限
        self.PUBLIC_EMBED_DIM = 200     # 公开训练好的Embedding向量维度
        self.WORD_EMBED_DIM = 100
        self.CHAR_EMBED_DIM = 100
        self.model_word2vec_file = './local/model_word2vec.w2v'         # 训练好的Word Embedding  
        self.model_char2vec_file = './local/model_char2vec.w2v'         # 训练好的Char Embedding
        
        
        # Vocabulary
        self.PAD_IDX = 0   # PAD约定取0，不要改变，以下UNK,SOS,EOS可以改变
        self.UNK_IDX = 1   # unknow word   # TODO 原本是没有UNK的？
        self.SOS_IDX = 2   # Start of sentence
        self.EOS_IDX = 3   # End of sentence 
        self.vocab_file = './local/vocab.pkl'       # 词汇表，包含word/char,idx,vector三者之间映射字典，Embedding Layer初始化权重
        
        
        # 结构化特征
        # TODO structured改成模型定义时参数！
        self.structured = 'word'        # word: word粒度  char: char粒度  both: word+char粒度  none: 无  
        self.word_svd_n_componets = 100
        self.char_svd_n_componets = 150
        self.word_tfidf_lsa_file = './local/word_tfidf_lsa.pkl'
        self.char_tfidf_lsa_file = './local/char_tfidf_lsa.pkl'
        
        
        # Bert相关
        self.bert_maxlen = 100
        self.bert_dim = 768
        self.bert_model_path = '/home/liuyao58/data/BERT/chinese_L-12_H-768_A-12/'
        self.bert_graph_tmpfile = './tmp_graph_xxx'
        self.data_bert_file = './local/bert_data.pkl'
        
        
        # 特征选择
        self.words_chi2_file = ''       # 基于卡方统计量筛选后的word
        self.chars_chi2_file = ''       # 基于卡方统计量筛选后的char
        
        
        # 数据预处理和编码
        self.data_file = './data/sku_qa_data_30000.csv'               # 处理好的标注数据，尚未编码
        self.data_encoded_file = './local/data_30000_encoded.pkl'     # 向量化编码后的训练数据
        self.WORD_MAXLEN = 100      # 57
        self.CHAR_MAXLEN = 200      # 126
        self.SENT_MAXLEN = 50       # 18
        
        
        # 训练
        self.BATCH_SIZE = 32
        self.n_folds = 5
        self.n_epochs = 10
        self.model_file = './local/model.h5'


        # 其他文件和路径
        self.annotation_file = './data/商品问答_手机_已标注_30000.xlsx'                # 原始的标注数据
        self.stopwords_files = ['./data/京东商城商品评论-Stopwords.txt', 
                                './data/京东商城商品评论-Stopwords-other_github.txt']  # 公开停用词
        self.cleaned_all_stopwords_file = './data/cleaned_all_stopwords.txt'          # 合并处理好的公开停用词
        self.config_file = './local/config.pkl'     # config文件



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
    