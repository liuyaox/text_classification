# -*- coding: utf-8 -*-
"""
Created:    2019-07-31 16:26:01
Author:     liuyao8
Descritipn: a. 通用Vocabulary：token支持char和word，(token, idx, vector)之间4种映射字典，Embedding Layer初始化权重
            b. 向量化编码工具：支持Padding和Truncating，支持X和Label，支持including和excluding
"""

import numpy as np
from functools import reduce
from tqdm import tqdm
from gensim.models import Word2Vec


# Default word tokens   # TODO 除这4个外，是否还应该有一些别的，比如空格？见P115
PAD_TOKEN = 0   # PAD约定取0，不要改变，以下UNK,SOS,EOS可以改变
UNK_TOKEN = 1   # unknow word   # TODO 原本是没有UNK的？
SOS_TOKEN = 2   # Start of sentence
EOS_TOKEN = 3   # End of sentence 


class Vocabulary(object):
    """token词汇表，token包括word和character"""
    
    def __init__(self):
        # 通用信息
        self.token2idx_init = {'PAD': PAD_TOKEN, 'UNK': UNK_TOKEN, 'SOS': SOS_TOKEN, 'EOS': EOS_TOKEN}
        self.idx2token_init = {PAD_TOKEN: 'PAD', UNK_TOKEN: 'UNK', SOS_TOKEN: 'SOS', EOS_TOKEN: 'EOS'}
        
        # Word Level
        self.word_trimmed = False       # 是否已过滤低频word
        self.word2idx = self.token2idx_init    # TODO 原来是没有{'PAD': PAD_TOKEN, ...}的？
        self.idx2word = self.idx2token_init    # TODO 只存在于word中么？
        self.word2count = {}
        self.num_words = 4
        
        self.word_emb_dim = 0
        self.word2vector = {}
        self.word_idx2vector = {}
        self.word_emb_matrix = None     # Embedding Layer Weights Matrix
        
        # Char Level
        self.char_trimmed = False       # 是否已过滤低频char
        self.char2idx = {}          # TODO self.token2idx_init ???
        self.idx2char = {}
        self.char2count = {}
        self.num_chars = 0
        
        self.char_emb_dim = 0
        self.char2vector = {}
        self.char_idx2vector = {}
        self.char_emb_matrix = None
        
        
    # 1. 创建词汇表
    # 1.1 挨个添加token：直接添加token，通过sentence添加token，通过document添加token
    def add_token(self, token, level='word'):
        """添加word或char，一个一个添加"""
        assert level in ['word', 'char']
        token = token.strip()
        if level == 'word':
            if token not in self.word2idx:
                self.word2idx[token] = self.num_words
                self.idx2word[self.num_words] = token
                self.num_words += 1
                self.word2count[token] = 1
            else:
                self.word2count[token] += 1
        else:
            if token not in self.char2idx:
                self.char2idx[token] = self.num_chars
                self.idx2char[self.num_chars] = token
                self.num_chars += 1
                self.char2count[token] = 1
            else:
                self.char2count[token] += 1
                
                
    def add_sentence(self, sentence, level='word', sep=' '):
        """按sentence添加word或char或both, sentence格式：sep分隔的分词字符串"""
        assert level in ['word', 'char', 'both']
        sentence = str(sentence)
        if level == 'word':
            for word in sentence.strip().split(sep):
                self.add_token(word, level='word')
        elif level == 'char':
            for char in list(sentence.replace(sep, '')):    # 删除分隔符后，变成字符列表
                self.add_token(char, level='char')
        else:
            for word in sentence.strip().split(sep):
                self.add_token(word, level='word')
            for char in list(sentence.replace(sep, '')):
                self.add_token(char, level='char')
        
        
    def add_document(self, document, level='word', sep=' '):
        """按document添加word或char或both"""
        assert level in ['word', 'char', 'both']
        for sentence in document:
            self.add_sentence(sentence, level=level, sep=sep)
            
    
    # 1.2 一次性添加所有token
    def add_all(self, corpus, level='word', sep=' ', min_count=None):
        """
        词汇表 Vocabulary：支持 char-level 和 word-level，以及两者的汇总
        统计 corpus 中 char/word 频率并倒序排序获得 idx，构建词汇字典：<char/word, idx>
        注意：
        其实也可不排序，直接随便赋给每个 char/word 一个 idx，只要保证唯一且固定即可
        比如按加入 Vocabulary 顺序依次赋值为1,2,3,...，0另有所用，比如当作 <PAD>、空格或 <UNK> 的 idx
        TODO idx=0 给谁？？怎么给？？ 也有把PAD和UNK赋值给词汇表里最后2个idx的
        """
        assert level in ['word', 'char', 'both']
        token2count = {}
        for line in corpus:
            tokens = line.strip().split(sep) if level == 'word' else list(line.strip())  # word时默认每一行是分词后分隔好的结果
            for token in tokens:
                token2count[token] = token2count.get(token, 0) + 1
        if min_count:       # 过滤低频字/词
            token2count = {word: num for (word, num) in token2count.items() if num >= min_count}
        
        token_sorted = sorted(token2count, key=token2count.get, reverse=True)       # 按token频率倒序排列
        token_list = token_sorted if ' ' in token_sorted else [' '] + token_sorted  # TODO 空格是否加入vocab？ 如何确定idx=0对应的term???
        
        if level == 'word':
            self.word2count = token2count
            self.word2idx = {word: idx + 4 for (idx, word) in enumerate(token_list)}.update(self.token2idx_init)
            self.idx2word = {idx: word for (word, idx) in self.word2idx.items()}
            self.num_words = len(self.word2idx)
        else:
            self.char2count = token2count
            self.char2idx = {char: idx for (idx, char) in enumerate(token_list)}
            self.idx2char = {idx: char for (char, idx) in self.char2idx.items()}
            self.num_chars = len(self.char2idx)
            
    
    # 2. 低频过滤
    def trim(self, min_count, level='word'):
        """过滤低频word或char"""
        assert level in ['word', 'char']
        if (level == 'word' and self.word_trimmed) or (level == 'char' and self.char_trimmed):
            return
        if level == 'word':
            kept = [word for word, cnt in self.word2count.items() if cnt >= min_count]
            print(f'kept words: {len(kept)} / {len(self.word2idx)} = {len(kept) / len(self.word2idx): .4f}')
            self.word2idx = self.token2idx_init
            self.idx2word = self.idx2token_init
            self.word2count = {}
            self.num_words = 4
            for word in kept:
                self.add_token(word, level='word')
            self.word_trimmed = True
        else:
            kept = [char for char, cnt in self.char2count.items() if cnt >= min_count]
            print(f'kept chars: {len(kept)} / {len(self.char2idx)} = {len(kept) / len(self.char2idx): .4f}')
            self.char2idx = {}
            self.idx2char = {}
            self.char2count = {}
            self.num_chars = 0
            for char in kept:
                self.add_token(char, level='char')
            self.char_trimmed = True
    
    
    # 3. 创建xxx2vector: word, char及其idx到vector的映射字典
    def init_vector(self, embedding=None, level='word'):
        """
        基于训练好的word/char embedding，初始化word2vector或char2vector及其对应的idx2vector
        其中embedding既可以是公开训练好的，也可以是自己训练好的，前者过于巨大，
        后者其实理论上就是word2vector，但实际中可能会因为语料不同步等原因，导致两者的word并不完全相同。
        另外后者可以是gensim.models.Word2Vec模型，也可以是普通字典
        不管前者后者，我们只选择感兴趣的word(word2idx中的word)
        TODO 优化点：增加备用 word embedding  如同get_word2vector_idx2vector一样！
        """
        assert level in ['word', 'char']
        if isinstance(embedding, Word2Vec):
            embedding = {token: embedding[token] for token in embedding.wv.vocab.keys()}
            
        emb_dim = len(list(embedding.values())[0])
        if level == 'word':
            self.word_emb_dim = emb_dim
            for word, idx in self.word2idx.items():
                if word in embedding:
                    vector = embedding.get(word)
                else:
                    vectors = [embedding.get(x, np.random.uniform(-0.01, 0.01, (emb_dim))) for x in list(word)]
                    vector = reduce(lambda x, y: x + y, vectors) / len(vectors)     # OOV时使用对应的若干字符向量的Average
                self.word2vector[word] = vector
                self.word_idx2vector[idx] = vector
        else:
            self.char_emb_dim = emb_dim
            for char, idx in self.char2idx.items():
                vector = embedding.get(char, np.random.uniform(-0.01, 0.01, (emb_dim)))
                self.char2vector[char] = vector
                self.char_idx2vector[idx] = vector


    # 4. 生成Embedding Layer的初始化权重
    def init_embedding_matrix(self, level='word'):
        """基于wordidx2vector或charidx2vector生成用于Embedding Layer的weights matrix
        TODO 总觉得似乎哪里不对？？？<PAD> <UNK>之类的如何处理？
        """
        assert level in ['word', 'char']
        if level == 'word':
            all_embs = np.stack(self.word_idx2vector.values())
            self.word_emb_matrix = np.random.normal(all_embs.mean(), all_embs.std(), size=(self.num_words, self.word_emb_dim))
            for idx, vector in tqdm(self.word_idx2vector.items()):
                self.word_emb_matrix[idx] = vector
        else:
            all_embs = np.stack(self.char_idx2vector.values())
            self.char_emb_matrix = np.random.normal(all_embs.mean(), all_embs.std(), size=(self.num_chars, self.char_emb_dim))
            for idx, vector in tqdm(self.char_idx2vector.items()):
                self.char_emb_matrix[idx] = vector


# 一些与Vocabulary相关的工具

def seq_to_idxs(seq, token2idx, token_maxlen, unk_idx=UNK_TOKEN, pad_idx=PAD_TOKEN, 
                padding='post', truncating='post', including=[], excluding=[]):
    """
    向量化编码：基于词汇表token2idx，把seq转化为idx向量，词汇表中不存在的token使用unk_idx进行编码，适用于特征编码和Label编码
    输入seq是分词/分字列表，如：['我', '们', '爱', '学', '习'] 或 ['我们', '爱', '学习']
    功能 = 向量化 + keras.sequence.pad_sequence
    ARGS
        including: 只关注这里面的token
        excluding: 排除这里面的token
    NOTE
        当including和excluding都存在时同时满足条件，即token in including and token not in excluding
    """
    if len(including) > 0:
        seq = [token for token in seq if token in including]
    seq = [token for token in seq if token not in excluding + ['', ' ']]    # TODO ['', ' ']???
    
    seq_vec = [token2idx.get(token, unk_idx) for token in seq]    # OOV的token标注为专门的unk_idx
    seq_vec = seq_vec[: token_maxlen] if truncating == 'post' else seq_vec[-token_maxlen:]  # 截断：前或后
    paddings = [pad_idx] * (token_maxlen - len(seq_vec))         	               # 小于向量长度的部分用pad_idx来padding
    return seq_vec + paddings if padding == 'post' else paddings + seq_vec      # PAD: 前或后



if __name__ == '__main__':
    # 创建word和char的词汇表，并保存本地
    
    import pandas as pd
    import pickle
    from config import Config
    config = Config()
    
    data = pd.read_csv(config.training_data_file, sep='\t', encoding='utf8')
    sentences_word, sentences_char = data['question_wordseg'], data['question_charseg']
    
    # 创建词汇表
    # TODO 仅仅使用当前任务的全量数据么？要不要加一些其他更全的语料库？应用时，遇到OOV的词汇咋整？
    vocab = Vocabulary()
    vocab.add_document(sentences_word, level='word')
    vocab.add_document(sentences_char, level='char')        # word与char-level使用的数据不一样(停用词不一样)，所以分别单独创建
    vocab.trim(min_count=config.min_count, level='word')    # min_count与训练Embedding时保持一致
    vocab.trim(min_count=config.min_count, level='char')
    # kept words: 5484 / 11692 =  0.4690
    # kept chars: 1594 / 2048 =  0.7783
    
    # 生成xxx2vector和Embedding Layer初始化权重
    model_word2vec = Word2Vec.load(config.model_word2vec_file)
    model_char2vec = Word2Vec.load(config.model_char2vec_file)
    vocab.init_vector(model_word2vec, level='word')
    vocab.init_vector(model_char2vec, level='char')
    vocab.init_embedding_matrix(level='word')
    vocab.init_embedding_matrix(level='char')
    
    # 保存词汇表
    pickle.dump(vocab, open(config.vocab_file, 'wb'))
    