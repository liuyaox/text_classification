# -*- coding: utf-8 -*-
"""
Created:    2019-07-31  14:51:45
Author:     liuyao8
Descritipn: 
"""

from numpy import array
from pandas import read_csv
import pickle
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

from Vocabulary import seq_to_idxs


def get_encoding_func(vocab, config):
    """生成word和char粒度的数据编码工具"""
    word_encoding = lambda x: seq_to_idxs(str(x).split(), vocab.word2idx, config.WORD_MAXLEN, config.UNK_IDX, config.PAD_IDX)
    char_encoding = lambda x: seq_to_idxs(str(x).split(), vocab.char2idx, config.CHAR_MAXLEN, config.UNK_IDX, config.PAD_IDX)
    return word_encoding, char_encoding


def get_sides(x, maxlen):
    """生成left和right原始数据(未编码) for TextRCNN  注意：只截断不补零"""
    xs = str(x).split()[: maxlen]   # 截断
    x_left = ' '.join(['UNK'] + xs[:-1])
    x_right = ' '.join(xs[1:] + ['UNK'])
    return x_left, x_right


def get_sides_encoding_func(vocab, config):
    """生成left和right原始数据，并编码 for TextRCNN"""
    word_encoding, char_encoding = get_encoding_func(vocab, config)
    word_left_encoding = lambda x: word_encoding(get_sides(x, config.WORD_MAXLEN)[0])
    word_right_encoding = lambda x: word_encoding(get_sides(x, config.WORD_MAXLEN)[1])
    char_left_encoding = lambda x: char_encoding(get_sides(x, config.CHAR_MAXLEN)[0])
    char_right_encoding = lambda x: char_encoding(get_sides(x, config.CHAR_MAXLEN)[1])
    return word_left_encoding, word_right_encoding, char_left_encoding, char_right_encoding


def get_bert_model(config):
    """预训练Bert模型，用于对raw文本编码，raw文本不需分词"""
    from model.Bert.extract_feature import BertVector
    bert_model = BertVector(pooling_strategy='NONE', 
                            max_seq_len=config.bert_maxlen, 
                            bert_model_path=config.bert_model_path, 
                            graph_tmpfile=config.bert_graph_tmpfile)
    return bert_model


def data_config_prepare(config):
    """特征编码，Label编码，Train/Test划分，Config生成，持久化"""
    # 0. 数据准备
    data = read_csv(config.data_file, sep='\t', encoding='utf8')
    data['labels'] = data['labels'].map(lambda x: [] if x == '&&' else x.split('&&'))
    x_raw, x_word_raw, x_char_raw, y_raw = data['question_raw'], data['question_wordseg'], data['question_charseg'], data['labels']
    
    vocab = pickle.load(open(config.vocab_file, 'rb'))      # 词汇表，映射字典，Embedding Layer初始化权重
    config.CHAR_VOCAB_SIZE = vocab.char_vocab_size
    config.WORD_VOCAB_SIZE = vocab.word_vocab_size
    config.char_embed_matrix = vocab.char_embed_matrix
    config.word_embed_matrix = vocab.word_embed_matrix
    
    
    # 1. Token筛选
    
    
    # 2. 特征和Label向量化编码
    # 特征
    config.WORD_MAXLEN = int(1.5 * x_word_raw.map(lambda x: len(str(x).split())).max())    # 57
    config.CHAR_MAXLEN = int(1.5 * x_char_raw.map(lambda x: len(str(x).split())).max())    # 126
    word_encoding, char_encoding = get_encoding_func(vocab, config)
    x_word = array(x_word_raw.map(word_encoding).tolist())
    x_char = array(x_char_raw.map(char_encoding).tolist())
    
    # left和right特征
    word_left_encoding, word_right_encoding, char_left_encoding, char_right_encoding = get_sides_encoding_func(vocab, config)
    x_word_left = array(x_word_raw.map(word_left_encoding).tolist())
    x_word_right = array(x_word_raw.map(word_right_encoding).tolist())
    x_char_left = array(x_char_raw.map(char_left_encoding).tolist())
    x_char_right = array(x_char_raw.map(char_right_encoding).tolist())
    
    # 结构化特征
    word_model_tfidf, x_word_tfidf, word_model_svd, x_word_lsa = pickle.load(open(config.word_tfidf_lsa_file, 'rb'))
    #char_model_tfidf, char_tfidf, char_model_svd, char_lsa = pickle.load(open(config.char_tfidf_lsa_file, 'rb'))
    
    # Bert模型编码
    bert_model = get_bert_model(config)
    bert_vectorizer = lambda x: csr_matrix(bert_model.encode([x])["encodes"][0])
    x_bert = array(x_raw.map(bert_vectorizer).tolist())
    
    # Label
    mlb = MultiLabelBinarizer()
    y_data = mlb.fit_transform(y_raw)       # TODO 使用训练数据还是所有数据来训练mlb？？？
    config.N_CLASSES = len(mlb.classes_)
    config.label_binarizer = mlb
    
    
    # 3. 划分并保存Train/Test    
    x_word_train, x_word_test, x_word_left_train, x_word_left_test, x_word_right_train, x_word_right_test, \
    x_char_train, x_char_test, x_char_left_train, x_char_left_test, x_char_right_train, x_char_right_test, \
    x_word_lsa_train, x_word_lsa_test, x_bert_train, x_bert_test, \
    y_train, y_test = train_test_split(
            x_word, x_word_left, x_word_right, 
            x_char, x_char_left, x_char_right, 
            x_word_lsa, x_bert, 
            y_data, test_size=0.2, random_state=2019)
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
    
    # 保存编码后数据
    pickle.dump((x_train, y_train, x_test, y_test), open(config.data_encoded_file, 'wb'))
    pickle.dump((x_bert_train, y_train, x_bert_test, y_test), open(config.data_bert_file, 'wb'))
    pickle.dump(config, open(config.config_file, 'wb'))


def data_augmentation():
    """数据增强"""
    pass


def example(bert_flag=False):
    import pickle
    from Vocabulary import Vocabulary
    from Config import Config
    config = Config()
    config.n_gpus = 1
    
    
    # Data和Config准备
    data_config_prepare(config)
    data_file = config.data_bert_file if bert_flag else config.data_encoded_file
    x_train, y_train, x_test, y_test = pickle.load(open(data_file, 'rb'))
    config = pickle.load(open(config.config_file, 'rb'))
    
    
    # 模型训练 评估 保存
    if not bert_flag:   # 一般模型
        from model.TextCNN import TextCNN
        textcnn = TextCNN(config)
        test_acc, scores, sims, vectors, _, _ = textcnn.train_predict(x_train, y_train, x_test, y_test, epochs=(2, 10))
        textcnn.model.save(config.model_file)
    
    else:               # Bert模型
        x_train = array([term.toarray() for term in x_train])
        x_test = array([term.toarray() for term in x_test])
        from model.TextBertGRU import TextBertGRU
        textbertgru = TextBertGRU(config)
        test_acc, scores, sims, vectors, history = textbertgru.train_evaluate(x_train, y_train, x_test, y_test)
    
    

if __name__ == '__main__':
    
    example()
    