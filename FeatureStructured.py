# -*- coding: utf-8 -*-
"""
Created:    2019-08-07 20:59:13
Author:     liuyao8
Descritipn: 结构化特征如TFIDF, LSA, LSI, LDA等
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline



class FeatureStructured(object):
    
    def __init__(self):
        pass 
        
        
    # 1. TFIDF特征
    @classmethod
    def tfidf_vectorizer(cls, data, ngram_range=(1, 1), vocabulary=None, stopwords=None, max_features=None):
        """训练TFIDF模型，并生成TFIDF特征"""
        # model_tfidf.vocabulary_是训练后的字典，是features   max_features=len(vocabulary_)
        model_tfidf = TfidfVectorizer(ngram_range=ngram_range, vocabulary=vocabulary, stop_words=stopwords, 
                                      sublinear_tf=True, max_features=max_features)
        data_tfidf = model_tfidf.fit_transform(data)    # .toarray()  (9, max_features)
        return model_tfidf, data_tfidf
    
    
    # 2. LSA特征
    # LSA转换 = TFIDF转换 + SVD转换
    # In particular, truncated SVD works on term count/tf-idf matrices as returned by the vectorizers 
    # in sklearn.feature_extraction.text. In that context, it is known as latent semantic analysis (LSA).
    
    # TODO **kawgs 实现
    @classmethod
    def lsa_vectorizer(cls, data, ngram_range=(1, 1), vocabulary=None, stopwords=None, 
                       max_features=None, n_components=2, n_iter=5):
        """
        训练LSA模型，并生成LSA特征
        ARGS
            data: iterable of sentence, sentence是空格分隔的分字/分词字符串
                形如 ['小猫咪 爱 吃肉', '我 有 一只 小猫咪', ...]  假设shape为(9, ) (即9个sentence)
            其他：参数及其默认值与 TfidfVectorizer 和 TruncatedSVD 保持一致
        USAGE  
            训练时，data既可以只是train，也可以是train+val+test，应用时分别应用于train/val/test
        """
        model_tfidf = TfidfVectorizer(ngram_range=ngram_range, vocabulary=vocabulary, stop_words=stopwords, 
                                      sublinear_tf=True, max_features=max_features)             # (9, ) -> (9, max_features)
        model_svd = TruncatedSVD(n_components=n_components, n_iter=n_iter, random_state=2019)   #       -> (9, n_components)
        model_lsa = make_pipeline(model_tfidf, model_svd)
        data_lsa = model_lsa.fit_transform(data)
        return model_lsa, data_lsa
    
    
    @classmethod
    def lsa_vectorizer_2steps(cls, data, ngram_range=(1, 1), vocabulary=None, stopwords=None, 
                              max_features=None, n_components=2, n_iter=5):
        """功能同lsa_vectorizer, 可返回训练好的TFIDF和SVD模型，假设 data 维度为(9, )"""
        # TFIDF 转换      (9, max_features)
        model_tfidf, data_tfidf = cls.tfidf_vectorizer(data, ngram_range=ngram_range, vocabulary=vocabulary, 
                                                       stopwords=stopwords, max_features=max_features)
        # SVD 转换
        model_svd = TruncatedSVD(n_components=n_components, n_iter=n_iter, random_state=2018)
        data_lsa = model_svd.fit_transform(data_tfidf)  # (9, n_components)  max_features维稀疏向量 -> n_components维稠密向量
        return model_tfidf, data_tfidf, model_svd, data_lsa
    
    
    
    # 3. LSI特征
    
    
    
    
    # 4. LDA特征




    # 5. Others




def example_lsa():
    """生成TFIDF特征、LSA特征"""
    import pandas as pd
    import pickle
    from config import Config
    config = Config()
    
    data = pd.read_csv(config.data_file, sep='\t', encoding='utf8')
    sentences_word, sentences_char = data['question_wordseg'].fillna(''), data['question_charseg'].fillna('')
    
    vocab = pickle.load(open(config.vocab_file, 'rb'))  # 在main中运行的话，必须 from Vocabulary import Vocabulary
    
    word_model_tfidf, word_tfidf, word_model_svd, word_lsa = FeatureStructured.lsa_vectorizer_2steps(
            sentences_word, vocabulary=vocab.word2idx, n_components=config.word_svd_n_componets)  # 指定vocabulary，保证全局一致性
    char_model_tfidf, char_tfidf, char_model_svd, char_lsa = FeatureStructured.lsa_vectorizer_2steps(
            sentences_char, vocabulary=vocab.char2idx, n_components=config.char_svd_n_componets)
    
    pickle.dump((word_model_tfidf, word_tfidf, word_model_svd, word_lsa), open(config.word_tfidf_lsa_file, 'wb'))
    pickle.dump((char_model_tfidf, char_tfidf, char_model_svd, char_lsa), open(config.char_tfidf_lsa_file, 'wb'))



if __name__ == '__main__':
    
    example_lsa()
