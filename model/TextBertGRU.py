# -*- coding: utf-8 -*-
"""
Created:    2019-08-23 16:28:24
Author:     liuyao8
Descritipn: 
"""

from keras.layers import Input, Masking, GRU, Dense, BatchNormalization
from keras.optimizers import Adam
from keras.models import Model

from model.BasicModel import BasicDeepModel
from model.Bert.extract_feature import BertVector


class TextBertGRU(BasicDeepModel):
    """
    Bert(Tensorflow实现)预训练模型对原始文本进行向量化编码，输入至RNN模型(Keras实现)里微调
    注意，Bert不参与模型搭建，更不参与训练！相当于提前训练好的Word Embedding那样使用
    TODO 设计成数据编码环节，可通用于其他所有模型！
    """
    
    def __init__(self, config=None, rnn_units=128, dense_units=128, **kwargs):
        self.rnn_units = rnn_units
        self.dense_units = dense_units
        self.bert_model = BertVector(pooling_strategy='NONE',
                                     max_seq_len=config.bert_maxlen, 
                                     bert_model_path=config.bert_model_path, 
                                     graph_tmpfile=config.bert_graph_tmpfile)
        name = 'TextBertGRU'
        BasicDeepModel.__init__(self, config=config, name=name, **kwargs)
        

    def build_model(self):
        """模型结构与BERT没任何关系，只是输入是BERT编码的向量"""
        X = self.bert_masking(self.bert_input)
        X = GRU(self.rnn_units, dropout=0.25, recurrent_dropout=0.25)(X)
        X = Dense(self.dense_units, activation='relu')(X)
        X = BatchNormalization()(X)
        out = Dense(self.n_classes, activation=self.activation)(X)
        self.model = Model(inputs=self.bert_input, outputs=out)
    
    
    def sentence_to_bert(self, sentence):
        """单个句子编码为向量"""
        return self.bert_model.encode([sentence])["encodes"][0]
    
    
    def sentences_to_bert(self, sentences):
        """多个句子编码为向量"""
        return [self.sentence_to_bert(sent.strip()) for sent in sentences]
        
    
    def data_generate(self, sentences, labels):
        """编码数据，生成器"""
        while True:
            for i in range(0, len(sentences), self.batch_size):
                X = self.sentences_to_bert(sentences[i: i + self.batch_size])
                Y = labels[i: i + self.batch_size]
                yield (X, Y)
        
    
    def data_prepare(self):
        """准备train/test/val，未编码"""
        X_train, y_train = None, None
        X_val, y_val = None, None
        X_test, y_test = None, None
        return X_train, y_train, X_val, y_val, X_test, y_test
        
        
    def train(self):
        X_train, y_train, X_val, y_val, X_test, y_test = self.data_generate()
        self.model.compile(loss=self.loss, optimizer=Adam(lr=0.001), metrics=self.metrics)
        self.model.fit_generator(self.data_generate(X_train, y_train),
                                 steps_per_epoch=int(len(X_train)/self.batch_size)+1,
                                 epochs=10,
                                 verbose=1,
                                 validation_data=(X_val, y_val),    # TODO 可不可以0.33这种？
                                 validation_steps=None)             # TODO steps???
        