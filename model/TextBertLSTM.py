# -*- coding: utf-8 -*-
"""
Created:    2019-08-23 16:28:24
Author:     liuyao8
Descritipn: 
"""

from keras.layers import Input, Masking, GRU, Dense, BatchNormalization
from keras.models import Model

from model.BasicModel import BasicDeepModel


class TextBertLSTM(BasicDeepModel):
    """
    Bert(Tensorflow实现)预训练模型对原始文本进行向量化编码，输入至RNN模型(Keras实现)里微调
    注意，Bert不参与模型搭建，更不参与训练！相当于提前训练好的Word Embedding那样使用
    TODO 设计成数据编码环节，可通用于其他所有模型！
    """
    
    def __init__(self, config=None, rnn_units=128, dense_units=128, **kwargs):
        self.rnn_units = rnn_units
        self.dense_units = dense_units
        name = 'TextBertLSTM'
        BasicDeepModel.__init__(self, config=config, name=name, **kwargs)
        
    
    def model_unit(self):
        inputs = Input(shape=(self.maxlen, 768, ), name='bert_input')      # TODO 输入数据是3维？？？  768什么鬼？
        X = Masking(mask_value=self.masking_value)(inputs)
        X = GRU(self.rnn_units, dropout=0.25, recurrent_dropout=0.25)(X)
        X = Dense(self.dense_units, activation='relu')(X)
        X = BatchNormalization()(X)
        out = Dense(self.n_classes, activation=self.activation)(X)
        self.model = Model(inputs=inputs, outputs=out)
        