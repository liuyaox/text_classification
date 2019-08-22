# -*- coding: utf-8 -*-
"""
Created:    2019-08-20 22:58:52
Author:     liuyao8
Descritipn: 
"""

from keras.layers import Input, BatchNormalization, Bidirectional, LSTM, TimeDistributed, Dropout, Concatenate, Dense
from keras.models import Model

from model.BasicModel import BasicDeepModel
from model.Layers import Attention


class TextHAN(BasicDeepModel):
    
    def __init__(self, config=None, rnn_units=128, **kwargs):
        self.rnn_units = rnn_units
        self.sent_maxlen = 30
        self.word_maxlen = config.WORD_MAXLEN
        name = 'TextHAN'
        BasicDeepModel.__init__(self, config=config, name=name, **kwargs)
        
        
    def build_model(self):
        # Word Part
        word_X = self.word_masking(self.word_input)
        word_X = self.word_embedding(word_X)
        word_X = BatchNormalization()(word_X)
        word_X = Bidirectional(LSTM(self.rnn_units, return_sequences=True))(word_X)
        word_out = Attention(self.word_maxlen)(word_X)                      # TODO 能不能使用AttentionAverageWeighted
        model_word = Model(inputs=self.word_input, outputs=word_out)
        
        # Sentence Part
        inputs = Input(shape=(self.sent_maxlen, self.word_maxlen), name='sentence') # TODO 长啥样的！？
        X = TimeDistributed(model_word)(inputs)
        X = BatchNormalization()(X)
        X = Bidirectional(LSTM(self.rnn_units, return_sequences=True))(X)
        X = Attention(self.sent_maxlen)(X)
        
        # 结构化特征
        if self.config.structured in ['word', 'char', 'both']:
            X = Concatenate()([X] + self.structured_input)
            inputs = [inputs, self.structured_input]
        
        # 模型结尾
        X = Dropout(0.5)(X)
        out = Dense(self.n_classes, activation=self.activation)(X)
        
        self.model = Model(inputs=inputs, outputs=out)