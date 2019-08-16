# -*- coding: utf-8 -*-
"""
Created:    2019-08-16 15:32:58
Author:     liuyao8
Descritipn: 
"""

from keras.layers import Bidirectional, LSTM, Concatenate, Dropout, Flatten, Dense, Lambda
from keras.models import Model
from keras import backend as K

from model.BasicModel import BasicDeepModel
from model.Layers import AttentionWeightedAverage


class TextRNN(BasicDeepModel):
    """TextRNN模型，支持char, word和both，支持Attention"""
    
    def __init__(self, config=None, n_rnns=None, rnn_units=64, with_sth='mean', **kwargs):
        if n_rnns is None:
            self.n_rnns = (1, 1) if config.token_level == 'both' else 1
        self.rnn_units = rnn_units
        assert with_sth in ('mean', 'flatten', 'attention')
        self.with_sth = with_sth
        name = 'TextRNN_' + with_sth + '_' + config.token_level
        BasicDeepModel.__init__(self, config=config, name=name, **kwargs)
        

    def model_unit(self, inputs, masking, embedding, n_rnns=1):
        """模型主体Unit"""
        X = masking(inputs)
        X = embedding(X)
        for _ in range(n_rnns):
            X = Bidirectional(LSTM(self.rnn_units, return_sequences=True))(X)  # TODO LSTM VS CuDNNLSTM   128需要动态变化？
            X = Dropout(0.5)(X)
            
        # X's shape = (None, word_maxlen, 2*rnn_units)  # TODO shape要变成2维的，才能输入到输出层！！！
        if self.with_sth == 'attention':
            X = AttentionWeightedAverage()(X)           # shape = (None, 2*rnn_units)
        elif self.with_sth == 'mean':
            X = Lambda(lambda x: K.mean(x, axis=1))(X)  # shape = (None, 2*rnn_units)  # TODO 不能写成 X=K.mean(X,axis=1)，会报错！
        elif self.with_sth == 'flatten':
            X = Flatten()(X)                            # shape = (None, word_maxlen*2*rnn_units)
        return X
        

    def build_model(self):
        # 模型主体
        if self.config.token_level == 'word':
            X = self.model_unit(self.word_input, self.word_masking, self.word_embedding, self.n_rnns)
            inputs = [self.word_input]
        elif self.config.token_level == 'word':
            X = self.model_unit(self.char_input, self.char_masking, self.char_embedding, self.n_rnns)
            inputs = [self.char_input]
        else:
            word_X = self.model_unit(self.word_input, self.word_masking, self.word_embedding, self.n_rnns[0])
            char_X = self.model_unit(self.char_input, self.char_masking, self.char_embedding, self.n_rnns[1])
            X = Concatenate()([word_X, char_X])
            inputs = [self.word_input, self.char_input]
        
        # 增加结构化特征
        if self.config.structured:
            X = Concatenate()([X] + self.structured_input)
            inputs = inputs + self.structured_input
        
        # 模型结尾
        X = Dropout(0.5)(X)
        out = Dense(self.n_classes, activation=self.activation)(X)
        
        self.model = Model(inputs=inputs, outputs=out)