# -*- coding: utf-8 -*-
"""
Created:    2019-08-16 15:32:58
Author:     liuyao8
Descritipn: 
"""

from keras.layers import BatchNormalization, Bidirectional, LSTM, Concatenate, Dropout, \
                        Flatten, Dense, Lambda
from keras.models import Model
from keras import backend as K

from model.BasicModel import BasicDeepModel
from model.Layers import AttentionWeightedAverage


class TextLSTM_Attn(BasicDeepModel):
    """TextLSTM模型，支持char, word和both，支持Attention"""
    
    def __init__(self, config=None, n_rnns=None, rnn_units=64, with_sth='mean', **kwargs):
        if n_rnns is None:
            self.n_rnns = (1, 1) if config.token_level == 'both' else 1
        self.rnn_units = rnn_units
        assert with_sth in ('mean', 'flatten', 'attention')
        self.with_sth = with_sth
        name = 'TextLSTM_Attn_' + with_sth + '_' + config.token_level
        BasicDeepModel.__init__(self, config=config, name=name, **kwargs)
        

    def model_unit(self, inputs, masking, embedding, n_rnns=None, rnn_units=None, with_sth=None):
        """模型主体Unit"""
        if n_rnns is None:
            n_rnns = self.n_rnns
        if rnn_units is None:
            rnn_units = [self.rnn_units] * n_rnns
        if isinstance(rnn_units, int):
            rnn_units = [rnn_units] * n_rnns
        if with_sth is None:
            with_sth = self.with_sth
        
        X = masking(inputs)
        X = embedding(X)
        X = BatchNormalization()(X)
        for i in range(n_rnns):
            X = Bidirectional(LSTM(rnn_units[i], return_sequences=True))(X)  # TODO LSTM VS CuDNNLSTM   128需要动态变化？
            X = Dropout(0.5)(X)     # TODO TextAttention此处为SpatialDropout1D？？？
            
        # X's shape = (None, word_maxlen, 2*rnn_units)  # TODO shape要变成2维的，才能输入到输出层！！！
        if with_sth == 'mean':
            X = Lambda(lambda x: K.mean(x, axis=1))(X)  # (None, 2*rnn_units)  # TODO 不能写成 X=K.mean(X,axis=1)，会报错！
        elif with_sth == 'flatten':
            X = Flatten()(X)                            # (None, word_maxlen*2*rnn_units)
        elif with_sth == 'attention':
            X = AttentionWeightedAverage()(X)           # (None, 2*rnn_units)
        return X
        

    def build_model(self):
        # 模型主体
        if self.config.token_level == 'word':
            X = self.model_unit(self.word_input, self.word_masking, self.word_embedding)
            inputs = [self.word_input]
            
        elif self.config.token_level == 'char':
            X = self.model_unit(self.char_input, self.char_masking, self.char_embedding)
            inputs = [self.char_input]
            
        else:
            word_X = self.model_unit(self.word_input, self.word_masking, self.word_embedding, self.n_rnns[0])
            char_X = self.model_unit(self.char_input, self.char_masking, self.char_embedding, self.n_rnns[1])
            X = Concatenate()([word_X, char_X])
            inputs = [self.word_input, self.char_input]
        
        
        # 结构化特征
        if self.config.structured in ['word', 'char', 'both']:
            X = Concatenate()([X] + self.structured_input)
            inputs = inputs + self.structured_input
        
        
        # 模型结尾
        X = Dropout(0.5)(X)
        out = Dense(self.n_classes, activation=self.activation)(X)
        
        self.model = Model(inputs=inputs, outputs=out)