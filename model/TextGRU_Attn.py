# -*- coding: utf-8 -*-
"""
Created:    2019-08-17 18:43:40
Author:     liuyao8
Descritipn: 
"""

from keras.layers import BatchNormalization, Bidirectional, GRU, SpatialDropout1D, Lambda, \
                        GlobalMaxPooling1D, GlobalAveragePooling1D, Concatenate, Dropout, Dense
from keras.models import Model

from model.BasicModel import BasicDeepModel
from model.Layers import AttentionWeightedAverage


class TextGRU_Attn(BasicDeepModel):
    
    def __init__(self, config=None, n_rnns=None, rnn_units=64, dropout_p=0.5, **kwargs):
        if n_rnns is None:
            self.n_rnns = (2, 2) if config.token_level == 'both' else 2
        self.rnn_units = rnn_units
        self.dropout_p = dropout_p
        name = 'TextGRU_Attn_' + config.token_level
        BasicDeepModel.__init__(self, config=config, name=name, **kwargs)
        
    
    def model_unit(self, inputs, masking, embedding, n_rnns=None, rnn_units=None, dropout_p=None):
        """模型主体Unit"""
        if n_rnns is None:
            n_rnns = self.n_rnns
        if rnn_units is None:
            rnn_units = [self.rnn_units] * n_rnns
        if isinstance(rnn_units, int):
            rnn_units = [rnn_units] * n_rnns
        if dropout_p is None:
            dropout_p = [self.dropout_p] * n_rnns
        if isinstance(dropout_p, float):
            dropout_p = [dropout_p] * n_rnns
        
        X = masking(inputs)
        X = embedding(X)
        X = BatchNormalization()(X)
        for i in range(n_rnns):
            X = Bidirectional(GRU(rnn_units[i], return_sequences=True))(X)
            X = SpatialDropout1D(dropout_p[i])(X)
        
        maxpool = GlobalMaxPooling1D()(X)
        avgpool = GlobalAveragePooling1D()(X)
        last = Lambda(lambda x: x[:, -1])(X)
        attn = AttentionWeightedAverage()(X)
        X = Concatenate()([maxpool, avgpool, last, attn])
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