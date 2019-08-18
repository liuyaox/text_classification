# -*- coding: utf-8 -*-
"""
Created:    2019-08-17 21:42:28
Author:     liuyao8
Descritipn: 
"""

from keras.layers import BatchNormalization, SpatialDropout1D, Bidirectional, LSTM, GRU, \
                        GlobalMaxPooling1D, GlobalAveragePooling1D, Concatenate, Dropout, Dense
from keras.models import Model

from model.BasicModel import BasicDeepModel
from model.Layers import AttentionWeightedAverage


class TextLSTMGRU(BasicDeepModel):
    
    def __init__(self, config=None, rnn_units=64, dropout_p=0.25, with_attention=False, **kwargs):
        self.rnn_units = rnn_units
        self.dropout_p = dropout_p
        self.with_attention = with_attention
        name = 'TextLSTMGRU_' + config.token_level
        BasicDeepModel.__init__(self, config=config, name=name, **kwargs)
        
        
    def model_unit(self, inputs, masking, embedding, dropout_p=None, rnn_units=None, with_attention=None):
        """模型主体Unit"""
        if dropout_p is None:
            dropout_p = (self.dropout_p, self.dropout_p)
        if isinstance(dropout_p, float):
            dropout_p = (dropout_p, dropout_p)
        if rnn_units is None:
            rnn_units = (self.rnn_units, self.rnn_units)
        if isinstance(rnn_units, int):
            rnn_units = (rnn_units, rnn_units)
        if with_attention is None:
            with_attention = self.with_attention
            
        X = masking(inputs)
        X = embedding(X)
        X = BatchNormalization()(X)
        X = SpatialDropout1D(dropout_p[0])(X)
        X = Bidirectional(LSTM(rnn_units[0], return_sequences=True))(X)
        X = SpatialDropout1D(dropout_p[1])(X)
        X = Bidirectional(GRU(rnn_units[1], return_sequences=True))(X)
        
        maxpool = GlobalMaxPooling1D()(X)
        avgpool = GlobalAveragePooling1D()(X)
        if with_attention:
            attn = AttentionWeightedAverage()(X)
            X = Concatenate()([maxpool, avgpool, attn])
        else:
            X = Concatenate()([maxpool, avgpool])
        return X
    
    
    def build_model(self):
        # 模型主体
        if self.config.token_level == 'word':
            X = self.model_unit(self.word_input, self.word_masking, self.word_embedding, 0.33, 100)
            inputs = [self.word_input]
            
        elif self.config.token_level == 'char':
            X = self.model_unit(self.char_input, self.char_masking, self.char_embedding, 0.2, 100)
            inputs = [self.char_input]
            
        else:
            word_X = self.model_unit(self.word_input, self.word_masking, self.word_embedding, (0.5, 0.1), 30)
            char_X = self.model_unit(self.char_input, self.char_masking, self.char_embedding, 0.2, 100)
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