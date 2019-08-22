# -*- coding: utf-8 -*-
"""
Created:    2019-08-20 23:14:02
Author:     liuyao8
Descritipn: 
"""

from keras.layers import BatchNormalization, Bidirectional, GRU, Flatten, Dropout, \
                        Concatenate, GlobalMaxPooling1D, GlobalAveragePooling1D, SpatialDropout1D, Dense
from keras.models import Model

from model.BasicModel import BasicDeepModel
from model.Layers import Capsule


class TextCapsule(BasicDeepModel):
    
    def __init__(self, config=None, rnn_units=30, dropout_p=0.2, n_capsule=10, dim_capsule=16, routings=5, share_weights=True, **kwargs):
        self.rnn_units = rnn_units
        self.dropout_p = dropout_p
        self.n_capsule = n_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        name = 'TextCapsule_' + config.token_level
        BasicDeepModel.__init__(self, config=config, name=name, **kwargs)
        
        
    def model_unit(self, inputs, masking, embedding, dropout_p=None, n_capsule=None, dim_capsule=None, routings=None, share_weights=None):
        """模型主体Unit"""
        if dropout_p is None:
            dropout_p = self.dropout_p
        if n_capsule is None:
            n_capsule = self.n_capsule
        if dim_capsule is None:
            dim_capsule = self.dim_capsule
        if routings is None:
            routings = self.routings
        if share_weights is None:
            share_weights = self.share_weights
            
        X = masking(inputs)
        X = embedding(X)
        X = BatchNormalization()(X)
        X = SpatialDropout1D(dropout_p)(X)
        X = Bidirectional(GRU(64, return_sequences=True))(X)
        capsule = Capsule(n_capsule=n_capsule, dim_capsule=dim_capsule, routings=routings, share_weights=share_weights)(X)
        X = Flatten()(capsule)
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
            # 对word进行特殊处理！
            word_X = self.word_masking(self.word_input)
            word_X = self.word_embedding(word_X)
            word_X = SpatialDropout1D(0.25)(word_X)
            word_X = Bidirectional(GRU(self.rnn_units, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(word_X)  # TODO ???
            word_X = Bidirectional(GRU(self.rnn_units, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(word_X)
            word_maxpool = GlobalMaxPooling1D()(word_X)
            word_avgpool = GlobalAveragePooling1D()(word_X)
            
            char_X = self.model_unit(self.char_input, self.char_masking, self.char_embedding)
            X = Concatenate()([word_maxpool, word_avgpool, char_X])
            inputs = [self.word_input, self.char_input]
        
        
        # 结构化特征
        if self.config.structured in ['word', 'char', 'both']:
            X = Concatenate()([X] + self.structured_input)
            inputs = inputs + self.structured_input
        
        
        # 模型结尾
        X = Dropout(0.5)(X)
        out = Dense(self.n_classes, activation=self.activation)(X)
        
        self.model = Model(inputs=inputs, outputs=out)