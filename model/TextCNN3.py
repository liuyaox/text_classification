# -*- coding: utf-8 -*-
"""
Created:    2019-08-14 13:35:44
Author:     liuyao8
Descritipn: 
"""

from keras.layers import BatchNormalization, SpatialDropout1D, Conv1D, GlobalMaxPooling1D, Concatenate, Dropout, Dense
from keras.models import Model

from model.BasicModel import BasicDeepModel


class TextCNN(BasicDeepModel):
    """TextCNN模型，支持char,word和both. both时char和word分别进行TextCNN，然后拼接结果"""
    
    def __init__(self, config=None, fsizes=(2, 5), n_filters=64, dropout_p=0.25, **kwargs):
        self.fsizes = fsizes
        self.n_filters = n_filters
        self.dropout_p = dropout_p
        name = 'TextCNN'
        BasicDeepModel.__init__(self, config=config, name=name, **kwargs)

        
    def model_unit(self, inputs, masking, embedding=None, dropout_p=None, fsizes=None, n_filters=None):
        """模型主体Unit"""
        if dropout_p is None:
            dropout_p = self.dropout_p
        if fsizes is None:
            fsizes = self.fsizes
        if n_filters is None:
            n_filters = [self.n_filters] * (fsizes[1] - fsizes[0] + 1)
        
        X = masking(inputs)
        if embedding:
            X = embedding(X)
        X = BatchNormalization()(X)
        X = SpatialDropout1D(dropout_p)(X)
        Xs = []
        for i, fsize in enumerate(range(fsizes[0], fsizes[1] + 1)):
            Xi = Conv1D(n_filters[i], fsize, activation='relu')(X)  # TODO Layer conv1d_5 does not support masking, but was passed an input_mask
            Xi = GlobalMaxPooling1D()(Xi)
            Xs.append(Xi)
        return Xs
    
    
    def build_model(self):
        # 模型主体
        if self.config.token_level == 'word':
            Xs = self.model_unit(self.word_input, self.word_masking, self.word_embedding)
            inputs = [self.word_input]
            
        elif self.config.token_level == 'char':
            Xs = self.model_unit(self.char_input, self.char_masking, self.char_embedding)
            inputs = [self.char_input]
            
        else:
            word_Xs = self.model_unit(self.word_input, self.word_masking, self.word_embedding)
            char_Xs = self.model_unit(self.char_input, self.char_masking, self.char_embedding)
            Xs = word_Xs + char_Xs
            inputs = [self.word_input, self.char_input]
        
        
        # 结构化特征
        if self.config.structured in ['word', 'char', 'both']:
            Xs = Xs + self.structured_input
            inputs = inputs + self.structured_input
        
        
        # 模型结尾
        X = Concatenate()(Xs) if len(Xs) > 1 else Xs[0]
        X = BatchNormalization()(X)
        X = Dropout(0.5)(X)
        # X = Dense(self.hidden_units, activation='relu')(X)  # TODO 不需要隐藏层！？
        out = Dense(self.n_classes, activation=self.activation)(X)
        
        self.model = Model(inputs=inputs, outputs=out)