# -*- coding: utf-8 -*-
"""
Created:    2019-08-15 20:08:42
Author:     liuyao8
Descritipn: 
"""

from keras.layers import BatchNormalization, SpatialDropout1D, Conv1D, GlobalMaxPooling1D, \
                        Concatenate, Dropout, Dense, Bidirectional, GRU, GlobalAveragePooling1D
from keras.models import Model

from model.BasicModel import BasicDeepModel


class TextCNN2(BasicDeepModel):
    """TextCNN模型，支持char,word和both. both时char进行TextCNN，word进行RNN，然后拼接结果"""
    
    def __init__(self, config=None, fsizes=(2, 5), n_filters=128, rnn_units=60, dropout_p=0.25, **kwargs):
        self.fsizes = fsizes
        self.n_filters = n_filters      # TODO 是否是BasicDeepModel通用？通用的话放在BasicDeepModel那里
        self.rnn_units = rnn_units
        self.dropout_p = dropout_p
        name = 'TextCNN2_' + config.token_level
        BasicDeepModel.__init__(self, config=config, name=name, **kwargs)
        
        
    def model_unit(self, inputs, masking, embedding):
        """模型主体Unit"""
        X = masking(inputs)
        X = embedding(X)
        X = BatchNormalization()(X)
        X = SpatialDropout1D(self.dropout_p)(X)
        Xs = []
        for fsize in range(self.fsizes[0], self.fsizes[1] + 1):
            Xi = Conv1D(self.n_filters, fsize, activation='relu')(X)
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
            # 对word进行特殊处理！  # TODO WHY???
            word_X = self.word_embedding(self.word_input)
            word_X = BatchNormalization()(word_X)
            for _ in range(2):
                word_X = SpatialDropout1D(0.2)(word_X)
                word_X = Bidirectional(GRU(self.rnn_units // 2, return_sequences=True))(word_X)
            word_maxpool = GlobalMaxPooling1D()(word_X)
            word_avgpool = GlobalAveragePooling1D()(word_X)
            char_Xs = self.model_unit(self.char_input, self.char_masking, self.char_embedding)
            Xs = [word_maxpool, word_avgpool] + char_Xs
            inputs = [self.word_input, self.char_input]
        
        # 增加结构化特征
        if self.config.structured:
            Xs = Xs + self.structured_input
            inputs = inputs + self.structured_input
        
        # 模型结尾
        X = Concatenate()(Xs) if len(Xs) > 1 else Xs[0]
        X = Dropout(0.5)(X)
        out = Dense(self.n_classes, activation=self.activation)(X)
        
        self.model = Model(inputs=inputs, outputs=out)