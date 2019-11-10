# -*- coding: utf-8 -*-
"""
Created:    2019-08-19 21:25:09
Author:     liuyao8
Descritipn: 
"""

from keras.layers import BatchNormalization, PReLU, Add, MaxPooling1D, Bidirectional, GRU, Dropout, \
                        Concatenate, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, SpatialDropout1D, Dense
from keras import regularizers
from keras import backend as K
from keras.models import Model

from model.BasicModel import BasicDeepModel


class TextDPCNN(BasicDeepModel):
    
    def __init__(self, config=None, rnn_units=30, n_filters=64, filter_size=3, dp=7, dense_units=256, **kwargs):
        self.rnn_units = rnn_units
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.dp = dp
        self.dense_units = dense_units
        name = 'TextDPCNN_' + config.token_level
        BasicDeepModel.__init__(self, config=config, name=name, **kwargs)
        
        
    def block(self, X, n_filters, filter_size, kernel_reg, bias_reg, first=False, last=False):
        """DPCNN网络结构中需要重复的block"""
        X1 = Conv1D(n_filters, kernel_size=filter_size, padding='same', kernel_regularizer=kernel_reg, bias_regularizer=bias_reg)(X)
        X1 = BatchNormalization()(X1)
        X1 = PReLU()(X1)
        X1 = Conv1D(n_filters, kernel_size=filter_size, padding='same', kernel_regularizer=kernel_reg, bias_regularizer=bias_reg)(X1)
        X1 = BatchNormalization()(X1)
        X1 = PReLU()(X1)            # (, 57, 64)
        
        if first:
            X = Conv1D(n_filters, kernel_size=1, padding='same', kernel_regularizer=kernel_reg, bias_regularizer=bias_reg)(X)   # (, 57, 64)
        
        X = Add()([X, X1])          # (, 57, 64)
        
        if last:
            X = GlobalMaxPooling1D()(X)
        else:
            X = MaxPooling1D(pool_size=3, strides=2)(X)     # (, 28, 64)
        return X
        
        
    def model_unit(self, inputs, masking, embedding, n_filters=None, filter_size=None, dp=None, dense_units=None):
        """模型主体Unit"""
        kernel_reg=regularizers.l2(0.00001)
        bias_reg=regularizers.l2(0.00001)
        if n_filters is None:
            n_filters = self.n_filters
        if filter_size is None:
            filter_size = self.filter_size
        if dp is None:
            dp = self.dp
        if dense_units is None:
            dense_units = self.dense_units
            
        # Region Embedding
        X = masking(inputs)
        X = embedding(X)
        X = BatchNormalization()(X)     # (, 57, 100)
        
        # 第1层 pre-activation
        X = self.block(X, n_filters, filter_size, kernel_reg, bias_reg, first=True)     # (, 28, 64)
        
        # 重复dp次: 不含第1层
        flag_last = False
        for i in range(dp):
            if i + 1 == dp or flag_last:        # 最后1层
                X = self.block(X, n_filters, filter_size, kernel_reg, bias_reg, last=True)
                break                           # 务必不要忘了break！！！
            else:                               # 中间层
                if K.int_shape(X)[1] // 2 < 8:  # 此次block操作后没法继续MaxPooling1D，下一层变为最后1层(GlobalMaxPooling1D)
                    flag_last = True
                X = self.block(X, n_filters, filter_size, kernel_reg, bias_reg)
        
        # 全连接层
        X = Dense(dense_units)(X)
        X = BatchNormalization()(X)
        X = PReLU()(X)
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
            word_X = self.word_embedding(self.word_input)
            word_X = SpatialDropout1D(0.25)(word_X)
            word_X = Bidirectional(GRU(self.rnn_units, return_sequences=True))(word_X)
            word_X = SpatialDropout1D(0.25)(word_X)
            word_X = Bidirectional(GRU(self.rnn_units, return_sequences=True))(word_X)
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