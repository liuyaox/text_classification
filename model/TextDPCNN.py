# -*- coding: utf-8 -*-
"""
Created:    2019-08-18 17:02:53
Author:     liuyao8
Descritipn: 
"""

from keras.layers import BatchNormalization, PReLU, Add, MaxPooling1D, Bidirectional, GRU, Dropout, \
                        Concatenate, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, SpatialDropout1D, Dense
from keras import regularizers
from keras.models import Model

from model.BasicModel import BasicDeepModel


class TextDPCNN(BasicDeepModel):
    
    def __init__(self, config=None, rnn_units=30, n_filters=64, filter_size=3, 
                 max_pool_size=3, max_pool_strides=2, dp=7, dense_units=256, **kwargs):
        self.rnn_units = rnn_units
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.max_pool_size = max_pool_size
        self.max_pool_strides = max_pool_strides
        self.dp = dp
        self.dense_units = dense_units
        name = 'TextDPCNN_' + config.token_level
        BasicDeepModel.__init__(self, config=config, name=name, **kwargs)
        
        
    def model_unit(self, inputs, masking, embedding, n_filters=None, filter_size=None, 
                   max_pool_size=None, max_pool_strides=None, dp=None, dense_units=None):
        """模型主体Unit"""
        kernel_reg=regularizers.l2(0.00001)
        bias_reg=regularizers.l2(0.00001)
        if n_filters is None:
            n_filters = self.n_filters
        if filter_size is None:
            filter_size = self.filter_size
        if max_pool_size is None:
            max_pool_size = self.max_pool_size
        if max_pool_strides is None:
            max_pool_strides = self.max_pool_strides
        if dp is None:
            dp = self.dp
        if dense_units is None:
            dense_units = self.dense_units
            
        X = masking(inputs)
        X = embedding(X)
        X = BatchNormalization()(X)
        
        X1 = Conv1D(n_filters, kernel_size=filter_size, padding='same', kernel_regularizer=kernel_reg, bias_regularizer=bias_reg)(X)
        X1 = BatchNormalization()(X1)
        X1 = PReLU()(X1)
        X1 = Conv1D(n_filters, kernel_size=filter_size, padding='same', kernel_regularizer=kernel_reg, bias_regularizer=bias_reg)(X1)
        X1 = BatchNormalization()(X1)
        X1 = PReLU()(X1)
        X2 = Conv1D(n_filters, kernel_size=1, padding='same', kernel_regularizer=kernel_reg, bias_regularizer=bias_reg)(X)
        
        X = Add()([X1, X2])
        X = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(X)
        
        for i in range(dp):
            X1 = Conv1D(n_filters, kernel_size=filter_size, padding='same', kernel_regularizer=kernel_reg, bias_regularizer=bias_reg)(X)
            X1 = BatchNormalization()(X1)
            X1 = PReLU()(X1)
            X1 = Conv1D(n_filters, kernel_size=filter_size, padding='same', kernel_regularizer=kernel_reg, bias_regularizer=bias_reg)(X1)
            X1 = BatchNormalization()(X1)
            X1 = PReLU()(X1)
            
            X2 = Add()([X1, X])
            if i + 1 != dp:     # 第dp次时不需要MaxPooling1D，之前每次都需要
                # TODO 此处报错，因为Negative Dimension，X2的shape不够MaxPool去缩减了
                X = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(X2)
            
        X = GlobalMaxPooling1D()(X2)
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
            word_X = SpatialDropout1D()(word_X)
            word_X = Bidirectional(GRU(self.rnn_units, return_sequences=True))(word_X)
            word_X = SpatialDropout1D()(word_X)
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