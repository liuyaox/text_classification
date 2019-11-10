# -*- coding: utf-8 -*-
"""
Created:    2019-08-17 21:11:18
Author:     liuyao8
Descritipn: 
"""

from keras.layers import BatchNormalization, SpatialDropout1D, Conv1D, Bidirectional, LSTM, GRU, \
                        GlobalMaxPooling1D, GlobalAveragePooling1D, Concatenate, Dropout, Dense
from keras.models import Model

from model.BasicModel import BasicDeepModel
from model.Layers import AttentionWeightedAverage


class TextConvLSTM2_Attn(BasicDeepModel):
    
    def __init__(self, config=None, n_filters=128, rnn_units=64, dropout_p=0.25, with_attention=True, **kwargs):
        self.n_filters = n_filters
        self.rnn_units = rnn_units
        self.dropout_p = dropout_p
        self.with_attention = with_attention
        name = 'TextConvLSTM2_Attn_' + config.token_level
        BasicDeepModel.__init__(self, config=config, name=name, **kwargs)
        
        
    def model_unit(self, inputs, masking, embedding, n_filters=None, rnn_units=None, dropout_p=None):
        """模型主体Unit"""
        if n_filters is None:
            n_filters = self.n_filters
        if rnn_units is None:
            rnn_units = [self.rnn_units] * 2
        if isinstance(rnn_units, int):
            rnn_units = [rnn_units] * 2
        if dropout_p is None:
            dropout_p = [self.dropout_p] * 2
        if isinstance(dropout_p, float):
            dropout_p = [dropout_p] * 2
        
        X = masking(inputs)
        X = embedding(X)
        X = BatchNormalization()(X)
        X = SpatialDropout1D(dropout_p[0])(X)
        # TODO Conv1D没有activation ???
        X = Conv1D(n_filters, kernel_size=3, padding="valid", kernel_initializer="glorot_uniform")(X) # 相比LSTMGRUModel，此处多了个Conv1D
        X = Bidirectional(LSTM(rnn_units[0], return_sequences=True))(X)
        X = SpatialDropout1D(dropout_p[1])(X)
        X = Bidirectional(GRU(rnn_units[1], return_sequences=True))(X)
        
        maxpool = GlobalMaxPooling1D()(X)
        avgpool = GlobalAveragePooling1D()(X)
        if self.with_attention:
            attn = AttentionWeightedAverage()(X)
            X = Concatenate()([maxpool, avgpool, attn])
        else:
            X = Concatenate()([maxpool, avgpool])
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
            # 与TextConvLSTMGRU对word进行特殊处理！  # TODO WHY???   与char相比，没有conv和attention
            word_X = self.word_masking(self.word_input)
            word_X = self.word_embedding(word_X)
            word_X = BatchNormalization()(word_X)
            word_X = SpatialDropout1D(0.2)(word_X)      # TODO 0.2  下面0.1 ？
            word_X = Bidirectional(GRU(self.rnn_units // 2, return_sequences=True))(word_X)
            word_X = SpatialDropout1D(0.1)(word_X)
            word_X = Bidirectional(GRU(self.rnn_units // 2, return_sequences=True))(word_X)
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