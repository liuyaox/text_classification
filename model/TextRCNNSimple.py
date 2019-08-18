# -*- coding: utf-8 -*-
"""
Created:    2019-08-18 14:52:38
Author:     liuyao8
Descritipn: 
"""

from keras.layers import Input, BatchNormalization, Bidirectional, GRU, Dropout, Lambda, \
                        Concatenate, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, Dense
from keras import backend as K
from keras.models import Model

from model.BasicModel import BasicDeepModel
from model.Layers import AttentionWeightedAverage


class TextRCNNSimple(BasicDeepModel):
    """简易版TextRCNN"""
    
    def __init__(self, config=None, rnn_units=64, n_filters=64, **kwargs):
        self.rnn_units = rnn_units
        self.n_filters = n_filters
        name = 'TextRCNNSimple_' + config.token_level
        BasicDeepModel.__init__(self, config=config, name=name, **kwargs)
        
        
    def model_unit(self, inputs, left_inputs, right_inputs, masking, embedding, rnn_units=None, n_filters=None):
        """模型主体Unit"""
        if rnn_units is None:
            rnn_units = [self.rnn_units] * 3
        if isinstance(rnn_units, int):
            rnn_units = [rnn_units] * 3
        if n_filters is None:
            n_filters = self.n_filters
        
        X = masking(inputs)
        X = embedding(X)
        X = BatchNormalization()(X)
        X = Bidirectional(GRU(rnn_units[0], return_sequences=True))(X)
        
        left_X = masking(left_inputs)
        left_X = embedding(left_X)
        left_X = BatchNormalization()(left_X)
        left_X = Bidirectional(GRU(rnn_units[1], return_sequences=True))(left_X)
        
        right_X = masking(right_inputs)
        right_X = embedding(right_X)
        right_X = BatchNormalization()(right_X)
        right_X = Dropout(0.5)(right_X)
        right_X = Bidirectional(GRU(rnn_units[2], return_sequences=True, go_backwards=True))(right_X)
        right_X = Lambda(lambda x: K.reverse(x, axes=1))(right_X)
        
        concat = Concatenate()([X, left_X, right_X])
        concat = Conv1D(n_filters, kernel_size=1, activation='relu')(concat)
        
        # TODO 为什么没有left_x与x交互的操作！！？？right_x与x同理！！？？
        # 比如上一个left与上一个word共同生成当前left？？？(详见论文中的公式1和2！！！)
        # 另外，与论文相比或与别的实现相比，下面这些是多余的，应该直接到output=Dense那里 ？？？
        maxpool = GlobalMaxPooling1D()(concat)
        avgpool = GlobalAveragePooling1D()(concat)
        attn = AttentionWeightedAverage()(concat)
        X = Concatenate()([maxpool, avgpool, attn])
        return X
        
        
    def build_model(self):
        # 额外的Input
        self.word_left_inputs = Input(shape=(self.word_maxlen, ), name='word_left')
        self.word_right_inputs = Input(shape=(self.word_maxlen, ), name='word_right')
        self.char_left_inputs = Input(shape=(self.char_maxlen, ), name='char_left')
        self.char_right_inputs = Input(shape=(self.char_maxlen, ), name='char_right')
        
        # 模型主体
        if self.config.token_level == 'word':
            X = self.model_unit(self.word_input, self.word_left_inputs, self.word_right_inputs, self.word_masking, self.word_embedding)
            inputs = [self.word_input, self.word_left_inputs, self.word_right_inputs]
            
        elif self.config.token_level == 'char':
            X = self.model_unit(self.char_input, self.char_left_inputs, self.char_right_inputs, self.char_masking, self.char_embedding)
            inputs = [self.char_input, self.char_left_inputs, self.char_right_inputs]
            
        else:
            word_X = self.model_unit(self.word_input, self.word_left_inputs, self.word_right_inputs, self.word_masking, self.word_embedding)
            char_X = self.model_unit(self.char_input, self.char_left_inputs, self.char_right_inputs, self.char_masking, self.char_embedding)
            X = Concatenate()([word_X, char_X])
            inputs = [self.word_input, self.word_left_inputs, self.word_right_inputs, \
                      self.char_input, self.char_left_inputs, self.char_right_inputs]
        
        
        # 结构化特征
        if self.config.structured in ['word', 'char', 'both']:
            X = Concatenate()([X] + self.structured_input)
            inputs = inputs + self.structured_input
        
        
        # 模型结尾
        X = Dropout(0.5)(X)
        out = Dense(self.n_classes, activation=self.activation)(X)
        
        self.model = Model(inputs=inputs, outputs=out)