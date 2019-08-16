# -*- coding: utf-8 -*-
"""
Created:    2019-08-16 13:37:44
Author:     liuyao8
Descritipn: 
"""

from keras.layers import Layer 
from keras import backend as K


class AttentionWeightedAverage(Layer):
    """
    A weighted Average of different channels across timesteps
    Reference: <https://blog.csdn.net/qq_40027052/article/details/78210253>
    """
    def __init__(self, return_attention=False, **kwargs):
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(**kwargs)
        
        
    def build(self, input_shape):
        """Define the weights"""
        assert len(input_shape) == 3
        self.W = self.add_weight(name=self.name+'_W', shape=(input_shape[2], 1), initializer='glorot_uniform')
        self.trainable_weights = [self.W]        
        super(AttentionWeightedAverage, self).build(input_shape)
        
        
    def call(self, x, mask=None):
        """
        Layer's logic:
            logit = W * x - max(W * x)    # 相当于小神经网络: x -> logit
            attn = softmax(logit) = exp(logit) / (sum(exp(logit)) + epsilon)
            result = sum(attn * x)
        """
        logit = K.dot(x, self.W)                                    # (i0, i1, i2) dot (i2, 1) -> (i0, i1, 1)
        logit = K.reshape(logit, (K.shape(x)[0], K.shape(x)[1]))    # -> (i0, i1)
        logit = logit - K.max(logit, axis=-1, keepdims=True)        # (i0, i1)
        ai = K.exp(logit)                                           # (i0, i1)
        
        # masked timesteps have 0 weight
        if mask:
            ai = ai * K.cast(mask, K.floatx())                   # (i0, i1)
        
        attn = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())    # (i0, i1)
        result = K.sum(x * K.expand_dims(attn), axis=1)                 # (i0, i1, i2) * (i0, i1, 1) -> (i0, i1, i2) -> (i0, i2)
        return [result, attn] if self.return_attention else result
        
    
    def compute_output_shape(self, input_shape):
        """The shape transformation logic"""
        if self.return_attention:
            return [(input_shape[0], input_shape[2]), (input_shape[0], input_shape[1])]
        return (input_shape[0], input_shape[2])
        


class Capsule(Layer):
    """
    
    """
    def __init__(self):
        pass
        
    def build(self, input_shape):
        pass
        
    def call(self, x):
        pass
        
    def compute_output_shape(self, input_shape):
        pass
        