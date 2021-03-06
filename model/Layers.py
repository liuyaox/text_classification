# -*- coding: utf-8 -*-
"""
Created:    2019-08-16 13:37:44
Author:     liuyao8
Descritipn: 
"""

from keras.layers import Layer, Activation
from keras import initializers, regularizers, constraints
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
            简写：result=sum(p(x)*x)  p(x)=softmax(Wx)  加性模型？？？
        """
        logit = K.dot(x, self.W)                                    # (i0, i1, i2) dot (i2, 1) -> (i0, i1, 1)
        logit = K.reshape(logit, (K.shape(x)[0], K.shape(x)[1]))    # -> (i0, i1)
        logit = logit - K.max(logit, axis=-1, keepdims=True)        # (i0, i1)
        ai = K.exp(logit)                                           # (i0, i1)
        
        # masked timesteps have 0 weight
        if mask:
            ai = ai * K.cast(mask, K.floatx())                      # (i0, i1)
        
        attn = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())    # (i0, i1)
        result = K.sum(x * K.expand_dims(attn), axis=1)                 # (i0, i1, i2) * (i0, i1, 1) -> (i0, i1, i2) -> (i0, i2)
        return [result, attn] if self.return_attention else result
        
    
    def compute_output_shape(self, input_shape):
        """The shape transformation logic"""
        if self.return_attention:
            return [(input_shape[0], input_shape[2]), (input_shape[0], input_shape[1])]
        return (input_shape[0], input_shape[2])
        


class Attention(Layer):
    """
    Keras Layer that implements an Attention mechanism for temporal data.
    Supports Masking.
    Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    :param kwargs:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Example:
        hidden = LSTM(64, return_sequences=True)(words)
        sentence = Attention()(hidden)
        # next add a Dense layer (for classification/regression) or whatever...
    代码源自：https://github.com/ShawnyXiao/TextClassification-Keras/blob/master/model/HAN/attention.py
    """
    def __init__(self, step_dim, W_regularizer=None, b_regularizer=None, W_constraint=None, b_constraint=None, bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)


    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight(shape=(input_shape[-1], ), name='{}_W'.format(self.name),
                                 initializer=self.init, regularizer=self.W_regularizer, constraint=self.W_constraint)
        self.features_dim = input_shape[-1]
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1], ), name='{}_b'.format(self.name),
                                     initializer='zero', regularizer=self.b_regularizer, constraint=self.b_constraint)
        else:
            self.b = None
        self.built = True


    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None


    def call(self, x, mask=None):
        """简写：result=sum(p(x)*x)   p(x)=softmax(tanh(Wx+b))  加性模型"""
        features_dim = self.features_dim
        step_dim = self.step_dim
        e = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))  # e = K.dot(x, self.W)
        if self.bias:
            e += self.b
        e = K.tanh(e)       # e = tanh(Wx + b)
        a = K.exp(e)        # a = exp(e)
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            a *= K.cast(mask, K.floatx())   # cast the mask to floatX to avoid float64 upcasting in theano
        # In some cases especially in the early stages of training, the sum may be almost zero and this results in NaN's.
        # A workaround is to add a very small positive number ε to the sum.
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())    # a = softmax(e) = softmax(tanh(Wx + b)) = p(x) 表示一种权重
        a = K.expand_dims(a)
        result = K.sum(a * x, axis=1)    # result = sum(p(x) * x)  对x加权求和
        return result


    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim
    
    
    
class AttentionSelf(Layer):
    """
    Self Attention, codes from: https://blog.csdn.net/xiaosongshine/article/details/90600028
    代码源自：https://github.com/yongzhuo/Keras-TextClassification/blob/master/keras_textclassification/keras_layers/attention_self.py
    """
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super().__init__(**kwargs)


    def build(self, input_shape):
        # Q、K and V
        self.kernel = self.add_weight(name='QKV', shape=(3, input_shape[2], self.output_dim), trainable=True,
                                      initializer='uniform', regularizer=regularizers.L1L2(0.0000032))
        super().build(input_shape)


    def call(self, x):
        '''简写：res=p(x)*Vx  p(x)=softmax(Qx*Kx)'''
        QX = K.dot(x, self.kernel[0])
        KX = K.dot(x, self.kernel[1])
        VX = K.dot(x, self.kernel[2])
        print("QX.shape", QX.shape)
        print("K.permute_dimensions(KX, [0, 2, 1]).shape", K.permute_dimensions(KX, [0, 2, 1]).shape)
        
        # batch_dot: 即batch-wise dot product，X与Y同一batch分别是Xi和Yi，则dot(Xi, Yi.T)为该batch的结果，遍历所有batch
        # 当axes!=None时另说
        QK = K.batch_dot(QX, K.permute_dimensions(KX, [0, 2, 1]))
        QK = QK / (64 ** 0.5)   # TODO 64是不是应该改为self.output_dim更合适一些？！？因为KX's shape=(, input_shape[1], output_dim)，KX的维度是output_dim
        QK = K.softmax(QK)
        print("QK.shape", QK.shape)
        
        res = K.batch_dot(QK, VX)
        return res


    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

        
        
def squash(x, axis=-1):
    # s_squared_norm is really small
    # s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    # scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    # return scale * x
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale

    

class Capsule(Layer):
    """
    Capsule  TODO 待研究！
    """
    def __init__(self, n_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True, activation=None, **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.n_capsule = n_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        self.activation =  Activation(activation) if activation else squash
        
        
    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_n_capsule = 1 if self.share_weights else input_shape[-2]
        input_dim_capsule = input_shape[-1]
        self.W = self.add_weight(name='capsule_kernel',
                                 shape=(input_n_capsule, input_dim_capsule, self.n_capsule * self.dim_capsule),
                                 initializer='glorot_uniform',
                                 trainable=True)


    def call(self, x):
        if self.share_weights:
            u_hat_vecs = K.conv1d(x, self.W)
        else:
            u_hat_vecs = K.local_conv1d(x, self.W, [1], [1])

        batch_size = K.shape(x)[0]
        input_n_capsule = K.shape(x)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_n_capsule, self.n_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, n_capsule, input_n_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, n_capsule, input_n_capsule]
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_n_capsule, n_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])
        return outputs


    def compute_output_shape(self, input_shape):
        return (None, self.n_capsule, self.dim_capsule)
