# -*- coding: utf-8 -*-
"""
Created:    2019-08-20 22:58:52
Author:     liuyao8
Descritipn: 
"""

from keras.layers import Input, BatchNormalization, Bidirectional, LSTM, TimeDistributed, Dropout, Dense, GRU, Masking, Flatten
from keras.models import Model
from keras import optimizers

from model.BasicModel import BasicDeepModel
from model.Layers import Attention, AttentionSelf


class TextHAN(BasicDeepModel):
    
    def __init__(self, config=None, rnn_units1=128, rnn_units2=128, **kwargs):
        self.rnn_units1 = rnn_units1
        self.rnn_units2 = rnn_units2
        self.sent_maxlen = config.SENT_MAXLEN
        self.word_maxlen = config.WORD_MAXLEN
        self.sent_input = Input(shape=(self.sent_maxlen, self.word_maxlen), dtype='int32', name='sentence1')  # (None, sent_maxlen, word_maxlen)
        name = 'TextHAN'
        BasicDeepModel.__init__(self, config=config, name=name, **kwargs)
        
    
    # 方法1：以下参考https://github.com/ShawnyXiao/TextClassification-Keras/blob/master/model/HAN/han.py
    # 脚本https://github.com/AlexYangLi/TextClassification/blob/master/models/keras_han_model.py与方法1其实是一样的，只是写法不同
    # 脚本https://github.com/richliao/textClassifier/blob/master/textClassifierHATT.py与方法1是一样的
    def build_model(self):
        # Sentence Part                                                       sent_input: (None, sent_maxlen, word_maxlen)
        X = TimeDistributed(self.word_encoder(), name='word_encoder')(self.sent_input)  # (None, sent_maxlen, 2*rnn_units1)
        # X = Masking()(X)  # 需不需要？？？
        X = BatchNormalization()(X)
        X = Bidirectional(LSTM(self.rnn_units2, return_sequences=True))(X)  # (None, sent_maxlen, 2*rnn_units2)
        X = Attention(self.sent_maxlen)(X)                                  # (None, 2*rnn_units2)
        
        X = Dropout(0.5)(X)
        # X = Dense(256, activation='relu')(X)  # 需不需要？？？
        out = Dense(self.n_classes, activation=self.activation)(X)          # (None, n_classes)
        self.model = Model(inputs=self.sent_input, outputs=out)  # TODO 注意inputs是Sentence Part处的inputs(而非Word Part处的word_input)！


    def word_encoder(self):
        # Word Part 模型，提供word level的编码功能
        word_X = self.word_masking(self.word_input)                 # (None, word_maxlen)
        word_X = self.word_embedding(word_X)                        # (None, word_maxlen, word_embed_dim)
        word_X = BatchNormalization()(word_X)
        word_X = Bidirectional(LSTM(self.rnn_units1, return_sequences=True))(word_X) # (None, word_maxlen, 2*rnn_units1)
        word_out = Attention(self.word_maxlen)(word_X)              # (None, 2*rnn_units1)  # TODO 能不能使用AttentionAverageWeighted
        return Model(inputs=self.word_input, outputs=word_out)


    def train_evaluate(self, x_train, y_train, x_test, y_test, lr=0.001, epochs=None):
        """
        模型训练和评估 only for TextHAN
        x_train/x_test是字典(key=Input创建时的name, value=Input对应的数据)，表示多输入
        """
        # 模型训练
        print('【' + self.name + '】')
        self.mode = 3
        epochs = epochs if epochs else (2, self.n_epochs)
        print('-------------------Step1: 前期冻结Word_Encoder层，编译和训练模型-------------------')
        self.model.get_layer('word_encoder').trainable = False        # TODO word_encoder由很多层组成，如何只设置其中的Embedding？？
        optimizer = optimizers.Adam(lr=lr, clipvalue=2.4)
        self.model.compile(loss=self.loss, optimizer=optimizer, metrics=self.metrics)
        history1 = self.model.fit(x_train,
                                  y_train, 
                                  batch_size=self.batch_size*self.config.n_gpus,
                                  epochs=epochs[0],
                                  validation_split=0.3)
        print('-------------Step2: 训练完参数后，解冻Word_Encoder层，再次编译和训练模型-------------')
        self.model.get_layer('word_encoder').trainable = True
        optimizer = optimizers.Adam(lr=lr, clipvalue=1.5)
        self.model.compile(loss=self.loss, optimizer=optimizer, metrics=self.metrics)
        history2 = self.model.fit(x_train,
                                  y_train, 
                                  batch_size=self.batch_size*self.config.n_gpus,
                                  epochs=epochs[1],
                                  validation_split=0.3,
                                  callbacks=None)
        self.plot_history(history2)
        history = (history1, history2)
            
        # 模型评估
        test_acc, scores, sims, vectors, test_pred = self._evaluate(x_test, y_test)
        pickle.dump(test_pred, open('./result/' + self.name + '_test_pred.pkl', 'wb'))
        return test_acc, scores, sims, vectors, history


    def han_data1(self, x_train, y_train, x_test, y_test):
        """脚本参考同build_model1"""
        # TODO 如何确定句子之间的分界！？之前的处理都不在意句子间的分界，把所有句子当成一个句子来处理。
        # TODO 以下做法似有不妥？
        # 1.强行在document(所有句子)后面padding一次，而不是在每个句子后面都padding一次 ？？？
        # -----------,------,--- ------------,-------- --,000000000000000000 00000000000000000000
        # 2.强行把document按sent_maxlen(假设为20)划分看，而非原本句子的自然划分 ？？？
        # -----------,------,---|------------,--------|--,000000000000000000|00000000000000000000
        # 1和2应该如下所示：  吧？？？
        # ----------- 000000 000|------000000 00000000|-- -------------00000|----------0000000000
        from keras.preprocessing.sequence import pad_sequences
        x_train = pad_sequences(x_train, maxlen=self.sent_maxlen * self.word_maxlen)    # 1
        x_test = pad_sequences(x_test, maxlen=self.sent_maxlen * self.word_maxlen)
        x_train = x_train.reshape((len(x_train), self.sent_maxlen, self.word_maxlen))   # 2
        x_test = x_test.reshape((len(x_test), self.sent_maxlen, self.word_maxlen))
        return x_train, x_test
    
    
    
    # 方法2：以下参考https://github.com/yongzhuo/Keras-TextClassification/blob/master/keras_textclassification/m12_HAN/graph.py
    # 方法1使用了Attention机制，而方法2使用了Self-Attention即Transformer机制！
    # TODO 输入是self.word_embedding.input？？？待研究！
    def build_model2(self):
        # Word Part
        word_X = self.word_embedding.output                         # (None, word_maxlen, word_embed_dim)
        word_X = Bidirectional(GRU(units=self.rnn_units1, return_sequences=True, activation='relu'))(word_X) # (None, word_maxlen, 2*rnn_units1)
        word_X = AttentionSelf(self.rnn_units*2)(word_X)            # (None, word_maxlen, 2*rnn_units)
        word_X = Dropout(0.5)(word_X)

        # Sentence Part
        X = Bidirectional(GRU(units=self.rnn_units2, return_sequences=True, activation='relu'))(word_X)      # (None, word_maxlen, 2*rnn_units2)
        X = AttentionSelf(self.word_embed_dim)(X)                   # (None, word_maxlen, word_embed_dim)
        X = Dropout(0.5)(X)

        X = Flatten()(X)                                            # (None, word_maxlen * word_embed_dim)
        out = Dense(self.n_classes, activation=self.activation)(X)  # (None, n_classes)
        self.model = Model(inputs=self.word_embedding.input, outputs=out)
        
        

if __name__ == '__main__':
    
    import pickle
    from Vocabulary import Vocabulary
    from Config import Config
    config = Config()
    
    # data和config准备
    config = pickle.load(open(config.config_file, 'rb'))
    x_train, y_train, x_test, y_test = pickle.load(open(config.data_encoded_file, 'rb'))
    
    # 根据实际情况修改，也可直接在Config.py里修改，推荐前者
    config.n_gpus = 1
    config.token_level = 'word'
    config.structured = 'none'
    config.bert_flag = False

    # 模型创建与训练    
    texthan = TextHAN(config)
    test_acc, scores, sims, vectors, history = texthan.train_evaluate(x_train, y_train, x_test, y_test)
    
    texthan.model.save(config.model_file)
    