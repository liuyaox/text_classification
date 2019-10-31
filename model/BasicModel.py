# -*- coding: utf-8 -*-
"""
Created:    2019-08-11 19:40:55
Author:     liuyao8
Descritipn: a. BasicModel: 模型基类，用于生成BasicStatModel和BasicDeepModel，目前仅提供功能：模型评估Metrics计算
            b. BasicStatModel: 传统模型基类，提供通用功能：
            c. BasicDeepModel: 深度模型基类，提供通用功能：
"""

import os
from functools import reduce
from collections import Counter
import numpy as np
import pickle
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold

from keras.layers import Input, Masking, Embedding
from keras.models import load_model
from keras.utils import multi_gpu_model, plot_model
from keras.optimizers import Adam, SGD
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


class BasicModel(object):
    
    def __init__(self):
        # TODO 其实可以放一些通用的变量如label数量等
        pass
    
    def build(self):
        pass


    # Metrics: Precision, Recall, F1-score, Distribution Similarity, ROC curve, ROC area, etc.
    # TODO 添加method
    def multilabel_precision_recall(self, ys_pred, ys_true):
        """
        多标签分类标准Metrics: Precision, Recall, F1-score
        ARGS
            ys_pred: 预测标签，iterable of iterable，形如：[['a', 'b', 'c'], ['a', 'd'], ['b'], ...]
            ys_true: 真实标签，格式同y_pred
        RETURN
            precision: 总命中标签数/总预测标签数
            recall:    总命中标签数/总真实标签数
            f1score:   (precision * recall) / (precision + recall)
        """
        assert len(ys_pred) == len(ys_true)
        ys_pred = self.label_binarizer.inverse_transform(ys_pred > 0.5)
        ys_true = self.label_binarizer.inverse_transform(ys_true)
        
        right_num, all_pred_num, all_true_num = 0, 0, 0     # 总命中标签数  总预测标注数  总真实标签数
        for y_pred, y_true in zip(ys_pred, ys_true):
            y_pred_set, y_true_set = set(y_pred), set(y_true)
            all_pred_num += len(y_pred_set)
            all_true_num += len(y_true_set)
            right_num += len(y_pred_set & y_true_set)       # 命中标签数：交集大小
        
        precision = float(right_num) / all_pred_num
        recall = float(right_num) / all_true_num
        f1score = (precision * recall) / (precision + recall)
        return round(precision, 4), round(recall, 4), round(f1score, 4)
        
    
    def roc_auc(self, ys_pred, ys_true, n_label):
        """
        ROC-AUC curve  ????
        ARGS
            ys_pred: 预测标签（的概率？）,iterable of iterable，原始预测结果，shape=(n_sample, n_label)
            ys_true: 真实标签？ shape同上
            n_label: 标签个数
        """
        # 为每个label计算ROC curve和ROC area
        fpr, tpr = {}, {}
        roc_auc = {}
        for i in range(n_label):
            fpr[i], tpr[i], _ = roc_curve(ys_true[:, i], ys_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        # 计算micro-average ROC curve and ROC area
        fpr['micro'], tpr['micro'], _ = roc_curve(ys_true.ravel(), ys_pred.ravel())
        roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
        
        
    def multilabel_distribution_similarity(self, ys_pred, ys_true):
        """
        多标签分类特定Metrics: 各标签分布余弦相似度和KL散度
        ARGS同上
        RETURN
            similarity: 基于各标签数据分布，预测结果与真实结果的余弦相似度 越小越好
            relative_entropy: KL散度/相对熵 越小越好
        """
        assert len(ys_pred) == len(ys_true)
        ys_pred = self.label_binarizer.inverse_transform(ys_pred > 0.5)
        ys_true = self.label_binarizer.inverse_transform(ys_true)
        
        ys_pred = Counter(reduce(lambda x, y: x + y, ys_pred))
        ys_true = Counter(reduce(lambda x, y: x + y, ys_true))
        keys = list(set(list(ys_pred.keys()) + list(ys_true.keys())))
        vec_pred = [ys_pred[k] for k in keys]
        vec_true = [ys_true[k] for k in keys]
        
        sim_cosine = cosine_similarity([vec_pred], [vec_true])[0, 0]    # 余弦相似度
        sim_entropy = entropy(vec_pred, vec_true)                       # KL散度/相对熵
        sim_eucliean = sum([(x - y) ** 2 for (x, y) in zip(vec_pred, vec_true)]) ** 0.5
        sim_manhattan = sum([abs(x - y) for (x, y) in zip(vec_pred, vec_true)])
        sims  = (round(sim_cosine, 4), round(sim_entropy, 4), round(sim_eucliean, 4), round(sim_manhattan, 4))
        return (vec_pred, vec_true), sims



class BasicStatModel(BasicModel):
    
    def __init__(self, n_fold=5, name='BasicStatModel', config=None):
        pass
    
    
    
class BasicDeepModel(BasicModel):
    
    def __init__(self, config=None, name='BasicDeepModel', model_summary=True, model_plot=False, 
                 token_level=None, structured=None, bert_flag=None):
        # 基本信息
        if token_level:
            config.token_level = token_level
        if structured:
            config.structured = structured
        if bert_flag:
            config.bert_flag = bert_flag
        self.config = config
        stru_postfix = '_stru-' + config.structured if config.structured != 'none' else ''
        bert_postfix = '_bert' if config.bert_flag else ''
        self.name = name + '_level-' + config.token_level + stru_postfix + bert_postfix
        
        
        # 任务类型决定了类别数量、激活函数和损失函数
        if config.task == 'binary':                     # 单标签二分类
            self.n_classes =  1
            self.activation = 'sigmoid'
            self.loss = 'binary_crossentropy'
            self.metrics = ['accuracy']
        elif config.task == 'categorical':              # 单标签多分类
            self.n_classes = config.N_CLASSES
            self.activation = 'softmax'
            self.loss = 'categorical_crossentropy'
            self.metrics = ['accuracy']                 # TODO ???
        elif config.task == 'multilabel':               # 多标签二分类(多标签多分类需转化为多标签二分类)
            self.n_classes = config.N_CLASSES
            self.activation = 'sigmoid'
            self.loss = 'binary_crossentropy'
            self.metrics = ['accuracy']
        
        
        # TODO 能不能删除这些self.xxx，而直接使用self.config.xxx来代替！？
        # word相关
        self.word_maxlen = config.WORD_MAXLEN
        self.word_vocab_size = config.WORD_VOCAB_SIZE
        self.word_embed_dim = config.WORD_EMBED_DIM
        self.word_embed_matrix = config.word_embed_matrix
        
        # char相关
        self.char_maxlen = config.CHAR_MAXLEN
        self.char_vocab_size = config.CHAR_VOCAB_SIZE
        self.char_embed_dim = config.CHAR_EMBED_DIM
        self.char_embed_matrix = config.char_embed_matrix
        
        # KFold相关
        self.n_folds = config.n_folds
        self.kfold = KFold(n_splits=config.n_folds, shuffle=True, random_state=10)
    
        # Model相关
        self.masking_value = config.PAD_IDX   # TODO mask PAD 突然想到：与PyTorch中的packed_padding和padded_packing相同功能？？？
        self.create_model(model_summary, model_plot)
        
        # Train相关
        self.n_epochs = 20
        self.batch_size = config.BATCH_SIZE
        self.init_lr = 0.001
        
        # Callback
        self.lr_schedule = None
        self.early_stopping = None
        self.snap_epochs = 10   # TODO ?
        self.snapshot = None
        self.checkpoint = None
        
        # Predict相关
        self.label_binarizer = config.label_binarizer
        
        
    def create_model(self, model_summary=True, model_plot=False):
        """调用当前类的build_layers生成通用layers，调用子类的build_model生成model"""
        self.build_layers()
        self.build_model()
        if self.config.n_gpus > 1:
            self.model = multi_gpu_model(self.model, gpus=self.config.n_gpus)
        if model_summary:
            self.model.summary()
        if model_plot:
            plot_model(self.model, to_file=self.name+'.png', show_shapes=True)
        
        
    def build_layers(self):
        """创建DeepModel通用的Layers: Input, Masking, Embedding"""
        if self.config.token_level == 'word':
            self.word_input = Input(shape=(self.word_maxlen, ), dtype='int32', name='word')
            self.word_masking = Masking(mask_value=self.masking_value)
            self.word_embedding = Embedding(self.word_vocab_size, self.word_embed_dim, weights=[self.word_embed_matrix], name='word_embedding')
        elif self.config.token_level == 'char':
            self.char_input = Input(shape=(self.char_maxlen, ), dtype='int32', name='char')
            self.char_masking = Masking(mask_value=self.masking_value)
            self.char_embedding = Embedding(self.char_vocab_size, self.char_embed_dim, weights=[self.char_embed_matrix], name='char_embedding')
        else:
            self.word_input = Input(shape=(self.word_maxlen, ), dtype='int32', name='word')
            self.char_input = Input(shape=(self.char_maxlen, ), dtype='int32', name='char')
            self.word_masking = Masking(mask_value=self.masking_value)
            self.char_masking = Masking(mask_value=self.masking_value)
            self.word_embedding = Embedding(self.word_vocab_size, self.word_embed_dim, weights=[self.word_embed_matrix], name='word_embedding')
            self.char_embedding = Embedding(self.char_vocab_size, self.char_embed_dim, weights=[self.char_embed_matrix], name='char_embedding')
        
        # 结构化特征
        word_structured = Input(shape=(self.config.word_svd_n_componets, ), dtype='float32', name='word_structured')
        char_structured = Input(shape=(self.config.char_svd_n_componets, ), dtype='float32', name='char_structured')
        if self.config.structured == 'word':
            # TODO 只支持LSA特征，暂不支持TFIDF特征，因为维度太大
            self.structured_input = [word_structured]   # 放在[]中是方便添加到别的列表中，比如Input列表和Tensor列表
        elif self.config.structured == 'char':
            self.structured_input = [char_structured]
        elif self.config.structured == 'both':
            self.structured_input = [word_structured, char_structured]
        
        # Bert编码向量
        if self.config.bert_flag:
            self.word_input = Input(shape=(self.config.bert_maxlen, self.config.bert_dim, ), dtype='float32', name='word_bert')  # 输入是2维！
            self.word_masking = Masking(mask_value=self.masking_value)
            self.word_embedding = None
        
        
    def lr_decay_poly(self, epoch, alpha=0.5, beta=12):
        """训练learning rate衰减schedular"""
        # TODO 哪种衰减？？？
        init_lr = self.init_lr
        lr = init_lr * alpha * ((1 + epoch) // beta)
        print(f'Epoch: {1 + epoch}, lr: {lr}, wd: {self.wd}')
        return lr
        
    
    def plot_history(self, history, i_fold=None):
        """绘制训练loss和accuracy，并保存图片"""
        if not isinstance(history, dict):
            history = history.history
        epochs = np.arange(0, len(history['loss']))
        plt.style.use('ggplot')
        plt.figure()
        plt.plot(epochs, history['loss'], label='train_loss')
        plt.plot(epochs, history['val_loss'], label='val_loss')
        plt.plot(epochs, history['acc'], label='train_acc')
        plt.plot(epochs, history['val_acc'], label='val_acc')
        plt.title(self.name + ' (mode=' + str(self.mode) + ')')
        plt.xlabel('Epoch #')
        plt.ylabel('Loss & Accuracy')
        plt.legend()
        os.makedirs('history', exist_ok=True)
        postfix = '-fold' + str(i_fold) if i_fold else ''
        plt.savefig('history/' + self.name + '-mode' + str(self.mode) + postfix + '.png')
        plt.close()
    
    
    def plot_histories(self, history1, history2, i_fold=None):
        """绘制两阶段训练的loss和accuracy，并保存图片"""
        history1, history2 = history1.history, history2.history
        history = {}
        history['loss'] = history1['loss'] + history2['loss']
        history['val_loss'] = history1['val_loss'] + history2['val_loss']
        history['acc'] = history1['acc'] + history2['acc']
        history['val_acc'] = history1['val_acc'] + history2['val_acc']
        self.plot_history(history, i_fold)
    
    
    def embedding_trainable(self, trainable=True):
        """是否解冻Embedding Layer"""
        if self.config.token_level == 'both':
            self.model.get_layer('char_embedding').trainable = trainable
            if not self.config.bert_flag:
                self.model.get_layer('word_embedding').trainable = trainable
        elif self.config.token_level == 'word':
            if not self.config.bert_flag:
                self.model.get_layer('word_embedding').trainable = trainable
        elif self.config.token_level == 'char':
            self.model.get_layer('char_embedding').trainable = trainable
        else:
            exit('Wrong Token Level')


    def _evaluate(self, x_test, y_test):
        """模型评估"""
        _, test_acc = self.model.evaluate(x_test, y_test)
        test_pred = self.model.predict(x_test, verbose=1)
        scores = self.multilabel_precision_recall(test_pred, y_test)
        vectors, sims = self.multilabel_distribution_similarity(test_pred, y_test)
        print('------------------ Final: Test Metrics: ------------------')
        print('Test Accuracy: ' + str(round(test_acc, 4)))
        print('Precision: ' + str(scores[0]) + '  Recall: ' + str(scores[1]) + '  F1score: ' + str(scores[2]))
        print('Cosine: ' + str(sims[0]) + '  Entropy: ' + str(sims[1]) + '  Euclidean: ' + str(round(sims[2], 1)) + '  Manhattan: ' + str(sims[3]))
        return test_acc, scores, sims, vectors, test_pred


    def train_evaluate(self, x_train, y_train, x_test, y_test, lr=1e-3, epochs=None):
        """
        模型训练和评估
        x_train/x_test是字典(key=Input创建时的name, value=Input对应的数据)，能够支持多输入
        """
        # 模型训练
        print('【' + self.name + '】')
        if self.config.bert_flag:           # 以Bert编码向量作为输入的模型
            epochs = epochs if epochs else self.n_epochs
            print('---------------------------------------------------------------------')
            optimizer = Adam(lr=lr)
            self.model.compile(loss=self.loss, optimizer=optimizer, metrics=self.metrics)
            history = self.model.fit(x_train, y_train, 
                                     batch_size=self.batch_size*self.config.n_gpus,
                                     epochs=epochs,
                                     validation_split=0.3)
        else:
            self.mode = 3
            epochs = epochs if epochs else (2, self.n_epochs)
            print('-------------------Step1: 前期冻结Embedding层，编译和训练模型-------------------')
            self.embedding_trainable(False)
            print('Embedding Trainable: ' + str(self.model.get_layer('word_embedding').trainable))
            optimizer = Adam(lr=lr, clipvalue=2.4)       # clipvalue不应该写死，或者使用默认值！下同
            self.model.compile(loss=self.loss, optimizer=optimizer, metrics=self.metrics)
            history1 = self.model.fit(x_train, y_train, 
                                      batch_size=self.batch_size*self.config.n_gpus,
                                      epochs=epochs[0],
                                      validation_split=0.3)
            print('-------------Step2: 训练完参数后，解冻Embedding层，再次编译和训练模型-------------')
            self.embedding_trainable(True)
            print('Embedding Trainable: ' + str(self.model.get_layer('word_embedding').trainable))
            optimizer = Adam(lr=lr, clipvalue=1.5)
            self.model.compile(loss=self.loss, optimizer=optimizer, metrics=self.metrics)
            #callbacks = [self.lr_schedule, self.checkpoint, ]       # TODO self.checkpoint???
            history2 = self.model.fit(x_train, y_train, 
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
        
    
    def model_compile_fit(self, data_fold, optimizer='adam', callbacks=None, epochs=None, model_file=None):
        """模型编译和训练Helper Function，支持各种配置"""
        x_train, y_train, x_val, y_val = data_fold
        epochs = epochs if epochs else self.n_epochs
        self.model.compile(loss=self.loss, optimizer=optimizer, metrics=self.metrics)   # TODO 多标签时accuracy含义是什么？
        history = self.model.fit(x_train, y_train, 
                                 batch_size=self.batch_size*self.config.n_gpus,
                                 epochs=epochs,
                                 validation_data=(x_val, y_val),
                                 callbacks=callbacks)
        if model_file:
            self.model.save_weights(model_file)
        return history


    def train_evaluate_cv(self, x_train, y_train, x_test, mode=3):
        """
        使用KFold方式训练模型，应用于x_train和x_test
        x_train/x_test是字典(key=Input创建时的name, value=Input对应的数据)，表示多输入
        model: 训练模式，包括各种Finetuning策略等
        """
        self.mode = mode
        checkpoint_path = 'checkpoint-mode' + str(mode) + '/' + self.name + '/'
        os.makedirs(checkpoint_path, exist_ok=True)
        # 先保存训练前的原始模型(参数和状态处于初始状态)，以便于后续KFold时每次加载的都是原始模型(line359)，保证起点一致，各Fold之间互不影响
        init_model_file = checkpoint_path + 'init_weight.h5'
        self.model.save_weights(init_model_file)
        
        # KFold循环前准备
        test_pred = np.zeros((len(x_test['word']), self.n_classes))     # K次预测结果的平均值(要对x_test预测K次)
        train_pred = np.zeros((len(x_train['word']), self.n_classes))   # K次预测结果不重不漏地覆盖所有x_train
        scores_pre, scores_rec, scores_f1, scores_sim = [], [], [], []
        
        for i_fold, (train_index, val_index) in enumerate(self.kfold.split(x_train['word'])):
            self.model.load_weights(init_model_file)      # 每次KFold开始时加载的都是原始模型
            
            # 取数：X和Y
            x_train_fold, x_val_fold = {}, {}
            # TODO 改到__init__里，自动取舍各name！
            for key in ['word', 'word_left', 'word_right', 'word_structured', 'char', 'char_left', 'char_right', 'char_structured']: # 对应model创建时Input的name
                x_train_fold[key] = x_train[key][train_index]
                x_val_fold[key] = x_train[key](val_index)
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
            data_fold = (x_train_fold, y_train_fold, x_val_fold, y_val_fold)
            
            
            # 创建Callbacks: checkpoint, snapshot
            model_prefix = checkpoint_path + '/' + str(i_fold)
            os.makedirs(model_prefix, exist_ok=True)
            model_file = model_prefix + '/k' + str(i_fold) + '_model.h5'
            checkpoint = ModelCheckpoint(model_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')    # TODO min???
            snapshot = self.snapshot.get_callbacks(model_save_place=model_prefix)
            # TODO 创建callbacks不规范，有的在__init__中，有的在每次KFold内各mode前，有的在某mode内！最好统一规范一下！
            
            
            # 模型编译和训练
            # 支持6种模式
            #     1 = 一直冻结，一次编译和训练
            #   2,3 = 前期冻结，后期解冻，两次编译和训练
            # 4,5,6 = 一直解冻，一次编译和训练
            if mode == 1:
                # 一直冻结Embedding，使用snapshot方式训练模型
                self.embedding_trainable(False)
                optimizer = Adam(lr=1e-3, clipvalue=2.0)
                callbacks = [snapshot, ]
                history = self.model_compile_fit(data_fold, optimizer, callbacks, epochs=self.snap_epoch, model_file=None)
                
            elif mode == 2:
                # 前期冻结Embedding层，模型编译和训练
                self.embedding_trainable(False)
                optimizer = Adam(lr=1e-3, clipvalue=2.0)
                history1 = self.model_compile_fit(data_fold, optimizer, epochs=6)
                # 训练好参数后，解冻Embedding层，再次编译，使用snapshot方式训练模型
                self.embedding_trainable(True)
                optimizer = 'adam'
                callbacks = [snapshot, ]
                history2 = self.model_compile_fit(data_fold, optimizer, callbacks, epochs=self.snap_epoch, model_file=None)
                
            elif mode == 3:
                # 前期冻结Embedding层，模型编译和训练
                self.embedding_trainable(False)
                optimizer = Adam(lr=1e-3, clipvalue=2.4)
                history1 = self.model_compile_fit(data_fold, optimizer, epochs=2, model_file=None)
                # 训练好参数后，解冻Embedding层，再次编译，训练模型
                self.embedding_trainable(True)
                optimizer = Adam(lr=1e-3, clipvalue=1.5)
                callbacks = [self.lr_schedule, checkpoint, ]
                history2 = self.model_compile_fit(data_fold, optimizer, callbacks, epochs=10, model_file=None)
                self.plot_histories(history1, history2, i_fold)
                
            elif mode == 4:
                # 一直解冻Embedding层，编译和训练模型
                if self.config.n_gpus == 1:             # TODO 为什么gpu=1时为True，=2时呢？为False??? 注意，默认为True
                    self.embedding_trainable(True)
                optimizer = SGD(lr=self.init_lr, momentum=0.9, decay=1e-6)
                callbacks = [LearningRateScheduler(self.poly_decay), self.early_stopping, ]
                history = self.model_compile_fit(data_fold, optimizer, callbacks, model_file=model_file)
                self.plot_history(history, i_fold)
                
            elif mode == 5:
                # 一直解冻Embedding层，编译和训练模型
                optimizer = Adam(lr=1e-3, clipnorm=1.0)
                callbacks = [self.lr_schedule, checkpoint, ]
                history = self.model_compile_fit(data_fold, optimizer, callbacks, epochs=20, model_file=None)
                self.plot_history(history, i_fold)
                
            elif mode == 6:
                # 一直解冻Embedding层，编译，使用snapshot方式训练模型
                if self.config.n_gpus == 1:
                    self.embedding_trainable(True)
                optimizer = Adam(lr=self.init_lr, decay=1e-6)
                callbacks = [snapshot, ]
                history = self.model_compile_fit(data_fold, optimizer, callbacks, model_file=None)
                self.plot_history(history, i_fold)
                
            else:
                exit('Wrong mode! mode must be in (1, 2, 3, 4, 5, 6)')
                
            
            # 模型评估
            h5models = [x for x in os.listdir(model_prefix) if '.h5' in x]
            print(h5models)
            test_pred_fold = np.zeros((len(x_test['word']), self.n_class))      # 预测test，按模型个数取平均值
            val_pred_fold = np.zeros((len(x_val_fold['word']), self.n_class))   # 预测val，按模型个数取平均值
            for h5file in h5models:
                self.model.load_weights(os.path.join(model_prefix, h5file))
                test_pred_fold += self.model.predict(x_test, verbose=1) / len(h5models)
                val_pred_fold += self.model.predict(x_val_fold, batch_size=64*self.config.n_gpus) / len(h5models)
            
            test_pred += test_pred_fold / self.n_folds      # 按KFold取平均值
            train_pred[val_index] = val_pred_fold
            
            precision, recall, f1score = self.multilabel_precision_recall(val_pred_fold, y_val_fold)
            vectors, sims = self.multilabel_distribution_similarity(val_pred_fold, y_val_fold)
            print('KFold CV precision = ' + str(precision))
            print('KFold CV recall = ' + str(recall))
            print('KFold CV f1score = ' + str(f1score))
            print('KFold CV similarity = ' + str(sims[0]))
            scores_pre.append(precision)
            scores_rec.append(recall)
            scores_f1.append(f1score)
            scores_sim.append(sims[0])
            
        
        # KFold结束后，保存预测结果
        print('Total precision = ' + str(np.mean(scores_pre)))
        print('Total recall = ' + str(np.mean(scores_rec)))
        print('Total f1score = ' + str(np.mean(scores_f1)))
        print('Total similarity = ' + str(np.mean(scores_sim)))
        result_prefix = './result/mode' + str(mode) + '_'
        result_postfix = 'f1_' + str(np.mean(scores_f1)) + 'pre_' + str(np.mean(scores_pre)) + 'rec_' + str(np.mean(scores_rec)) + '.pkl'
        #os.makedirs(result_prefix, exist_ok=True)
        pickle.dump(train_pred, open(result_prefix + self.name + '_oof_' + result_postfix, 'wb'))
        pickle.dump(test_pred, open(result_prefix + self.name + '_test_' + result_postfix, 'wb'))
        
        
    def load_model(self, model_file):
        """加载模型及权重"""
        self.model = load_model(model_file)
        
        
    def load_weights(self, weights_file):
        """加载模型的权重"""
        self.model.load_weights(weights_file)
        
        
    