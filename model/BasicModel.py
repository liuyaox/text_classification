# -*- coding: utf-8 -*-
"""
Created:    2019-08-11 19:40:55
Author:     liuyao8
Descritipn: a. BasicModel: 模型基类，用于生成BasicStatModel和BasicDeepModel，目前仅提供功能：模型评估Metrics计算
            b. BasicStatModel: 传统模型基类，提供通用功能：
            c. BasicDeepModel: 深度模型基类，提供通用功能：
"""


from functools import reduce
from collections import Counter
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc



class BasicModel(object):
    
    def __init__(self):
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
        """
        assert len(ys_pred) == len(ys_true)
        
        right_num, all_pred_num, all_true_num = 0, 0, 0     # 总命中标签数  总预测标注数  总真实标签数
        for y_pred, y_true in zip(ys_pred, ys_true):
            y_pred_set, y_true_set = set(y_pred), set(y_true)
            all_pred_num += len(y_pred_set)
            all_true_num += len(y_true_set)
            right_num += len(y_pred_set & y_true_set)       # 命中标签数：交集大小
        
        precision = float(right_num) / all_pred_num
        recall = float(right_num) / all_true_num
        f1 = (precision * recall) / (precision + recall)
        return precision, recall, f1
        
    
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
        
        
    def multilabel_eval_distribution(self, ys_pred, ys_true):
        """
        多标签分类特定Metrics: 各标签分布余弦相似度和KL散度
        ARGS同上
        RETURN
            similarity: 基于各标签数据分布，预测结果与真实结果的余弦相似度 越小越好
            relative_entropy: KL散度/相对熵 越小越好
        """
        assert len(ys_pred) == len(ys_true)
        
        ys_pred = Counter(reduce(lambda x, y: x + y, ys_pred))
        ys_true = Counter(reduce(lambda x, y: x + y, ys_true))
        keys = list(set(list(ys_pred.keys()) + list(ys_true.keys())))
        vec_pred = [ys_pred[k] for k in keys]
        vec_true = [ys_true[k] for k in keys]
        
        similarity = cosine_similarity([vec_pred], [vec_true])[0, 0]    # 余弦相似度
        relative_entropy = entropy(vec_pred, vec_true)                  # KL散度/相对熵
        return similarity, relative_entropy



class BasicStatModel(BasicModel):
    
    def __init__(self, n_fold=5, name='BasicStatModel', config=None):
        pass
    
    
    
    
class BasicDeepModel(BasicModel):
    
    def __init__(self, n_fold=5, name='BasicDeepModel', config=None):
        pass
    
    
    
