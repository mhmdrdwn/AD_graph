#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 07:34:26 2025

@author: mohamedr
"""


import torch

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.base import TransformerMixin,BaseEstimator
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import copy
import networkx as nx
import numpy as np
import copy



def label2skip(train_graphs, train_X, train_y, skip_label):
    train_graphs_, train_X_, train_y_ =  [], [], []
    for g, x, y in zip(train_graphs, train_X, train_y):
        if y[0] == skip_label:
            pass
        else:
            train_graphs_.append(g)
            train_X_.append(x)
            train_y_.append(y)
       
    if skip_label == 0:
        train_y_ = [y-1 for y in train_y_]
    elif skip_label == 1:
        train_y_ = [y-1 if y[0]==2 else y for y in train_y_]

    train_graphs, train_X, train_y = train_graphs_, train_X_, train_y_
    train_graphs, train_X, train_y = np.array(train_graphs), np.array(train_X), np.array(train_y)
    return train_graphs, train_X, train_y


from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.base import TransformerMixin,BaseEstimator

#https://stackoverflow.com/questions/50125844/how-to-standard-scale-a-3d-matrix
class StandardScaler3D(BaseEstimator,TransformerMixin):
    #batch, sequence, channels
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X.reshape(-1, X.shape[2]*X.shape[3]))
        return self

    def transform(self,X):
        return self.scaler.transform(X.reshape(-1, X.shape[2]*X.shape[3])).reshape(X.shape)

def standardize_data(train_X, test_X):
    train_X = np.moveaxis(train_X, 1, 2)
    test_X = np.moveaxis(test_X, 1, 2)
    scaler = StandardScaler3D()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)
    train_X = np.moveaxis(train_X, 1, 2)
    test_X = np.moveaxis(test_X, 1, 2)
    return train_X, test_X



def standardize_data(train_X, test_X):
    feat_dim = train_X.shape[-1]
    ch_dim = train_X.shape[1]
    spec_dim = train_X.shape[2]
    for i in range(feat_dim):
        for j in range(ch_dim):
            for k in range(spec_dim):
                mean_ = train_X[:, j, k, i].mean()
                std_ = train_X[:, j, k, i].std()
                train_X[:, j, k, i] = (train_X[:, j, k, i] - mean_)/(std_)
                test_X[:, j, k, i] = (test_X[:, j, k, i] - mean_)/(std_)
    return train_X, test_X



def standardize_data(train_X, test_X):
    feat_dim = train_X.shape[-1]
    ch_dim = train_X.shape[1]
    for i in range(feat_dim):
        for j in range(ch_dim):
            mean_ = train_X[:, j, :, i].mean()
            std_ = train_X[:, j, :, i].std()
            train_X[:, j, :, i] = (train_X[:, j, :, i] - mean_)/(std_)
            test_X[:, j, :, i] = (test_X[:, j, :, i] - mean_)/(std_)
    return train_X, test_X

