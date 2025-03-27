#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 07:34:26 2025

@author: mohamedr
"""


import torch 
from torch.nn import functional as F

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, auc, roc_auc_score
import numpy as np


def evaluate_model(model, data_iter):
    model.eval()
    loss_func = nn.BCEWithLogitsLoss()
    loss_sum, n = 0, 0
    with torch.no_grad():
        for batch in data_iter:
            x, idx, attr, y, batch = batch.x, batch.edge_index, batch.edge_attr, batch.y, batch.batch
            attr = attr.float()
            x = x.float()
            idx = idx.long()
            yhat = model(x, idx, attr, batch)
            loss = loss_func(yhat, y)
            loss_sum += loss.item()
            n += 1
        return loss_sum / n
    
        

from sklearn.metrics import roc_auc_score
def cal_accuracy(model, data_iter):
    ytrue = []
    ypreds = []
    y_score = []
    ypreds_real = []
    
    model.eval()
    #threeshold = 0.5
    with torch.no_grad():
        for batch in data_iter:
            x, idx, attr, y, batch = batch.x, batch.edge_index, batch.edge_attr, batch.y, batch.batch
            x = x.float()
            attr = attr.float()
            idx = idx.long()
            yhat = model(x, idx, attr, batch)
            #yhat = torch.softmax(yhat, dim=-1)
            y = torch.argmax(y, dim=-1)
            y = y.detach().cpu().numpy()
            yhat = torch.argmax(yhat, dim=-1)
            yhat = yhat.detach().cpu().numpy()
            ypreds_real.extend([yhat_i for yhat_i in yhat])
            ytrue.extend([y_i for y_i in y])
            ypreds.extend([yhat_i for yhat_i in yhat])

    return (accuracy_score(ytrue, ypreds), 
            confusion_matrix(ytrue, ypreds), 
            precision_score(ytrue, ypreds, zero_division=0, average=None), 
            recall_score(ytrue, ypreds, zero_division=0, average=None),
            f1_score(ytrue, ypreds, zero_division=0, average=None))


from collections import Counter

def cal_accuracy_loso(ytrue, ypreds):    
    return (accuracy_score(ytrue, ypreds), 
            confusion_matrix(ytrue, ypreds), 
            precision_score(ytrue, ypreds, zero_division=0, average=None), 
            recall_score(ytrue, ypreds, zero_division=0, average=None),
            f1_score(ytrue, ypreds, zero_division=0, average=None))



def cal_auc(model, data_iter):
    ytrue = []
    ypreds = []
    y_score = []
    ypreds_real = []
    
    model.eval()
    threeshold = 0.5
    with torch.no_grad():
        for batch in data_iter:
            x, idx, attr, y, batch = batch.x, batch.edge_index, batch.edge_attr, batch.y, batch.batch
            x = x.float()
            attr = attr.float()
            idx = idx.long()
            yhat = model(x, idx, attr, batch)
            yhat = torch.sigmoid(yhat)
            y = y.detach().cpu().numpy()
            yhat = yhat.detach().cpu().numpy()
            ypreds_real.extend([yhat_i for yhat_i in yhat])
            ytrue.extend([y_i for y_i in y])
            
    return roc_auc_score(ytrue, ypreds_real)


def avg_accuracy(all_cm, all_y):
    """
    calculate average accuracy from confusion matrices of all datasets
    """
    TN = []
    TP = []
    FP = []
    FN = []
    for k, v in all_cm.items():
        cm = all_cm[k]
        if cm.shape == (2, 2):
            TP.append(cm[1, 1])
            FP.append(cm[1, 0])
            FN.append(cm[0, 1])
            TN.append(cm[0, 0])
        elif cm.shape == (1, 1):
            # if the matrix is (1x1), this means it only contain TP or TN
            # accuracy is 100% in this case
            if all_y[k][0] == 0:
                TN.append(cm[0, 0])
            elif all_y[k][0] == 1 or all_y[k][0] == 2:
                TP.append(cm[0, 0])

    TP = np.sum(TP)
    TN = np.sum(TN)
    FP = np.sum(FP)
    FN = np.sum(FN)
    acc = (TP+TN)/(TP+TN+FP+FN)
    sen = (TP)/(TP+FN) #recall
    spec = (TN)/(TN+FP)
    prec = (TP)/(TP+FP)
    f1 = (2*prec*sen)/(prec+sen)
    return acc, sen, spec, prec, f1



