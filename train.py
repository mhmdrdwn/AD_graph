#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 07:34:26 2025

@author: mohamedr
"""

from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split
from collections import Counter

from evaluate import cal_accuracy


def predict(model, data_iter):
    ytrue = []
    ypreds = []
    
    model.eval()
    cn_threeshold = 0.5
    with torch.no_grad():
        for batch in data_iter:
            x, idx, attr, y, batch = batch.x, batch.edge_index, batch.edge_attr, batch.y, batch.batch
            x = x.float()
            attr = attr.float()
            idx = idx.long()
            yhat = model(x, idx, attr, batch)
            y = torch.argmax(y, dim=-1)
            y = y.detach().cpu().numpy()
            yhat = torch.argmax(yhat, dim=-1)
            yhat = yhat.detach().cpu().numpy()
            ytrue.extend([y_i for y_i in y])
            ypreds.extend([yhat_i for yhat_i in yhat])
    print(ypreds)
    len_ypreds = len(ypreds)
    
    non_pathological = 0
    pathological = 1
    
    pathological_count = ypreds.count(1) # count predict labels 1
    no_pathological_count = ypreds.count(0)
    if no_pathological_count >= cn_threeshold*len_ypreds: #if some pathological trails in subject
        ypreds = [non_pathological for _ in range(len_ypreds)]
    else:
        ypreds = [pathological for _ in range(len_ypreds)]
    
    return ytrue, ypreds

def trainer(num_epochs, model, train_iter, val_iter, kfold_num="None", 
            lr=0.001, report_accuracy=False):
    print("Training Model....")
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    all_train_acc = []
    all_val_acc = []
    for epoch in range(num_epochs):
        model.train()
        for t, batch in enumerate(train_iter):
            x, idx, attr, y, batch = batch.x, batch.edge_index, batch.edge_attr, batch.y, batch.batch
            x = x.float()
            attr = attr.float()
            
            idx = idx.long()
            optimizer.zero_grad()
            y_pred = model(x, idx, attr, batch)
            y_pred = y_pred.float()
            y = y.float()
            loss = loss_func(y_pred, y)
            loss.backward()
            optimizer.step() 

        if report_accuracy:
            train_acc = cal_accuracy(model, train_iter)
            all_train_acc.append(train_acc)
            val_acc = cal_accuracy(model, val_iter)
            all_val_acc.append(val_acc)
            print("Train Acc:", train_acc)
            print("Val Acc:", val_acc) 
            torch.save(model.state_dict(), "saved_models/kfold_model_"+str(kfold_num)+"_"+str(epoch)+"_.pt")
    return model, all_train_acc, all_val_acc       
        
def trainer_regression(num_epochs, model, train_iter, val_iter, kfold_num="None", 
            lr=0.001, report_accuracy=False):
    print("Training Model....")
    loss_func = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for epoch in range(num_epochs):
        model.train()
        for t, batch in enumerate(train_iter):
            x, idx, attr, y, batch = batch.x, batch.edge_index, batch.edge_attr, batch.y, batch.batch
            x = x.float()
            attr = attr.float()
            idx = idx.long()
            optimizer.zero_grad()
            y_pred = model(x, idx, attr, batch)
            y_pred = y_pred.float()
            y = y.float()
            loss = loss_func(y_pred, y)
            loss.backward()
            optimizer.step() 

    return model  


def train_test_split_subjects(files, files_data, labels, random_seed=2025, 
                              skip_label=2, num_test_subjects=6):
    all_labels = [labels[i] for i in files]
    #stratification = []
    files_ = []
    labels = []
    for file, y_label in zip(files, all_labels):
        if y_label[0] != skip_label:
            files_.append(file)
            labels.append(y_label[0])
            
    train_files, test_files = train_test_split(files_, 
                                               random_state=random_seed,
                                               test_size=num_test_subjects)

    return train_files, test_files
