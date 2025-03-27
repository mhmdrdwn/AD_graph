#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 07:34:26 2025

@author: mohamedr
"""

import scipy.sparse as sp
import numpy as np

from torch_geometric.data import Data
import torch
#from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin,BaseEstimator


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = (adj - adj.min())/(adj.max() - adj.min())
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)) # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten() # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # D^-0.5
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo() # D^-0.5AD^0.5

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model 
    and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0])).toarray()
    return adj_normalized

def normalize_all_adj(graphs):
    for graph_idx in range(graphs.shape[0]):
        for band_idx in range(graphs.shape[-1]):
            graphs[graph_idx, :, :, band_idx] = preprocess_adj(graphs[graph_idx, :, :, band_idx])
    return graphs


def build_pyg_dl(x, a, y, num_edge_features, device):
    """Convert features and adjacency to PyTorch Geometric Dataloader"""
    a = torch.from_numpy(a)
    a = a + 1e-10 
    edge_attr = []
    for edge_attr_idx in range(num_edge_features):
        Af = a[:, :, edge_attr_idx]
        Af.fill_diagonal_(1)
        edge_index, attrf = dense_to_sparse(Af)
        edge_attr.append(attrf)
    
    edge_attr = torch.stack(edge_attr)
    edge_attr = torch.moveaxis(edge_attr, 0, 1).to(device)
    edge_index = edge_index.to(device)
    x = torch.from_numpy(x).to(device)
    y = torch.tensor([y], dtype=torch.float).to(device)
    data = Data(x=x, edge_index=edge_index, 
                edge_attr=edge_attr, 
                y=y)
    return data


#https://stackoverflow.com/questions/50125844/how-to-standard-scale-a-3d-matrix
class StandardScaler3D(BaseEstimator,TransformerMixin):
    #batch, sequence, channels
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self,X,y=None):
        X = X.reshape(-1, X.shape[1]*X.shape[3])
        self.scaler.fit(X)
        return self

    def transform(self,X):
        return self.scaler.transform(X.reshape(-1, X.shape[1]*X.shape[3])).reshape(X.shape)
        
def std_data(train_X, val_X):
    scaler = StandardScaler3D()
    train_X = scaler.fit_transform(train_X)
    val_X = scaler.transform(val_X)
    return train_X, val_X


def fill_diag(x):
    num_channels = x.shape[1]
    num_bands = x.shape[0]
    x_diag = np.zeros((num_bands, num_channels, num_channels))
    for band_idx, band in enumerate(x):
        for idx, i in enumerate(band):
            x_diag[band_idx, idx, idx] = 0
            if idx == 0:
                x_diag[band_idx, idx, 1:] = i
            elif idx > 0 and idx < num_channels-1:
                x_diag[band_idx, idx, idx+1:] = i[idx:]
                x_diag[band_idx, idx, :idx] = i[:idx]
            elif idx == num_channels-1:
                x_diag[band_idx, idx, :-1] = i
    return x_diag
