#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 07:34:26 2025

@author: mohamedr
"""



from torch_geometric.nn import BatchNorm, global_mean_pool, global_add_pool, GATConv, NNConv
from torch.nn import BatchNorm1d
import torch.nn.functional as F
import torch.nn as nn

import torch
import numpy as np


class GATblock(nn.Module):
    def __init__(self, num_feat, nheads, hidden_dim, output_dim, num_edges):
        super(GATblock, self).__init__()
        self.nheads = nheads
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim 
        self.num_edges = num_edges
        self.conv1 = GATConv(num_feat, hidden_dim, heads=nheads,  
                             edge_dim=num_edges)
        self.conv2 = GATConv(hidden_dim * nheads, output_dim, 
                             heads=nheads, edge_dim=num_edges)
        self.conv3 = GATConv(output_dim * nheads, output_dim, 
                             edge_dim=num_edges)
        self.BN1 = BatchNorm(hidden_dim*nheads)
        self.BN2 = BatchNorm(output_dim*nheads)
        self.BN3 = BatchNorm(output_dim) 
        
    def forward(self, x, edge_index, edge_weigth):
        x = self.conv1(x, edge_index, edge_weigth)
        x = self.BN1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weigth)
        x = self.BN2(x)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_weigth)
        x = self.BN3(x)
        x = F.relu(x)
        return x

class GraphAttTrans(nn.Module):
    def __init__(self, num_nodes, num_bands, num_conns, num_outputs, 
                 timepoints, num_edges, device):
        super(GraphAttTrans, self).__init__()
        self.enc = GATblock(num_feat=num_bands, nheads=1,
                            hidden_dim=32, output_dim=16, 
                            num_edges=num_edges).to(device)
        self.num_nodes = num_nodes
        self.BN = nn.BatchNorm1d(self.num_nodes)
        self.dropout = nn.Dropout(p=0.3)
        self.trans_out_dim = 32
        self.LSTM_layers = 1
        self.timepoints = timepoints
        self.num_bands = num_bands
        self.GATout = 16
        self.Convout = 373
        self.fc = nn.Linear(self.trans_out_dim, num_outputs)
        self.num_edges = num_edges
        self.BN = BatchNorm(self.trans_out_dim)
        self.conv1 = nn.Conv1d(self.GATout, self.GATout , 3, stride=3)
        self.conv2 = nn.Conv1d(self.GATout, self.GATout , 3, stride=3)
        self.conv3 = nn.Conv1d(self.GATout, self.GATout , 3, stride=3)
        self.Trans = TransformerModel(input_dim=self.GATout)

    def forward(self, X, idx, attr, batch):
        batch_size = int(X.shape[0]/self.num_nodes)
        X = X.view(X.shape[0]*self.timepoints, self.num_bands)
        HS = self.enc(X, idx, attr)
        HS = nn.functional.relu(HS)
        HS = HS.view(batch_size*self.num_nodes, self.timepoints, self.GATout)
        HS = HS.permute(0,2,1)
        HS = self.conv1(HS)
        HS = self.conv2(HS)
        HS = self.conv3(HS)
        HS = HS.permute(0,2,1)
        HS = self.Trans(HS)
        HS = self.BN(HS)
        output = global_mean_pool(HS, batch)
        output = self.fc(output)
        output = nn.Sigmoid()(output)
        return output
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, input_dim=16, d_model=64, nhead=1, num_layers=1, dropout=0.2):
        super(TransformerModel, self).__init__()

        self.encoder = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, 32)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.decoder(x[:, -1, :])
        return x


