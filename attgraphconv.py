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

class GraphAtt3DConv(nn.Module):
    def __init__(self, num_nodes, num_bands, num_conns, num_out, 
                 timepoints, num_edges, device):
        super(GraphAtt3DConv, self).__init__()
        self.enc = GATblock(num_feat=num_bands, nheads=1,
                            hidden_dim=32, output_dim=16, 
                            num_edges=num_edges).to(device)
        self.num_nodes = num_nodes
        self.BN = nn.BatchNorm1d(self.num_nodes)
        self.dropout = nn.Dropout(p=0.3)
        self.trans_out_dim = 1920
        self.LSTM_layers = 1
        self.timepoints = timepoints
        self.num_bands = num_bands
        self.GATout = 16
        self.Convout = 373
        self.fc1 = nn.Linear(self.trans_out_dim, 64)
        self.fc2 = nn.Linear(64, num_out)
        self.num_edges = num_edges
        self.BN = BatchNorm(self.trans_out_dim)
        self.conv1 = nn.Conv1d(self.GATout, self.GATout , 3, stride=3)
        self.conv2 = nn.Conv1d(self.GATout, self.GATout , 3, stride=3)
        self.conv3 = nn.Conv1d(self.GATout, self.GATout , 3, stride=3)
        self.ThreeDenc = CONV2DBlock(num_features=self.GATout) # adopted from EEGNet

    def forward(self, X, idx, attr, batch):
        batch_size = int(X.shape[0]/self.num_nodes)
        X = X.view(X.shape[0]*self.timepoints, self.num_bands)
        HS = self.enc(X, idx, attr)
        HS = nn.functional.relu(HS)
        HS = HS.view(batch_size*self.num_nodes, self.timepoints, self.GATout) # [3, 19, 3840, 16]
        HS = HS.permute(0,2,1)
        HS = HS.unsqueeze(1) #[3, 1, 16, 19, 3840]
        HS = self.ThreeDenc(HS)
        HS = self.BN(HS)
        output = global_mean_pool(HS, batch)
        output = self.fc1(output)
        output = self.fc2(output)
        output = nn.Sigmoid()(output)
        return output

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class CONV2DBlock(nn.Module):
    def __init__(self, num_features):
        super(CONV2DBlock, self).__init__()

        self.F1 = 8
        self.F2 = 16
        self.D = 2
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(self.F1)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.F1, self.D*self.F1, (num_features, 1), groups=self.F1, bias=False),
            nn.BatchNorm2d(self.D*self.F1),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.5)
        )
        
        self.Conv3 = nn.Sequential(
            nn.Conv2d(self.D*self.F1, self.D*self.F1, (1, 16), padding=(0, 8), groups=self.D*self.F1, bias=False),
            nn.Conv2d(self.D*self.F1, self.F2, (1, 1), bias=False),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.5)
        )
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.Conv3(x)
        x = torch.flatten(x, start_dim=1)
        return x


