from torch_geometric.nn import GATv2Conv, GeneralConv, GMMConv, GENConv, BatchNorm, global_mean_pool, TransformerConv, global_add_pool, GATConv, NNConv, ResGatedGraphConv, global_max_pool
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
        
        self.conv1 = TransformerConv(num_feat, hidden_dim, heads=nheads,
                             edge_dim=num_edges)
        self.conv2 = TransformerConv(nheads*hidden_dim , output_dim , heads=nheads,
                                     edge_dim=num_edges)
        self.conv3 = TransformerConv(nheads*output_dim, output_dim, heads=nheads,
                             edge_dim=num_edges)
        
        self.BN1 = BatchNorm(hidden_dim*nheads)
        self.BN2 = BatchNorm(output_dim*nheads)
        self.BN3 = BatchNorm(output_dim*nheads) 
    
    def forward(self, x, edge_index, edge_weigth):
        x1 = self.conv1(x, edge_index, edge_weigth)
        x1 = self.BN1(x1)
        x1 = F.relu(x1)
        x2 = self.conv2(x1, edge_index, edge_weigth)
        x2 = self.BN2(x2)
        x2 = F.relu(x2)
        x3 = self.conv3(x2, edge_index, edge_weigth)
        x3 = self.BN3(x3)
        x3 = F.relu(x3)
        return x1, x2, x3


class GraphModel1(nn.Module):
    def __init__(self, num_nodes, num_signals, num_conns, 
                 timepoints, num_edges, device):
        super(GraphModel1, self).__init__()
        self.GATout = 32
        self.GATin = 50
        self.GAThidden = 32
        self.num_signals = num_signals
        self.num_edges = num_edges
        self.num_nodes = num_nodes
        nheads = 5
        self.enc = GATblock(num_feat=self.GATin, nheads=nheads,
                            hidden_dim=self.GAThidden, output_dim=self.GATout, 
                            num_edges=num_edges).to(device)
        
        self.BN2 = BatchNorm(nheads*32*3)
        self.avgpool = nn.MaxPool2d(1, 3)
        self.fc1 = nn.Linear(nheads*32*3, 64)
        #self.BN1 = BatchNorm(64)
        #self.BN_2 = BatchNorm(10)
        #self.BN = BatchNorm(self.GATin)
                
    def forward(self, X, idx, attr, batch):
        #X = X[:, :, :]
        X = torch.flatten(X, start_dim=1)
        x1, x2, x3 = self.enc(X, idx, attr)
        output1 = global_mean_pool(x1, batch)
        output2 = global_mean_pool(x2, batch)
        output3 = global_mean_pool(x3, batch)
        output = torch.cat((output1, output2, output2), dim=1) 
        output = self.BN2(output)
        output = self.fc1(output)
        return output


class OutputLayer(nn.Module):
    def __init__(self, num_feat, num_classes):
        super(OutputLayer, self).__init__()
        self.fc2 = nn.Linear(num_feat, num_classes)

    def forward(self, feat):
        out = F.dropout(F.relu(feat), training=self.training)
        out = self.fc2(out)
        return out


class PredictiveClassifier(nn.Module):
    def __init__(self, num_nodes, num_signals, num_edges, timepoints,
                 num_conns, num_out_feat, num_classes, device):
        super(PredictiveClassifier, self).__init__()
        self.outputlayer =  OutputLayer(num_out_feat, num_classes)
        self.num_edges = num_conns*num_signals
        self.device = device
        self.enc = GraphModel1(num_nodes=num_nodes, timepoints=timepoints,
                 num_signals=num_signals, num_conns=num_conns,
                 num_edges=self.num_edges, device=device)
        
    def forward(self, x, idx, attr, batch):
        out = self.enc(x, idx, attr, batch)
        out = self.outputlayer(out)
        return out