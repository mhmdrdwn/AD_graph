#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 07:34:26 2025

@author: mohamedr
"""

from captum.attr import Saliency, IntegratedGradients, GuidedGradCam, GuidedBackprop, InputXGradient
import torch


TEST_MODEL = None
def build_maps(data_iter, map_kind, test_model):
    global TEST_MODEL

    masks = []
    for t, batch in enumerate(data_iter):
        x, idx, attr, y, batch = batch.x, batch.edge_index, batch.edge_attr, batch.y, batch.batch
        x = x.float()
        attr = attr.float()
        idx = idx.long()
        y = torch.argmax(y.int(), -1)
    
        if map_kind == "node":
            TEST_MODEL = test_model
            x.requires_grad=True
            attr.requires_grad=False
            sal = Saliency(model_forward1)
            mask = sal.attribute(x, target=y, additional_forward_args=(attr, idx, batch))
    
        elif map_kind == "edge":
            TEST_MODEL = test_model
            x.requires_grad=False
            attr.requires_grad=True
            sal = Saliency(model_forward2)
            mask = sal.attribute(attr, target=y, additional_forward_args=(x, idx, batch))
        
        mask = mask.squeeze().detach().numpy()
        masks.extend(mask)
    return masks

def model_forward1(X, attr, idx, batch):
    out = TEST_MODEL(X, idx, attr, batch)
    return out  

def model_forward2(attr, X, idx, batch): 
    out = TEST_MODEL(X, idx, attr, batch)
    return out