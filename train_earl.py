#!/usr/bin/env python
# coding: utf-8
import sys
import json
from datetime import datetime
from math import sqrt
import torch
import numpy as np
import scanpy
from collections import defaultdict
import pickle
import random
from collections import Counter
import torch.nn.functional as F
from torch.nn import Linear, LeakyReLU, Sigmoid, BCELoss
from torch_geometric.nn import GATConv, HeteroConv, SAGEConv
from torch_geometric.data import HeteroData
import torch_geometric
from torch import tensor
from torch.distributions import Bernoulli


# ## EARL = Expression and Representation Learner

# ✓ For each cell, create a data vector.
# 
# ✓ Data level batching
# 
# Graph level batching (is this necessary?)
# 
# ✓ Metapath or TransE for featureless (all) nodes?
# 
# Random masking (self supervision)
# 
# ✓ Backprop loss of just unknown
# 
# ✓ Make most graphs undirected. Remove incoming edges to known nodes?
# 
# ✓ Create a GNN
# 
# Train GNN


def proteins_to_idxs(data):
    indexes = []
    proteins = data.var.index.to_list()
    for protein_name in proteins:
        protein_name = protein_name.upper()
        if protein_name in node_idxs['gene_name']:
            indexes.append(node_idxs['gene_name'][protein_name])
        else:
            indexes.append(None)
    return indexes, data.X

def genes_to_idxs(data):
    indexes = []
    genes = data.var['gene_ids'].to_list()
    for gene_id in genes:
        if gene_id in node_idxs['gene']:
            indexes.append(node_idxs['gene'][gene_id])
        else:
            indexes.append(None)
    return indexes, data.X

def atac_to_idxs(data):
    indexes = {}
    regions = data.var.index.to_list()
    for region in regions:
        if region in node_idxs['atac_region']:
            indexes.append(node_idxs['atac_region'][region])
        else:
            indexes.append(None)
    return indexes, data.X

def append_expression(graph, cell_idx):
    newgraph = HeteroData()
    
    gene = tensor(gene_expression[cell_idx].todense())*128
    protein = tensor(protein_expression[cell_idx].todense())
        
    expression = dict()

    for node_type in ['gene_name', 'gene', 'atac_region']:
        expression[node_type] = torch.ones((
            len(node_idxs[node_type]),
            1
        ),device=device)*-1
    
    for i in range(gene.shape[1]):
        if gene_idxs[i]:
            expression['gene'][gene_idxs[i]] = gene[:,i]
    
    for i in range(protein.shape[1]):
        if protein_idxs[i]:
            expression['gene_name'][protein_idxs[i]] = protein[:,i]

    newgraph['gene_name'].y = expression['gene_name']
    newgraph['gene_name'].x = torch.ones((len(node_idxs['gene_name']),1),device=device)*-1

    newgraph['gene'].x = torch.cat([
        graph['gene'].x,
        expression['gene']
    ], dim=1)
        
    newgraph['atac_region'].x = torch.cat([
        graph['atac_region'].x,
        expression['atac_region']
    ], dim=1)
    
    newgraph['tad'].x = graph['tad'].x
    newgraph['protein'].x = graph['protein'].x
    
    for edge_type, store in graph._edge_store_dict.items():
        for k,v in store.items():
            newgraph[edge_type][k]=v
    
    return newgraph


class EaRL(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.sigmoid = Sigmoid()
        # This is a little shim that gets around us not being able to json dump 
        # the class of the layer (i.e GATConv) in the param dictionary
        layer_classes = {'GATConv':GATConv, 'SAGEConv':SAGEConv}
        for layer_type, params in layers:
            layer_type = layer_classes[layer_type]
            conv = HeteroConv({
                ('tad', 'overlaps', 'atac_region'): layer_type(**params),
                ('atac_region', 'rev_overlaps', 'tad'): layer_type(**params),
                ('tad', 'overlaps', 'gene'): layer_type(**params),
                ('gene', 'rev_overlaps', 'tad'): layer_type(**params),
                ('atac_region', 'overlaps', 'gene'): layer_type(**params),
                ('gene', 'rev_overlaps', 'atac_region'): layer_type(**params),
                ('protein', 'coexpressed', 'protein'): layer_type(**params),
                #('protein', 'trrust_interacts', 'gene'): layer_type(**params),
                #('gene', 'rev_trrust_interacts', 'protein'): layer_type(**params),
                ('protein', 'tf_interacts', 'gene'): layer_type(**params),
                ('gene', 'rev_tf_interacts', 'protein'): layer_type(**params),
                ('protein', 'rev_associated', 'gene'): layer_type(**params),
                ('gene', 'associated', 'protein'): layer_type(**params),
            })

            self.convs.append(conv)
        # TODO fiddle with these final layers?
        self.name_conv = HeteroConv({('protein', 'is_named', 'gene_name'): SAGEConv((-1, -1), 1)})
        self.zero_conv = HeteroConv({('protein', 'is_named', 'gene_name'): SAGEConv((-1, -1), 1)})

    def forward(self, x_dict, edge_index_dict):
        #gene_names = x_dict['gene_name']
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        x_dict['gene_name'] = torch.ones((len(node_idxs['gene_name']),1),device=device)*-1
        #gene_prediction = self.relu(self.name_conv(x_dict, edge_index_dict)['gene_name'])
        gene_prediction = self.name_conv(x_dict, edge_index_dict)['gene_name']
        gene_zero_prob = self.sigmoid(self.zero_conv(x_dict, edge_index_dict)['gene_name'])

        x_dict['p_gene_zero'] = gene_zero_prob
        x_dict['gene_zero'] = Bernoulli(gene_zero_prob).sample()
        x_dict['gene_value'] = gene_prediction

        return x_dict

now = datetime.strftime(datetime.now(), format='%Y%m%d-%H%M')

log = open(f'logs/train_earl_{now}.log','w')
param_file = open(f'logs/earl_params_{now}.json','w')

params = {
    'lr':.0005,
    'n_epochs':10,
    'layers':[#('GATConv', {'heads':2, 'in_channels':(-1,-1), 'out_channels':256}), 
              ('SAGEConv', {'in_channels':(-1,-1), 'out_channels':256}),
              ('SAGEConv', {'in_channels':(-1,-1), 'out_channels':256}),
              ('SAGEConv', {'in_channels':(-1,-1), 'out_channels':256}),
              ('SAGEConv', {'in_channels':(-1,-1), 'out_channels':256})],
    'train_batch_size':5,
    'validation_batch_size': 100
}

json.dump(params, param_file)

device=sys.argv[1]

graph = torch.load('input/graph_with_embeddings.torch')
node_idxs = pickle.load(open('input/nodes_by_type.pickle','rb'))

datadir = 'output/datasets/predict_modality/openproblems_bmmc_cite_phase1_mod2/'
protein_data = scanpy.read_h5ad(datadir+'openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_train_mod1.h5ad')
protein_idxs, protein_expression = proteins_to_idxs(protein_data)

gene_data = scanpy.read_h5ad(datadir+'openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_train_mod2.h5ad')
gene_idxs, gene_expression = genes_to_idxs(gene_data)

graph = graph.to('cpu')
graph = torch_geometric.transforms.ToUndirected()(graph)
graph = graph.to(device)

protein_mask = torch.zeros((len(node_idxs['gene_name']),1), dtype=bool, device=device)
protein_mask[protein_idxs] = 1
graph['gene_name']['mask'] = protein_mask

gene_mask = torch.zeros((len(node_idxs['gene']),1), dtype=bool, device=device)
gene_mask[[idx for idx in gene_idxs if idx]] = 1
graph['gene']['mask'] = gene_mask

# ## TODO trim the gene names, we have way more gene names than we have in the data

num_cells = gene_expression.shape[0]

train_set_size = int(num_cells*.7)

val_set_size = num_cells - train_set_size

earl = EaRL(layers=params['layers'])

earl = earl.to(device)
optimizer = torch.optim.Adam(params=earl.parameters(), lr=params['lr'])
earl.train()

n_epochs = params['n_epochs']
checkpoint = 10

cell_idxs = list(range(gene_expression.shape[0]))
random.shuffle(cell_idxs)
train_idxs = cell_idxs[:train_set_size]
validation_idxs = cell_idxs[train_set_size:]

train_batch_size = params['train_batch_size']
validation_batch_size = params['validation_batch_size']

bce_loss = BCELoss()

best_validation_loss = float('inf')

def predict(earl, idxs, mask, eval=False):
    with torch.inference_mode(eval):
        num_predictions = len(idxs)
        predictions = torch.zeros((num_predictions, mask.sum()), device=device)
        p_zeros = torch.zeros((num_predictions, mask.sum()), device=device)
        zero_ones = torch.zeros((num_predictions, mask.sum()), device=device)
        ys = torch.zeros((num_predictions, mask.sum()), device=device)

        for i,idx in enumerate(idxs):
            newgraph = append_expression(graph, idx)
            output = earl(newgraph.x_dict, newgraph.edge_index_dict)
            predictions[i] = output['gene_value'][mask].flatten()
            p_zeros[i] = output['p_gene_zero'][mask].flatten()
            zero_ones[i] = output['gene_zero'][mask].flatten()
            ys[i] = newgraph['gene_name'].y[mask]

        return predictions, p_zeros, zero_ones, ys

for epoch in range(n_epochs):
    mask = graph['gene_name']['mask']
    batch_start = 0
    batch_end = train_batch_size
    
    batch_idx = 0
    while batch_end < len(cell_idxs)+train_batch_size:
        optimizer.zero_grad()
        predictions, p_zeros, zero_ones, ys = predict(earl, train_idxs[batch_start:batch_end], mask)
        y_zero_one = ys > .00001
        zero_one_loss = bce_loss(p_zeros, y_zero_one.float())
        value_loss = ((predictions[y_zero_one] - ys[y_zero_one])**2).mean()
        prediction_loss = float(((predictions*zero_ones - ys)**2).mean())
        loss = zero_one_loss + value_loss
        print(f'Batch: {batch_idx}', file=log)
        print(f'train zero one loss {sqrt(float(zero_one_loss))}', file=log) 
        print(f'train value loss {sqrt(float(value_loss))}',flush=True, file=log)
        print(f'train prediction loss {sqrt(prediction_loss)}',flush=True, file=log)
        loss.backward()
        optimizer.step()
        batch_start += train_batch_size
        batch_end += train_batch_size
        batch_idx += 1
        if batch_idx % checkpoint == 0:
            idxs = random.sample(validation_idxs, k=validation_batch_size)
            predictions, p_zeros, zero_ones, ys = predict(earl, idxs, mask, eval=True)
            y_zero_one = ys > .00001
            zero_one_loss = bce_loss(p_zeros, y_zero_one.float())
            value_loss = ((predictions[y_zero_one] - ys[y_zero_one])**2).mean()
            validation_loss = float(((predictions*zero_ones - ys)**2).mean())
            print(f'validation zero one loss {sqrt(float(zero_one_loss))}', file=log) 
            print(f'validation value loss {sqrt(float(value_loss))}',flush=True, file=log)
            print(f'validation prediction loss {sqrt(validation_loss)}',flush=True, file=log)

            stacked = torch.vstack([predictions[0,:]*zero_ones[0,:], ys[0,:]])
            prediction_log = open(f'logs/train_earl_prediction_sample_{now}.log','w')
            for i in range(stacked.shape[1]):
                print(f'batch {batch_idx} {i:<6d} pred,y: {float(stacked[0,i]):>7.3f} {float(stacked[1,i]):.3f}', file=prediction_log)
            prediction_log.flush()

            if validation_loss < best_validation_loss:
                torch.save(earl.state_dict(), f'models/best_earl_{now}.model')
        
    print(f'Epoch: {epoch}', file=log)
    torch.save(earl, 'earl.model')

