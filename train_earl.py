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
from torch_geometric.graphgym.models import MLP


def proteins_to_idxs(data):
    indexes = []
    proteins = data.var.index.to_list()
    for protein_name in proteins:
        protein_name = protein_name.upper()
        if protein_name in node_idxs['protein_name']:
            indexes.append(node_idxs['protein_name'][protein_name])
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

    for node_type in ['protein_name', 'gene', 'atac_region']:
        # initialize all values to -1
        expression[node_type] = torch.ones((
            len(node_idxs[node_type]),
            1),device=device)*-1
    
    for i in range(gene.shape[1]):
        if gene_idxs[i]:
            expression['gene'][gene_idxs[i]] = gene[:,i]
    
    for i in range(protein.shape[1]):
        if protein_idxs[i]:
            expression['protein_name'][protein_idxs[i]] = protein[:,i]

    newgraph['protein_name'].y = expression['protein_name']
    newgraph['protein_name'].x = torch.ones((len(node_idxs['protein_name']),1),device=device)*-1

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
    def __init__(self, gnn_layers, out_mlp):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.sigmoid = Sigmoid()
        # This is a little shim that gets around us not being able to json dump 
        # the class of the layer (i.e GATConv) in the param dictionary
        layer_classes = {'GATConv':GATConv, 'SAGEConv':SAGEConv}
        for layer_type, params in gnn_layers:
            layer_type = layer_classes[layer_type]
            conv = HeteroConv({
                ('tad', 'overlaps', 'atac_region'): layer_type(**params, in_channels=(-1,-1)),
                ('atac_region', 'rev_overlaps', 'tad'): layer_type(**params, in_channels=(-1,-1)),
                ('tad', 'overlaps', 'gene'): layer_type(**params, in_channels=(-1,-1)),
                ('gene', 'rev_overlaps', 'tad'): layer_type(**params, in_channels=(-1,-1)),
                ('atac_region', 'overlaps', 'gene'): layer_type(**params, in_channels=(-1,-1)),
                ('gene', 'rev_overlaps', 'atac_region'): layer_type(**params, in_channels=(-1,-1)),
                # Not bipartite so we have just (-1) for in channels, same feature sizes for both
                ('protein', 'coexpressed', 'protein'): layer_type(**params, in_channels=-1),
                ('protein', 'tf_interacts', 'gene'): layer_type(**params, in_channels=(-1,-1)),
                ('gene', 'rev_tf_interacts', 'protein'): layer_type(**params, in_channels=(-1,-1)),
                ('protein', 'rev_associated', 'gene'): layer_type(**params, in_channels=(-1,-1)),
                ('gene', 'associated', 'protein'): layer_type(**params, in_channels=(-1,-1)),
            })

            self.convs.append(conv)

        last_layer, last_params = gnn_layers[-1]
        last_layer = layer_classes[last_layer]
        hidden = last_params['out_channels']

        # TODO maybe add a final small MLP after conv?
        self.protein_layer = HeteroConv({('protein', 'is_named', 'protein_name'): 
                                        last_layer(in_channels=(-1, -1), **last_params)})

        self.protein_zero  = MLP(**out_mlp) 
        self.protein_value = MLP(**out_mlp)

        # TODO maybe make this a small NN?
        #self.gene_layer = Linear(hidden,1)
        #self.gene_zero  = Linear(hidden,1)

        #self.atac_layer = Linear(hidden,1)

    def encode(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        return x_dict

    #def gene(self, x_dict, edge_index_dict):
    #    x_dict = self.encode(x_dict, edge_index_dict)
    #    x_dict['p_gene_zero'] = self.sigmoid(self.gene_zero(x_dict['gene']))
    #    x_dict['gene_zero'] = Bernoulli(x_dict['p_gene_zero']).sample()
    #    x_dict['gene_value'] = self.gene(x_dict['gene'])
    #    return x_dict

    #def atac(self, x_dict, edge_index_dict):
    #    x_dict = self.encode(x_dict, edge_index_dict)
    #    x_dict['atac_value'] = self.sigmoid(self.atac(x_dict['atac_region']))
    #    return x_dict

    def protein(self, x_dict, edge_index_dict):
        x_dict = self.encode(x_dict, edge_index_dict)

        x_dict['protein_name'] = torch.ones((len(node_idxs['protein_name']),1),device=device)*-1

        protein_name_embedding = self.protein_layer(x_dict, edge_index_dict)['protein_name']
        x_dict['p_protein_zero'] = self.sigmoid(self.protein_zero(protein_name_embedding))
        x_dict['protein_zero'] = Bernoulli(x_dict['p_protein_zero']).sample()
        x_dict['protein_value'] = self.protein_value(protein_name_embedding)

        return x_dict
    

now = datetime.strftime(datetime.now(), format='%Y%m%d-%H%M')
print(now)

# Device is first command line arg
device=sys.argv[1]
# Provide anything as a second command line argument after device to log to stdout
if len(sys.argv) == 3:
    log = None
    prediction_log = None
else:
    log = open(f'logs/train_earl_{now}.log','w')
    prediction_log = open(f'logs/train_earl_prediction_sample_{now}.log','w')



params = {
    'lr':.001,
    'n_epochs':10,
    #'layers':[('GATConv', {'heads':2, 'out_channels':64}), 
    #          ('GATConv', {'heads':2, 'out_channels':64}), 
    #          ('GATConv', {'heads':2, 'out_channels':64}), 
    'layers':[('SAGEConv', {'out_channels':256}),
              ('SAGEConv', {'out_channels':256}),
              ('SAGEConv', {'out_channels':256}),
              ('SAGEConv', {'out_channels':256})],
    'out_mlp':{'dim_in':256, 'dim_out':1, 'bias':True, 
               'dim_inner': 512, 'num_layers':3},
    'train_batch_size': 20,
    'validation_batch_size': 100,
    'checkpoint': 50,
    'device': device,
}

if log:
    with open(f'logs/earl_params_{now}.json','w') as param_file:
        json.dump(params, param_file)

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

protein_mask = torch.zeros((len(node_idxs['protein_name']),1), dtype=bool, device=device)
protein_mask[protein_idxs] = 1
graph['protein_name']['mask'] = protein_mask

gene_mask = torch.zeros((len(node_idxs['gene']),1), dtype=bool, device=device)
gene_mask[[idx for idx in gene_idxs if idx]] = 1
graph['gene']['mask'] = gene_mask

# TODO trim the gene names, we have way more gene names than we have in the data

num_cells = gene_expression.shape[0]

train_set_size = int(num_cells*.7)

val_set_size = num_cells - train_set_size

earl = EaRL(gnn_layers=params['layers'], out_mlp=params['out_mlp'])

earl = earl.to(device)
optimizer = torch.optim.Adam(params=earl.parameters(), lr=params['lr'])
earl.train()

n_epochs = params['n_epochs']
checkpoint = params['checkpoint']

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
            output = earl.protein(newgraph.x_dict, newgraph.edge_index_dict)
            predictions[i] = output['protein_value'][mask].flatten()
            p_zeros[i] = output['p_protein_zero'][mask].flatten()
            zero_ones[i] = output['protein_zero'][mask].flatten()
            ys[i] = newgraph['protein_name'].y[mask]

        return predictions, p_zeros, zero_ones, ys

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

batches = chunks(train_idxs, train_batch_size)

for epoch in range(n_epochs):
    print(f'Epoch: {epoch}', file=log)
    mask = graph['protein_name']['mask']
    
    for batch_idx, batch in enumerate(batches):
        earl.train()
        optimizer.zero_grad()
        batch_zero_one_loss = 0.0
        batch_value_loss = 0.0
        batch_prediction_loss = 0.0
        for idx in batch:
            predictions, p_zeros, zero_ones, ys = predict(earl, [idx], mask)
            y_zero_one = ys > .00001
            zero_one_loss = bce_loss(p_zeros, y_zero_one.float())
            value_loss = ((predictions[y_zero_one] - ys[y_zero_one])**2).mean()
            prediction_loss = float(((predictions*zero_ones - ys)**2).mean())
            loss = zero_one_loss + value_loss
            loss.backward()

            batch_zero_one_loss += float(zero_one_loss)/len(batch)
            batch_value_loss += float(value_loss)/len(batch)
            batch_prediction_loss += float(prediction_loss)/len(batch)

        print(f'Batch: {batch_idx}', file=log)
        print(f'train zero one loss {sqrt(batch_zero_one_loss)}', file=log) 
        print(f'train value loss {sqrt(batch_value_loss)}',flush=True, file=log)
        print(f'train prediction loss {sqrt(batch_prediction_loss)}',flush=True, file=log)
        optimizer.step()

        # Checkpoint
        if batch_idx % checkpoint == 0:
            earl.eval()
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

            if prediction_log:
                prediction_log.truncate(0)
            for i in range(stacked.shape[1]):
                print(f'batch {batch_idx} {i:<6d} pred,y: '+
                      f'{float(stacked[0,i]):>7.3f} {float(stacked[1,i]):.3f}', 
                      file=prediction_log, flush=True)

            if validation_loss < best_validation_loss:
                torch.save(earl.state_dict(), f'models/best_earl_{now}.model')

