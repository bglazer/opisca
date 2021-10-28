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
from torch_geometric.nn import GATConv, HeteroConv, SAGEConv, GATv2Conv, TransformerConv
from torch_geometric.data import HeteroData
import torch_geometric
from torch import tensor
from torch.distributions import Bernoulli
from torch_geometric.graphgym.models import MLP


def proteins_to_idxs(data):
    indexes = []
    proteins = data.var.index.to_list()
    for i,protein_name in enumerate(proteins):
        protein_name = protein_name.upper()
        if protein_name in node_idxs['protein_name']:
            indexes.append((i,node_idxs['protein_name'][protein_name]))
    return tensor(indexes,device=device), data.X

def genes_to_idxs(data):
    indexes = []
    genes = data.var['gene_ids'].to_list()
    for i,gene_id in enumerate(genes):
        if gene_id in node_idxs['gene']:
            indexes.append((i,node_idxs['gene'][gene_id]))
    return tensor(indexes,device=device), data.X

def atacs_to_idxs(data):
    indexes = {}
    regions = data.var.index.to_list()
    for i,region in enumerate(regions):
        if region in node_idxs['atac_region']:
            indexes.append((i,node_idxs['atac_region'][region]))
    return tensor(indexes,device=device), data.X

def expand_for_data(graph):
    with torch.no_grad():
        for node_type in ['gene', 'atac_region']:
            graph[node_type].x = torch.cat([
                graph[node_type].x,
                torch.ones((len(node_idxs[node_type]),1),device=device)
            ], dim=1)
        return graph
        

def add_expression(graph, cell_idx, task):
    with torch.no_grad():
        graph['protein_name'].x = torch.ones((len(node_idxs['protein_name']),1), device=device)*-1
        # 128 is the feature size from the node2vec
        # multiply by 128 so the expression has roughly the same magnitude as 
        # the rest of the features combined
        src_expr = tensor(expression[task][0][cell_idx].todense(),device=device)*128
        tgt_expr = tensor(expression[task][1][cell_idx].todense(),device=device)
        src_idxs = idxs[task][0]
        tgt_idxs = idxs[task][1]

        graph[source].x[src_idxs[:,1],-1] = src_expr[0,src_idxs[:,0]]

        graph[target].y = torch.ones((len(node_idxs[target]),1), device=device)*-1
        graph[target].y[tgt_idxs[:,1],-1] = tgt_expr[0,tgt_idxs[:,0]]

        for node_type in ['gene','protein_name','atac_region']:
            if node_type not in (source, target):
                graph[node_type].x[:,-1] = torch.ones((len(node_idxs[node_type])), device=device)*-1

        return graph
    

class EaRL(torch.nn.Module):
    def __init__(self, gnn_layers, out_mlp):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.sigmoid = Sigmoid()
        # This is a little shim that gets around us not being able to json dump 
        # the class of the layer (i.e GATConv) in the param dictionary
        layer_classes = {'GATConv':GATConv, 'SAGEConv':SAGEConv,
                         'TransformerConv':TransformerConv}

        self.methods = {'protein_name':self.protein, 'gene':self.gene, 'atac_region':self.atac}

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
                ('protein', 'is_named', 'protein_name'): layer_type(**params, in_channels=(-1,-1)),
                ('protein_name', 'rev_is_named', 'protein'): layer_type(**params, in_channels=(-1,-1)),
            })

            self.convs.append(conv)

        last_layer, last_params = gnn_layers[-1]
        last_layer = layer_classes[last_layer]
        hidden = last_params['out_channels']

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
    
    def forward(self, x_dict, edge_index_dict, task):
        source, target = task
        if target == 'gene':
            x = self.gene(x_dict, edge_index_dict)
        if target == 'protein_name':
            x = self.protein_name(x_dict, edge_index_dict)
        if target == 'atac_region':
            x = self.atac_region(x_dict, edge_index_dict)

        return x


print('Starting')
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
    'layers':[('SAGEConv', {'out_channels':128}),
              ('SAGEConv', {'out_channels':128}),
              ('TransformerConv', {'out_channels':32, 'heads':2})],
    'out_mlp':{'dim_in':64, 'dim_out':1, 'bias':True, 
               'dim_inner': 512, 'num_layers':3},
    'train_batch_size': 20,
    'validation_batch_size': 100,
    'checkpoint': 50,
    'device': device,
}

if log:
    with open(f'logs/earl_params_{now}.json','w') as param_file:
        json.dump(params, param_file)

print('Loading graph')
graph = torch.load('input/graph_with_embeddings.torch')
graph.to(device)

node_idxs = pickle.load(open('input/nodes_by_type.pickle','rb'))
graph = expand_for_data(graph)

print('Loading protein/gene data')
datadir = 'output/datasets/predict_modality/openproblems_bmmc_cite_phase1_mod2/'
datafile = 'openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_train_mod1.h5ad'
protein_data = scanpy.read_h5ad(datadir+datafile)
protein_idxs, protein_expression = proteins_to_idxs(protein_data)

datafile = 'openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_train_mod2.h5ad'
gene_data = scanpy.read_h5ad(datadir+datafile)
gene_idxs, gene_expression = genes_to_idxs(gene_data)

expression[('protein','gene')] = (protein_expression,gene_expression)
idxs[('protein','gene')] = (protein_idxs,gene_idxs)
expression[('gene','protein')] = (gene_expression,protein_expression)
idxs[('gene','protein')] = (gene_idxs,protein_idxs)

print('Loading atac/gene data')
datadir = 'output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_mod2/'
datafile = 'openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_train_mod1.h5ad'
atac_data = scanpy.read_h5ad(datadir+datafile)
atac_idxs, atac_expression = atacs_to_idxs(atac_data)

datafile = 'openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_train_mod2.h5ad'
gene_data = scanpy.read_h5ad(datadir+datafile)
gene_idxs, gene_expression = genes_to_idxs(gene_data)

expression[('atac','gene')] = (protein_expression,gene_expression)
idxs[('atac','gene')] = (protein_idxs,gene_idxs)
expression[('gene','atac')] = (gene_expression,protein_expression)
idxs[('gene','atac')] = (gene_idxs,protein_idxs)

print('Making graph undirected')
graph = graph.to('cpu')
graph = torch_geometric.transforms.ToUndirected()(graph)
graph = graph.to(device)

def set_mask(graph, task):
    for i,node_type in enumerate(task):
        mask = torch.zeros((len(node_idxs[node_type]),1), dtype=bool, device=device)
        mask[idxs[task][i][:,1]] = 1
        graph[node_type]['mask'] = mask
    return graph


print('Initializing EaRL')
earl = EaRL(gnn_layers=params['layers'], out_mlp=params['out_mlp'])
earl = earl.to(device)

optimizer = torch.optim.Adam(params=earl.parameters(), lr=params['lr'])
earl.train()

n_epochs = params['n_epochs']
checkpoint = params['checkpoint']

# TODO trim the gene names, we have way more gene names than we have in the data

train_cell_idxs = {}
validation_cell_idxs = {}
for task in expression:
    num_cells = expression[task][0].shape[0]
    train_set_size = int(num_cells*.7)
    val_set_size = num_cells - train_set_size
    cell_idxs = list(len(expression[task]))
    random.shuffle(cell_idxs)
    train_cell_idxs[task] = cell_idxs[:train_set_size]
    validation_cell_idxs[task] = cell_idxs[train_set_size:]

train_batch_size = params['train_batch_size']
validation_batch_size = params['validation_batch_size']

bce_loss = BCELoss()

best_validation_loss = float('inf')

# TODO make this take a task as an input variable
def predict(earl, task, idxs, mask, eval=False):
    with torch.inference_mode(eval):
        num_predictions = len(idxs)
        predictions = torch.zeros((num_predictions, mask.sum()), device=device)
        p_zeros = torch.zeros((num_predictions, mask.sum()), device=device)
        zero_ones = torch.zeros((num_predictions, mask.sum()), device=device)
        ys = torch.zeros((num_predictions, mask.sum()), device=device)

        for i,idx in enumerate(idxs):
            newgraph = add_expression(graph, idx, task)
            output = earl(task, newgraph.x_dict, newgraph.edge_index_dict)
            predictions[i] = output['protein_value'][mask].flatten()
            p_zeros[i] = output['p_protein_zero'][mask].flatten()
            zero_ones[i] = output['protein_zero'][mask].flatten()
            ys[i] = newgraph['protein_name'].y[mask]

        return predictions, p_zeros, zero_ones, ys

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

batches = {task:list(chunks(idxs, train_batch_size)) for task,idxs in train_cell_idxs.items()}

print('Starting training')
for epoch in range(n_epochs):
    print(f'Epoch: {epoch}', file=log)
    mask = graph['protein_name']['mask']
    
    # TODO MAML here?
    # TODO this is messed up
    # TODO step through batches?
    for batch_idx, batch in enumerate(batches):
        for task,batch in batches.items():
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
            print(f'train zero one loss {task} {batch_zero_one_loss}', file=log) 
            print(f'train value loss {task} {batch_value_loss}',flush=True, file=log)
            print(f'train prediction loss {task} {batch_prediction_loss}',flush=True, file=log)
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
                print(f'validation zero one loss {task} {float(zero_one_loss)}', file=log) 
                print(f'validation value loss {task} {float(value_loss)}',flush=True, file=log)
                print(f'validation prediction loss {task} {validation_loss}',flush=True, file=log)

                stacked = torch.vstack([predictions[0,:]*zero_ones[0,:], ys[0,:]])

                if prediction_log:
                    prediction_log.truncate(0)
                for i in range(stacked.shape[1]):
                    print(f'batch {batch_idx} {i:<6d} pred,y: '+
                          f'{float(stacked[0,i]):>7.3f} {float(stacked[1,i]):.3f}', 
                          file=prediction_log, flush=True)

                if validation_loss < best_validation_loss and log:
                    torch.save(earl.state_dict(), f'models/best_earl_{now}.model')

