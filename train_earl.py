#!/usr/bin/env python
# coding: utf-8
import sys
import json
from datetime import datetime
from math import sqrt
from collections import defaultdict
import pickle
import random
from collections import Counter
from statistics import mean

import scanpy
import numpy as np

import torch_geometric
from torch_geometric.nn import GATConv, HeteroConv, SAGEConv, GATv2Conv, TransformerConv
from torch_geometric.data import HeteroData
#from torch_geometric.graphgym.models import MLP

import torch
from torch import tensor
from torch.distributions import Bernoulli
from torch.nn import Linear, LeakyReLU, Sigmoid, BCELoss, Sequential as Seq, ReLU
import torch.nn.functional as F

from loader import HeteroPathSampler, remap

def MLP(channels):
    hidden = Seq(*[Seq(
                    Linear(channels[i - 1], channels[i]), 
                    ReLU())
                   for i in range(1, len(channels)-1)])
    final = Linear(channels[len(channels)-2], channels[len(channels)-1])
    return Seq(hidden, final)

# protein_idxs is a (n_proteins x 2) 
# The first column is an index to the data
# the second index is an index to the graph
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
    indexes = []
    regions = data.var.index.to_list()
    for i,region in enumerate(regions):
        if region in node_idxs['atac_region']:
            indexes.append((i,node_idxs['atac_region'][region]))
    return tensor(indexes,device=device), data.X

def expand_for_data(graph):
    with torch.no_grad():
        for node_type in ['gene', 'atac_region', 'protein_name']:
            graph[node_type].x = torch.cat([
                graph[node_type].x,
                torch.ones((len(node_idxs[node_type]),1),device=device)
            ], dim=1)
        return graph
        

def add_expression(graph, cell_idx, task):
    source, target = task
    with torch.no_grad():
        # 128 is the feature size from the node2vec
        # multiply by 128 so the expression has roughly the same magnitude as 
        # the rest of the features combined
        src_expr = tensor(expression[task][0][cell_idx].todense(),device=device)*128
        tgt_expr = tensor(expression[task][1][cell_idx].todense(),device=device)
        src_idxs = graph_idxs[task][0]
        tgt_idxs = graph_idxs[task][1]

        # src_idxs[:,1] is the index of the nodes in the graph corresponding to the data
        # src_idxs[:,0] is the index of the source data
        # same for target
        graph[source].x[src_idxs[:,1],-1] = src_expr[0,src_idxs[:,0]]

        graph[target].y = torch.ones((len(node_idxs[target]),1), device=device)*-1
        graph[target].y[tgt_idxs[:,1],-1] = tgt_expr[0,tgt_idxs[:,0]]

        for node_type in ['gene','atac_region','protein_name']:
            if node_type not in (source, target):
                graph[node_type].x[:,-1] = torch.ones((len(node_idxs[node_type])), device=device)*-1

        return graph
    

class EaRL(torch.nn.Module):
    def __init__(self, gnn_params, out_mlp, layer_mlp):
        super().__init__()

        self.sigmoid = Sigmoid()
        # This is a little shim that gets around us not being able to json dump 
        # the class of the layer (i.e GATConv) in the param dictionary
        conv_classes = {'GATConv':GATConv, 'SAGEConv':SAGEConv,
                         'TransformerConv':TransformerConv}

        conv_name, conv_params = gnn_params
        conv = conv_classes[conv_name]
        self.convs = {
            ('tad', 'overlaps', 'atac_region'): conv(**conv_params, in_channels=(-1,-1)),
            ('atac_region', 'rev_overlaps', 'tad'): conv(**conv_params, in_channels=(-1,-1)),
            ('tad', 'overlaps', 'gene'): conv(**conv_params, in_channels=(-1,-1)),
            ('gene', 'rev_overlaps', 'tad'): conv(**conv_params, in_channels=(-1,-1)),
            ('atac_region', 'overlaps', 'gene'): conv(**conv_params, in_channels=(-1,-1)),
            ('gene', 'rev_overlaps', 'atac_region'): conv(**conv_params, in_channels=(-1,-1)),
            # Not bipartite so we have just (-1) for in channels, same feature sizes for both
            ('protein', 'coexpressed', 'protein'): conv(**conv_params, in_channels=-1),
            ('protein', 'tf_interacts', 'gene'): conv(**conv_params, in_channels=(-1,-1)),
            ('gene', 'rev_tf_interacts', 'protein'): conv(**conv_params, in_channels=(-1,-1)),
            ('protein', 'rev_associated', 'gene'): conv(**conv_params, in_channels=(-1,-1)),
            ('gene', 'associated', 'protein'): conv(**conv_params, in_channels=(-1,-1)),
            ('protein', 'is_named', 'protein_name'): conv(**conv_params, in_channels=(-1,-1)),
            ('protein_name', 'rev_is_named', 'protein'): conv(**conv_params, in_channels=(-1,-1)),
            ('enhancer','overlaps','atac_region'): conv(**conv_params, in_channels=(-1,-1)),
            ('atac_region','rev_overlaps','enhancer'): conv(**conv_params, in_channels=(-1,-1)),
            ('enhancer','associated','gene'): conv(**conv_params, in_channels=(-1,-1)),
            ('gene','rev_associated','enhancer'): conv(**conv_params, in_channels=(-1,-1)),
            ('atac_region','neighbors','gene'): conv(**conv_params, in_channels=(-1,-1)),
            ('gene','rev_neighbors','atac_region'): conv(**conv_params, in_channels=(-1,-1)),
        }

        # TODO input dimensionality needs to be a parameter or inferred from data
        # TODO it's hardcoded right now to be +1 of the conv output dimensionality, which is likely to break
        # TODO don't need to change out_channels, that should be the same as the input to the convs
        self.input_linear = {
            'protein_name': Linear(129, conv_params['out_channels']),
            'atac_region': Linear(257, conv_params['out_channels']),
            'gene': Linear(129, conv_params['out_channels']),
        }

        self.protein_dropout  = MLP(out_mlp)
        self.protein_value = MLP(out_mlp)

        self.gene_value = MLP(out_mlp) 
        self.gene_dropout  = MLP(out_mlp) 

        self.atac_dropout = MLP(out_mlp) 

        self.node_types = ['atac_region', 'gene', 'tad', 'protein', 'protein_name', 'enhancer']
        self.mlp = {node_type:MLP(layer_mlp) for node_type in self.node_types} 

    def to(self, device):
        for k,conv in self.convs.items():
            self.convs[k] = conv.to(device)
        for k,layer in self.input_linear.items():
            self.input_linear[k] = layer.to(device)
        self.protein_dropout = self.protein_dropout.to(device)
        self.protein_value = self.protein_value.to(device)
        self.gene_value = self.gene_value.to(device)
        self.gene_dropout = self.gene_dropout.to(device)
        self.atac_dropout = self.atac_dropout.to(device)
        for task, mlp in self.mlp.items():
            self.mlp[task] = mlp.to(device)

    def encode(self, layers):
        for layer in layers:
            for data_type, data in layer.x_dict.items():
                if data_type in self.input_linear:
                    layer[data_type].x = self.input_linear[data_type](layer[data_type].x)
            
        layer_idx = 0
        while layer_idx < len(layers)-1:
            layer = layers[layer_idx]
            subconv = HeteroConv({relation: conv for relation,conv in self.convs.items()
                                  if relation in layer._edge_store_dict})

            new_x = subconv(layer.x_dict, layer.edge_index_dict)
            
            new_x = {dst: self.mlp[dst](x) for dst,x in new_x.items()}

            next_layer = layers[layer_idx+1]
            for relation in layer._edge_store_dict:
                src,_,dst = relation
                dst_mask = (layer[dst].x_map.unsqueeze(0).T == next_layer[dst].x_map).sum(dim=0).bool()
                src_mask = (next_layer[dst].x_map.unsqueeze(0).T == layer[dst].x_map).sum(dim=0).bool()
                next_layer[dst].x[dst_mask] = new_x[dst][src_mask]
            layer_idx+=1

        layer = layers[-1]
        subconv = HeteroConv({relation: conv for relation,conv in self.convs.items()
                              if relation in layer._edge_store_dict})

        new_x = subconv(layer.x_dict, layer.edge_index_dict)

        return new_x, layer[target].x_map

    def gene(self, x_dict):
        out = dict()
        out['p_dropout'] = self.sigmoid(self.gene_dropout(x_dict['gene']))
        out['dropouts'] = Bernoulli(out['p_dropout']).sample()
        out['values'] = self.gene_value(x_dict['gene'])
        return out

    def atac(self, x_dict):
        out = dict()
        out['p_dropout'] = self.sigmoid(self.atac_dropout(x_dict['atac_region']))
        return out

    def protein_name(self, x_dict):
        out = dict()
        out['p_dropout'] = self.sigmoid(self.protein_dropout(x_dict['protein_name']))
        out['dropouts'] = Bernoulli(out['p_dropout']).sample()
        out['values'] = self.protein_value(x_dict['protein_name'])
        return out
    
    def forward(self, layers, task):
        source, target = task
        x_dict, x_map = self.encode(layers)
        if target == 'gene':
            x = self.gene(x_dict)
        if target == 'protein_name':
            x = self.protein_name(x_dict)
        if target == 'atac_region':
            x = self.atac(x_dict)

        return x, x_map


now = datetime.strftime(datetime.now(), format='%Y%m%d-%H%M')

# Device is first command line arg
device=sys.argv[1]
# Provide anything as a second command line argument after device to log to stdout
if len(sys.argv) == 3:
    log = None
    prediction_log = None
else:
    log = open(f'logs/train_earl_{now}.log','w')
    prediction_log = open(f'logs/train_earl_prediction_sample_{now}.log','w')

print('Starting', file=log)
print(now, file=log)

params = {
    'lr':.001,
    'n_steps':15000,
    'gnn_params':('TransformerConv', {'out_channels':128, 'heads':3}),
    'out_mlp': [128*3, 256, 1],
    'layer_mlp': [128*3, 256, 128],
    'checkpoint': 25,
    'atac_ones_weight': 1,
    'gene_ones_weight': 1,
    'sample_depth': [3,4,5,6],
    'sampling_factors': [1.4, 1.5, 1.6],
    'cells_per_batch': 50,
    'targets_per_sample': 1000,
    'device': device,
}

if log:
    with open(f'logs/earl_params_{now}.json','w') as param_file:
        json.dump(params, param_file)

print('Loading graph', file=log)
node_idxs = pickle.load(open('input/nodes_by_type.pickle','rb'))
graph = torch.load('input/graph_with_embeddings.torch').to(device)
graph = expand_for_data(graph)

expression = {}
# For each task
# match the index of the data to the index of the graph
graph_idxs = {}
print('Loading protein/gene data', file=log)
datadir = 'output/datasets/predict_modality/openproblems_bmmc_cite_phase1_mod2/'
datafile = 'openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_train_mod1.h5ad'
protein_data = scanpy.read_h5ad(datadir+datafile)
protein_idxs, protein_expression = proteins_to_idxs(protein_data)

datafile = 'openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_train_mod2.h5ad'
gene_data = scanpy.read_h5ad(datadir+datafile)
gene_idxs, gene_expression = genes_to_idxs(gene_data)

expression[('protein_name','gene')] = (protein_expression,gene_expression)
graph_idxs[('protein_name','gene')] = (protein_idxs,gene_idxs)
expression[('gene','protein_name')] = (gene_expression,protein_expression)
graph_idxs[('gene','protein_name')] = (gene_idxs,protein_idxs)

print('Loading atac/gene data', file=log)
datadir = 'output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_mod2/'
datafile = 'openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_train_mod1.h5ad'
atac_data = scanpy.read_h5ad(datadir+datafile)
atac_idxs, atac_expression = atacs_to_idxs(atac_data)

datafile = 'openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_train_mod2.h5ad'
gene_data = scanpy.read_h5ad(datadir+datafile)
gene_idxs, gene_expression = genes_to_idxs(gene_data)

expression[('atac_region','gene')] = (atac_expression,gene_expression)
graph_idxs[('atac_region','gene')] = (atac_idxs,gene_idxs)
expression[('gene','atac_region')] = (gene_expression,atac_expression)
graph_idxs[('gene','atac_region')] = (gene_idxs,atac_idxs)

print('Making graph undirected', file=log)
graph = graph.to('cpu')
graph = torch_geometric.transforms.ToUndirected()(graph)
graph = graph.to(device)

print('Initializing EaRL', file=log)
earl = EaRL(gnn_params=params['gnn_params'], 
            out_mlp=params['out_mlp'],
            layer_mlp=params['layer_mlp'])

earl.to(device)

optimizer = torch.optim.Adam(params=earl.parameters(), lr=params['lr'])
earl.train()

n_steps = params['n_steps']
checkpoint = params['checkpoint']

train_cell_idxs = {}
validation_cell_idxs = {}
for task in expression:
    num_cells = expression[task][0].shape[0]
    train_set_size = int(num_cells*.7)
    val_set_size = num_cells - train_set_size
    cell_idxs = list(range(num_cells))
    random.shuffle(cell_idxs)
    train_cell_idxs[task] = cell_idxs[:train_set_size]
    validation_cell_idxs[task] = cell_idxs[train_set_size:]


bce_loss = BCELoss()

best_validation_loss = float('inf')

loader = HeteroPathSampler(graph, device)

def predict(earl, graph, task, cell_idxs, target_idxs, eval=False):
    source,target = task
    with torch.inference_mode(eval):
        predictions = []

        for cell_idx in cell_idxs:
            newgraph = add_expression(graph, cell_idx, task)
            # sample a subgraph
            sample_depth = random.choice(params['sample_depth'])
            sampling_factor = random.choice(params['sampling_factors'])
            #print(f'sample_depth={sample_depth}', file=log)
            #print(f'sampling_factor={sampling_factor}', file=log)
            subgraph = loader.sample(task, 
                                     target_idxs, 
                                     n_steps=sample_depth, 
                                     sampling_factor=sampling_factor)
            if subgraph[0].num_edges == 0:
                print(f'Empty graph={cell_idx}')
                continue
            if task[1] in subgraph[-1].x_map_dict:
                print(f'N targets predicted={len(subgraph[-1][task[1]].x_map.unique())}', file=log)

            output, x_map = earl(subgraph, task)
            predictions.append((output, graph[target].y[x_map], x_map))

        return predictions

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def compute_loss(prediction, y, target):
    y_one = y > .00001
    # TODO normalize loss so that genes/atacs dont swamp protein gradients?
    p_dropout = prediction['p_dropout']
    
    if target == 'atac_region':
        bce_loss = BCELoss(weight=1+y_one*params['atac_ones_weight'])
        loss = bce_loss(p_dropout, y_one.float())
        return loss, loss, 0, loss

    if target in ['gene', 'protein_name']:
        if target=='gene':
            bce_loss = BCELoss(weight=1+y_one*params['gene_ones_weight'])
        else:
            bce_loss = BCELoss()
        values = prediction['values']
        dropouts = prediction['dropouts']
        dropout_loss = bce_loss(p_dropout, y_one.float()).mean()
        if len(y[y_one]) > 0:
            value_loss = ((values[y_one] - y[y_one])**2).mean()
            loss = dropout_loss + value_loss
        else:
            value_loss = 0.0
            loss = dropout_loss
        prediction_loss = float(((values*dropouts - y)**2).mean())
        return loss, prediction_loss, value_loss, dropout_loss

tasks = list(train_cell_idxs.keys())

print('Starting training', file=log)
for batch_idx in range(n_steps):
    #task = random.choice(tasks)
    optimizer.zero_grad()
    for task in tasks:
        cell_batch = random.sample(train_cell_idxs[task], k=params['cells_per_batch'])
        # grabs a sample of the indexes of the nodes that have data
        n_max = len(graph_idxs[task][1])
        n = min(params['targets_per_sample'], n_max)
        rnd = torch.randint(len(graph_idxs[task][1]), (n,))
        target_batch = graph_idxs[task][1][rnd,1]
        
        source,target = task
        batch_dropout_loss = []
        batch_value_loss = []
        batch_prediction_loss = []
        predictions = predict(earl, graph, task, cell_idxs=cell_batch, target_idxs=target_batch)
        for prediction in predictions:
            prediction, y, prediction_idxs = prediction
            losses = compute_loss(prediction, y, target) 
            loss, prediction_loss, value_loss, dropout_loss = losses
            loss.backward()

            batch_dropout_loss.append(float(dropout_loss))
            batch_value_loss.append(float(value_loss))
            batch_prediction_loss.append(float(prediction_loss))

        print(f'Batch={batch_idx}', file=log)
        print(f'train dropout one loss {task}={mean(batch_dropout_loss)}', file=log) 
        print(f'train value loss {task}={mean(batch_value_loss)}',flush=True, file=log)
        print(f'train prediction loss {task}={mean(batch_prediction_loss)}',flush=True, file=log)
    optimizer.step()

    # Checkpoint
    if (batch_idx) % checkpoint == 0:
        print('Checkpoint', file=log)
        earl.eval()
        if prediction_log:
            prediction_log.truncate(0)

        for task in tasks:
            source,target = task
            total_validation_loss = 0.0

            cell_batch = random.sample(validation_cell_idxs[task], k=params['cells_per_batch'])
            rnd = torch.randint(len(graph_idxs[task][1]), (params['targets_per_sample'],))
            target_batch = graph_idxs[task][1][rnd,1]

            validation_loss = []
            val_dropout_loss = []
            val_prediction_loss = []
            val_value_loss = []
            predictions = predict(earl, graph, task, cell_idxs=cell_batch, target_idxs=target_batch)
            print('-'*80, file=prediction_log)
            print(f'Task: {task}', file=prediction_log)
            print('-'*80, file=prediction_log)
            for prediction in predictions:
                prediction, y, prediction_idxs = prediction
                losses = compute_loss(prediction, y, target) 
                loss, prediction_loss, value_loss, dropout_loss = losses

                val_dropout_loss.append(float(dropout_loss))
                val_value_loss.append(float(value_loss))
                val_prediction_loss.append(float(prediction_loss))
                validation_loss.append(float(loss))

            print(f'Batch={batch_idx}', file=log)
            print(f'validation dropout one loss {task}={mean(val_dropout_loss)}', file=log) 
            print(f'validation value loss {task}={mean(val_value_loss)}',flush=True, file=log)
            print(f'validation prediction loss {task}={mean(val_prediction_loss)}',flush=True, file=log)
            if target == 'atac_region':
                p_dropout = prediction['p_dropout']
                #dropouts = prediction['dropouts']
                stacked = torch.hstack([p_dropout, y])
                for i in range(min(stacked.shape[0], 100)):
                    print(f'batch {batch_idx} {i:<6d} pred,y: '+
                          f'{float(stacked[i,0]):>7.3f} {float(stacked[i,1]):.0f}', 
                          file=prediction_log)
                print('', flush=True)

            if target in ['gene', 'protein_name']:
                dropouts = prediction['dropouts']
                values = prediction['values']
                stacked = torch.hstack([values*dropouts, y])

                for i in range(min(stacked.shape[0], 100)):
                    print(f'batch {batch_idx} {i:<6d} pred,y: '+
                          f'{float(stacked[i,0]):>7.3f} {float(stacked[i,1]):.3f}', 
                          file=prediction_log)
                print('', flush=True)

            total_validation_loss += mean(validation_loss)


        if log:
            torch.save(earl.state_dict(), f'models/latest_earl_{now}.model')
        print(f'total_validation_loss={total_validation_loss}',file=log)
        print(f'best_validation_loss={best_validation_loss}',file=log)
        if total_validation_loss < best_validation_loss and log:
            torch.save(earl.state_dict(), f'models/best_earl_{now}.model')
            best_validation_loss = total_validation_loss

