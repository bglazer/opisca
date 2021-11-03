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
from copy import deepcopy

import numpy as np
import scanpy

import torch
from torch import tensor
import torch.nn.functional as F
from torch.nn import Linear, LeakyReLU, Sigmoid, BCELoss
from torch.distributions import Bernoulli

import torch_geometric
from torch_geometric.nn import GATConv, HeteroConv, SAGEConv, GATv2Conv, TransformerConv
from torch_geometric.data import HeteroData
from torch_geometric.graphgym.models import MLP
from torch_geometric.loader import NeighborLoader, HGTLoader


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
        for node_type in ['gene', 'atac_region']:
            graph[node_type].x = torch.cat([
                graph[node_type].x,
                torch.ones((len(node_idxs[node_type]),1),device=device)
            ], dim=1)
        return graph
        

def add_expression(graph, cell_idx, task):
    source, target = task
    with torch.no_grad():
        graph['protein_name'].x = torch.ones((len(node_idxs['protein_name']),1), device=device)*-1
        # 128 is the feature size from the node2vec
        # multiply by 128 so the expression has roughly the same magnitude as 
        # the rest of the features combined
        src_expr = tensor(expression[task][0][cell_idx].todense(),device=device)*128
        tgt_expr = tensor(expression[task][1][cell_idx].todense(),device=device)
        src_idxs = graph_idxs[task][0]
        tgt_idxs = graph_idxs[task][1]

        graph[source].x[src_idxs[:,1],-1] = src_expr[0,src_idxs[:,0]]

        y = torch.ones((len(node_idxs[target]),1), device=device)*-1
        y[tgt_idxs[:,1],-1] = tgt_expr[0,tgt_idxs[:,0]]

        for node_type in ['gene','atac_region']:
            if node_type not in (source, target):
                graph[node_type].x[:,-1] = torch.ones((len(node_idxs[node_type])), device=device)*-1

        return graph, y
    

class EaRL(torch.nn.Module):
    def __init__(self, gnn_layers, out_mlp):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.sigmoid = Sigmoid()
        # This is a little shim that gets around us not being able to json dump 
        # the class of the layer (i.e GATConv) in the param dictionary
        layer_classes = {'GATConv':GATConv, 'SAGEConv':SAGEConv,
                         'TransformerConv':TransformerConv}

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
                ('enhancer','overlaps','atac_region'): layer_type(**params, in_channels=(-1,-1)),
                ('atac_region','rev_overlaps','enhancer'): layer_type(**params, in_channels=(-1,-1)),
                ('enhancer','associated','gene'): layer_type(**params, in_channels=(-1,-1)),
                ('gene','rev_associated','enhancer'): layer_type(**params, in_channels=(-1,-1)),
                ('atac_region','neighbors','gene'): layer_type(**params, in_channels=(-1,-1)),
                ('gene','rev_neighbors','atac_region'): layer_type(**params, in_channels=(-1,-1)),

            })

            self.convs.append(conv)

        self.protein_zero  = MLP(**out_mlp) 
        self.protein_value = MLP(**out_mlp)

        self.gene_value = MLP(**out_mlp) 
        self.gene_zero  = MLP(**out_mlp) 

        self.atac_zero = MLP(**out_mlp) 

    def encode(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        return x_dict

    def gene(self, x_dict, edge_index_dict):
        x_dict = self.encode(x_dict, edge_index_dict)
        x_dict['p_zero'] = self.sigmoid(self.gene_zero(x_dict['gene']))
        x_dict['zeros'] = Bernoulli(x_dict['p_zero']).sample()
        x_dict['values'] = self.gene_value(x_dict['gene'])
        return x_dict

    def atac(self, x_dict, edge_index_dict):
        x_dict = self.encode(x_dict, edge_index_dict)
        x_dict['p_zero'] = self.sigmoid(self.atac_zero(x_dict['atac_region']))
        x_dict['zeros'] = Bernoulli(x_dict['p_zero']).sample()
        return x_dict

    def protein_name(self, x_dict, edge_index_dict):
        x_dict = self.encode(x_dict, edge_index_dict)
        x_dict['p_zero'] = self.sigmoid(self.protein_zero(x_dict['protein_name']))
        x_dict['zeros'] = Bernoulli(x_dict['p_zero']).sample()
        x_dict['values'] = self.protein_value(x_dict['protein_name'])
        return x_dict
    
    def forward(self, x_dict, edge_index_dict, task):
        source, target = task
        if target == 'gene':
            x = self.gene(x_dict, edge_index_dict)
        if target == 'protein_name':
            x = self.protein_name(x_dict, edge_index_dict)
        if target == 'atac_region':
            x = self.atac(x_dict, edge_index_dict)

        return x


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
    'mode':'reptile',
    'sampling':True,
    'inner_lr': .01,
    'outer_lr': .1,
    'inner_steps': 5,
    'n_steps': 10000,
    'layers':[('GATConv', {'out_channels':32, 'heads':2, 'concat':False}),
              ('GATConv', {'out_channels':32, 'heads':2, 'concat':False}),
              ('GATConv', {'out_channels':32, 'heads':2, 'concat':False}),
              ('GATConv', {'out_channels':32, 'heads':2, 'concat':False})],
              #('SAGEConv', {'out_channels':64}),
              #('SAGEConv', {'out_channels':128})],
    'out_mlp':{'dim_in':32, 'dim_out':1, 'bias':True, 
               'dim_inner': 512, 'num_layers':3},
    'train_batch_cells': 10,
    'validation_batch_cells': 10,
    'checkpoint': 40,
    'atac_ones_weight': 1,
    'gene_ones_weight': 1,
    'target_heads': False,
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

def get_mask(graph, task):
    for i,node_type in enumerate(task):
        graph[node_type]['mask'] = mask
    return graph


print('Initializing EaRL', file=log)
earl = EaRL(gnn_layers=params['layers'], out_mlp=params['out_mlp'])
earl = earl.to(device)


earl.train()

n_steps = params['n_steps']
checkpoint = params['checkpoint']

# TODO trim the gene names, we have way more gene names than we have in the data

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

train_batch_cells = params['train_batch_cells']
validation_batch_cells = params['validation_batch_cells']

best_validation_loss = float('inf')

def predict(earl, graph, task, eval=False):
    context = torch.no_grad() if eval else torch.enable_grad()
    with context:
        output = earl(graph.x_dict, graph.edge_index_dict, task)

        return output
        

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def compute_loss(prediction, y, target, mask):
    y_zero = y > .00001
    # TODO normalize loss so that genes/atacs dont swamp protein gradients?
    p_zeros = prediction['p_zero'][mask]
    
    if target == 'atac_region':
        bce_loss = BCELoss(weight=1+y_zero*params['atac_ones_weight'])
        loss = bce_loss(p_zeros, y_zero.float())
        zeros = prediction['zeros'][mask]
        prediction_loss = float(((zeros - y)**2).mean())
        return loss, prediction_loss, 0, loss

    if target in ['gene', 'protein_name']:
        if target=='gene':
            bce_loss = BCELoss(weight=1+y_zero*params['gene_ones_weight'])
        else:
            bce_loss = BCELoss()
        values = prediction['values'][mask]
        zeros = prediction['zeros'][mask]
        zero_loss = bce_loss(p_zeros, y_zero.float())
        value_loss = ((values[y_zero] - y[y_zero])**2).mean()
        prediction_loss = float(((values*zeros - y)**2).mean())
        loss = zero_loss + value_loss
        return loss, prediction_loss, value_loss, zero_loss



def inner_loop(earl, task, batch, batch_type, verbose=True):
    source,target = task

    mask = torch.zeros((len(node_idxs[target]),1), dtype=bool, device=device)
    #mask[graph_idxs[task][1][:,1]] = 1
    mask[2] = 1 

    inner_optimizer = torch.optim.Adam(params=earl.parameters(), lr=params['inner_lr'])
    inner_optimizer.zero_grad()

    batch_zero_loss = 0.0
    batch_value_loss = 0.0
    batch_prediction_loss = 0.0

    for cell_idx in batch:
        newgraph, y = add_expression(graph, cell_idx, task)
        breakpoint()
        # TODO need to find masked items that aren't connected to anything
        train_loader = NeighborLoader(newgraph,
                                      #num_samples=[10]*2,
                                      num_neighbors=[1],
                                      batch_size=1,
                                      input_nodes=(target, mask.flatten())
                                     )

        for minibatch in train_loader:
            # only one prediction at a time, minimizes memory usage
            prediction = predict(earl, newgraph, task)
            losses = compute_loss(prediction, y[mask], target, mask) 
            loss, prediction_loss, value_loss, zero_loss = losses
            loss.backward()

            batch_zero_loss += float(zero_loss)/len(batch)
            batch_value_loss += float(value_loss)/len(batch)
            batch_prediction_loss += float(prediction_loss)/len(batch)

    inner_optimizer.step()
    if verbose:
        print(f'{batch_type} zero one loss {task}={batch_zero_loss}', file=log) 
        print(f'{batch_type} value loss {task}={batch_value_loss}',flush=True, file=log)
        print(f'{batch_type} prediction loss {task}={batch_prediction_loss}',flush=True, file=log)



tasks = list(train_cell_idxs.keys())
task_iter = iter(tasks)
outer_optimizer = torch.optim.Adam(params=earl.parameters(), lr=params['outer_lr'])

print('Starting training', file=log)
outer_lr = params['outer_lr']

# dummy batch to initialize parameters
task = tasks[0]
source,target = task
idx = 0
mask = torch.ones((len(node_idxs[target]),1), dtype=bool, device=device)
add_expression(graph, idx, task)
predict(earl, graph, task, eval=True)


#for iteration 1,2,3,…do
for step_idx in range(n_steps):
    # ParallelDataLoader here?
    # some data loader from learn2learn?
    #  Randomly sample a task T
    task = next(task_iter, None)
    if task is None:
        task_iter = iter(random.sample(tasks, k=len(tasks)))
        task = next(task_iter, None)
    source, target = task

    weights_before = deepcopy(earl.state_dict())
    # Inner updates
    print(f'Outer step={step_idx}', file=log)
    for inner_step in range(params['inner_steps']-1):
        print(f'Inner step={inner_step}', file=log, flush=True)
        batch = random.sample(train_cell_idxs[task], k=train_batch_cells)
        inner_loop(earl, task, batch, 'train', verbose=False)
    batch = random.sample(train_cell_idxs[task], k=train_batch_cells)
    inner_loop(earl, task, batch, 'train', verbose=True)

    weights_after = earl.state_dict()
    # TODO evaluation batch after inner loops?

    # Outer update
    # Update: Φ←Φ+ϵ(W−Φ)
    # TODO explicitly set grad to diff, pass to ADAM
    new_state = {}
    for name in weights_before:
        before = weights_before[name]
        after = weights_after[name]
        diff = after-before
        # Don't apply outer loop to task specific heads
        if params['target_heads'] and f'{target}_value' in name or f'{target}_zero' in name:
            new_state[name] = after
            # TODO for ADAM just set requires_grad = False
        else:
            updated = before + diff * outer_lr
            new_state[name] = updated
    earl.load_state_dict(new_state)
    # TODO learning rate decay
    
    # Checkpoint
    if (step_idx+1) % checkpoint == 0:
        # Save state before validation training
        weights_before = deepcopy(earl.state_dict())
        print('Checkpoint', file=log)
        #earl.eval()
        if prediction_log:
            prediction_log.truncate(0)
        for task in tasks:
            source,target = task
            total_validation_loss = 0.0

            # Inner updates
            for inner_step in range(params['inner_steps']-1):
                batch = random.sample(validation_cell_idxs[task], k=validation_batch_cells)
                weights_after = inner_loop(earl, task, batch, 'validation', verbose=False)
            batch = random.sample(validation_cell_idxs[task], k=validation_batch_cells)

            #idx = random.sample(validation_cell_idxs[task], k=1)
            #mask = torch.zeros((len(node_idxs[target]),1), dtype=bool, device=device)
            #mask[graph_idxs[task][1][:,1]] = 1
            #earl.eval()
            #prediction = predict(earl, graph, task, eval=True)
            #y=y[mask]

            ## Print a sample of predictions
            #if target == 'atac_region':
            #    zeros = prediction['zeros'][mask]
            #    p_zeros = prediction['p_zero'][mask]
            #    stacked = torch.vstack([p_zeros, y])

            #if target in ['gene', 'protein_name']:
            #    zeros = prediction['zeros'][mask]
            #    values = prediction['values'][mask]
            #    stacked = torch.vstack([values*zeros, y])

            #print('-'*80, file=prediction_log)
            #print(f'Task: {task}', file=prediction_log)
            #print('-'*80, file=prediction_log)
            #for i in range(min(stacked.shape[1], 300)):
            #    print(f'batch {step_idx} {i:<6d} pred,y: '+
            #          f'{float(stacked[0,i]):>7.3f} {float(stacked[1,i]):.3f}', 
            #          file=prediction_log, flush=True)

            # Reset weights to initial state before tuning (inner loops) on this task
            earl.load_state_dict(weights_before)

        print(f'Total validation loss={total_validation_loss}')
        if total_validation_loss < best_validation_loss and log:
            torch.save(earl.state_dict(), f'models/best_earl_{now}.model')
            best_validation_loss = total_validation_loss

