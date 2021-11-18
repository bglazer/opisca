import sys
import json
from datetime import datetime
from math import sqrt
from collections import defaultdict
import pickle
import random
from collections import Counter
import logging
from pprint import pformat, pprint

import torch
from torch import tensor
from torch.distributions import Bernoulli
from torch.nn import Linear, LeakyReLU, Sigmoid, BCELoss
import torch.nn.functional as F

import torch_geometric
from torch_geometric.nn import GATConv, HeteroConv, SAGEConv, GATv2Conv, TransformerConv
from torch_geometric.data import HeteroData
from torch_geometric.graphgym.models import MLP

from scipy.sparse import csc_matrix
import numpy as np
import scanpy
import anndata as ad

logging.basicConfig(level=logging.INFO)

method_id = "EaRL-joint"

## VIASH START
# Anything within this block will be removed by `viash` and will be
# replaced with the parameters as specified in your config.vsh.yaml.
par = {
    'input_train_mod1': 'output/datasets_phase1v2/predict_modality/openproblems_bmmc_cite_phase1v2_mod2/openproblems_bmmc_cite_phase1v2_mod2.censor_dataset.output_train_mod1.h5ad',
    'input_train_mod2': 'output/datasets_phase1v2/predict_modality/openproblems_bmmc_cite_phase1v2_mod2/openproblems_bmmc_cite_phase1v2_mod2.censor_dataset.output_train_mod2.h5ad',
    'input_test_mod1': 'output/datasets_phase1v2/predict_modality/openproblems_bmmc_cite_phase1v2_mod2/openproblems_bmmc_cite_phase1v2_mod2.censor_dataset.output_test_mod1.h5ad',
    'output': 'output.h5ad',
}

meta = {
    'resources_dir': './',
}
## VIASH END

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
        

def add_expression(graph, cell_idx, task, eval=False):
    source, target = task

    with torch.no_grad():

        if not eval:
            mode='train'
        else:
            mode='test'

        src_expr = tensor(expression[mode][0][cell_idx].todense(),device=device)*128
        src_idxs = graph_idxs[0]
        graph[source].x[src_idxs[:,1],-1] = src_expr[0,src_idxs[:,0]]

        if not eval:
            tgt_idxs = graph_idxs[1]
            tgt_expr = tensor(expression[mode][1][cell_idx].todense(),device=device)
            graph[target].y = torch.ones((len(node_idxs[target]),1), device=device)*-1
            graph[target].y[tgt_idxs[:,1],-1] = tgt_expr[0,tgt_idxs[:,0]]

        for node_type in ['gene','atac_region', 'protein_name']:
            if node_type not in (source, target):
                graph[node_type].x[:,-1] = torch.ones((len(node_idxs[node_type])), device=device)*-1

        return graph


def predict(earl, graph, task, idx, mask, eval=False):
    with torch.inference_mode(eval):
        source,target = task

        newgraph = add_expression(graph, idx, task, eval)
        output = earl(newgraph.x_dict, newgraph.edge_index_dict, task)

        return output


def compute_loss(prediction, y, target, mask):
    y = y[mask]
    y_zero = y > .00001
    # TODO normalize loss so that genes/atacs dont swamp protein gradients?
    p_zeros = prediction['p_zero'][mask]
    
    if target == 'atac_region':
        bce_loss = BCELoss(weight=1+y_zero*params['atac_ones_weight'])
        loss = bce_loss(p_zeros, y_zero.float())
        return loss, loss, 0, loss

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


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
    

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
        x_dict['prediction'] = x_dict['values'] * x_dict['zeros']
        return x_dict

    def atac(self, x_dict, edge_index_dict):
        x_dict = self.encode(x_dict, edge_index_dict)
        x_dict['p_zero'] = self.sigmoid(self.atac_zero(x_dict['atac_region']))
        x_dict['prediction'] = Bernoulli(x_dict['p_zero']).sample()
        return x_dict

    def protein_name(self, x_dict, edge_index_dict):
        x_dict = self.encode(x_dict, edge_index_dict)
        x_dict['p_zero'] = self.sigmoid(self.protein_zero(x_dict['protein_name']))
        x_dict['zeros'] = Bernoulli(x_dict['p_zero']).sample()
        x_dict['values'] = self.protein_value(x_dict['protein_name'])
        x_dict['prediction'] = x_dict['values'] * x_dict['zeros']
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
device='cuda:0'

logging.info('Starting')
logging.info(now)

tmstp = '20211117-2359'
logging.info(f'Using EaRL version: {tmstp}')

params = json.load(open(meta['resources_dir'] + f'earl_params_{tmstp}.json'))

logging.info('EaRL parameters:')
logging.info(pformat(params))

logging.info('Loading graph')
node_idxs = pickle.load(open('input/nodes_by_type.pickle','rb'))
graph = torch.load('input/graph_with_embeddings.torch').to(device)
graph = expand_for_data(graph)

expression = {}
# For each task
# match the index of the data to the index of the graph
graph_idxs = {}
logging.info('Loading protein/gene data')

input_train_mod1 = ad.read_h5ad(par['input_train_mod1'])
input_train_mod2 = ad.read_h5ad(par['input_train_mod2'])
input_test_mod1 = ad.read_h5ad(par['input_test_mod1'])

input_modality = input_train_mod1.var.feature_types.to_list()[0]
output_modality = input_train_mod2.var.feature_types.to_list()[0]

feature_to_task = {'ATAC': 'atac_region',
                   'GEX': 'gene',
                   'ADT': 'protein_name'}

if input_modality == 'ATAC' and output_modality == 'GEX':
    task = ('atac_region','gene')
    input_idxs, input_expression = atacs_to_idxs(input_train_mod1)
    output_idxs, output_expression = genes_to_idxs(input_train_mod2)
    _, test_input_expression = atacs_to_idxs(input_test_mod1)

if input_modality == 'ADT' and output_modality == 'GEX':
    task = ('protein_name','gene')
    input_idxs, input_expression = proteins_to_idxs(input_train_mod1)
    output_idxs, output_expression = genes_to_idxs(input_train_mod2)
    _, test_input_expression = proteins_to_idxs(input_test_mod1)

if input_modality == 'GEX' and output_modality == 'ATAC':
    task = ('gene','atac_region')
    input_idxs, input_expression = genes_to_idxs(input_train_mod1)
    output_idxs, output_expression = atacs_to_idxs(input_train_mod2)
    _, test_input_expression = genes_to_idxs(input_test_mod1)

if input_modality == 'GEX' and output_modality == 'ADT':
    task = ('gene','protein_name')
    input_idxs, input_expression = genes_to_idxs(input_train_mod1)
    output_idxs, output_expression = proteins_to_idxs(input_train_mod2)
    _, test_input_expression = genes_to_idxs(input_test_mod1)

expression['train'] = (input_expression, output_expression)
expression['test'] = (test_input_expression, None)
graph_idxs = (input_idxs, output_idxs)

logging.info('Making graph undirected')
graph = graph.to('cpu')
graph = torch_geometric.transforms.ToUndirected()(graph)
graph = graph.to(device)


train_cell_idxs = list(range(expression['train'][0].shape[0]))
test_cell_idxs = list(range(expression['test'][0].shape[0]))

train_batch_size = params['train_batch_size']

logging.info('Initializing EaRL')
earl = EaRL(gnn_layers=params['layers'], out_mlp=params['out_mlp'])
earl = earl.to(device)

# Dummy batch so that we can initialize the parameters to the correct shape and load the state dict
source,target = task
idx = 0
mask = torch.ones((len(node_idxs[target]),1), dtype=bool, device=device)
predict(earl, graph, task, idx, mask, eval=False)

# TODO make sure this matches the config.vsh.yaml
earl.load_state_dict(torch.load(meta['resources_dir'] + f'latest_earl_{tmstp}.model'), strict=True)
earl = earl.to(device)

n_steps = 100
lr = .00001

optimizer = torch.optim.Adam(params=earl.parameters(), lr=lr)
earl.train()

# Finetuning
#############
logging.info('Starting training')

mode = 'train'
source,target = task
for batch_idx in range(n_steps):
    optimizer.zero_grad()
    batch = random.sample(train_cell_idxs, k=train_batch_size)
    
    mask = torch.zeros((len(node_idxs[target]),1), dtype=bool, device=device)
    mask[graph_idxs[1][:,1]] = 1
    batch_zero_loss = 0.0
    batch_value_loss = 0.0
    batch_prediction_loss = 0.0
    for idx in batch:
        # only one prediction at a time, minimizes memory usage
        prediction = predict(earl, graph, task, idx, mask)
        y = graph[target].y
        losses = compute_loss(prediction, y, target, mask) 
        loss, prediction_loss, value_loss, zero_loss = losses
        loss.backward()

        batch_zero_loss += float(zero_loss)/len(batch)
        batch_value_loss += float(value_loss)/len(batch)
        batch_prediction_loss += float(prediction_loss)/len(batch)

    logging.info(f'Batch={batch_idx}')
    logging.info(f'train zero one loss {task}={batch_zero_loss}') 
    logging.info(f'train value loss {task}={batch_value_loss}')
    logging.info(f'train prediction loss {task}={batch_prediction_loss}')
    optimizer.step()

mode = 'test'
# Evaluation on test set
########################
logging.info('Starting evaluation')
test_batch_size = 100
y_pred = torch.zeros((input_test_mod1.shape[0], input_train_mod2.shape[1]), device=device)
for idx in test_cell_idxs:
    mask = torch.zeros((len(node_idxs[target]),1), dtype=bool, device=device)
    prediction = predict(earl, graph, task, idx, mask, eval=True)
    y_pred[idx,graph_idxs[1][:,0]] = prediction['prediction'][graph_idxs[1][:,1]].T
    if idx % test_batch_size == 0:
        logging.info(f'Done with cell {idx}')


# Store as sparse matrix to be efficient. Note that this might require
# different classifiers/embedders before-hand. Not every class is able
# to support such data structures.
y_pred = csc_matrix(y_pred.cpu().numpy())

adata = ad.AnnData(
    X=y_pred,
    obs=input_test_mod1.obs,
    var=input_train_mod2.var,
    uns={
        'dataset_id': input_train_mod1.uns['dataset_id'],
        'method_id': method_id,
    },
)

logging.info('Storing annotated data...')
adata.write_h5ad(par['output'], compression = "gzip")
