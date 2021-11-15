import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import torch_geometric
from torch import tensor
import scanpy
import pickle
from loader import HeteroPathSampler
import random

def proteins_to_idxs(data):
    with torch.no_grad():
        indexes = []
        proteins = data.var.index.to_list()
        for i,protein_name in enumerate(proteins):
            protein_name = protein_name.upper()
            if protein_name in node_idxs['protein_name']:
                indexes.append((i,node_idxs['protein_name'][protein_name]))
        return tensor(indexes,device=device), data.X

def genes_to_idxs(data):
    with torch.no_grad():
        indexes = []
        genes = data.var['gene_ids'].to_list()
        for i,gene_id in enumerate(genes):
            if gene_id in node_idxs['gene']:
                indexes.append((i,node_idxs['gene'][gene_id]))
        return tensor(indexes,device=device), data.X

def atacs_to_idxs(data):
    with torch.no_grad():
        indexes = []
        regions = data.var.index.to_list()
        for i,region in enumerate(regions):
            if region in node_idxs['atac_region']:
                indexes.append((i,node_idxs['atac_region'][region]))
        return tensor(indexes,device=device), data.X

device = 'cuda:3'
log=None

print('Loading graph')
node_idxs = pickle.load(open('input/nodes_by_type.pickle','rb'))
graph = torch.load('input/graph_with_embeddings.torch').to(device)
graph = graph.to('cpu')
graph = torch_geometric.transforms.ToUndirected()(graph)
graph = graph.to(device)

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

with torch.no_grad():
    loader = HeteroPathSampler(graph, device)

    #sampling_factors = [1.2 ,1.3,1.4,1.5,1.6,1.7]
    #sampling_factors = [1.4, 1.6, 1.8, None]
    sampling_factors = [1.2, 1.4]
    tasks = expression.keys()
    #n_targets = 1000
    n_steps = [3,4,5,6]
    tasks = [('atac_region', 'gene')]

    targets= ['atac_region', 'gene', 'protein_name']
    n_totals = {task: len(graph_idxs[task][1]) for task in tasks}
    target_counts = {task: torch.zeros((len(node_idxs[task[1]]),), device=device, dtype=int)
                     for task in tasks}
    q=torch.tensor([.5], device=device)
    threshold=2000

    for task in tasks:        
        i = 0
        while True:
            sampling_factor = random.choice(sampling_factors)
            n_step = random.choice(n_steps)
            n_total = n_totals[task]
            n = min(1000, n_total)
            rnd = torch.randint(n_total, (n,))
            idx = graph_idxs[task][1][rnd,1]
            sampled_graph = loader.sample(task, idx, n_step, sampling_factor)
            total_edges=0
            
            print(i,'task',task,'sampling_factor', sampling_factor, 'n_step',n_step, 'n_targets',n, flush=True)
            for j,layer in enumerate(sampled_graph):
                total_edges += layer.num_edges
                n_sources=0
                n_targets=0
                if task[0] in layer.x_map_dict:
                    n_sources = int(layer[task[0]].x_map.unique().shape[0])
                if task[1] in layer.x_map_dict:
                    n_targets = int(layer[task[1]].x_map.unique().shape[0])
                print('layer',j,'num_sources',n_sources,'num targets',n_targets,flush=True)
                
            if task[1] in layer.x_map_dict:
                target_counts[task][layer[task[1]].x_map.unique()] += 1

            print('total_edges', total_edges)
            #torch.save(sampled_graph, f'{task[0]}_{task[1]}_{i}.graph')
            i+=1
            # If the nth percentile has been hit at least 10 times, then we stop
            cnts = target_counts[task][graph_idxs[task][1][:,1]]
            nth_quantile = torch.quantile(cnts.float(), q)
            nth_percentile = cnts[cnts <= nth_quantile].max()
            nonzero_cnts = int((cnts>0).sum())
            if nonzero_cnts > threshold:
                breakpoint()
            print('nth percentile', int(nth_percentile))
            print('nonzero counts', nonzero_cnts)
            if nth_percentile > 10:
                break
            print('-='*40)
