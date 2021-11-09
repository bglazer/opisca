import torch
import torch_geometric
from torch import tensor
import scanpy
import pickle
from loader import HeteroLoader

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

device = 'cuda:3'

print('Loading graph')
node_idxs = pickle.load(open('input/nodes_by_type.pickle','rb'))
graph = torch.load('input/graph_with_embeddings.torch').to(device)
graph = graph.to('cpu')
graph = torch_geometric.transforms.ToUndirected()(graph)
graph = graph.to(device)

print('Loading protein/gene data')
datadir = 'output/datasets/predict_modality/openproblems_bmmc_cite_phase1_mod2/'
datafile = 'openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_train_mod1.h5ad'
protein_data = scanpy.read_h5ad(datadir+datafile)
protein_idxs, protein_expression = proteins_to_idxs(protein_data)

datafile = 'openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_train_mod2.h5ad'
gene_data = scanpy.read_h5ad(datadir+datafile)
gene_idxs, gene_expression = genes_to_idxs(gene_data)

loader = HeteroLoader(graph, device)

task = ('gene','protein_name')
for idx in protein_idxs[:,0]:
    print('protein idx', idx)
    sampled_graph = loader.sample(task, [idx], 6, random_sample=True)
    print(sampled_graph)
    print('-='*40)
    #torch.save(sampled_graph, f'input/graph_samples/{task[0]}_{task[1]}_{idx}.graph')

#task = ('protein_name','gene')
#for idx in gene_idxs[:,0]:
#    print('gene idx', idx)
#    sampled_graph = loader.sample(task, [idx], 3)
#    print(sampled_graph)
#    print('-='*40)
#    torch.save(sampled_graph, f'input/graph_samples/{task[0]}_{task[1]}_{idx}.graph')
