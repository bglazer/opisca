from torch_geometric.data import HeteroData
import torch_geometric
import torch
from math import log, ceil, sqrt

class HeteroPathSampler():
    def __init__(self, graph, device):
        self.graph = graph
        self.device = device

    def sample(self, task, ids, n_steps, sampling_factor=None):
        layers = self.forward(task, ids, n_steps, sampling_factor)
        layers = self.backward(task, ids, layers)
        return layers

    def forward(self, task, ids, n_steps, sampling_factor):
        with torch.no_grad():
            sources = {task[1]: ids}
            all_nodes = {}
            layers = []
            relations = self.graph._edge_store_dict

            for step in range(n_steps):
                # TODO is this the right place to reinit visited?
                visited = {relation:set() for relation in relations}
                next_sources = {}
                edge_masks = {}
                for source, next_ids in sources.items():
                    for relation, edges in relations.items():
                        next_source, _, dst = relation

                        if step == n_steps - 1:
                            if dst != task[0]:
                                continue

                        if next_source == source:
                            edge_index = edges.edge_index
                            #TODO batching ids 
                            for idxs in next_ids.split(10):
                                #if idx not in visited[relation]:
                                #visited[relation].add(idx)
                                match_mask = (edge_index[0]==idxs.unsqueeze(0).T).sum(axis=0).bool()
                                self.update_mask(edge_masks, relation, match_mask)

                new_graph = HeteroData()
                for relation in edge_masks:
                    mask = edge_masks[relation]
                    if sampling_factor:
                        true_idxs = torch.arange(0,len(mask), device=self.device, dtype=int)
                        true_idxs = true_idxs[mask]
                        n_samples = ceil(sampling_factor**log(len(true_idxs), 2))
                        sample_idxs = torch.randperm(len(true_idxs), device=self.device)[:n_samples]
                        mask = torch.zeros(mask.shape, device=self.device, dtype=bool)
                        mask[true_idxs[sample_idxs]] = True
                        
                    new_graph[relation].edge_index = self.graph[relation].edge_index[:,mask]
                    neighbors = new_graph[relation].edge_index[1]
                    next_source, _, dst = relation
                    self.append_ids(next_sources, dst, neighbors)

                layers.append(new_graph)
                sources = next_sources
                for src in sources:
                    sources[src] = sources[src].unique()


        return layers
        
    # TODO visited
    # TODO associate data
    def backward(self, task, ids, layers):
        source, target = task
        new_layers = []
        step = len(layers)-1
        with torch.no_grad():
            sources = {}
            while step >= 0:
                edge_masks = {}
                layer = layers[step]
                srcs = {}
                dests = {}
                relations = layer._edge_store_dict
                for relation, edges in relations.items():
                    edge_index = edges.edge_index
                    dst, _, src = relation
                    if src == source:
                        full_mask = torch.ones_like(edge_index[0], dtype=bool, device=self.device)
                        self.update_mask(edge_masks, relation, full_mask)
                        self.append_ids(dests, dst, edge_index[0])
                        self.append_ids(srcs, src, edge_index[1])
                    if src in sources:
                        ids = sources[src]
                        for idx in ids:
                            match_mask = (edge_index[1]==idx).flatten()
                            matching_edges = edge_index[:,match_mask]
                            self.update_mask(edge_masks, relation, match_mask)
                            self.append_ids(dests, dst, matching_edges[0])
                            self.append_ids(srcs, src, matching_edges[1])

                new_graph = HeteroData()
                sources = dests
                all_nodes = {}
                
                for node_type in srcs:
                    # NOTE not sure what the unique is doing but don't remove it
                    self.append_ids(all_nodes, node_type, srcs[node_type].unique())

                for node_type in sources:
                    sources[node_type] = sources[node_type].unique()
                    self.append_ids(all_nodes, node_type, sources[node_type])
                    all_nodes[node_type] = all_nodes[node_type].unique()

                for node_type in all_nodes:
                    new_graph[node_type].x = self.graph[node_type].x[all_nodes[node_type]]
                    new_graph[node_type].x_map = all_nodes[node_type]

                for relation in edge_masks:
                    mask = edge_masks[relation]
                    edge_index = relations[relation].edge_index[:,mask]
                    relation, edge_index = self.reverse(relation, edge_index)
                    src,_,dst = relation
                    edge_index[0] = remap(edge_index[0], new_graph[src].x_map)
                    edge_index[1] = remap(edge_index[1], new_graph[dst].x_map)
                    new_graph[relation].edge_index = edge_index

                new_layers.append(new_graph)
                
                step -= 1 

        return new_layers

    def reverse(self, relation, edge_index):
        src, type, dst = relation
        if src != dst:
            if type.startswith('rev'):
                type = type[4:]
            else:
                type = 'rev_' + type

            new_relation = (dst, type, src)
        else:
            new_relation = relation

        # switch rows in-place
        new_edge_index = edge_index.clone()
        new_edge_index = new_edge_index[[1,0]]

        return new_relation, new_edge_index

    def update_mask(self, edge_masks, relation, edges):
        if edges.sum() == 0:
            return 

        if relation not in edge_masks:
            edge_masks[relation] = edges
            return

        edge_masks[relation] += edges

    def append_ids(self, ids, node_type, new):
        if len(new) == 0:
            return

        if node_type not in ids:
            ids[node_type] = torch.empty((0,), device=self.device, dtype=int)
        ids[node_type] = torch.cat([ids[node_type], new])

def remap(keys, ids):
    return torch.bucketize(keys, ids)

