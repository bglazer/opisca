from torch_geometric.data import HeteroData
import torch_geometric
import torch

class HeteroLoader():
    def __init__(self, graph, device):
        self.graph = graph
        self.device = device

    def sample(self, task, ids, n_steps):
        layers = self.forward(task, ids, n_steps)
        layers = self.backward(task, ids, layers)
        return layers

    def forward(self, task, ids, n_steps): #, batch_size=None):
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
                            for idx in next_ids:
                                if idx not in visited[relation]:
                                    visited[relation].add(idx)
                                    match_mask = (edge_index[0]==idx).flatten()
                                    matching_edges = edge_index[:,match_mask]
                                    self.update_mask(edge_masks, relation, match_mask)
                                    self.append_sources(next_sources, dst, matching_edges[1])

                new_graph = HeteroData()
                for relation in edge_masks:
                    mask = edge_masks[relation]
                    new_graph[relation].edge_index = self.graph[relation].edge_index[:,mask]

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
                next_sources = {}
                relations = layer._edge_store_dict
                for relation, edges in relations.items():
                    edge_index = edges.edge_index
                    src, _, dst = relation
                    if dst == source:
                        full_mask = torch.ones_like(edge_index[0], dtype=bool, device=self.device)
                        self.update_mask(edge_masks, relation, full_mask)
                        self.append_sources(next_sources, src, edge_index[0])
                    if dst in sources:
                        ids = sources[dst]
                        for idx in ids:
                            match_mask = (edge_index[1]==idx).flatten()
                            matching_edges = edge_index[:,match_mask]
                            self.update_mask(edge_masks, relation, match_mask)
                            self.append_sources(next_sources, src, matching_edges[0])
                        #breakpoint()

                new_graph = HeteroData()
                for relation in edge_masks:
                    mask = edge_masks[relation]
                    edge_index = relations[relation].edge_index
                    relation, edge_index = self.reverse(relation, edge_index)
                    new_graph[relation].edge_index = edge_index[:,mask]

                new_layers.append(new_graph)
                sources = next_sources

                for src in sources:
                    sources[src] = sources[src].unique()
                
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

    def append_sources(self, next_sources, src, neighbors):
        if len(neighbors) == 0:
            return

        if src not in next_sources:
            next_sources[src] = torch.empty((0,), device=self.device, dtype=int)
        next_sources[src] = torch.cat([next_sources[src], neighbors])
