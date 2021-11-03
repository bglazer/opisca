from torch_geometric.data import HeteroData

class HeteroLoader():
    def __init__(self, graph):
        self.graph = graph

    def sample(self, node_type, ids, n_steps): #, batch_size=None):
        with torch.no_grad():
            newgraph = HeteroData()
            sources = {node_type: ids}
            all_nodes = {}
            for step in n_steps:
                next_sources = {}
                for source, ids in sources:
                    for relation, edges in graph._edge_stores_dict.items():
                        source, _, target = relation
                        if source == node_type and relation not in newgraph:
                            edge_index = edges.edge_index
                            for idx in ids:
                                neighbors = edge_index[:,(edge_index[0]==idx).flatten()]
                                newgraph[relation].edge_index = neighbors

                            if target not in all_nodes:
                                all_nodes[target] = neighbors[1].unique()
                            else:
                                all_nodes[target] = torch.cat(all_nodes[target], neighbors[1].unique(), axis=0).unique()

                            if target not in next_sources:
                                next_sources[target] = neighbors[1].unique()
                            else:
                                next_sources[target] = torch.cat(next_sources[target], neighbors[1].unique(), axis=0).unique()
                sources = next_sources

            # TODO load data, make association from subgraph to full graph indexes
            for node_name, nodes in all_nodes.items():
                newgraph[node_name].x = 
                for relation, edges in newgraph._edge_stores_dict.items():
                    newgraph[relation]
