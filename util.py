# Copied from the newest version of pytorch_geometric because the old 
# version didn't support adding node_types to the homogeneous data object


from typing import Union, Tuple, List, Dict, Any, Optional, NamedTuple
from torch_geometric.typing import NodeType, EdgeType, QueryType
from torch_geometric.data.data import BaseData, Data, size_repr
from torch_geometric.data.storage import (BaseStorage, NodeStorage,
                                          EdgeStorage)

import copy
import re
from itertools import chain
from collections.abc import Mapping
from collections import namedtuple, defaultdict

import torch
from torch import Tensor
from torch_sparse import SparseTensor

def to_homogeneous(self, node_attrs: Optional[List[str]] = None,
                       edge_attrs: Optional[List[str]] = None,
                       add_node_type: bool = True,
                       add_edge_type: bool = True) -> Data:
        """Converts a :class:`~torch_geometric.data.HeteroData` object to a
        homogeneous :class:`~torch_geometric.data.Data` object.
        By default, all features with same feature dimensionality across
        different types will be merged into a single representation, unless
        otherwise specified via the :obj:`node_attrs` and :obj:`edge_attrs`
        arguments.
        Furthermore, attributes named :obj:`node_type` and :obj:`edge_type`
        will be added to the returned :class:`~torch_geometric.data.Data`
        object, denoting node-level and edge-level vectors holding the
        node and edge type as integers, respectively.

        Args:
            node_attrs (List[str], optional): The node features to combine
                across all node types. These node features need to be of the
                same feature dimensionality. If set to :obj:`None`, will
                automatically determine which node features to combine.
                (default: :obj:`None`)
            edge_attrs (List[str], optional): The edge features to combine
                across all edge types. These edge features need to be of the
                same feature dimensionality. If set to :obj:`None`, will
                automatically determine which edge features to combine.
                (default: :obj:`None`)
            add_node_type (bool, optional): If set to :obj:`False`, will not
                add the node-level vector :obj:`node_type` to the returned
                :class:`~torch_geometric.data.Data` object.
                (default: :obj:`True`)
            add_edge_type (bool, optional): If set to :obj:`False`, will not
                add the edge-level vector :obj:`edge_type` to the returned
                :class:`~torch_geometric.data.Data` object.
                (default: :obj:`True`)
        """
        def _consistent_size(stores: List[BaseStorage]) -> List[str]:
            sizes_dict = defaultdict(list)
            for store in stores:
                for key, value in store.items():
                    if key in ['edge_index', 'adj_t']:
                        continue
                    if isinstance(value, Tensor):
                        dim = self.__cat_dim__(key, value, store)
                        size = value.size()[:dim] + value.size()[dim + 1:]
                        sizes_dict[key].append(tuple(size))
            return [
                k for k, sizes in sizes_dict.items()
                if len(sizes) == len(stores) and len(set(sizes)) == 1
            ]

        data = Data(**self._global_store.to_dict())

        # Iterate over all node stores and record the slice information:
        node_slices, cumsum = {}, 0
        node_type_names, node_types = [], []
        for i, (node_type, store) in enumerate(self._node_store_dict.items()):
            num_nodes = store.num_nodes
            node_slices[node_type] = (cumsum, cumsum + num_nodes)
            node_type_names.append(node_type)
            cumsum += num_nodes

            if add_node_type:
                kwargs = {'dtype': torch.long}
                node_types.append(torch.full((num_nodes, ), i, **kwargs))
        data._node_type_names = node_type_names

        if len(node_types) > 1:
            data.node_type = torch.cat(node_types, dim=0)
        elif len(node_types) == 1:
            data.node_type = node_types[0]

        # Combine node attributes into a single tensor:
        if node_attrs is None:
            node_attrs = _consistent_size(self.node_stores)
        for key in node_attrs:
            values = [store[key] for store in self.node_stores]
            dim = self.__cat_dim__(key, values[0], self.node_stores[0])
            value = torch.cat(values, dim) if len(values) > 1 else values[0]
            data[key] = value

        if len([
                key for key in node_attrs
                if (key in {'x', 'pos', 'batch'} or 'node' in key)
        ]) == 0 and not add_node_type:
            data.num_nodes = cumsum

        # Iterate over all edge stores and record the slice information:
        edge_slices, cumsum = {}, 0
        edge_indices, edge_type_names, edge_types = [], [], []
        for i, (edge_type, store) in enumerate(self._edge_store_dict.items()):
            src, _, dst = edge_type
            num_edges = store.num_edges
            edge_slices[edge_type] = (cumsum, cumsum + num_edges)
            edge_type_names.append(edge_type)
            cumsum += num_edges

            kwargs = {'dtype': torch.long, 'device': store.edge_index.device}
            offset = [[node_slices[src][0]], [node_slices[dst][0]]]
            offset = torch.tensor(offset, **kwargs)
            edge_indices.append(store.edge_index + offset)
            if add_edge_type:
                edge_types.append(torch.full((num_edges, ), i, **kwargs))
        data._edge_type_names = edge_type_names

        if len(edge_indices) > 1:
            data.edge_index = torch.cat(edge_indices, dim=-1)
        elif len(edge_indices) == 1:
            data.edge_index = edge_indices[0]

        if len(edge_types) > 1:
            data.edge_type = torch.cat(edge_types, dim=0)
        elif len(edge_types) == 1:
            data.edge_type = edge_types[0]

        # Combine edge attributes into a single tensor:
        if edge_attrs is None:
            edge_attrs = _consistent_size(self.edge_stores)
        for key in edge_attrs:
            values = [store[key] for store in self.edge_stores]
            dim = self.__cat_dim__(key, values[0], self.edge_stores[0])
            value = torch.cat(values, dim) if len(values) > 1 else values[0]
            data[key] = value

        return data
