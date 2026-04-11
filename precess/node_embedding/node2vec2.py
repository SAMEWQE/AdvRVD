from cogdl.models.emb.node2vec import Node2vec
from cogdl.data import Graph
import numpy as np
import torch

edge_index = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
edge_index = torch.from_numpy(edge_index)
edge_index = (edge_index[:, 0], edge_index[:, 1])
graph = Graph(edge_index=edge_index)
feat_graph = Graph(edge_index=edge_index, x=torch.randn(8, 16))

node2vec = Node2vec(dimension=64, walk_length=20, walk_num=40, window_size=5, worker=1, iteration=10, p=1, q=1)
emb = node2vec(graph, return_dict=True)

print(emb)
