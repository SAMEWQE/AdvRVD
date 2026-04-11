import numpy as np
import networkx as nx
import scipy.sparse as sp
from sklearn import preprocessing
from base_model import BaseModel


class Spectral(BaseModel):
    r"""The Spectral clustering model from the `"Leveraging social media networks for classiﬁcation"
    <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.481.5392&rep=rep1&type=pdf>`_ paper

    Args:
        hidden_size (int) : The dimension of node representation.
    """

    def __init__(self, hidden_size):
        super(Spectral, self).__init__()
        self.dimension = hidden_size

    def forward(self, graph, return_dict=True):
        # nx_g = graph.to_networkx()
        nx_g = graph
        matrix = nx.normalized_laplacian_matrix(nx_g).todense()
        matrix = np.eye(matrix.shape[0]) - np.asarray(matrix)
        ut, s, _ = sp.linalg.svds(matrix, self.dimension)
        emb_matrix = ut * np.sqrt(s)
        embeddings = preprocessing.normalize(emb_matrix, "l2")

        if return_dict:
            features_matrix = dict()
            for vid, node in enumerate(nx_g.nodes()):
                features_matrix[node] = embeddings[vid]
        else:
            features_matrix = np.zeros((graph.num_nodes, embeddings.shape[1]))
            nx_nodes = nx_g.nodes()
            features_matrix[nx_nodes] = embeddings[np.arange(graph.num_nodes)]
        return features_matrix
