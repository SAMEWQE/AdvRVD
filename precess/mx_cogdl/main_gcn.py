import dgl
import networkx as nx
import sent2vec
import warnings

import torch
from cogdl.datasets import generate_random_graph

warnings.filterwarnings(action="ignore")
import numpy as np

from gcn2 import GCN
from cogdl.data import Graph

from emb.node2vec import Node2vec
from emb.netmf import NetMF
from emb.netsmf import NetSMF
from emb.prone import ProNE
from emb.line import LINE
from emb.deepwalk import DeepWalk
from emb.spectral import Spectral
from emb.hope import HOPE
from emb.grarep import GraRep
from emb.sdne import SDNE


# node2vec = Node2vec(dimension=128, walk_num=200, walk_length=30, window_size=5, worker=96, iteration=5, p=1.0, q=1.0)
# hope = HOPE(dimension=128, beta=0.01) # dim 128 < node num
# spectral = Spectral(hidden_size=128)
# deepwalk = DeepWalk(dimension=128, walk_num=200, walk_length=30, window_size=5, worker=96, iteration=5)
# line = LINE(dimension=128, walk_length=30, walk_num=200, batch_size=1000, negative=5, alpha=0.025, order=3)
# netsmf = NetSMF(dimension=128, window_size=5, negative=1, worker=96, num_round=100)
# prone = ProNE(dimension=128, step=5, mu=0.2, theta=0.5)
# grarep = GraRep(dimension=128, step=5)
# sdne = SDNE(hidden_size1=1000, hidden_size2=128, droput=0.5, alpha=1e-1, beta=5, nu1=1e-4, nu2=1e-3, epochs=10, lr=0.05, cpu=False)

# gcn = GCN(in_feats=128, hidden_size=128, out_feats=128, num_layers=2, dropout=0.5)
global sent2vec_model
sent2vec_model = sent2vec.Sent2vecModel()
sent2vec_model.load_model('../../datasets/vulcnn/data_model.bin')


from step3_train_sent2vec import tokenize

def graph_extraction(dot):
    graph = nx.drawing.nx_pydot.read_dot(dot)
    return graph


def sentence_embedding(sentence):
    # global sent2vec_model
    emb = sent2vec_model.embed_sentence(sentence)
    return emb[0]


def image_generation(dot):
    # try:
    pdg = graph_extraction(dot)
    labels_dict = nx.get_node_attributes(pdg, 'label')
    labels_code = dict()
    nodes_label = dict()
    for label, all_code in labels_dict.items():
        # code = all_code.split('code:')[1].split('\\n')[0]
        code = all_code[all_code.index(",") + 1:-2].split('\\n')[0]
        code = code.replace("static void", "void")
        code = tokenize(code)
        labels_code[label] = code
        sent_vec = sentence_embedding(code)
        nodes_label[label] = sent_vec

    # G = nx.DiGraph()
    # G.add_nodes_from(pdg.nodes())
    # G.add_edges_from(pdg.edges())

    nodes_map = dict()
    index = 0
    embeddings = []
    for node in pdg.nodes():
        nodes_map[node] = index
        embeddings.append(nodes_label[node])
        index = index + 1
    edges = []
    for edge in pdg.edges():
        e1 = nodes_map.get(edge[0])
        e2 = nodes_map.get(edge[1])
        edges.append([e1, e2])
    edge_index = np.array(edges)
    edge_index = torch.from_numpy(edge_index)

    embs = np.array(embeddings)
    embs = torch.from_numpy(embs)


    print(nodes_map)
    # nodes = pdg.nodes()
    # edges = pdg.edges()

    # G = dgl.from_networkx(pdg)

    # emb1 = node2vec(G)
    # emb2 = deepwalk(G)
    # emb3 = line(G)
    #
    # print(emb1)


if __name__ == '__main__':
    image_generation("../../datasets/d2a/cfgs/pdgs/1000_1.dot")
