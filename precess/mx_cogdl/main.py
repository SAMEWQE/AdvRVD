import networkx as nx
import sent2vec
import warnings

warnings.filterwarnings(action="ignore")
import numpy as np

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

node2vec = Node2vec(dimension=128, walk_num=200, walk_length=30, window_size=5, worker=96, iteration=5, p=1.0, q=1.0)
hope = HOPE(dimension=128, beta=0.01) # dim 128 < node num
spectral = Spectral(hidden_size=128)
deepwalk = DeepWalk(dimension=128, walk_num=200, walk_length=30, window_size=5, worker=96, iteration=5)
line = LINE(dimension=128, walk_length=30, walk_num=200, batch_size=1000, negative=5, alpha=0.025, order=3)
netsmf = NetSMF(dimension=128, window_size=5, negative=1, worker=96, num_round=100)
prone = ProNE(dimension=128, step=5, mu=0.2, theta=0.5)
grarep = GraRep(dimension=128, step=5)
sdne = SDNE(hidden_size1=1000, hidden_size2=128, droput=0.5, alpha=1e-1, beta=5, nu1=1e-4, nu2=1e-3, epochs=10, lr=0.05, cpu=False)


def graph_extraction(dot):
    graph = nx.drawing.nx_pydot.read_dot(dot)
    return graph


def sentence_embedding(sentence):
    # global sent2vec_model
    sent2vec_model = sent2vec.Sent2vecModel()
    sent2vec_model.load_model(
        '/data/bhtian2/win_linux_mapping/Adversarial_Reprogramming-master/datasets/vulcnn/data_model.bin')
    emb = sent2vec_model.embed_sentence(sentence)
    return emb[0]


def image_generation(dot):
    # try:
    pdg = graph_extraction(dot)
    labels_dict = nx.get_node_attributes(pdg, 'label')
    labels_code = dict()
    for label, all_code in labels_dict.items():
        # code = all_code.split('code:')[1].split('\\n')[0]
        code = all_code[all_code.index(",") + 1:-2].split('\\n')[0]
        code = code.replace("static void", "void")
        labels_code[label] = code

    G = nx.DiGraph()
    G.add_nodes_from(pdg.nodes())
    G.add_edges_from(pdg.edges())

    emb1 = node2vec(G)
    emb2 = deepwalk(G)
    emb3 = line(G)

    print(emb1)


if __name__ == '__main__':
    image_generation("/data/bhtian2/win_linux_mapping/three_fusion/data2/d2a/cfgs/pdgs/1000_1.dot")
