import networkx as nx
import argparse
import os
import sent2vec
import pickle
import glob
from multiprocessing import Pool
from functools import partial
import warnings
from step3_train_sent2vec import tokenize

warnings.filterwarnings(action="ignore")
import numpy as np

from mx_cogdl.emb.node2vec import Node2vec
from mx_cogdl.emb.deepwalk import DeepWalk
from mx_cogdl.emb.line import LINE

# 原始配置 (等价于DeepWalk)
# node2vec = Node2vec(dimension=128, ..., p=1.0, q=1.0)

# 建议的新配置
#node2vec = Node2vec(dimension=128, walk_num=100, walk_length=30, window_size=5, worker=96, iteration=5, p=0.25, q=4.0)
node2vec = Node2vec(dimension=128, walk_num=100, walk_length=30, window_size=5, worker=96, iteration=5, p=0.5, q=2.0)
#node2vec = Node2vec(dimension=128, walk_num=100, walk_length=30, window_size=5, worker=96, iteration=5, p=2.0, q=0.5)
deepwalk = DeepWalk(dimension=128, walk_num=100, walk_length=30, window_size=5, worker=96, iteration=5)
line = LINE(dimension=128, walk_length=30, walk_num=200, batch_size=1000, negative=5, alpha=0.025, order=3)


def parse_options():
    parser = argparse.ArgumentParser(description='Image-based Vulnerability Detection.')
    parser.add_argument('-i', '--input',
                        default='../datasets/d2a1/valid/pdgs/No-Vul',
                        help='The path of a dir which consists of some dot_files')
    parser.add_argument('-o', '--out',
                        default='../datasets/d2a1/valid/channel_vec/No-Vul',
                        help='The path of output.', required=False)
    parser.add_argument('-m', '--model',
                        # default='../sent2vec/devign_model_Chk3.ckpt.bin',
                        default='../sent2vec/d2a1_model.bin',
                        help='The path of model.', required=False)
    args = parser.parse_args()
    return args


def graph_extraction(dot):
    try:
        graph = nx.drawing.nx_pydot.read_dot(dot)
    except:
        return None
    return graph


def sentence_embedding(sentence):
    emb = sent2vec_model.embed_sentence(sentence)
    return emb[0]


def image_generation(dot):
    global emb2, emb3, emb1, line_vec
    pdg = graph_extraction(dot)
    if pdg is None:
        os.system("rm " + dot)
        return
    labels_dict = nx.get_node_attributes(pdg, 'label')
    labels_code = dict()
    for label, all_code in labels_dict.items():
        code = all_code[all_code.index(",") + 1:-2].split('\\n')[0]
        code = code.replace("static void", "void")
        code = tokenize(code)
        print(code)
        labels_code[label] = code
    graph = nx.DiGraph()
    graph.add_nodes_from(pdg.nodes())
    graph.add_edges_from(pdg.edges())
    #   设定条件----------------------------------------------------
    if len(pdg.nodes) > 0:
    # if len(pdg.nodes) < 0:
        node2vec_emb = node2vec(graph)
        deepwalk_emb = deepwalk(graph)
        line_emb = line(graph)
    else:
        node2vec_emb = None
        deepwalk_emb = None
        line_emb = None
    sent_channels = []
    node_channels = []
    deepwalk_channels = []
    line_channels = []
    test = []
    for label, code in labels_code.items():
        line_vec = sentence_embedding(code)
        line_vec = np.array(line_vec)
        test.append(line_vec)
        if node2vec_emb is None:
            emb1 = np.zeros_like(line_vec)
            emb2 = np.zeros_like(line_vec)
            emb3 = np.zeros_like(line_vec)
        else:
            emb1 = node2vec_emb[label]
            emb2 = deepwalk_emb[label]
            emb3 = line_emb[label]
        sent_channels.append(line_vec)
        node_channels.append(emb1)
        deepwalk_channels.append(emb2)
        line_channels.append(emb3)

        # res = line_vec + emb1
    # return (sent_node_channels, sent_deepwalk_channels, sent_line_channels)
    return (sent_channels, node_channels, deepwalk_channels, line_channels)
    # return (line_vec, line_vec, line_vec)
    # return (emb1, emb1, emb1)



def write_to_pkl(dot, out):
    dot_name = dot.split('/')[-1].split('.dot')[0]
    channels = image_generation(dot)
    if channels == None:
        return None
    else:
        out_pkl = out + dot_name + '.pkl'
        (sent_channel, node2vec_channel, deepwalk_channel, line_channel) = channels
        data = [sent_channel, node2vec_channel, deepwalk_channel, line_channel]
        with open(out_pkl, 'wb') as f:
            pickle.dump(data, f)



def main():
    args = parse_options()
    dir_name = args.input
    out_path = args.out
    trained_model_path = args.model
    global sent2vec_model
    sent2vec_model = sent2vec.Sent2vecModel()
    sent2vec_model.load_model(trained_model_path)

    if dir_name[-1] == '/':
        dir_name = dir_name
    else:
        dir_name += "/"
    dotfiles = glob.glob(dir_name + '*.dot')

    if out_path[-1] == '/':
        out_path = out_path
    else:
        out_path += '/'

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    pool = Pool(12)
    pool.map(partial(write_to_pkl, out=out_path), dotfiles)



if __name__ == '__main__':
    main()
