import networkx as nx
import argparse
import os
import sent2vec
import pickle
import glob
from multiprocessing import Pool
from functools import partial
import warnings

warnings.filterwarnings(action="ignore")
import numpy as np

operators3 = {'<<=', '>>='}
operators2 = {
    '->', '++', '--',
    '!~', '<<', '>>', '<=', '>=',
    '==', '!=', '&&', '||', '+=',
    '-=', '*=', '/=', '%=', '&=', '^=', '|='
}
operators1 = {
    '(', ')', '[', ']', '.',
    '+', '-', '*', '&', '/',
    '%', '<', '>', '^', '|',
    '=', ',', '?', ':', ';',
    '{', '}'
}

def tokenize(line):
    tmp, w = [], []
    i = 0
    while i < len(line):
        # Ignore spaces and combine previously collected chars to form words
        if line[i] == ' ':
            tmp.append(''.join(w))
            tmp.append(line[i])
            w = []
            i += 1
        # Check operators and append to final list
        elif line[i:i + 3] in operators3:
            tmp.append(''.join(w))
            tmp.append(line[i:i + 3])
            w = []
            i += 3
        elif line[i:i + 2] in operators2:
            tmp.append(''.join(w))
            tmp.append(line[i:i + 2])
            w = []
            i += 2
        elif line[i] in operators1:
            tmp.append(''.join(w))
            tmp.append(line[i])
            w = []
            i += 1
        # Character appended to word list
        else:
            w.append(line[i])
            i += 1
    tmp.append(''.join(w))
    # Filter out irrelevant strings
    res = list(filter(lambda c: c != '', tmp))
    res = list(filter(lambda c: c != ' ', res))
    # return list(filter(lambda c: c != ' ', res))
    return ' '.join(res)



def parse_options():
    parser = argparse.ArgumentParser(description='Image-based Vulnerability Detection.')
    parser.add_argument('-i', '--input',
                        default='../../datasets/d2a/train/pdgs/Vul',
                        help='The path of a dir which consists of some dot_files')
    parser.add_argument('-o', '--out',
                        default='../../datasets/d2a/train/corpus.txt',
                        help='The path of output.', required=False)
    parser.add_argument('-m', '--model',
                        default='../../datasets/vulcnn/data_model.bin',
                        help='The path of model.', required=False)
    args = parser.parse_args()
    return args

def graph_extraction(dot):
    graph = nx.drawing.nx_pydot.read_dot(dot)
    return graph


def generate_corpus(dot):
    pdg = graph_extraction(dot)
    labels_dict = nx.get_node_attributes(pdg, 'label')
    with open('../../datasets/d2a/train/corpus.txt', 'a') as f:
        for label, all_code in labels_dict.items():
            code = all_code[all_code.index(",") + 1:-2].split('\\n')[0]
            code = code.replace("static void", "void")
            code2 = tokenize(code)
            print(code2)
            f.write(code2+"\n")






def main():
    args = parse_options()
    dir_name = args.input

    if dir_name[-1] == '/':
        dir_name = dir_name
    else:
        dir_name += "/"
    dotfiles = glob.glob(dir_name + '*.dot')

    pool = Pool(96)
    pool.map(partial(generate_corpus), dotfiles)

    # for dot in dotfiles:
    #     generate_corpus(dot)



if __name__ == '__main__':
    main()
    # out = tokenize('VAR2->VAR9')
    # print(out)
    #
    # generate_corpus("../../datasets/d2a/cfgs/pdgs/1000_1.dot")
