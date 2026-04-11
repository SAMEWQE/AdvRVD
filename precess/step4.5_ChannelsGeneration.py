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
import time
import datetime
def log_time(log_file, message):
    """记录时间日志"""
    with open(log_file, 'a') as log:
        log.write(message + "\n")
# node2vec = Node2vec(dimension=128, walk_num=200, walk_length=30, window_size=5, worker=96, iteration=5, p=1.0, q=1.0)
# deepwalk = DeepWalk(dimension=128, walk_num=200, walk_length=30, window_size=5, worker=96, iteration=5)
# line = LINE(dimension=128, walk_length=30, walk_num=200, batch_size=1000, negative=5, alpha=0.025, order=3)


def parse_options():
    parser = argparse.ArgumentParser(description='Image-based Vulnerability Detection.')
    parser.add_argument('-i', '--input',
                        default='/root/shared-nvme/pxyang/AdvRVD/datasets/megavul/valid/channel_vec/Vul',
                        help='The path of output.', required=False)
    parser.add_argument('-o', '--out',
                        default='/root/shared-nvme/pxyang/AdvRVD/datasets/megavul/valid/images/Vul',
                        help='The path of output.', required=False)
    parser.add_argument('-f', '--fusion_mode',
                        default='add_sent_base',  # 新增融合模式参数
                        choices=['add_sent_base', 'weighted_sent_base', 'separate_channels', 'average_all', 'weighted_all', 'sent_only'],
                        help='Feature fusion strategy')
    args = parser.parse_args()
    return args



def load_data(filename):
    print("开始读取数据于：", filename)
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()
    return data

def vec_add(vec1: list, vec2: list):
    """高效的向量加法融合"""
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)
    result = vec1_np + vec2_np
    return result.tolist()

def vec_weighted_add(vec1: list, vec2: list, weight1=0.7, weight2=0.3):
    """高效的加权加法融合"""
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)
    result = vec1_np * weight1 + vec2_np * weight2
    return result.tolist()

def vec_average(vectors: list):
    """高效的平均融合"""
    if not vectors:
        return []
    vectors_np = np.array(vectors)
    result = np.mean(vectors_np, axis=0)
    return result.tolist()

def vec_weighted_average(vectors: list, weights: list):
    """高效的加权平均融合"""
    if not vectors or len(vectors) != len(weights):
        return []
    vectors_np = np.array(vectors)
    weights_np = np.array(weights)
    # 使用广播机制进行加权平均
    weighted_sum = np.sum(vectors_np * weights_np[:, np.newaxis], axis=0)
    return weighted_sum.tolist()

def generate_channels(filename, out, fusion_mode='add_sent_base'):
    dot_name = filename.split('/')[-1].split('.pkl')[0]
    data = load_data(filename)
    
    # data[0]: sent2vec, data[1]: node2vec, data[2]: deepwalk, data[3]: line
    sent_vec = data[0]
    node_vec = data[1] 
    deepwalk_vec = data[2]
    line_vec = data[3]
    
    if fusion_mode == 'add_sent_base':
        # 原始方式：以sent2vec为基础，分别与其他向量相加
        channel1 = vec_add(sent_vec, node_vec)
        channel2 = vec_add(sent_vec, deepwalk_vec)
        channel3 = vec_add(sent_vec, line_vec)
        
    elif fusion_mode == 'weighted_sent_base':
        # 加权融合：sent2vec权重更高
        channel1 = vec_weighted_add(sent_vec, node_vec, 0.7, 0.3)
        channel2 = vec_weighted_add(sent_vec, deepwalk_vec, 0.7, 0.3)
        channel3 = vec_weighted_add(sent_vec, line_vec, 0.7, 0.3)
        
    elif fusion_mode == 'separate_channels':
        # 通道分离：每个通道使用不同的特征表示
        channel1 = sent_vec  # 语义特征
        channel2 = node_vec  # 结构特征
        channel3 = deepwalk_vec  # 路径特征
        
    elif fusion_mode == 'average_all':
        # 所有特征平均融合到每个通道
        avg_vec = vec_average([sent_vec, node_vec, deepwalk_vec, line_vec])
        channel1 = avg_vec
        channel2 = avg_vec  
        channel3 = avg_vec
        
    elif fusion_mode == 'weighted_all':
        # 所有特征加权融合：sent2vec权重最高，其他递减
        weights = [0.4, 0.3, 0.2, 0.1]  # sent2vec, node2vec, deepwalk, line
        weighted_vec = vec_weighted_average([sent_vec, node_vec, deepwalk_vec, line_vec], weights)
        channel1 = weighted_vec
        channel2 = weighted_vec
        channel3 = weighted_vec
        
    elif fusion_mode == 'sent_only':
        # 只使用sent2vec特征（消融研究）
        channel1 = sent_vec
        channel2 = sent_vec
        channel3 = sent_vec
        
    else:
        # 默认使用原始方式
        channel1 = vec_add(sent_vec, node_vec)
        channel2 = vec_add(sent_vec, deepwalk_vec)
        channel3 = vec_add(sent_vec, line_vec)
    
    data = [channel1, channel2, channel3]
    out_pkl = out + dot_name + '.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump(data, f)


def main():
    args = parse_options()
    dir_name = args.input
    out_path = args.out
    fusion_mode = args.fusion_mode

    print(f"使用特征融合模式: {fusion_mode}")

    if dir_name[-1] == '/':
        dir_name = dir_name
    else:
        dir_name += "/"

    pklfiles = glob.glob(dir_name+'*.pkl')

    if out_path[-1] == '/':
        out_path = out_path
    else:
        out_path += '/'

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    pool = Pool(96)
    pool.map(partial(generate_channels, out=out_path, fusion_mode=fusion_mode), pklfiles)



if __name__ == '__main__':
    start_time = time.time()  # 记录总运行时间
    try:
        main()
    except Exception as e:
        # 捕获异常并输出错误信息
        print(f"An error occurred: {e}")
    finally:
        elapsed_time = time.time() - start_time  # 计算总时间
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_time("/data/pxyang/AdvRVD/total_time_log.txt",
                 f"{timestamp} - Total execution time: {elapsed_time:.2f} seconds")

