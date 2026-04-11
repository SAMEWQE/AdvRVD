import pickle, os, glob
import argparse
import pandas as pd
from collections import Counter

from sklearn.utils import shuffle
import numpy
import time
import datetime

def log_time(log_file, message):
    """记录时间日志"""
    with open(log_file, 'a') as log:
        log.write(message + "\n")

def sava_data(filename, data):
    print("开始保存数据至于：", filename)
    f = open(filename, 'wb')
    pickle.dump(data, f)
    f.close()


def load_data(filename):
    print("开始读取数据于：", filename)
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()
    return data


def parse_options():
    parser = argparse.ArgumentParser(description='Generate and split train dataset data with optional SMOTE.')
    parser.add_argument('-i', '--input', default='../datasets/reveal/valid/images',
                        help='The path of a dir which consists of some pkl_files')
    parser.add_argument('-o', '--out', default='../datasets/bigvul/',
                        help='The path of output.', required=False)
    parser.add_argument('-t', '--type', default='valid.pkl', required=False)
    parser.add_argument('--enable-smote', action='store_true',
                        help='Enable SMOTE data balancing')
    parser.add_argument('--smote-strategy', default='auto',
                        help='SMOTE sampling strategy (auto, minority, or specific ratio)')
    parser.add_argument('--smote-k', type=int, default=5,
                        help='Number of k neighbors for SMOTE (default: 5)')
    parser.add_argument('--vector-mode', action='store_true',
                        help='Use vector mode for SMOTE (flatten data to vectors)')
    args = parser.parse_args()
    return args


def apply_smote_to_data(final_dic, smote_strategy='auto', k_neighbors=5, vector_mode=False):
    """
    对数据应用SMOTE处理
    
    Args:
        final_dic: 包含data和label的DataFrame
        smote_strategy: SMOTE采样策略
        k_neighbors: K近邻数量
        vector_mode: 是否使用向量模式
    
    Returns:
        处理后的DataFrame
    """
    from imblearn.over_sampling import SMOTE
    
    print("\n=== 开始SMOTE数据平衡处理 ===")
    
    # 分析原始数据分布
    print("原始数据标签分布:")
    print(final_dic['label'].value_counts())
    
    data = final_dic['data']
    label = final_dic['label']
    
    if vector_mode:
        print("使用向量模式处理数据...")
        # 向量模式：将多维数据展平为向量
        vec_list = []
        
        for idx, data_item in enumerate(data):
            item_vec_list = []
            
            if isinstance(data_item, list) and len(data_item) > 0:
                # 处理多通道数据
                for channel in data_item:
                    if len(channel) > 128:
                        channel = channel[:128]
                    else:
                        # 填充到128长度
                        for pad_idx in range(128 - len(channel)):
                            channel.append(numpy.zeros((50,)))
                    
                    # 展平通道数据
                    chs = []
                    for ch in channel:
                        if hasattr(ch, 'flatten'):
                            chs.extend(ch.flatten())
                        else:
                            chs.extend(ch)
                    
                    channel_vec = numpy.array(chs)
                    item_vec_list.extend(channel_vec)
            else:
                # 处理单个数据项
                if hasattr(data_item, 'flatten'):
                    item_vec_list = data_item.flatten().tolist()
                else:
                    item_vec_list = [data_item] if not isinstance(data_item, list) else data_item
            
            vec_list.append(numpy.array(item_vec_list))
        
        # 确保所有向量长度一致
        max_len = max(len(vec) for vec in vec_list)
        for i, vec in enumerate(vec_list):
            if len(vec) < max_len:
                # 用零填充
                padded_vec = numpy.zeros(max_len)
                padded_vec[:len(vec)] = vec
                vec_list[i] = padded_vec
        
        arr_data = numpy.array(vec_list)
        print(f"向量化后的数据形状: {arr_data.shape}")
        
    else:
        print("使用标准模式处理数据...")
        # 标准模式：直接处理数据
        data_list = []
        for data_item in data:
            if hasattr(data_item, 'flatten'):
                data_list.append(data_item.flatten())
            elif isinstance(data_item, list):
                # 如果是列表，尝试展平
                flat_item = []
                for item in data_item:
                    if hasattr(item, 'flatten'):
                        flat_item.extend(item.flatten())
                    else:
                        flat_item.extend(item if isinstance(item, list) else [item])
                data_list.append(numpy.array(flat_item))
            else:
                data_list.append(data_item)
        
        # 确保所有数据长度一致
        max_len = max(len(item) for item in data_list)
        for i, item in enumerate(data_list):
            if len(item) < max_len:
                padded_item = numpy.zeros(max_len)
                padded_item[:len(item)] = item
                data_list[i] = padded_item
        
        arr_data = numpy.array(data_list)
        print(f"标准模式数据形状: {arr_data.shape}")
    
    # 检查数据是否适合SMOTE
    label_counts = Counter(label)
    min_samples = min(label_counts.values())
    
    if k_neighbors >= min_samples:
        new_k = max(1, min_samples - 1)
        print(f"⚠️  调整k_neighbors从 {k_neighbors} 到 {new_k} (最小类别样本数: {min_samples})")
        k_neighbors = new_k
    
    # 应用SMOTE
    try:
        print(f"应用SMOTE (strategy={smote_strategy}, k_neighbors={k_neighbors})...")
        smo = SMOTE(random_state=42, sampling_strategy=smote_strategy, k_neighbors=k_neighbors)
        X_smo, y_smo = smo.fit_resample(arr_data, label)
        
        print("SMOTE处理完成!")
        print("处理后数据标签分布:")
        print(Counter(y_smo))
        print(f"原始样本数: {len(label)} -> 处理后样本数: {len(y_smo)}")
        
        if vector_mode:
            # 将向量重新reshape为原始格式
            print("重新构造数据格式...")
            # 这里假设原始数据是3通道128x50的格式
            try:
                vecs = [vec.reshape(3, 128, -1) for vec in X_smo]
                processed_data = vecs
            except:
                # 如果reshape失败，保持向量格式
                print("⚠️  无法重新构造为原始格式，保持向量格式")
                processed_data = [vec for vec in X_smo]
        else:
            processed_data = [vec for vec in X_smo]
        
        # 创建新的DataFrame
        smote_df = pd.DataFrame({
            'data': processed_data,
            'label': y_smo
        })
        
        return smote_df
        
    except Exception as e:
        print(f"❌ SMOTE处理失败: {str(e)}")
        print("返回原始数据...")
        return final_dic


def generate_dataframe(input_path, save_path, enable_smote=False, smote_strategy='auto', 
                      smote_k=5, vector_mode=False):
    input_path = input_path + "/" if input_path[-1] != "/" else input_path
    save_path = save_path + "/" if save_path[-1] != "/" else save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    dic = []
    print(f"处理路径: {input_path}")
    
    for type_name in os.listdir(input_path):
        dicname = input_path + type_name
        print(f"处理类别: {type_name}")
        filename = glob.glob(dicname + "/*.pkl")
        
        for file in filename:
            data = load_data(file)
            dic.append({
                "filename": file.split("/")[-1].rstrip(".pkl"),
                "length": len(data[0]) if isinstance(data, list) and len(data) > 0 else len(data),
                "data": data,
                "label": 0 if type_name == "No-Vul" else 1
            })
    
    final_dic = pd.DataFrame(dic)
    print(f"加载完成，总样本数: {len(final_dic)}")

    # 应用SMOTE处理（如果启用）
    if enable_smote:
        final_dic = apply_smote_to_data(final_dic, smote_strategy, smote_k, vector_mode)
    else:
        print("跳过SMOTE处理")

    # 原有的数据平衡逻辑（注释掉，因为我们使用SMOTE）
    # listType = final_dic['label'].unique()
    # data0 = final_dic[final_dic['label'].isin([0])]
    # data0 = shuffle(data0)
    # data1 = final_dic[final_dic['label'].isin([1])]
    # data0 = data0[0:len(data1)]
    # final_dic = shuffle(pd.concat([data0, data1]), random_state=44)
    # print(final_dic['label'].value_counts())

    # 保存数据
    output_file = save_path + ("data_smote.pkl" if enable_smote else "data.pkl")
    sava_data(output_file, final_dic)
    
    return final_dic


def main():
    args = parse_options()
    input_path = args.input
    output_path = args.out
    type_name = args.type
    enable_smote = args.enable_smote
    smote_strategy = args.smote_strategy
    smote_k = args.smote_k
    vector_mode = args.vector_mode
    
    print("=== 数据生成参数 ===")
    print(f"输入路径: {input_path}")
    print(f"输出路径: {output_path}")
    print(f"输出类型: {type_name}")
    print(f"启用SMOTE: {enable_smote}")
    if enable_smote:
        print(f"SMOTE策略: {smote_strategy}")
        print(f"K近邻数: {smote_k}")
        print(f"向量模式: {vector_mode}")
    
    # 生成数据
    final_dic = generate_dataframe(input_path, output_path, enable_smote, 
                                  smote_strategy, smote_k, vector_mode)
    
    # 读取生成的数据
    data_file = output_path + ("data_smote.pkl" if enable_smote else "data.pkl")
    all_data = pd.read_pickle(data_file)
    
    # 打乱数据
    train = shuffle(all_data, random_state=44)
    train = train.reset_index(drop=True)
    
    # 保存最终数据 - 修改文件名逻辑
    if enable_smote:
        # 如果启用了SMOTE，在文件名中添加_smote标识
        if type_name.endswith('.pkl'):
            # 如果type_name已经包含.pkl后缀，在.pkl前插入_smote
            base_name = type_name[:-4]  # 去掉.pkl
            final_output = output_path + base_name + "_smote.pkl"
        else:
            # 如果type_name不包含.pkl后缀，直接添加_smote.pkl
            final_output = output_path + type_name + "_smote.pkl"
    else:
        # 如果没有启用SMOTE，使用原始文件名
        final_output = output_path + type_name
    
    sava_data(final_output, train)
    
    print(f"\n✅ 数据生成完成!")
    print(f"最终数据保存至: {final_output}")
    print(f"最终数据形状: {train.shape}")
    print("最终标签分布:")
    print(train['label'].value_counts())

    # vulcnn分割逻辑（可选）
    if False:  # 如果需要分割数据，将此处改为True
        split_size = int(len(all_data) * 0.8)
        train_split = all_data[:split_size]
        test_split = all_data[split_size:]
        
        test_split = test_split.reset_index(drop=True)
        train_split = train_split.reset_index(drop=True)
        
        # 修改分割文件的命名逻辑
        if enable_smote:
            sava_data(output_path + "train_smote.pkl", train_split)
            sava_data(output_path + "valid_smote.pkl", test_split)
        else:
            sava_data(output_path + "train.pkl", train_split)
            sava_data(output_path + "valid.pkl", test_split)
        print(f"数据已分割 - 训练集: {len(train_split)}, 验证集: {len(test_split)}")


if __name__ == '__main__':
    start_time = time.time()  # 记录总运行时间
    try:
        main()
    except Exception as e:
        # 捕获异常并输出错误信息
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        elapsed_time = time.time() - start_time  # 计算总时间
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_time("../total_time_log.txt",
                 f"{timestamp} - Total execution time: {elapsed_time:.2f} seconds")



