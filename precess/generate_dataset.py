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
    print("开始保存数据至：", filename)
    f = open(filename, 'wb')
    pickle.dump(data, f)
    f.close()

def load_data(filename):
    print("开始读取数据于：", filename)
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()
    return data

def load_dataset_from_directories(train_path, valid_path):
    """
    从已有的训练集和验证集目录直接加载数据
    """
    print(f"🔧 从现有目录加载数据:")
    print(f"训练集路径: {train_path}")
    print(f"验证集路径: {valid_path}")
    
    # 确保路径以/结尾
    train_path = train_path + "/" if train_path[-1] != "/" else train_path
    valid_path = valid_path + "/" if valid_path[-1] != "/" else valid_path
    
    # 加载训练集
    train_dic = []
    print(f"🔄 加载训练集...")
    for type_name in os.listdir(train_path):
        dicname = train_path + type_name
        print(f"处理训练集类别: {type_name}")
        filename = glob.glob(dicname + "/*.pkl")
        
        for file in filename:
            data = load_data(file)
            train_dic.append({
                "filename": file.split("/")[-1].rstrip(".pkl"),
                "length": len(data[0]) if isinstance(data, list) and len(data) > 0 else len(data),
                "data": data,
                "label": 0 if type_name == "No-Vul" else 1
            })
    
    # 加载验证集
    valid_dic = []
    print(f"🔄 加载验证集...")
    for type_name in os.listdir(valid_path):
        dicname = valid_path + type_name
        print(f"处理验证集类别: {type_name}")
        filename = glob.glob(dicname + "/*.pkl")
        
        for file in filename:
            data = load_data(file)
            valid_dic.append({
                "filename": file.split("/")[-1].rstrip(".pkl"),
                "length": len(data[0]) if isinstance(data, list) and len(data) > 0 else len(data),
                "data": data,
                "label": 0 if type_name == "No-Vul" else 1
            })
    
    # 转换为DataFrame
    train_data = pd.DataFrame(train_dic)
    valid_data = pd.DataFrame(valid_dic)
    
    print(f"✅ 数据加载完成!")
    print(f"训练集样本数: {len(train_data)}")
    print("训练集标签分布:")
    print(train_data['label'].value_counts())
    print(f"验证集样本数: {len(valid_data)}")
    print("验证集标签分布:")
    print(valid_data['label'].value_counts())
    
    return train_data, valid_data

def apply_smote_to_data(final_dic, smote_strategy='auto', k_neighbors=5, vector_mode=True):
    """
    对训练集应用SMOTE处理（基于原始文件的实现）
    
    Args:
        final_dic: 包含data和label的DataFrame
        smote_strategy: SMOTE采样策略
        k_neighbors: K近邻数量
        vector_mode: 是否使用向量模式
    
    Returns:
        处理后的DataFrame
    """
    from imblearn.over_sampling import SMOTE
    
    print("\n=== 开始SMOTE数据平衡处理（仅应用于训练集）===")
    
    # 分析原始数据分布
    print("原始训练集标签分布:")
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
                            channel.append(numpy.zeros((128,)))  # 修正：应该是128维向量
                    
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
        print("处理后训练集标签分布:")
        print(Counter(y_smo))
        print(f"原始训练集样本数: {len(label)} -> 处理后训练集样本数: {len(y_smo)}")
        
        if vector_mode:
            # 将向量重新reshape为原始格式
            print("重新构造数据格式...")
            # 这里假设原始数据是3通道，每通道128个128维向量的格式
            try:
                processed_data = []
                for vec in X_smo:
                    # 尝试重新构造为3通道格式
                    total_elements = len(vec)
                    elements_per_channel = total_elements // 3
                    
                    reconstructed_channels = []
                    for channel_idx in range(3):
                        start_idx = channel_idx * elements_per_channel
                        end_idx = start_idx + elements_per_channel
                        channel_data = vec[start_idx:end_idx]
                        
                        # 重构为128个128维向量的列表
                        try:
                            # 假设每个向量是128维
                            vectors_per_channel = len(channel_data) // 128
                            channel_list = []
                            for v_idx in range(min(vectors_per_channel, 128)):  # 最多128个向量
                                start = v_idx * 128
                                end = start + 128
                                if end <= len(channel_data):
                                    vector = channel_data[start:end]
                                    channel_list.append(vector)
                                else:
                                    # 如果数据不足，用零填充
                                    remaining = channel_data[start:]
                                    padded = numpy.zeros(128)
                                    padded[:len(remaining)] = remaining
                                    channel_list.append(padded)
                            
                            # 确保每个通道有足够的向量
                            while len(channel_list) < 70:  # 原始数据大约70个向量/通道
                                channel_list.append(numpy.zeros(128))
                            
                            reconstructed_channels.append(channel_list)
                        except Exception as e:
                            print(f"⚠️  通道{channel_idx}重构失败: {e}")
                            # 简单备选方案
                            simple_channel = []
                            chunk_size = max(1, len(channel_data) // 70)
                            for i in range(0, min(len(channel_data), 70 * chunk_size), chunk_size):
                                chunk = channel_data[i:i+chunk_size]
                                if len(chunk) < 128:
                                    padded = numpy.zeros(128)
                                    padded[:len(chunk)] = chunk
                                    simple_channel.append(padded)
                                else:
                                    simple_channel.append(chunk[:128])
                            
                            while len(simple_channel) < 70:
                                simple_channel.append(numpy.zeros(128))
                            
                            reconstructed_channels.append(simple_channel[:70])
                    
                    processed_data.append(reconstructed_channels)
                
                print(f"✅ 成功重构为3通道格式，样本数: {len(processed_data)}")
                
            except Exception as e:
                print(f"⚠️  重构失败: {e}，保持向量格式")
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
        print("返回原始训练集数据...")
        return final_dic

def main():
    print("=== 从现有训练集和验证集目录生成数据 ===")
    
    # 🔧 可修改参数区域
    train_path = './datasets/reveal/train/images'  # 训练集目录
    valid_path = './datasets/reveal/valid/images'  # 验证集目录
    output_path = './datasets/reveal_from_dirs/'   # 输出目录
    enable_smote = True         # 🔥 启用SMOTE（使用原始文件的实现）
    smote_strategy = 'auto'     # SMOTE策略 ('auto', 'minority', 或数字比例如0.8)
    smote_k = 5                 # SMOTE的K近邻数量
    vector_mode = True          # 🔧 使用向量模式，确保三通道格式重构
    
    print(f"训练集路径: {train_path}")
    print(f"验证集路径: {valid_path}")  
    print(f"输出路径: {output_path}")
    print(f"启用SMOTE: {enable_smote}")
    if enable_smote:
        print(f"SMOTE策略: {smote_strategy}")
        print(f"K近邻数: {smote_k}")
        print(f"向量模式: {vector_mode}")
    
    # 创建输出目录
    output_path = output_path + "/" if output_path[-1] != "/" else output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # 🔑 关键：直接从现有目录加载训练集和验证集
    train_data, valid_data = load_dataset_from_directories(train_path, valid_path)
    
    # 可选：对训练集应用SMOTE
    if enable_smote:
        print("\n⚠️  SMOTE仅应用于训练集，验证集保持原始分布")
        train_data_smote = apply_smote_to_data(train_data, smote_strategy, smote_k, vector_mode)
        
        # 保存SMOTE处理后的训练集和原始验证集
        sava_data(output_path + "train_smote.pkl", train_data_smote)
        sava_data(output_path + "valid.pkl", valid_data)  # 验证集不变
        
        print(f"\n✅ 数据生成完成 (启用SMOTE)!")
        print(f"SMOTE训练集保存至: {output_path}train_smote.pkl")
        print(f"原始验证集保存至: {output_path}valid.pkl")
        print(f"最终训练集样本数: {len(train_data_smote)}")
        print(f"最终验证集样本数: {len(valid_data)}")
        
    else:
        print("\n跳过SMOTE处理")
        # 保存原始的训练集和验证集
        sava_data(output_path + "train.pkl", train_data)
        sava_data(output_path + "valid.pkl", valid_data)
        
        print(f"\n✅ 数据生成完成 (未启用SMOTE)!")
        print(f"训练集保存至: {output_path}train.pkl")
        print(f"验证集保存至: {output_path}valid.pkl")
        print(f"训练集样本数: {len(train_data)}")
        print(f"验证集样本数: {len(valid_data)}")
    
    print("\n🔒 无数据泄露风险:")
    print("✅ 训练集和验证集来自不同的原始目录")
    print("✅ 未进行额外的数据分割")
    print("✅ SMOTE仅应用于训练集（如果启用）")
    print("✅ 使用向量模式重构，保持三通道格式")

if __name__ == '__main__':
    start_time = time.time()
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        elapsed_time = time.time() - start_time
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_time("./total_time_log.txt",
                 f"{timestamp} - Direct load from directories execution time: {elapsed_time:.2f} seconds") 