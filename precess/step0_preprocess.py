import pandas as pd
import os
import time
import datetime

def log_time(log_file, message):
    """记录时间日志"""
    with open(log_file, 'a') as log:
        log.write(message + "\n")

def process_dataset(df, base_out, split_name, ds_name):
    # 自动识别代码列
    if 'code' in df.columns:
        code_col = 'code'
    elif 'text' in df.columns:
        code_col = 'text'
    elif 'functionSource' in df.columns:
        code_col = 'functionSource'
    else:
        code_col = df.columns[0]
        
    # 自动识别标签列
    label_col = 'label' if 'label' in df.columns else 'target'
    
    # 自动识别 ID 列，如果没有则用索引
    if 'id' in df.columns:
        id_col = 'id'
    elif 'hash' in df.columns:
        id_col = 'hash'
    else:
        id_col = None

    os.makedirs(f"{base_out}/{split_name}/Vul", exist_ok=True)
    os.makedirs(f"{base_out}/{split_name}/No-Vul", exist_ok=True)

    print(f"正在生成 {ds_name} - {split_name} ({len(df)} 条)")

    for i, row in df.iterrows():
        file_name = f"{row[id_col]}" if id_col else f"{i}"
        label = row[label_col]
        code_text = row[code_col]
        
        if label == 1:
            path = f"{base_out}/{split_name}/Vul/Bad_{file_name}.c"
        elif label == 0:
            path = f"{base_out}/{split_name}/No-Vul/Good_{file_name}.c"
        else:
            continue
            
        with open(path, 'w', encoding='utf-8') as f:
            f.write(str(code_text))

def d2a():
    base_in = '../datasets/d2a'
    base_out = '../datasets/d2a'
    
    splits = [('train', 'train'), ('val', 'val'), ('test', 'test')]
    for split_dir, split_suffix in splits:
        file_path = f"{base_in}/d2a_{split_suffix}.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, encoding='utf-8')
            process_dataset(df, base_out, split_dir, 'd2a')

def devign():
    base_in = '../datasets/devign'
    base_out = '../datasets/devign'
    
    splits = [('train', 'train'), ('val', 'val'), ('test', 'test')]
    for split_dir, split_suffix in splits:
        file_path = f"{base_in}/devign_{split_suffix}.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, encoding='utf-8')
            process_dataset(df, base_out, split_dir, 'devign')

def reveal():
    base_in = '../datasets/reveal'
    base_out = '../datasets/reveal'
    
    splits = [('train', 'train'), ('val', 'val'), ('test', 'test')]
    for split_dir, split_suffix in splits:
        file_path = f"{base_in}/reveal_{split_suffix}.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, encoding='utf-8')
            process_dataset(df, base_out, split_dir, 'reveal')

if __name__ == '__main__':
    start_time = time.time()
    
    print("=== 开始执行 step0 数据预处理 ===")
    d2a()
    devign()
    reveal()
    
    elapsed_time = time.time() - start_time
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, "../../total_time_log.txt")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_time(log_dir, f"{timestamp} - Preprocessing execution time: {elapsed_time:.2f} seconds")
    print("=== step0 预处理完成 ===")
