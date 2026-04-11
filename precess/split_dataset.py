import pandas as pd
import os
from sklearn.model_selection import train_test_split

# 使用匿名项目内的绝对路径以防路径错误
base_dir = '../datasets'
datasets = ['d2a', 'devign', 'reveal']

for ds in datasets:
    ds_file = os.path.join(base_dir, ds, f"{ds}_dataset.csv")
    if not os.path.exists(ds_file):
        print(f"⚠️ 跳过 {ds}: 找不到合并文件 {ds_file}")
        continue
        
    print(f"🔄 正在划分数据集: {ds.upper()}")
    df = pd.read_csv(ds_file)
    
    # 确定标签列名
    label_col = 'label' if 'label' in df.columns else 'target'
    if label_col not in df.columns:
        print(f"⚠️ 找不到 {ds} 的标签列 (label 或 target)，跳过。\n")
        continue

    # 对分类数据集开启分层抽样 (stratify)，保证划分后的正负样本比例跟总比例完全一致
    # 第一步：划分 80% 训练集, 20% 临时测试集
    df_train, df_temp = train_test_split(df, test_size=0.2, random_state=42, stratify=df[label_col])
    
    # 第二步：将 20% 的临时测试集平分为两半，获得 10% 验证集(Val) 和 10% 测试集(Test)
    df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42, stratify=df_temp[label_col])
    
    # 命名规范：[dataset_name]_[split].csv，方便后续主代码或生成脚本遍历
    train_path = os.path.join(base_dir, ds, f'{ds}_train.csv')
    val_path = os.path.join(base_dir, ds, f'{ds}_val.csv')
    test_path = os.path.join(base_dir, ds, f'{ds}_test.csv')
    
    # 保存结果
    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)
    
    print(f"  ✓ 原始总数: {len(df)}")
    print(f"  ✓ 训练集 (Train): {len(df_train)} 条 ({len(df_train)/len(df)*100:.1f}%)")
    print(f"  ✓ 验证集 (Val):   {len(df_val)} 条 ({len(df_val)/len(df)*100:.1f}%)")
    print(f"  ✓ 测试集 (Test):  {len(df_test)} 条 ({len(df_test)/len(df)*100:.1f}%)")
    print()

print("🎉 8:1:1 数据集重新划分全部完成！")
