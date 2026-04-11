# -*- coding:utf-8 -*-
# 模型评估程序 - 用于验证已训练模型在验证集上的性能
# 使用方法: python evaluate.py -d [dataset_name] -m [model_path]

import numpy as np
import importlib
import torch
from torch.autograd import Variable
import os
import argparse
from tqdm import tqdm
import pickle
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, classification_report, confusion_matrix
import warnings
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")

# 导入必要的模块
from tradition_dataset import TraditionalDataset
from program import Program


class ModelEvaluator:
    """模型评估器 - 用于评估已训练的模型"""
    
    def __init__(self, args):
        self.dataset_name = args.dataset
        self.model_path = args.model
        
        # 动态加载配置文件
        self.cfg = self._load_config(self.dataset_name)
        
        # GPU设置
        self.gpu_id = self.cfg.gpu_id
        self.device = self.cfg.device
        self.use_gpu = self.cfg.use_gpu
        
        if self.use_gpu and torch.cuda.is_available():
            torch.cuda.set_device(self.gpu_id)
            torch.cuda.empty_cache()
            print(f"🔧 使用GPU: {self.device}")
        else:
            self.use_gpu = False
            print(f"🔧 使用CPU")
        
        self.gpu = [self.gpu_id] if self.use_gpu else []
        
        print(f"🔍 模型评估配置:")
        print(f"   数据集: {self.dataset_name}")
        print(f"   模型路径: {self.model_path}")
        print(f"   数据集路径: {self.cfg.dataset_path}")
        print(f"   批次大小: {self.cfg.batch_size_per_gpu}")
        
        # 初始化数据集和模型
        self.init_dataset()
        self.Program = Program(self.cfg, self.gpu)
        self.load_model()
        self.set_mode_and_gpu()
        
    def _load_config(self, dataset_name):
        """动态加载配置文件"""
        config_module_name = f'config_{dataset_name}'
        
        try:
            print(f"🔍 正在加载配置: {config_module_name}.py")
            config_module = importlib.import_module(config_module_name)
            
            if hasattr(config_module, 'get_config'):
                cfg = config_module.get_config(dataset_name=dataset_name)
                print(f"✅ 成功加载配置: {config_module_name}.get_config()")
            elif hasattr(config_module, 'cfg'):
                cfg = config_module.cfg
                
                # 自动补全缺失参数
                if not hasattr(cfg, 'dataset_name'):
                    cfg.dataset_name = dataset_name
                if not hasattr(cfg, 'dataset_path'):
                    cfg.dataset_path = f'/data/pxyang/AdvRVD/datasets/{dataset_name}'
                if not hasattr(cfg, 'use_gpu'):
                    cfg.use_gpu = True
                if not hasattr(cfg, 'gpu_id'):
                    cfg.gpu_id = 0
                if not hasattr(cfg, 'device'):
                    cfg.device = 'cuda:0'
                
                print(f"✅ 成功加载配置: {config_module_name}.cfg")
            else:
                raise AttributeError(f"配置模块缺少 get_config() 或 cfg")
            
            return cfg
            
        except (ImportError, AttributeError) as e:
            print(f"⚠️  警告: 无法加载 {config_module_name}.py ({str(e)})")
            print(f"🔄 回退到默认配置: config_primevul.py")
            
            from config_primevul import get_config
            cfg = get_config(dataset_name=dataset_name)
            print(f"✅ 使用默认配置处理数据集: {dataset_name}")
            return cfg
    
    def init_dataset(self):
        """初始化验证数据集"""
        print(f"📂 加载验证集: {self.cfg.dataset_path}")
        # 只加载验证集，不需要训练集
        eval = self.load_valid_data(pathname=self.cfg.dataset_path)
        
        X_valid = eval['data']
        y_valid = eval['label']
        
        print("验证集标签分布:")
        print(y_valid.value_counts())
        
        test_set = TraditionalDataset(X_valid, y_valid, self.cfg.h2, self.cfg.w2)
        
        # 数据加载器配置
        # drop_last=True 丢弃最后不足一个batch的数据，与训练时保持一致
        kwargs = {'num_workers': 96, 'pin_memory': True, 'drop_last': True}
        
        # 使用与训练时相同的batch size配置，参照main.py
        if self.use_gpu:
            self.test_loader = DataLoader(
                test_set,
                batch_size=self.cfg.batch_size_per_gpu * len(self.gpu),
                shuffle=False,  # 评估时不需要shuffle
                **kwargs
            )
        else:
            self.test_loader = DataLoader(
                test_set,
                batch_size=self.cfg.batch_size_per_gpu,
                shuffle=False,
                **kwargs
            )
    
    def load_data(self, filename):
        """加载pickle格式的数据文件"""
        print(f"加载数据文件: {filename}")
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data
    
    def load_valid_data(self, pathname: str):
        """只加载验证集数据文件（评估时不需要训练集）"""
        pathname = pathname + "/" if pathname[-1] != "/" else pathname
        eval_df = self.load_data(pathname + "valid.pkl")
        return eval_df
    
    def load_model(self):
        """加载训练好的模型权重"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"❌ 模型文件不存在: {self.model_path}")
        
        print(f"📂 正在加载模型: {self.model_path}")
        
        if self.use_gpu:
            checkpoint = torch.load(self.model_path, map_location=self.device)
        else:
            checkpoint = torch.load(self.model_path, map_location='cpu')
        
        # 处理不同的检查点格式
        if 'W' in checkpoint:
            # 如果是保存的字典格式（包含对抗重编程的扰动矩阵 W）
            # 找到模型中对应的可训练参数并加载
            W_tensor = checkpoint['W']
            
            # 将 W 加载到模型的可训练参数中
            loaded = False
            for name, param in self.Program.named_parameters():
                if param.requires_grad:
                    # 找到可训练参数（通常是对抗重编程的扰动矩阵）
                    if param.shape == W_tensor.shape:
                        param.data = W_tensor
                        loaded = True
                        print(f"✅ 成功加载参数: {name}, shape: {param.shape}")
                        break
            
            if not loaded:
                print("⚠️  警告: 未找到匹配的可训练参数，尝试直接赋值到第一个可训练参数")
                for param in self.Program.parameters():
                    if param.requires_grad:
                        try:
                            param.data = W_tensor
                            print(f"✅ 成功加载到可训练参数, shape: {param.shape}")
                            loaded = True
                            break
                        except:
                            continue
            
            if not loaded:
                raise RuntimeError("❌ 无法加载检查点：未找到匹配的参数")
                
        else:
            # 直接加载完整的state_dict
            self.Program.load_state_dict(checkpoint, strict=False)
            print(f"✅ 加载完整的 state_dict")
        
        print(f"✅ 模型加载成功")
    
    def set_mode_and_gpu(self):
        """设置评估模式和GPU"""
        # 设置损失函数
        if hasattr(self.cfg, 'use_crossentropy') and self.cfg.use_crossentropy:
            print("🔧 使用 CrossEntropy Loss")
            self.BCE = torch.nn.CrossEntropyLoss()
            self.loss_type = 'crossentropy'
        else:
            print("🔧 使用 BCE Loss")
            self.BCE = torch.nn.BCELoss()
            self.loss_type = 'bce'
        
        # 将模型设置为评估模式
        self.Program.eval()
        
        # GPU设置
        if self.use_gpu:
            with torch.cuda.device(self.gpu_id):
                self.BCE.cuda(self.gpu_id)
                self.Program.cuda(self.gpu_id)
        
        if len(self.gpu) > 1:
            self.Program = torch.nn.DataParallel(self.Program, device_ids=list(range(len(self.gpu))))
    
    def tensor2var(self, tensor, requires_grad=False, volatile=False):
        """将Tensor转换为Variable"""
        if self.use_gpu:
            with torch.cuda.device(self.gpu_id):
                tensor = tensor.cuda(self.gpu_id)
        return Variable(tensor, requires_grad=requires_grad, volatile=volatile)
    
    def compute_loss(self, out, label):
        """计算损失"""
        if hasattr(self.cfg, 'use_crossentropy') and self.cfg.use_crossentropy:
            loss = self.BCE(out, label.long())
        else:
            if self.use_gpu:
                label_onehot = torch.zeros(out.size(0), 2).cuda(self.gpu_id).scatter_(1, label.view(-1, 1), 1)
            else:
                label_onehot = torch.zeros(out.size(0), 2).scatter_(1, label.view(-1, 1), 1)
            
            label_onehot = self.tensor2var(label_onehot)
            loss = self.BCE(out, label_onehot)
        
        return loss
    
    def evaluate(self):
        """在验证集上评估模型"""
        print()
        print("=" * 80)
        print("🚀 开始模型评估")
        print("=" * 80)
        
        preds = []
        labels = []
        total_loss = 0
        
        # 禁用梯度计算以节省内存
        with torch.no_grad():
            progress_bar = tqdm(enumerate(self.test_loader), total=len(self.test_loader), desc='评估中')
            
            for j, data in progress_bar:
                image = data["vector"]
                label = data["targets"]
                image = self.tensor2var(image)
                
                if self.use_gpu:
                    label = label.cuda(self.gpu_id)
                
                # 前向传播
                out = self.Program(image)
                
                # 计算损失
                loss = self.compute_loss(out, label)
                total_loss += loss.item()
                
                # 获取预测结果
                pred = out.data.cpu().numpy().argmax(1)
                preds += pred.tolist()
                labels += label.tolist()  # 参照main.py的写法
                
                # 更新进度条
                progress_bar.set_description(f'评估中 - Loss: {loss.item():.4f}')
        
        # 计算平均损失
        avg_loss = total_loss / len(self.test_loader)
        
        # 计算评估指标
        print()
        print("=" * 80)
        print("📊 评估结果")
        print("=" * 80)
        
        print(f"\n📉 平均损失: {avg_loss:.6f}")
        print()
        
        # 详细的分类报告
        print("📋 分类报告:")
        print("-" * 80)
        print(classification_report(labels, preds, target_names=['非漏洞', '漏洞']))
        
        # 混淆矩阵
        print("📊 混淆矩阵:")
        print("-" * 80)
        cm = confusion_matrix(labels, preds)
        print(f"真负例(TN): {cm[0][0]:<8} 假正例(FP): {cm[0][1]}")
        print(f"假负例(FN): {cm[1][0]:<8} 真正例(TP): {cm[1][1]}")
        print()
        
        # 计算各项指标
        accuracy = accuracy_score(y_true=labels, y_pred=preds)
        precision = precision_score(y_true=labels, y_pred=preds, average='binary')
        f1 = f1_score(y_true=labels, y_pred=preds, average='binary')
        recall = recall_score(y_true=labels, y_pred=preds, average='binary')
        
        # 输出汇总指标
        print("🎯 性能指标汇总:")
        print("-" * 80)
        print(f"准确率 (Accuracy):  {accuracy:.5f}")
        print(f"精确率 (Precision): {precision:.5f}")
        print(f"F1分数 (F1-Score):  {f1:.5f}")
        print(f"召回率 (Recall):    {recall:.5f}")
        print("=" * 80)
        
        # 保存评估结果到文件
        self.save_evaluation_results(accuracy, precision, f1, recall, avg_loss, cm)
        
        return accuracy, precision, f1, recall, avg_loss
    
    def save_evaluation_results(self, accuracy, precision, f1, recall, avg_loss, cm):
        """保存评估结果到文件"""
        # 创建结果目录
        model_dir = os.path.dirname(self.model_path)
        model_name = os.path.basename(self.model_path).replace('.pt', '')
        
        result_file = os.path.join(model_dir, f'evaluation_results_{model_name}.txt')
        
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("模型评估报告\n")
            f.write("=" * 80 + "\n")
            f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据集: {self.dataset_name}\n")
            f.write(f"模型路径: {self.model_path}\n")
            f.write("-" * 80 + "\n")
            f.write("\n性能指标:\n")
            f.write("-" * 80 + "\n")
            f.write(f"准确率 (Accuracy):  {accuracy:.5f}\n")
            f.write(f"精确率 (Precision): {precision:.5f}\n")
            f.write(f"F1分数 (F1-Score):  {f1:.5f}\n")
            f.write(f"召回率 (Recall):    {recall:.5f}\n")
            f.write(f"平均损失 (Loss):    {avg_loss:.6f}\n")
            f.write("-" * 80 + "\n")
            f.write("\n混淆矩阵:\n")
            f.write("-" * 80 + "\n")
            f.write(f"真负例 (TN): {cm[0][0]:<8} 假正例 (FP): {cm[0][1]}\n")
            f.write(f"假负例 (FN): {cm[1][0]:<8} 真正例 (TP): {cm[1][1]}\n")
            f.write("=" * 80 + "\n")
        
        print(f"✅ 评估结果已保存到: {result_file}")


def main():
    parser = argparse.ArgumentParser(description='模型评估程序 - 验证已训练模型在验证集上的性能')
    parser.add_argument('-d', '--dataset', required=True, type=str,
                        choices=['megavul', 'reveal', 'bigvul', 'devign', 'primevul', 'd2a'],
                        help='数据集选择')
    parser.add_argument('-m', '--model', required=True, type=str,
                        help='训练好的模型文件路径 (例如: /path/to/W_050.pt)')
    parser.add_argument('-g', '--gpu', default=['0'], nargs='+', type=str,
                        help='指定GPU设备ID')
    
    args = parser.parse_args()
    
    # 设置GPU环境变量
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)
    
    # 打印配置信息
    print()
    print("=" * 80)
    print("🔍 模型评估程序")
    print("=" * 80)
    print(f"数据集: {args.dataset}")
    print(f"模型路径: {args.model}")
    print(f"GPU设备: {args.gpu}")
    print("=" * 80)
    print()
    
    # 创建评估器并运行评估
    evaluator = ModelEvaluator(args)
    evaluator.evaluate()
    
    print()
    print("🎉 评估完成!")
    print()


if __name__ == "__main__":
    main()

