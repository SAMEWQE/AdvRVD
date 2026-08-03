# -*- coding:utf-8 -*-
# Enhanced Adversarial Reprogramming with Visualization and Early Stopping
# Created Time: Thu 05 Jul 2018 10:00:41 PM CST

import numpy as np
import importlib
import torch
from torch.autograd import Variable
import os
import argparse
from tqdm import tqdm, trange
import pickle
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import warnings
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
# 解决中文显示问题 
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
# 🔧 移除硬编码的GPU设置，改为从配置文件读取
warnings.filterwarnings("ignore")
from tradition_dataset import TraditionalDataset
from program import Program
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
from torch.utils.tensorboard import SummaryWriter  # 导入 TensorBoard 的 SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm
import timm.optim
import timm.scheduler
from torchvision import datasets, transforms
# 🔧 移除全局的GPU缓存清理，改为在初始化时处理

class EarlyStopping:
    """早停类"""
    def __init__(self, patience=15, min_delta=0.001, metric_name='f1'):
        self.patience = patience
        self.min_delta = min_delta
        self.metric_name = metric_name
        self.wait = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
        elif current_score <= self.best_score + self.min_delta:
            self.wait += 1
            if self.wait >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = current_score
            self.wait = 0
        return False
class Adversarial_Reprogramming(object):
    def _load_config(self, dataset_name):
        """
        🔧 动态加载配置文件
        优先级: config_{dataset_name}.py > config_primevul.py (默认)
        """
        config_module_name = f'config_{dataset_name}'
        
        try:
            # 尝试导入对应数据集的配置模块
            print(f"🔍 正在加载配置: {config_module_name}.py")
            config_module = importlib.import_module(config_module_name)
            
            # 尝试调用 get_config 函数
            if hasattr(config_module, 'get_config'):
                cfg = config_module.get_config(dataset_name=dataset_name)
                print(f"✅ 成功加载配置: {config_module_name}.get_config()")
            # 如果没有 get_config 函数，尝试直接使用 cfg 对象
            elif hasattr(config_module, 'cfg'):
                cfg = config_module.cfg
                
                # 🔧 增强旧格式配置的兼容性
                # 确保 cfg 有 dataset_name 属性
                if not hasattr(cfg, 'dataset_name'):
                    cfg.dataset_name = dataset_name
                
                # 确保 dataset_path 存在且正确
                if not hasattr(cfg, 'dataset_path'):
                    cfg.dataset_path = f'./datasets/{dataset_name}'
                    print(f"   🔧 自动添加 dataset_path: {cfg.dataset_path}")
                elif dataset_name not in cfg.dataset_path:
                    cfg.dataset_path = f'./datasets/{dataset_name}'
                    print(f"   🔧 自动修正 dataset_path: {cfg.dataset_path}")
                
                # 确保早停参数存在（使用默认值）
                if not hasattr(cfg, 'early_stop_patience'):
                    cfg.early_stop_patience = 20
                    print(f"   🔧 自动添加 early_stop_patience: {cfg.early_stop_patience}")
                if not hasattr(cfg, 'early_stop_min_delta'):
                    cfg.early_stop_min_delta = 0.0005
                    print(f"   🔧 自动添加 early_stop_min_delta: {cfg.early_stop_min_delta}")
                if not hasattr(cfg, 'early_stop_metric'):
                    cfg.early_stop_metric = 'f1'
                    print(f"   🔧 自动添加 early_stop_metric: {cfg.early_stop_metric}")
                
                # 确保优化器参数存在
                if not hasattr(cfg, 'warmup_epochs'):
                    cfg.warmup_epochs = 5
                if not hasattr(cfg, 'use_gpu'):
                    cfg.use_gpu = True
                if not hasattr(cfg, 'gpu_id'):
                    cfg.gpu_id = 0
                if not hasattr(cfg, 'device'):
                    cfg.device = 'cuda:0'
                
                print(f"✅ 成功加载配置: {config_module_name}.cfg (已自动补全缺失参数)")
            else:
                raise AttributeError(f"配置模块 {config_module_name} 缺少 get_config() 函数或 cfg 对象")
            
            return cfg
            
        except (ImportError, AttributeError) as e:
            # 如果导入失败，回退到默认配置
            print(f"⚠️  警告: 无法加载 {config_module_name}.py ({str(e)})")
            print(f"🔄 回退到默认配置: config_primevul.py")
            
            try:
                from config_primevul import get_config
                cfg = get_config(dataset_name=dataset_name)
                print(f"✅ 使用默认配置 (config_primevul) 处理数据集: {dataset_name}")
                return cfg
            except Exception as fallback_error:
                raise RuntimeError(f"❌ 无法加载任何配置文件: {fallback_error}")
    
    def __init__(self, args):
        # 根据命令行参数获取配置
        self.dataset_name = args.dataset if hasattr(args, 'dataset') else 'megavul'
        
        # 🔧 动态加载对应数据集的配置文件
        self.cfg = self._load_config(self.dataset_name)
        
        # 🔧 统一GPU设置管理 - 从配置文件读取GPU设置
        self.gpu_id = self.cfg.gpu_id
        self.device = self.cfg.device
        self.use_gpu = self.cfg.use_gpu
        
        # 🔧 设置GPU设备和清理缓存
        if self.use_gpu and torch.cuda.is_available():
            torch.cuda.set_device(self.gpu_id)
            torch.cuda.empty_cache()
            print(f"🔧 使用GPU: {self.device}")
        else:
            print(f"🔧 使用CPU")
            
        # 兼容旧版本参数（从命令行传入的gpu参数）
        self.gpu = [self.gpu_id] if self.use_gpu else []
        
        self.best_f1 = 0  # 记录最佳F1-score，用于保存最佳模型
        self.restore = args.restore  # 检查点文件路径
        self.restore_epoch = 0  # 恢复的epoch数字，用于学习率调度器
        
        # 🔧 初始化早停机制
        self.early_stopping = EarlyStopping(
            patience=self.cfg.early_stop_patience,
            min_delta=self.cfg.early_stop_min_delta,
            metric_name=self.cfg.early_stop_metric
        )
        
        # 🔧 用于记录训练历史的列表（用于可视化）
        self.train_losses = []
        self.train_accuracies = []
        self.train_f1_scores = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_f1_scores = []
        self.epochs = []
        
        print(f"🔧 初始化增强版对抗重编程实验:")
        print(f"   数据集名称: {self.dataset_name}")
        print(f"   数据集路径: {self.cfg.dataset_path}")
        print(f"   GPU设备: {self.device}")
        print(f"   训练目录: {self.cfg.train_dir}")
        print(f"   批次大小: {self.cfg.batch_size_per_gpu}")
        print(f"   学习率: {self.cfg.lr}")
        print(f"   正则化强度: {self.cfg.lmd}")
        print(f"   最大训练轮数: {self.cfg.max_epoch}")
        print(f"   早停耐心值: {self.cfg.early_stop_patience}")
        
        self.init_dataset() # 获取配置文件中的超参数
        self.Program = Program(self.cfg, self.gpu) # 获取配置文件中的超参数
        self.restore_from_file()  # 恢复模型参数（若指定检查点）
        self.set_mode_and_gpu() # 设置训练模式和GPU环境
        self.writer = SummaryWriter(log_dir=self.cfg.train_dir+'/loss')  # 设置日志目录
        self.writer2 = SummaryWriter(log_dir=self.cfg.train_dir + '/f1acc')  # 设置日志目录
        self.transform = transforms.Compose([
            #transforms.RandomHorizontalFlip(),  # 随机水平翻转
            #transforms.RandomRotation(30),  # 随机旋转
            #transforms.RandomResizedCrop(self.cfg.h2, scale=(0.8, 1.0)),  # 随机裁剪并调整大小
            #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 随机调整颜色
            #transforms.ToTensor(),  # 转换为 Tensor 格式
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
            transforms.RandomResizedCrop(
                size=(384, 384),  # 图像裁剪后的尺寸
                scale=(0.8, 1.0),  # 随机裁剪的缩放比例范围
                ratio=(3 / 4, 4 / 3)  # 随机裁剪的宽高比范围
            )
        ])
    def init_dataset(self):
        # 🔧 加载训练集和验证集 - 支持动态数据集路径
        print(f"📂 Loading dataset from: {self.cfg.dataset_path}")
        train, eval = self.get_dataset(pathname=self.cfg.dataset_path)  # 使用动态数据集路径
        X_train = train['data']
        #x_train是train的数据
        # y_train = train['label']
        # X_train = train['Unnamed 0']50:100
        #t_train是train的标签
        y_train = train['label']
        #X_train = X_train[:10]
        #y_train = y_train[:10]
        #x_valid是验证集的标签
        X_valid = eval['data']
        #y_valid是验证集的标签
        y_valid = eval['label']
        # X_train = eval['data']
        # y_train = eval['label']
        print(y_train.value_counts())# 输出训练集标签的分布情况
        #class_counts = np.bincount(y_train)
        #print(class_counts)
        print(y_valid.value_counts())# 输出验证集标签的分布情况
        
        # 🚫 注释掉额外的下采样，因为修复后的数据已经正确处理
        # sampling_ratio = 0.2  # 正负样本比例
        # sampling_ratio = 1.0  # 正负样本比例
        # undersampler = RandomUnderSampler(sampling_strategy=sampling_ratio, random_state=42)
        # # 如果 X_train 是一维的 pandas Series，转换为二维数组
        # X_train = X_train.values.reshape(-1, 1) if len(X_train.shape) == 1 else X_train
        # # y_train 确保是数组格式
        # y_train = y_train.values if isinstance(y_train, pd.Series) else y_train
        # # 对训练数据进行欠采样
        # X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)
        # # 打印欠采样后的结果
        # print("原始训练集分布：", np.bincount(y_train))
        # print("欠采样后的训练集分布：", np.bincount(y_train_resampled))
        # # 如果 X_train_resampled 是 DataFrame，将其转换为 Series（假设原始只有一个特征）
        # X_train_resampled = X_train_resampled.squeeze()  # 转换为 Series（如果是单列）

        # # 如果 X_train_resampled 是 Series，且原始的 X_train 是 Series，你可以保留格式
        # # 不需要改变，继续使用原始的 X_train_resampled 格式。

        # # 确保 y_train_resampled 是 Series 类型
        # y_train_resampled = pd.Series(y_train_resampled)

        # # 打印新的数据分布
        # print(f'欠采样后的训练集分布： {y_train_resampled.value_counts().values}')
        # train_set = TraditionalDataset(X_train_resampled, y_train_resampled, self.cfg.h2, self.cfg.w2)
        
        print("使用修复后的数据集，无需额外的采样处理")
        train_set = TraditionalDataset(X_train, y_train, self.cfg.h2, self.cfg.w2)
        test_set = TraditionalDataset(X_valid, y_valid, self.cfg.h2, self.cfg.w2)
        # self.train_loader = DataLoader(train_set, batch_size=self.cfg.batch_size_per_gpu, shuffle=True)
        # self.valid_loader = DataLoader(test_set, batch_size=self.cfg.batch_size_per_gpu, shuffle=True)
        # 定义加载器参数
        # 使用高并发加载（96个worker进程）
        kwargs = {'num_workers': 16, 'pin_memory': True, 'drop_last': True}
        # 🔧 判断是否使用GPU并初始化数据加载器
        if self.use_gpu:
            self.train_loader = torch.utils.data.DataLoader(train_set,
                                                            batch_size=self.cfg.batch_size_per_gpu * len(self.gpu), # 根据GPU数量调整批量大小
                                                            shuffle=True, **kwargs)
            self.test_loader = torch.utils.data.DataLoader(test_set,
                                                           batch_size=self.cfg.batch_size_per_gpu * len(self.gpu),
                                                           shuffle=False, num_workers=kwargs['num_workers'],
                                                           pin_memory=kwargs['pin_memory'], drop_last=False)
        else:
            self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.cfg.batch_size_per_gpu,
                                                            shuffle=True, **kwargs)
            self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.cfg.batch_size_per_gpu,
                                                           shuffle=False, num_workers=kwargs['num_workers'],
                                                           pin_memory=kwargs['pin_memory'], drop_last=False)

    def calculate_class_weights(self,labels):
        total_samples = len(labels)
        class_counts = np.bincount(labels)  # 每个类别的样本数
        num_classes = len(class_counts)  # 类别总数
        weights = total_samples / (num_classes * class_counts)
        return torch.tensor(weights, dtype=torch.float32)
    # 加载pickle格式的数据文件（带进度显示）
    def load_data(self, filename):
        import time
        import os
        
        file_size = os.path.getsize(filename)
        file_size_mb = file_size / (1024 * 1024)
        file_size_gb = file_size / (1024 * 1024 * 1024)
        
        if file_size_gb >= 1:
            size_str = f"{file_size_gb:.2f} GB"
        else:
            size_str = f"{file_size_mb:.1f} MB"
        
        print(f"📂 Loading data file: {os.path.basename(filename)}")
        print(f"   📊 File size: {size_str}")
        print(f"   ⏳ Loading (this may take a while for large files)...")
        
        start_time = time.time()
        
        # 使用 tqdm 显示文件读取进度
        with open(filename, 'rb') as f:
            # 创建进度条，以字节为单位
            with tqdm(total=file_size, unit='B', unit_scale=True, 
                     desc='   Loading', ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
                
                # 包装文件对象以更新进度条
                class ProgressFileWrapper:
                    def __init__(self, file_obj, progress_bar):
                        self.file_obj = file_obj
                        self.progress_bar = progress_bar
                        self.bytes_read = 0
                    
                    def read(self, size=-1):
                        data = self.file_obj.read(size)
                        self.bytes_read += len(data)
                        self.progress_bar.update(len(data))
                        return data
                    
                    def readline(self):
                        data = self.file_obj.readline()
                        self.bytes_read += len(data)
                        self.progress_bar.update(len(data))
                        return data
                    
                    def __getattr__(self, name):
                        return getattr(self.file_obj, name)
                
                wrapped_file = ProgressFileWrapper(f, pbar)
                data = pickle.load(wrapped_file)
        
        elapsed_time = time.time() - start_time
        print(f"   ✅ Loaded successfully! Time: {elapsed_time:.2f}s")
        print()
        
        return data

    # 加载训练集和验证集数据文件
    def get_dataset(self, pathname: str):
        pathname = pathname + "/" if pathname[-1] != "/" else pathname
        train_df = self.load_data(pathname + "train.pkl")
        eval_df = self.load_data(pathname + "valid.pkl")
        return train_df, eval_df

    def restore_from_file(self):
        # 从指定的检查点文件恢复训练
        if self.restore is not None:
            ckpt = self.restore  # 直接使用提供的文件路径
            
            # 检查文件是否存在
            if not os.path.exists(ckpt):
                raise FileNotFoundError(f"❌ 检查点文件不存在: {ckpt}")
            
            print(f"📂 正在加载检查点: {ckpt}")
            
            # 加载模型参数（自动映射到当前设备）
            if self.gpu:
                # 🔧 使用 map_location 将检查点映射到当前 GPU
                checkpoint = torch.load(ckpt, map_location=self.device)
                self.Program.load_state_dict(checkpoint, strict=False)
                print(f"✅ 检查点已映射到设备: {self.device}")
            else:
                # 使用CPU加载参数
                checkpoint = torch.load(ckpt, map_location='cpu')
                self.Program.load_state_dict(checkpoint, strict=False)
                print(f"✅ 检查点已映射到设备: CPU")
            
            # 尝试从文件名中提取epoch数字（用于学习率调度器）
            import re
            match = re.search(r'W_(\d+)', os.path.basename(ckpt))
            if match:
                self.restore_epoch = int(match.group(1))
                self.start_epoch = self.restore_epoch + 1
                print(f"✅ 已加载 Epoch {self.restore_epoch} 的检查点，将从 Epoch {self.start_epoch} 继续训练")
            else:
                # 如果无法提取epoch数字，从epoch 1开始
                self.restore_epoch = 0
                self.start_epoch = 1
                print(f"⚠️  无法从文件名提取epoch信息，将从 Epoch 1 开始（但使用加载的权重）")
        else:
            self.restore_epoch = 0
            self.start_epoch = 1
            print("🆕 从头开始训练（不加载任何检查点）")

    def set_mode_and_gpu(self):
        # 设置模型训练模式以及GPU
        # 🔧 根据配置选择损失函数并输出
        if hasattr(self.cfg, 'use_crossentropy') and self.cfg.use_crossentropy:
            print("🔧 Using CrossEntropy Loss")
            print("📊 Loss Function: torch.nn.CrossEntropyLoss()")
            self.BCE = torch.nn.CrossEntropyLoss()
            self.loss_type = 'crossentropy'
        else:
            print("🔧 Using BCEWithLogits Loss") 
            print("📊 Loss Function: torch.nn.BCEWithLogitsLoss()")
            self.BCE = torch.nn.BCEWithLogitsLoss()
            self.loss_type = 'bce'
        
        # 输出详细配置信息
        print(f"📋 Loss Configuration:")
        print(f"   - Loss Type: {self.loss_type}")
        print(f"   - Expected Input: {'logits' if self.loss_type == 'crossentropy' else 'probabilities'}")
        print(f"   - Expected Labels: {'class indices' if self.loss_type == 'crossentropy' else 'one-hot encoded'}")
    
        # 优化器设置
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.Program.parameters()),
                                          lr=self.cfg.lr, betas=(0.9, 0.999))
        
        # 学习率调度器
        self.lr_scheduler = timm.scheduler.CosineLRScheduler(optimizer=self.optimizer, t_initial=self.cfg.max_epoch, lr_min=1e-4, warmup_t=5, warmup_lr_init=1e-3)
        # 如果从检查点恢复，同步学习率调度器
        if self.restore_epoch > 0:
            for i in range(self.restore_epoch):
                self.lr_scheduler.step(epoch=i)  # 🔧 传入 epoch 参数
            print(f"📊 学习率调度器已同步至 Epoch {self.restore_epoch}")
            print(f"📊 当前学习率: {self.lr_scheduler._get_lr(self.restore_epoch)[0]:.6f}")
        # 🔧 统一GPU设备管理
        if self.use_gpu:
            with torch.cuda.device(self.gpu_id):
                self.BCE.cuda(self.gpu_id)
                self.Program.cuda(self.gpu_id)
        if len(self.gpu) > 1:
            self.Program = torch.nn.DataParallel(self.Program, device_ids=list(range(len(self.gpu))))

    @property
    #获取可训练参数的梯度
    def get_W(self):
        for p in self.Program.parameters():
            if p.requires_grad:
                return p
    #获取可训练参数的梯度
    def imagenet_label2_mnist_label(self, imagenet_label):
        # return imagenet_label[:, :10]
        return imagenet_label[:, :self.cfg.n_classes]
    # """将Tensor转换为Variable"""
    def tensor2var(self, tensor, requires_grad=False, volatile=False):
        # 🔧 使用统一的GPU设备管理
        if self.use_gpu:
            with torch.cuda.device(self.gpu_id):
                tensor = tensor.cuda(self.gpu_id)
        return Variable(tensor, requires_grad=requires_grad, volatile=volatile)
    # """计算损失值"""
    def compute_loss(self, out, label):
        """🔧 修改损失计算以支持CrossEntropy"""
        if hasattr(self.cfg, 'use_crossentropy') and self.cfg.use_crossentropy:
            # 🔧 使用CrossEntropy: 输入logits，标签为类别索引
            loss = self.BCE(out, label.long())
        else:
            # 🔧 使用BCE: 需要one-hot编码
            if self.use_gpu:
                # 🔧 使用统一的GPU设备管理
                label_onehot = torch.zeros(out.size(0), 2).cuda(self.gpu_id).scatter_(1, label.view(-1, 1), 1)
            else:
                label_onehot = torch.zeros(out.size(0), 2).scatter_(1, label.view(-1, 1), 1)
            
            label_onehot = self.tensor2var(label_onehot)
            loss = self.BCE(out, label_onehot)
        
        # 添加正则化项
        regularization = self.cfg.lmd * torch.norm(self.get_W) ** 2
        
        return loss + regularization
    #"""验证模型"""
    #"""输出模型的评估指标"""
    def log_metrics(self, epoch, accuracy, precision, f1, recall, phase, loss):
        log_file = os.path.join(self.cfg.train_dir, 'training_metrics.log')
        
        # 🔧 如果是第一轮训练，记录超参数信息
        if epoch == 1 and phase == 'train':
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write("=" * 100 + "\n")
                f.write("ENHANCED ADVERSARIAL REPROGRAMMING EXPERIMENT\n")
                f.write("=" * 100 + "\n")
                f.write(f"Experiment Time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Dataset: {self.dataset_name}\n")
                f.write(f"Dataset Path: {self.cfg.dataset_path}\n")
                f.write("-" * 50 + " HYPERPARAMETERS " + "-" * 50 + "\n")
                f.write(f"Network: {self.cfg.net}\n")
                f.write(f"Batch Size: {self.cfg.batch_size_per_gpu}\n")
                f.write(f"Learning Rate: {self.cfg.lr}\n")
                f.write(f"Regularization (lambda): {self.cfg.lmd}\n")
                f.write(f"Max Epochs: {self.cfg.max_epoch}\n")
                f.write(f"Image Size: {self.cfg.w1}x{self.cfg.h1} -> {self.cfg.w2}x{self.cfg.h2}\n")
                f.write(f"Early Stop Patience: {self.cfg.early_stop_patience}\n")
                f.write(f"Early Stop Metric: {self.cfg.early_stop_metric}\n")
                f.write(f"Min Delta: {self.cfg.early_stop_min_delta}\n")
                f.write(f"Warmup Epochs: {self.cfg.warmup_epochs}\n")
                f.write("=" * 100 + "\n")
                f.write("TRAINING LOGS:\n")
                f.write("-" * 100 + "\n")
        
        # 记录训练指标
        if phase == 'train':
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"train:Epoch {epoch:03d}: "
                        f"Accuracy: {accuracy:.5f}, "
                        f"Precision: {precision:.5f}, "
                        f"F1-score: {f1:.5f}, "
                        f"Recall: {recall:.5f}, "
                        f"Loss: {loss:.4f}\n")
        if phase == 'test':
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"test:Epoch {epoch:03d}: "
                        f"Accuracy: {accuracy:.5f}, "
                        f"Precision: {precision:.5f}, "
                        f"F1-score: {f1:.5f}, "
                        f"Recall: {recall:.5f}, "
                        f"Loss: {loss:.4f}\n")
    def eval(self, labels, preds):
        rst_str = classification_report(labels,preds)
        print(rst_str)
        print('-' * 100)
        print('Accuracy: ' + str('%.5f' % accuracy_score(y_true=labels, y_pred=preds)), end="\t")
        print('Precision: ' + str('%.5f' % precision_score(y_true=labels, y_pred=preds, average='binary')), end="\t")
        print('F-measure: ' + str('%.5f' % f1_score(y_true=labels, y_pred=preds, average='binary')), end="\t")
        print('Recall: ' + str('%.5f' % recall_score(y_true=labels, y_pred=preds, average='binary')), end="\t")
        return (accuracy_score(y_true=labels, y_pred=preds),precision_score(y_true=labels, y_pred=preds, average='binary'),
                f1_score(y_true=labels, y_pred=preds, average='binary'),recall_score(y_true=labels, y_pred=preds, average='binary'))
    def validate(self):
        preds = []
        labels = []
        total_loss = 0  # 初始化总损失
        progress_bar = tqdm(enumerate(self.test_loader), total=len(self.test_loader), desc='Validating')
        for j, data in progress_bar:
            image = data["vector"]
            label = data["targets"]
            image = self.tensor2var(image)
            # 🔧 使用统一的GPU设备管理
            if self.use_gpu:
                label = label.cuda(self.gpu_id)
            self.out = self.Program(image)
            # 计算损失
            loss = self.compute_loss(self.out, label)
            total_loss += loss.item()  # 累加损失
            pred = self.out.data.cpu().numpy().argmax(1)
            preds += pred.tolist()
            labels += label.tolist()
            # 更新进度条
            progress_bar.set_description(f'Validating - Loss: {loss.item():.4f}')
            
            # 计算平均损失
        avg_loss = total_loss / len(self.test_loader)
        # 记录验证集的平均损失到 TensorBoard
        self.writer.add_scalar('Loss/avg_validate', avg_loss, getattr(self, "epoch", 0))
        print('validateloss: %.6f' % (avg_loss))
        print("validate", end="  ")
        accuracy,precision,f1,recall=self.eval(labels, preds)
        print()
        print("", '-' * 100)
        return accuracy,precision,f1,recall,avg_loss
    
    def log_hyperparameters(self):
        """🔧 记录超参数到独立日志文件"""
        log_file = os.path.join(self.cfg.train_dir, 'hyperparameters.log')
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + " HYPERPARAMETERS " + "=" * 60 + "\n")
            f.write(f"实验时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据集名称: {self.dataset_name}\n")
            f.write(f"数据集路径: {self.cfg.dataset_path}\n")
            f.write(f"GPU设备: {self.gpu}\n")
            f.write("-" * 40 + " 模型参数 " + "-" * 40 + "\n")
            f.write(f"网络结构: {self.cfg.net}\n")
            f.write(f"批次大小: {self.cfg.batch_size_per_gpu}\n")
            f.write(f"学习率: {self.cfg.lr}\n")
            f.write(f"正则化强度 (lambda): {self.cfg.lmd}\n")
            f.write(f"最大训练轮数: {self.cfg.max_epoch}\n")
            f.write(f"图像尺寸: {self.cfg.w1}x{self.cfg.h1} -> {self.cfg.w2}x{self.cfg.h2}\n")
            f.write("-" * 40 + " 早停参数 " + "-" * 40 + "\n")
            f.write(f"早停耐心值: {self.cfg.early_stop_patience}\n")
            f.write(f"早停监控指标: {self.cfg.early_stop_metric}\n")
            f.write(f"最小改善阈值: {self.cfg.early_stop_min_delta}\n")
            f.write("-" * 40 + " 优化参数 " + "-" * 40 + "\n")
            f.write(f"学习率衰减: {self.cfg.decay}\n")
            f.write(f"预热轮数: {self.cfg.warmup_epochs}\n")
            f.write("=" * 133 + "\n")

    def plot_loss_curves(self):
        """🔧 绘制损失曲线（单独文件）"""
        if not self.epochs:
            return
            
        plt.figure(figsize=(12, 8))
        plt.plot(self.epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2, alpha=0.8, marker='o', markersize=4)
        plt.plot(self.epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2, alpha=0.8, marker='s', markersize=4)
        
        plt.title(f'Loss Curves - Dataset: {self.dataset_name}', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 添加统计信息
        min_train_loss = min(self.train_losses)
        min_val_loss = min(self.val_losses)
        plt.text(0.02, 0.98, f'Min Train Loss: {min_train_loss:.4f}\nMin Val Loss: {min_val_loss:.4f}', 
                transform=plt.gca().transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        save_path = os.path.join(self.cfg.train_dir, f'loss_curves_{self.dataset_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"📊 Loss curves saved: {save_path}")
        
    def plot_accuracy_f1_curves(self):
        """🔧 绘制准确率和F1分数曲线（单独文件）"""
        if not self.epochs:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 准确率曲线
        ax1.plot(self.epochs, self.train_accuracies, 'b-', label='Training Accuracy', linewidth=2, alpha=0.8, marker='o', markersize=4)
        ax1.plot(self.epochs, self.val_accuracies, 'r-', label='Validation Accuracy', linewidth=2, alpha=0.8, marker='s', markersize=4)
        ax1.set_title('Accuracy Curves', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
        
        # 添加最佳准确率信息
        max_train_acc = max(self.train_accuracies)
        max_val_acc = max(self.val_accuracies)
        ax1.text(0.02, 0.98, f'Max Train Acc: {max_train_acc:.4f}\nMax Val Acc: {max_val_acc:.4f}', 
                transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # F1分数曲线
        ax2.plot(self.epochs, self.train_f1_scores, 'b-', label='Training F1-Score', linewidth=2, alpha=0.8, marker='o', markersize=4)
        ax2.plot(self.epochs, self.val_f1_scores, 'r-', label='Validation F1-Score', linewidth=2, alpha=0.8, marker='s', markersize=4)
        ax2.set_title('F1-Score Curves', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('F1-Score', fontsize=12)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        
        # 添加最佳F1信息
        max_train_f1 = max(self.train_f1_scores)
        max_val_f1 = max(self.val_f1_scores)
        ax2.text(0.02, 0.98, f'Max Train F1: {max_train_f1:.4f}\nMax Val F1: {max_val_f1:.4f}', 
                transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.suptitle(f'Accuracy & F1-Score - Dataset: {self.dataset_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.cfg.train_dir, f'accuracy_f1_curves_{self.dataset_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"📊 Accuracy & F1 curves saved: {save_path}")
        
    def plot_training_summary(self):
        """🔧 绘制训练总结图（包含关键信息）"""
        if not self.epochs:
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 损失对比
        ax1.plot(self.epochs, self.train_losses, 'b-', label='Training', linewidth=2, alpha=0.8)
        ax1.plot(self.epochs, self.val_losses, 'r-', label='Validation', linewidth=2, alpha=0.8)
        ax1.set_title('Loss Comparison', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 准确率对比
        ax2.plot(self.epochs, self.train_accuracies, 'b-', label='Training', linewidth=2, alpha=0.8)
        ax2.plot(self.epochs, self.val_accuracies, 'r-', label='Validation', linewidth=2, alpha=0.8)
        ax2.set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        
        # F1分数对比
        ax3.plot(self.epochs, self.train_f1_scores, 'b-', label='Training', linewidth=2, alpha=0.8)
        ax3.plot(self.epochs, self.val_f1_scores, 'r-', label='Validation', linewidth=2, alpha=0.8)
        ax3.set_title('F1-Score Comparison', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('F1-Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1])
        
        # 实验信息
        ax4.axis('off')
        info_text = f"""EXPERIMENT SUMMARY
        
Dataset: {self.dataset_name}
Total Epochs: {len(self.epochs)}
        
HYPERPARAMETERS:
Learning Rate: {self.cfg.lr}
Batch Size: {self.cfg.batch_size_per_gpu}
Regularization: {self.cfg.lmd}
Early Stop Patience: {self.cfg.early_stop_patience}
        
BEST RESULTS:
Best Val F1: {max(self.val_f1_scores):.4f}
Best Val Acc: {max(self.val_accuracies):.4f}
Min Val Loss: {min(self.val_losses):.4f}
        
STATUS:
{'Early Stopped' if self.early_stopping.early_stop else 'Completed'}
"""
        ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle(f'Training Summary - {self.dataset_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.cfg.train_dir, f'training_summary_{self.dataset_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"📊 Training summary saved: {save_path}")

    def plot_training_curves(self):
        """🔧 绘制所有训练曲线"""
        print(f"📊 Generating detailed training curves...")
        self.plot_loss_curves()
        self.plot_accuracy_f1_curves() 
        self.plot_training_summary()

    def train(self):
        # 🔧 输出完整的实验配置
        print("=" * 80)
        print("🚀 ENHANCED ADVERSARIAL REPROGRAMMING EXPERIMENT")
        print("=" * 80)
        print(f"📊 Dataset: {self.dataset_name}")
        print(f"📊 Loss Function: {self.loss_type.upper()}")
        print(f"📊 Network: {self.cfg.net}")
        print(f"📊 Batch Size: {self.cfg.batch_size_per_gpu}")
        print(f"📊 Learning Rate: {self.cfg.lr}")
        print(f"📊 Max Epochs: {self.cfg.max_epoch}")
        
        if hasattr(self.cfg, 'use_crossentropy'):
            print(f"📊 CrossEntropy Enabled: {self.cfg.use_crossentropy}")
        
        print("=" * 80)
        
        # 🔧 记录超参数
        self.log_hyperparameters()
        
        try:
            for self.epoch in range(self.start_epoch, self.cfg.max_epoch + 1):
                # 🔧 使用统一的GPU缓存管理
                if self.use_gpu:
                    torch.cuda.empty_cache()
                preds = []
                labels = []
                total_loss = 0  # 初始化总损失
                print()
                print('-' * 100)
                print(f'🚀 epoch: {self.epoch:03d}/{self.cfg.max_epoch:03d} | 数据集: {self.dataset_name}')
                current_lr = self.lr_scheduler._get_lr(self.epoch)[0]
                print(f'📊 Learning Rate: {current_lr:.6f}')
                
                progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
                for j, data in progress_bar:
                    image = data["vector"]
                    label = data["targets"]
                    image = self.tensor2var(image)
                    
                    # 🔧 使用统一的GPU设备管理
                    if self.use_gpu:
                        label = label.cuda(self.gpu_id)
                    
                    self.out = self.Program(image)
                    self.loss = self.compute_loss(self.out, label)
                    self.optimizer.zero_grad()
                    self.loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, self.Program.parameters()),
                        max_norm=1.0
                    )
                    self.optimizer.step()
                    # 累加损失
                    total_loss += self.loss.item()
                    pred = self.out.data.cpu().numpy().argmax(1)
                    preds += pred.tolist()
                    labels += label.tolist()
                    # 更新进度条显示
                    progress_bar.set_description(f'Loss: {self.loss.item():.4f}, LR: {current_lr:.6f}')
                    
                print(len(self.train_loader))
                # 记录损失到 TensorBoard
                avg_loss = total_loss / len(self.train_loader)  # 计算平均损失
                self.writer.add_scalar('Loss/train', avg_loss, self.epoch)  # 记录损失
                
                # 🔧 显示训练集指标 - 增强显示效果
                print()
                print("=" * 50 + " TRAINING METRICS " + "=" * 50)
                print('📊 Average Training Loss: %.6f' % (avg_loss))
                print("🎯 Training Performance:", end="  ")
                train_accuracy,train_precision,train_f1_score,train_recall=self.eval(labels, preds)
                
                # 添加到TensorBoard
                self.writer2.add_scalars('Metrics/train', {
                    'Accuracy': train_accuracy,
                    'Precision': train_precision,
                    'F1_score': train_f1_score,
                    'Recall': train_recall
                }, self.epoch)
                
                self.log_metrics(self.epoch, train_accuracy, train_precision, train_f1_score, train_recall,'train',loss=avg_loss)
                
                self.writer.add_scalar('learning rate',self.lr_scheduler._get_lr(self.epoch)[0],self.epoch)
                torch.save({'W': self.get_W}, os.path.join(self.cfg.train_dir, 'W_%03d.pt' % self.epoch))
                
                # 🔧 验证阶段
                print()
                print("=" * 50 + " VALIDATION METRICS " + "=" * 50)
                val_accuracy,val_precision,val_f1_score,val_recall,avg_loss2=self.validate()
                
                self.writer2.add_scalars('Metrics/validate', {
                    'Accuracy': val_accuracy,
                    'Precision': val_precision,
                    'F1_score': val_f1_score,
                    'Recall': val_recall
                }, self.epoch)
                
                self.log_metrics(self.epoch, val_accuracy, val_precision, val_f1_score, val_recall,'test',loss=avg_loss2)
                
                # 🔧 记录训练历史（用于绘图）
                self.epochs.append(self.epoch)
                self.train_losses.append(avg_loss)
                self.train_accuracies.append(train_accuracy)
                self.train_f1_scores.append(train_f1_score)
                self.val_losses.append(avg_loss2)
                self.val_accuracies.append(val_accuracy)
                self.val_f1_scores.append(val_f1_score)
                
                # 保存最佳模型
                if val_f1_score > self.best_f1:
                    self.best_f1 = val_f1_score
                    model_path = os.path.join(self.cfg.train_dir,'W_%03dbestmodel.pt' %self.epoch)
                    torch.save({'W': self.get_W}, model_path)
                    print(f"🎯 Best model saved at epoch {self.epoch}, F1-score: {val_f1_score:.5f}")
                
                # 🔧 早停检查
                if self.early_stopping(val_f1_score):
                    print()
                    print("🛑 " + "=" * 20 + " EARLY STOPPING TRIGGERED " + "=" * 20)
                    print(f"   Val {self.cfg.early_stop_metric.upper()} not improved for {self.cfg.early_stop_patience} epochs")
                    print(f"   Current best {self.cfg.early_stop_metric.upper()}: {self.best_f1:.5f}")
                    print(f"   Training stopped at epoch {self.epoch}")
                    print("=" * 65)
                    break
                
                # 🔧 对比显示训练集和验证集指标
                print()
                print("=" * 50 + " EPOCH SUMMARY " + "=" * 50)
                print(f"📊 Epoch {self.epoch:03d} Summary (Dataset: {self.dataset_name}):")
                print(f"   Training   -> Acc: {train_accuracy:.5f}, P: {train_precision:.5f}, F1: {train_f1_score:.5f}, R: {train_recall:.5f}, Loss: {avg_loss:.4f}")
                print(f"   Validation -> Acc: {val_accuracy:.5f}, P: {val_precision:.5f}, F1: {val_f1_score:.5f}, R: {val_recall:.5f}, Loss: {avg_loss2:.4f}")
                print(f"   Best F1: {self.best_f1:.5f}")
                print(f"   Early Stop Wait: {self.early_stopping.wait}/{self.cfg.early_stop_patience}")
                print("=" * 115)
                
                self.lr_scheduler.step(self.epoch)
        finally:
            # 无论训练如何结束，都尝试执行以下操作
            print()
            print("=" * 50 + " FINALIZING EXPERIMENT " + "=" * 50)

            # 检查是否有数据可供绘图
            if self.epochs:
                print("📊 Generating visualization from collected training data...")
                self.plot_training_curves()
            else:
                print("⚠️ No epochs were completed. Skipping visualization.")

            # 关闭 TensorBoard writer
            print("✍️ Closing TensorBoard writers...")
            self.writer.close()
            self.writer2.close()
            
            # 输出最终总结
            final_status = "Early Stopped" if self.early_stopping.early_stop else "Completed"
            print()
            print(f"🎉 Enhanced adversarial reprogramming experiment {final_status}!")
            if self.epochs:
                print(f"📊 Experiment config: Dataset={self.dataset_name}")
                print(f"🏆 Best F1-score: {self.best_f1:.5f}")
                print(f"📈 Total training epochs: {len(self.epochs)}")
                if self.early_stopping.early_stop:
                    print(f"⏱️  Early stopping triggered at epoch {self.epoch}")
            else:
                print("No training was completed.")
            print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Enhanced Adversarial Reprogramming with Visualization and Early Stopping')
    parser.add_argument('-d', '--dataset', default='megavul', type=str, 
                        choices=['d2a','megavul', 'reveal1', 'bigvul', 'devign', 'primevul'],
                        help='数据集选择 (megavul, reveal1, bigvul, devign, primevul)')
    parser.add_argument('-r', '--restore', default=None, type=str,
                        help='从指定检查点文件路径恢复训练（例如：/path/to/W_010.pt）')
    parser.add_argument('-g', '--gpu', default=['0'], nargs='+', type=str, help='指定GPU设备ID')
    
    args = parser.parse_args()
    
    # 打印实验配置
    print("🚀 启动增强版对抗重编程实验")
    print("=" * 80)
    print(f"📊 实验配置:")
    print(f"   数据集: {args.dataset}")
    print(f"   GPU设备: {args.gpu}")
    if args.restore:
        print(f"   恢复检查点: {args.restore}")
    print("=" * 80)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)
    AR = Adversarial_Reprogramming(args)
    AR.train()


if __name__ == "__main__":
    main()
