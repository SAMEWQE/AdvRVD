import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np
import os
import timm


class Program(nn.Module):
    def __init__(self, cfg, gpu):
        super(Program, self).__init__()
        self.cfg = cfg
        self.gpu = gpu
        self.init_net()
        self.init_mask()
        self.W = Parameter(torch.randn(self.M.shape), requires_grad=True)

        # 标记labels已经按照从高到低排序
        self._labels_already_sorted_desc = True

        print("Initializing smart label mapping for vulnerability detection...")
        print("Using blank image to determine highest scored ImageNet categories...")
        print("Using global perturbation mask: the full canvas is trainable.")

        # 设置为漏洞检测专用配置
        if not hasattr(self.cfg, 'n_classes'):
            self.cfg.n_classes = 2  # 二分类：有无漏洞
        if not hasattr(self.cfg, 'm_per_class'):
            self.cfg.m_per_class = 10  # 每类分配10个ImageNet标签

        # 显示映射配置
        use_smart = getattr(self.cfg, 'use_smart_mapping', False)
        reduction = getattr(self.cfg, 'mapping_reduction', 'mean')
        print(f"\n=== Mapping Configuration ===")
        print(f"  Use smart mapping: {use_smart}")
        print(f"  Reduction method: {reduction}")
        print(f"  Classes: {self.cfg.n_classes}")
        print(f"  Labels per class: {self.cfg.m_per_class}")
        print(f"==============================\n")

        self.image_net_labels = self.get_imagenet_label_list(self.net, None, self.cfg.w1)
        self.class_mapping = self.create_label_mapping(
            self.cfg.n_classes, self.cfg.m_per_class, self.image_net_labels
        )

        # 验证映射质量
        self.validate_mapping_detailed()

        print("Program initialization completed with vulnerability-specific smart mapping!")

    def init_net(self):
        if self.cfg.net == 'vit_base_patch16_384':
            print("Loading pretrained vit_base_patch16_384 model ......waiting ")
            self.net = timm.create_model("vit_base_patch16_384", pretrained=True)
            img_mean = (0.485, 0.456, 0.406)
            img_std = (0.229, 0.224, 0.225)
            self.mean = torch.tensor(img_mean)[None, :, None, None]
            self.std = torch.tensor(img_std)[None, :, None, None]
        elif self.cfg.net == 'vit_large_patch16_384':
            print("Loading pretrained vit_large_patch16_384 model ......waiting ")
            self.net = timm.create_model("vit_large_patch16_384", pretrained=True)
            img_mean = (0.485, 0.456, 0.406)
            img_std = (0.229, 0.224, 0.225)
            self.mean = torch.tensor(img_mean)[None, :, None, None]
            self.std = torch.tensor(img_std)[None, :, None, None]
        elif self.cfg.net == 'vit_large_patch14_clip_336':
            print("Loading pretrained vit_large_patch14_clip_336 model ......waiting ")
            self.net = timm.create_model("vit_large_patch14_clip_336", pretrained=True)
            img_mean = (0.485, 0.456, 0.406)
            img_std = (0.229, 0.224, 0.225)
            self.mean = torch.tensor(img_mean)[None, :, None, None]
            self.std = torch.tensor(img_std)[None, :, None, None]
        elif self.cfg.net == 'beit_large_patch16_512':
            print("Loading pretrained beit_large_patch16_512 model ......waiting ")
            self.net = timm.create_model("beit_large_patch16_512", pretrained=True)
            img_mean = (0.485, 0.456, 0.406)
            img_std = (0.229, 0.224, 0.225)
            self.mean = torch.tensor(img_mean)[None, :, None, None]
            self.std = torch.tensor(img_std)[None, :, None, None]
        elif self.cfg.net == 'eva02_large_patch14_448':
            print("Loading pretrained eva02_large_patch14_448 model ......waiting ")
            self.net = timm.create_model("eva02_large_patch14_448", pretrained=True)
            img_mean = (0.485, 0.456, 0.406)
            img_std = (0.229, 0.224, 0.225)
            self.mean = torch.tensor(img_mean)[None, :, None, None]
            self.std = torch.tensor(img_std)[None, :, None, None]
        elif self.cfg.net == 'resnet50':
            print("Loading pretrained resnet50 model ......waiting ")
            self.net = timm.create_model("resnet50", pretrained=True)
            img_mean = (0.485, 0.456, 0.406)
            img_std = (0.229, 0.224, 0.225)
            self.mean = torch.tensor(img_mean)[None, :, None, None]
            self.std = torch.tensor(img_std)[None, :, None, None]
        elif self.cfg.net == 'tf_efficientnet_b7':
            print("Loading pretrained tf_efficientnet_b7 model ......waiting ")
            self.net = timm.create_model("tf_efficientnet_b7", pretrained=True)
            img_mean = (0.485, 0.456, 0.406)
            img_std = (0.229, 0.224, 0.225)
            self.mean = torch.tensor(img_mean)[None, :, None, None]
            self.std = torch.tensor(img_std)[None, :, None, None]
        elif self.cfg.net == 'tf_efficientnet_b4':
            print("Loading pretrained tf_efficientnet_b4 model ......waiting ")
            self.net = timm.create_model("tf_efficientnet_b4", pretrained=True)
            img_mean = (0.485, 0.456, 0.406)
            img_std = (0.229, 0.224, 0.225)
            self.mean = torch.tensor(img_mean)[None, :, None, None]
            self.std = torch.tensor(img_std)[None, :, None, None]
        elif self.cfg.net == 'inception_v3':
            print("Loading pretrained inception_v3 model ......waiting ")
            self.net = timm.create_model("inception_v3", pretrained=True)
            img_mean = (0.485, 0.456, 0.406)
            img_std = (0.229, 0.224, 0.225)
            self.mean = torch.tensor(img_mean)[None, :, None, None]
            self.std = torch.tensor(img_std)[None, :, None, None]
        elif self.cfg.net == 'swin_base_patch4_window12_384':
            print("Loading pretrained swin_base_patch4_window12_384 model ......waiting ")
            self.net = timm.create_model("swin_base_patch4_window12_384", pretrained=True)
            img_mean = (0.485, 0.456, 0.406)
            img_std = (0.229, 0.224, 0.225)
            self.mean = torch.tensor(img_mean)[None, :, None, None]
            self.std = torch.tensor(img_std)[None, :, None, None]
        else:
            raise ValueError(f"Unsupported network: {self.cfg.net}. Please check the model name in your config.")
        self.net.eval()
        for param in self.net.parameters():
            param.requires_grad = False
        print("Vision model Frozen!")

    def init_mask(self):
        # Global mode: make every position trainable instead of masking only the border.
        M = torch.ones(3, self.cfg.h1, self.cfg.w1)
        self.M = Parameter(M, requires_grad=False)

    def imagenet_label2_mnist_label(self, imagenet_label):
        return imagenet_label[:, :2]

    def get_mapped_logits(self, logits, class_mapping, multi_label_remapper):
        """
        logits : Tensor of shape (batch_size, 1000) # imagenet class logits
        class_mapping: class_mapping[i] = list of image net labels for text class i
        reduction : max or mean (从配置读取)
        """
        if multi_label_remapper is None:
            reduction = getattr(self.cfg, 'mapping_reduction', 'mean')

            mapped_logits = []
            for class_no in range(len(class_mapping)):
                if reduction == "max":
                    class_logits, _ = torch.max(logits[:, class_mapping[class_no]], dim=1)
                elif reduction == "mean":
                    class_logits = torch.mean(logits[:, class_mapping[class_no]], dim=1)
                else:
                    raise ValueError(f"Unknown reduction method: {reduction}. Use 'max' or 'mean'.")

                mapped_logits.append(class_logits)
            return torch.stack(mapped_logits, dim=1)
        else:
            orig_prob_scores = nn.Softmax(dim=-1)(logits)
            mapped_logits = multi_label_remapper(orig_prob_scores)
            return mapped_logits

    def create_label_mapping(self, n_classes, m_per_class, image_net_labels=None):
        if image_net_labels is None:
            image_net_labels = range(1000)

        if hasattr(self, '_labels_already_sorted_desc'):
            sorted_labels = list(image_net_labels)
        else:
            sorted_labels = list(reversed(image_net_labels))

        print(f"Creating smart label mapping with {n_classes} classes, {m_per_class} labels per class")
        print(f"Using top scored ImageNet labels: {sorted_labels[:20]}...")

        class_mapping = [[] for i in range(n_classes)]
        for i in range(n_classes * m_per_class):
            target_class = i % n_classes
            if i < len(sorted_labels):
                class_mapping[target_class].append(sorted_labels[i])
            else:
                class_mapping[target_class].append(sorted_labels[i % len(sorted_labels)])

        print("\n=== Round-Robin Label Mapping ===")
        for class_id, mapping in enumerate(class_mapping):
            class_name = "Non-vulnerable" if class_id == 0 else "Vulnerable"
            print(f"Class {class_id} ({class_name}):")
            print(f"  Labels: {mapping}")
        print("==================================\n")
        return class_mapping

    def get_imagenet_label_list(self, vision_model, base_image, img_size):
        if base_image is None:
            base_image = torch.zeros(3, img_size, img_size).to("cuda")
            base_image = base_image.type(torch.FloatTensor)
            print("Using blank (zero) image as base for label mapping...")

        print("Getting ImageNet predictions for smart label mapping...")
        with torch.no_grad():
            logits = vision_model(base_image[None])[0]
            sorted_scores, sorted_indices = torch.sort(logits, descending=True)
            label_list = sorted_indices.detach().cpu().numpy().tolist()

        print(f"Top 10 highest scored ImageNet labels: {label_list[:10]}")
        print(f"Top 10 scores: {sorted_scores[:10].detach().cpu().numpy().round(3)}")
        print(f"Top 20 highest scored ImageNet labels: {label_list[:20]}")
        print(f"Top 20 scores: {sorted_scores[:20].detach().cpu().numpy().round(3)}")

        return label_list

    def forward(self, image):
        X = image.data.new(self.cfg.batch_size_per_gpu, 3, self.cfg.h1, self.cfg.w1)
        X[:] = 0
        X[:, :, int((self.cfg.h1 - self.cfg.h2) // 2):int((self.cfg.h1 + self.cfg.h2) // 2),
        int((self.cfg.w1 - self.cfg.w2) // 2):int((self.cfg.w1 + self.cfg.w2) // 2)] = image.data.clone()
        X = Variable(X, requires_grad=True)
        P = self.W * self.M
        X_adv = X + torch.tanh(P)
        self.mean = self.mean.to(X_adv.device)
        self.std = self.std.to(X_adv.device)
        X_adv = (X_adv - self.mean) / self.std
        X_adv = X_adv.type(torch.cuda.FloatTensor)
        Y_adv = self.net(X_adv)
        Y_adv = F.softmax(Y_adv, 1)

        if hasattr(self.cfg, 'use_smart_mapping') and self.cfg.use_smart_mapping:
            out = self.get_mapped_logits(Y_adv, self.class_mapping, None)
        else:
            out = self.imagenet_label2_mnist_label(Y_adv)

        return out

    def validate_mapping(self):
        print("\n=== Validating Label Mapping Quality ===")

        torch.manual_seed(42)
        base_image = 2 * torch.rand(3, self.cfg.w1, self.cfg.w1).to("cuda") - 1.0

        with torch.no_grad():
            logits = self.net(base_image[None])[0]

        for class_id, mapping in enumerate(self.class_mapping):
            scores = [logits[label_idx].item() for label_idx in mapping]
            print(f"Class {class_id}: labels {mapping[:3]}..., scores {[round(s, 3) for s in scores[:3]]}...")
            print(f"  Score range: {min(scores):.3f} to {max(scores):.3f}")

        print("==========================================\n")

    def validate_mapping_detailed(self):
        print("\n=== Detailed Label Mapping Validation ===")

        base_image = torch.zeros(3, self.cfg.w1, self.cfg.w1).to("cuda")
        base_image = base_image.type(torch.FloatTensor)

        with torch.no_grad():
            logits = self.net(base_image[None])[0]

        print("Mapping validation using blank image:")
        for class_id, mapping in enumerate(self.class_mapping):
            scores = [logits[label_idx].item() for label_idx in mapping]
            class_name = "Non-vulnerable" if class_id == 0 else "Vulnerable"

            print(f"\nClass {class_id} ({class_name}):")
            print(f"  ImageNet labels: {mapping}")
            print(f"  Corresponding scores: {[round(s, 4) for s in scores]}")
            print(f"  Score statistics:")
            print(f"    - Mean: {np.mean(scores):.4f}")
            print(f"    - Max:  {max(scores):.4f}")
            print(f"    - Min:  {min(scores):.4f}")
            print(f"    - Std:  {np.std(scores):.4f}")

        print("\n==========================================\n")
