# AdvRVD: Enhanced Adversarial Reprogramming for Vulnerability Detection

This repository contains the official PyTorch implementation for **AdvRVD**, a framework that applies Enhanced Adversarial Reprogramming to adapt pre-trained image models for the task of Source Code Vulnerability Detection. 


## 📊 Datasets & Distribution

To ensure robustness, the model is evaluated on three prominent vulnerability datasets. We have strictly applied a **Stratified 8:1:1 Split (Train : Validation : Test)** to maintain an identical ratio of vulnerable vs. non-vulnerable samples across all subsets.


*All dataset partitions are provided in standard CSV formats within the `datasets/` directory.*

## ⚙️ Data Preprocessing Pipeline

Our data processing is fully automated and modularized under the `precess/` directory.

- `split_dataset.py`: Implements Stratified 8:1:1 Sampling, slicing the raw vulnerability dataset into `train`, `val`, and `test` CSVs while preserving exact positive/negative label ratios.
- `step0_preprocess.py`: Extracts raw C code according to the 8:1:1 CSV splits into structured `.c` source code files.
- `step1_normalization.py`: Normalizes the source code (removing comments, standardizing formats).
- `step2_joern_graph_gen.py`: Generates Code Property Graphs (CPG, PDG) using Joern.
- `step3_train_sent2vec.py`: Trains node embeddings.
- `step4_ImageGeneration.py`: Converts code graphs/embeddings into spatial image representations.
- `step4.5_ChannelsGeneration.py`: Further enhances image channels for the adversarial framework.
- `step5_generate_train_test_data.py`: Assembles the final `.pkl` tensors for fast GPU loading.

## 🚀 Training & Evaluation

All hyper-parameters (such as batch size, learning rate, lambda for regularization, etc.) and GPU device allocations correspond to configurations globally set in `config.py` (and dynamically scaled via `config_{dataset_name}.py`). 

### Core Arguments for Scripts
- `-d` / `--dataset`: Specify the dataset name (`d2a`, `devign`, or `reveal`).
- `-g` / `--gpu`: Specify the GPU device ID(s) to use (e.g., `0`, or `0 1`).
- `-r` / `--restore`: Provide a path to a `.pt` checkpoint file to resume training or perform evaluation.

### Example Commands
Wait for the completion of the data preprocessing pipeline.
Build and train the core mechanism using `main.py`:
```bash
python main.py --dataset megavul  # or d2a / devign / reveal
```

Train with specific GPU mapping:
```bash
python main.py --dataset d2a -g 0
```

Resume training from a specific checkpoint:
```bash
python main.py --dataset devign --restore ./train_log_devign/W_05.pt
```

To specifically evaluate a checkpoint or measure metrics on the test set:
```bash
python evaluate.py --dataset reveal --restore ./train_log_reveal/W_best.pt
```

## 📦 Requirements

The essential dependencies carefully extracted from our environment are:
- **Python** (>= 3.8)
- **PyTorch** (`torch>=2.9.0`, `torchvision>=0.24.0`)
- **Graph & Embeddings**: `dgl>=2.1.0`, `cogdl>=0.6`, `gensim>=4.4.0`
- **Deep Learning / NLP**: `transformers>=4.51.3`, `timm>=1.0.15`
- **Data Science**: `scikit-learn>=1.5.2`, `pandas>=2.2.3`, `numpy>=1.26.4`, `networkx>=3.5`
- **Imbalance Handling**: `imbalanced-learn>=0.12.4`
- **Parsing**: `tree-sitter>=0.20.4`
- **Joern** (for structural Code Property Graph extraction)

> *Note: Please see `requirements.txt` for the full comprehensive list of environment packages.*
