# Qwen3-0.6B Medical Finetuning (Qwen3-0.6B 医疗问答微调)

这是一个基于 **[Qwen3-0.6B](https://modelscope.cn/models/Qwen/Qwen3-0.6B)** 小参数语言模型进行的医疗领域微调项目。

本项目旨在运行医疗问答模型的微调训练，支持自动数据下载、自动模型下载。

## 🚀 快速开始 (Quick Start)

### 1. 环境准备
```bash
# 建议使用 python 3.9+
conda create -n qwen_finetuning python=3.9
conda activate qwen_finetuning
pip install swanlab modelscope==1.22.0 "transformers>=4.50.0" datasets==3.2.0 accelerate pandas addict
```

### 2. 运行训练
直接运行主程序即可：
```bash
python train.py
```
* **首次运行**：代码会自动下载约 1GB 的模型文件和数据集，请保持网络畅通。
* **GPU 用户**：训练速度快，显存占用低。
* **CPU 用户**：训练速度较慢，但在任何机器上都能跑通流程。

## 📂 文件说明 (File Structure)
| 文件名 | 说明 |
| :--- | :--- |
| `config.py` | **配置文件**：模型路径、数据集 ID、超参数 (Batch Size, LR) 等 |
| `train.py` | **主程序**：负责初始化 Trainer，启动训练循环 |
| `data_helper.py` | **数据处理**：自动下载数据集，清洗并转换为 ChatML 格式 |
| `model_helper.py` | **模型工具**：自动下载模型，智能判断设备 (CPU/GPU) 并加载权重 |

