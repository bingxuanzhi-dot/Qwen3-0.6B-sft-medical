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

2. 运行训练
