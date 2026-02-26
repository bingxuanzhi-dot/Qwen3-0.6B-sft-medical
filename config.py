import os

# ================= 项目配置 =================
SWANLAB_PROJECT = "qwen3-sft-medical"
SWANLAB_RUN_NAME = "qwen3-0.6B-finetune"

# ================= 模型配置 =================
MODEL_ID = "Qwen/Qwen3-0.6B" 
MODEL_PATH = "./Qwen/Qwen3-0.6B"

# ================= 数据配置 =================
DATASET_ID = "krisfu/delicate_medical_r1_data"

PROMPT = "你是一个医学专家, 你需要根据用户的问题, 给出带有思考的过程。"
MAX_LENGTH = 2048

TRAIN_FILE = "train.jsonl"
VAL_FILE = "val.jsonl"
TRAIN_FORMAT_FILE = "train_format.jsonl"
VAL_FORMAT_FILE = "val_format.jsonl"

# ================= 训练参数 =================
OUTPUT_DIR = "./output_qwen"
LEARNING_RATE = 1e-4
NUM_EPOCHS = 2
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4
SAVE_STEPS = 400
LOGGING_STEPS = 10
