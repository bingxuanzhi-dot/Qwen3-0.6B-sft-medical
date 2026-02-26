import json
import os
import pandas as pd
from datasets import Dataset
from modelscope.msdatasets import MsDataset
from config import PROMPT, MAX_LENGTH, DATASET_ID, TRAIN_FILE, VAL_FILE

def download_and_split_data():
    """
    如果本地没有 jsonl 文件，则从 ModelScope 下载并切分
    """
    if os.path.exists(TRAIN_FILE) and os.path.exists(VAL_FILE):
        print("✅ 检测到本地数据集文件，跳过下载。")
        return

    print(f"⬇️ 正在从 ModelScope 下载数据集: {DATASET_ID} ...")
    try:
        # 下载数据集
        ds = MsDataset.load(DATASET_ID, subset_name='default', split='train')
        data_list = list(ds)
        
        # 简单切分 9:1
        split_idx = int(len(data_list) * 0.9)
        train_data = data_list[:split_idx]
        val_data = data_list[split_idx:]

        # 保存为 jsonl
        with open(TRAIN_FILE, 'w', encoding='utf-8') as f:
            for item in train_data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        
        with open(VAL_FILE, 'w', encoding='utf-8') as f:
            for item in val_data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
                
        print(f"✅ 数据集下载并切分完成！训练集: {len(train_data)}, 验证集: {len(val_data)}")
    except Exception as e:
        print(f"❌ 数据集下载失败: {e}")
        raise e

def dataset_jsonl_transfer(origin_path, new_path):
    """
    将原始数据转换为标准格式
    """
    messages = []
    try:
        with open(origin_path, "r", encoding="utf-8") as file:
            for line in file:
                data = json.loads(line)
                input_text = data.get("question", "") or data.get("input", "")
                think_text = data.get("think", "")
                answer_text = data.get("answer", "") or data.get("output", "")
                output_text = "<think>{data['think']}</think> \n {data['answer']}"
                
                message = {
                    "instruction": PROMPT,
                    "input": input_text,
                    "output": output_text,
                }
                messages.append(message)
    except FileNotFoundError:
        print(f"❌ 找不到文件 {origin_path}")
        return

    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")
    print(f"✅ 格式化数据已保存至: {new_path}")

def process_func(example, tokenizer):
    """
    Tokenize 处理
    """
    instruction = tokenizer(
        f"<|im_start|>system\n{PROMPT}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
        
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def load_and_process_data(jsonl_path, tokenizer):
    df = pd.read_json(jsonl_path, lines=True)
    ds = Dataset.from_pandas(df)
    dataset = ds.map(process_func, fn_kwargs={"tokenizer": tokenizer}, remove_columns=ds.column_names)
    return dataset, df
