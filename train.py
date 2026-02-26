import os
import torch
# import swanlab
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq

import config
from data_helper import download_and_split_data, dataset_jsonl_transfer, load_and_process_data
from model_helper import load_model_tokenizer, predict

def main():
    print("🚀 初始化训练流程...")

    use_gpu = torch.cuda.is_available()
    
    # 1. 初始化 SwanLab
    # swanlab.init(
    #     project=config.SWANLAB_PROJECT,
    #     mode="local",  # 开启离线模式，日志只保存在本地，不上传
    #     config={
    #         "model_id": config.MODEL_ID,
    #         "device": "cuda" if use_gpu else "cpu"
    #     }
    # )


    # 2. 准备数据 (下载 -> 切分 -> 格式化)
    download_and_split_data()
    
    if not os.path.exists(config.TRAIN_FORMAT_FILE):
        dataset_jsonl_transfer(config.TRAIN_FILE, config.TRAIN_FORMAT_FILE)
    if not os.path.exists(config.VAL_FORMAT_FILE):
        dataset_jsonl_transfer(config.VAL_FILE, config.VAL_FORMAT_FILE)

    # 3. 加载模型 (下载 -> 加载)
    model, tokenizer = load_model_tokenizer()

    # 4. 处理数据集
    print("⏳ 正在 Tokenize 数据集...")
    train_dataset, _ = load_and_process_data(config.TRAIN_FORMAT_FILE, tokenizer)
    eval_dataset, test_df = load_and_process_data(config.VAL_FORMAT_FILE, tokenizer)

    # 5. 设置训练参数
    args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM_STEPS,
        eval_strategy="steps",
        logging_steps=config.LOGGING_STEPS,
        num_train_epochs=config.NUM_EPOCHS,
        save_steps=config.SAVE_STEPS,
        learning_rate=config.LEARNING_RATE,
        save_on_each_node=True,
        gradient_checkpointing=True if use_gpu else False, 
        report_to="none",
        run_name=config.SWANLAB_RUN_NAME,
        
        # =========== 关键：CPU/GPU 自动切换 ===========
        fp16=(use_gpu and not torch.cuda.is_bf16_supported()), 
        bf16=(use_gpu and torch.cuda.is_bf16_supported()),
        use_cpu=(not use_gpu),
        dataloader_num_workers=4,
        # ============================================
    )

    # 6. 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    # 7. 开始训练
    print(f"🔥 开始训练 (模式: {'GPU 🚀' if use_gpu else 'CPU 🐢'})...")
    trainer.train()

    # 8. 测试
    print("📝 训练完成，生成测试结果...")
    test_samples = test_df[:3]
    test_text_list = []

    for index, row in test_samples.iterrows():
        instruction = row['instruction']
        input_value = row['input']
        messages = [
            {"role": "system", "content": f"{instruction}"},
            {"role": "user", "content": f"{input_value}"}
        ]
        response = predict(messages, model, tokenizer)
        response_text = f"Question: {input_value}\n\nLLM Response:\n{response}"
        print("-" * 50)
        print(response_text)
        # test_text_list.append(swanlab.Text(response_text))

    # swanlab.log({"Prediction": test_text_list})
    # swanlab.finish()
    print("✅ 任务结束！")

if __name__ == "__main__":
    main()
