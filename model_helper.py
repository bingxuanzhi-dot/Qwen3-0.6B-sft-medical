import torch
import os
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MAX_LENGTH, MODEL_ID, MODEL_PATH

def get_device_and_dtype():
    """
    自动判断当前环境支持的设备和精度
    """
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        device_map = "auto"
        print(f"✨ 检测到 GPU，将使用设备: {device}, 精度: {dtype}")
    else:
        device = "cpu"
        dtype = torch.float32  # CPU 必须用 float32，否则报错
        device_map = None      # CPU 模式下 device_map 设为 None
        print(f"🐢 未检测到 GPU，将使用设备: {device}, 精度: {dtype}")
    
    return device, dtype, device_map

def load_model_tokenizer():
    """
    下载(如不存在)并加载模型
    """
    # 1. 如果本地没有模型，先下载
    if not os.path.exists(MODEL_PATH):
        print(f"⬇️ 本地未找到模型，正在从 ModelScope 下载: {MODEL_ID} ...")
        try:
            snapshot_download(MODEL_ID, cache_dir="./", local_dir=MODEL_PATH)
        except Exception as e:
            print(f"❌ 下载失败！请检查 '{MODEL_ID}' 是否存在于 ModelScope。")
            raise e
    else:
        print(f"✅ 检测到本地模型文件: {MODEL_PATH}")

    # 2. 获取设备配置
    device, dtype, device_map = get_device_and_dtype()
    
    print("📂 正在加载模型权重...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, 
            use_fast=False, 
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, 
            device_map=device_map, 
            torch_dtype=dtype,
            trust_remote_code=True
        )
        
        if dtype != torch.float32:
            model.enable_input_require_grads() 
            
        print("✅ 模型加载成功！")
        return model, tokenizer
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        raise e

def predict(messages, model, tokenizer):
    """
    推理函数
    """
    device = model.device
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=MAX_LENGTH,
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response
