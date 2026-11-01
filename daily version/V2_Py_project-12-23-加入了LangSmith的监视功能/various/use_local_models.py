# 以下程序用于使用本地模型模块测试,用的是ybw_AgentRAG环境
# 这个也得用三张卡，2张会报错 CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`
# 3张卡依旧非常慢，不考虑使用本地部署了
# 2025-12-17 15:58

import torch
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from langchain_huggingface import HuggingFacePipeline

# ================= 配置区域 =================
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"

MODEL_ROOT_DIR = "/home/lab/HDD_data_slow/models"

LOCAL_MODEL_MAP = {
    "qwen3-8B": "Qwen3-8B",
    "llama3.1-8b-instruct": "llama3.1-8b-instruct",
    "llama3.1-70b-instruct": "llama3.1-70b-instruct",
}

# 切换你想测试的模型
CURRENT_MODEL_KEY = "llama3.1-70b-instruct" 

# ================= 模型加载函数 =================
def get_response_model(model_key):
    model_dir_name = LOCAL_MODEL_MAP[model_key]
    model_path = os.path.join(MODEL_ROOT_DIR, model_dir_name)
    
    print(f"🚀 Initializing Local Model: {model_key}")
    print(f"📂 Path: {model_path}")
    print(f"✅ Visible GPU count: {torch.cuda.device_count()}")

    #4-bit NF4 量化
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, # 3080支持bf16，也可以改成 torch.bfloat16
        bnb_4bit_use_double_quant=True,
    )
    model_kwargs = {"quantization_config": quantization_config}

    # 1. 加载 Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer from {model_path}. Error: {e}")

    # 2. 加载 Model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto", 
            trust_remote_code=True,
            **model_kwargs 
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}. Error: {e}")

    # 3. 创建 Pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=4096,
        temperature=0.1, 
        do_sample=True,
        repetition_penalty=1.1,
        return_full_text=False # 只返回生成的答案，不包含 Prompt
    )

    # 4. 封装为 LangChain 对象
    return HuggingFacePipeline(pipeline=pipe), tokenizer

# 获取模型和分词器 (分词器用于处理 Chat Template)
RESPONSE_MODEL, TOKENIZER = get_response_model(CURRENT_MODEL_KEY)

# ================= 交互测试循环 =================
print(f"\n✨ 模型加载完成！当前模型: {CURRENT_MODEL_KEY}")
print("--------------------------------------------------")

while True:
    try:
        user_input = input("\n👤 User: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("👋 程序结束")
            break
        if not user_input: continue
        
        # 1. 构造消息列表
        messages = [{"role": "user", "content": user_input}]
        
        # 2. 🔥 关键步骤：应用 Chat Template
        # 这会将 [{"role": "user"}] 转换为 Qwen/Llama 特定的 Prompt 字符串
        # 例如: "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n"
        prompt = TOKENIZER.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 3. 调用模型 (传入处理好的 String)
        print("🤖 AI Thinking...", end="", flush=True)
        response = RESPONSE_MODEL.invoke(prompt)
        
        # 4. 打印结果 (LangChain Pipeline 返回的通常是纯文本)
        print(f"\r🤖 AI: {response}")

    except KeyboardInterrupt:
        print("\n👋 用户强制中断")
        break
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")