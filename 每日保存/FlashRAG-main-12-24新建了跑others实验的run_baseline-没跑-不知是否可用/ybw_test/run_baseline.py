# 跑其RAG方法的实验

import sys
import os
import time
import argparse
import torch
from flashrag.config import Config
from flashrag.utils import get_dataset
# 导入官方内置的 Pipeline
from flashrag.pipeline import SequentialPipeline, SelfRAGPipeline, REPLUGPipeline, FLAREPipeline

parser = argparse.ArgumentParser(description="Run Baseline RAG experiments")
parser.add_argument("--method", type=str, required=True, choices=['naive', 'selfrag', 'replug', 'flare'], help="Baseline method")
parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., nq, hotpotqa)")
parser.add_argument("--model", type=str, default="qwen3-8b", help="Generator model key (for API)")
parser.add_argument("--num", type=int, default=300, help="Number of samples")
parser.add_argument("--gpu", type=str, default="0", help="GPU ID")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

DASHSCOPE_API_KEY = "sk-757aab3d1cdf47d2896bfef1e45efa1d"   
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

MODEL_MAP = {
        "qwen3-8b": "qwen3-8b",
        "qwen3-32b": "qwen3-32b",
        "qwen3-next-80b-a3b-instruct": "qwen3-next-80b-a3b-instruct"
}
generator_path = MODEL_MAP.get(args.model, "qwen3-8b")

def main():
    print(f"Lanuch baseline: method=[{args.method}] | model=[{args.model}] | dataset=[{args.dataset}]")
    DATA_DIR = "/home/lab/ybw/FlashRAG-main/datasets_flashRAG"
    SAVE_DIR = f"/home/lab/ybw/FlashRAG-main/ybw_test/results_baselines/{args.method}/{args.dataset}"

    # 检索的嵌入模型
    RETRIEVER_PATH = "/home/lab/Shared_model/embeddinggemma-300m"
    INDEX_PATH = "/home/lab/ybw/Generate_VectorsDB/DB/faiss_index/gemma_300m_ivfpq.index"
    CORPUS_PATH = "/home/lab/ybw/Generate_VectorsDB/DB/raw_data/psgs_w100.jsonl"

    config_dict = {
        "device": "cuda",
        "gpu_id": "0",
        "data_dir": DATA_DIR, 
        "dataset_name": args.dataset,
        "split": ["dev"],
        "test_sample_num": args.num,
        "random_sample": True,
        "seed": 2026,
         
        #Gemma和FAISS
        "retrieval_method": "sentence_transformer",
        "model_path": RETRIEVER_PATH,
        "index_path": INDEX_PATH,
        "corpus_path": CORPUS_PATH,
        "retrieval_topk": 200,   # <--------------------
        "use_reranker": True,
        "rerank_model_path": "/home/lab/Shared_model/bge-reranker-v2-m3",
        "rerank_topk": 3,
        "use_fp16": True,
        "instruction": "Given a search query, retrieve relevant passages:",

        # 大模型
        "framework": "openai",
        "generator_model": "openai",
        "generator_model_path": generator_path,
        "openai_api_key": DASHSCOPE_API_KEY,
        "openai_api_base": DASHSCOPE_BASE_URL,
        "generation_params": {"temperature": 0.1, "max_tokens": 1024},

        "metrics": ["em", "f1", "acc"],
        "save_dir": SAVE_DIR,
        "save_intermediate_data": True
    }

    print("⚙️ 加载配置...")
    config = Config(config_dict=config_dict)
    print("📥 加载数据集...")
    dataset = get_dataset(config)["dev"]
    print(f"🛠️ 初始化 Pipeline: {args.method.upper()}...")

    if args.method == 'naive':
        # Naive RAG 就是 SequentialPipeline
        pipeline = SequentialPipeline(config)
    elif args.method == 'selfrag':
        # Self-RAG 需要特殊的 token 处理，FlashRAG 已内置
        pipeline = SelfRAGPipeline(config)
    elif args.method == 'replug':
        pipeline = REPLUGPipeline(config)
    elif args.method == 'flare':
        pipeline = FLAREPipeline(config)
        
    print("🏃 运行...")
    start_time = time.time()
    
    # 运行并评估
    pipeline.run(dataset, do_eval=True)
    
    end_time = time.time()
    print(f"⏱️ 耗时: {end_time - start_time:.2f} 秒")
    print(f"📊 结果已保存至: {SAVE_DIR}")

if __name__ == "__main__":
    main()
