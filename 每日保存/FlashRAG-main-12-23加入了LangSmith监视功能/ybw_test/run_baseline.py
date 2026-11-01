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



