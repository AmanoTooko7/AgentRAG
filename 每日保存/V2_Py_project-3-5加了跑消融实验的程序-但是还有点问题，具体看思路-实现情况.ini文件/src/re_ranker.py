import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

RERANKER_PATH = "/home/lab/Shared_model/bge-reranker-v2-m3"
target_device = "cuda:4" 
_RERANKER_MODEL = None      # 全局变量，缓存模型
_RERANKER_TOKENIZER = None  # 全局变量，缓存分词器
related_chunks = 3

def get_ranker_score(query, docs, batch_size=16):
    global _RERANKER_MODEL, _RERANKER_TOKENIZER
    
    # 1. 懒加载：第一次调用时才加载模型，避免一开始就占显存
    if _RERANKER_MODEL is None:
        # print(f"⚡ Loading Reranker from {RERANKER_PATH}...")
        _RERANKER_TOKENIZER = AutoTokenizer.from_pretrained(RERANKER_PATH)
        _RERANKER_MODEL = AutoModelForSequenceClassification.from_pretrained(RERANKER_PATH)
        _RERANKER_MODEL.eval()
        if torch.cuda.is_available():
            _RERANKER_MODEL.to(target_device)

    # 2. 输入对
    if not docs: return []
    pairs = [[query, doc] for doc in docs]
    all_scores = []
    print(f" Re-ranking on {target_device} | Docs: {len(pairs)} | Batch: {batch_size}")
    # 3. 分批次推理
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i : i + batch_size]
            
            # 🔥 关键：输入数据也必须搬运到 target_device (cuda:1)
            inputs = _RERANKER_TOKENIZER(
                batch_pairs, 
                padding=True, 
                truncation=True, 
                return_tensors='pt', 
                max_length=512
            ).to(target_device)
            batch_scores = _RERANKER_MODEL(**inputs, return_dict=True).logits.view(-1,).float()
            all_scores.append(batch_scores.detach().cpu().numpy())
            
            # 清理缓存 (可选)
            del inputs, batch_scores
    scores = np.concatenate(all_scores)
    top_k_indices = scores.argsort()[::-1][:related_chunks] # 返回前三个语义相同的文档块
    # print(f"top_k_indics:{top_k_indices}")
    return [docs[i] for i in top_k_indices]

