# 来着Gemini3生成代码
#用于检测faiss GPU条件下检索是否会更快，嵌入模型是embeddinggemma-300m
# 执行环境是ybw_DBGenerate_2

import os
import time
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import csv

def load_vectors(vec_path, id_path):
    print(f"Loading vectors from {vec_path}...")
    # 1. 读取向量 (Float32)
    vecs = np.fromfile(vec_path, dtype='float32').reshape(-1, 768)
    # 2. 读取 ID (Int64)
    ids = np.fromfile(id_path, dtype='int64')
    
    # 计算大小
    size_in_gb = vecs.nbytes / (1024**3)
    print(f"--> Loaded {vecs.shape[0]} vectors.")
    print(f"--> Vector data size in RAM: {size_in_gb:.2f} GB")
    return vecs, ids

def get_text_by_id(tsv_path, target_ids):
    # (这个函数保持不变，用于最后查文本)
    print(f"Searching for text in {tsv_path}...")
    found_map = {}
    target_set = set(target_ids)
    reader = pd.read_csv(tsv_path, sep='\t', header=None, names=['id', 'text', 'title'], chunksize=100000, quoting=csv.QUOTE_NONE)
    
    for chunk in reader:
        chunk['id'] = pd.to_numeric(chunk['id'], errors='coerce')
        chunk = chunk.dropna(subset=['id'])
        chunk['id'] = chunk['id'].astype(int)
        matched = chunk[chunk['id'].isin(target_set)]
        for _, row in matched.iterrows():
            content = f"Title: {row['title']} | Text: {row['text']}"
            found_map[row['id']] = content
        if len(found_map) == len(target_set):
            break
    return found_map

def main():
    # ================= 配置路径 =================
    base_dir = "/home/lab/ybw/Generate_VectorsDB/DB/vector_DB_Embeddinggemma-300m"
    vec_file = os.path.join(base_dir, "embeddings_shard_0.bin")
    id_file = os.path.join(base_dir, "ids_shard_0.bin")
    raw_data_path = "/home/lab/ybw/Generate_VectorsDB/DB/raw_data/psgs_w100.tsv"
    model_path = "/home/lab/Shared_model/embeddinggemma-300m"
    # ===========================================

    # 1. 加载数据到内存
    vecs, ids = load_vectors(vec_file, id_file)

    # 2. 构建 GPU 索引 (核心变化部分)
    print("\n=== Building GPU Index ===")
    
    # 获取可用 GPU 数量
    ngpus = faiss.get_num_gpus()
    print(f"--> Found {ngpus} GPUs available.")
    if ngpus < 2:
        print("Warning: 4.2M vectors need ~12.8GB VRAM. Single 3080 (10GB) might fail.")

    # 定义资源对象 (Resources)
    # 我们使用 index_cpu_to_all_gpus 让 Faiss 自动把数据分片到所有 GPU 上
    # flat_config = faiss.GpuMultipleClonerOptions()
    # flat_config.useFloat16 = False # 保持精度，如果显存不够可以设为 True

    # 建立一个 CPU 基础索引
    cpu_index = faiss.IndexFlatIP(768)

    # 配置 GPU 选项
    co = faiss.GpuMultipleClonerOptions()
    
    # 1. 开启分片 (Shard)：把数据切分给不同的卡，而不是每张卡存一份
    co.shard = True  
    
    # 2. (可选) 开启 Float16：为了以后跑全量 2100万条做准备
    # 3080 显卡跑 Float16 速度更快，显存占用减半，精度损失可忽略
    # 目前跑 400万条 Float32 够用，但跑全量必须开这个，建议现在就加上
    co.useFloat16 = True 
    
    print("--> Moving index to GPUs (Sharding enabled)...")
    
    # 传入 co 参数
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index, co=co)

    # 添加数据到 GPU
    # 注意：GPU 索引通常不支持直接 add_with_ids (除非用特定配置)，
    # 为了通用性，我们让 Faiss 自动生成 0,1,2... 的内部索引，
    # 检索回来后，我们自己用 Python 里的 `ids` 数组做映射。
    t0 = time.time()
    gpu_index.add(vecs)
    t1 = time.time()
    print(f"--> Added {gpu_index.ntotal} vectors to GPU index in {t1-t0:.2f} seconds.")

    # 3. 加载模型
    print("\n=== Loading Query Model ===")
    # 注意：模型本身也需要显存。如果 Faiss 占满了所有卡，模型加载可能会挤。
    # 但 3080 应该还有剩余。为了安全，可以指定让模型只用 GPU 0，或者 cpu (如果显存极度紧张)
    model = SentenceTransformer(model_path, device="cuda:0", trust_remote_code=True)
    
    # 4. 准备测试
    test_idx = 100
    target_real_id = ids[test_idx] # 这是真实的 TSV ID
    print(f"\n[Test Case] Verifying Real ID: {target_real_id} (Internal Index: {test_idx})")

    # 获取原文
    text_map = get_text_by_id(raw_data_path, [target_real_id])
    ground_truth_text = text_map.get(target_real_id, "Text not found")
    print(f"--- Original Text ---\n{ground_truth_text[:200]}...\n---------------------")

    # 构造查询 (Gemma 模型通常需要 Instruction)
    # 官方建议查询加前缀，文档不加
    query_raw = ground_truth_text.split('| Text:')[1].strip()[:50]
    instruction = "Given a search query, retrieve relevant passages: "
    query_text = instruction + query_raw
    print(f"Query: '{query_text}'")

    # 5. 编码查询
    query_vec = model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)

    # 6. 执行 GPU 检索
    print("\n=== Executing GPU Search ===")
    k = 5
    
    # 计时开始
    start_search = time.time()
    D, I = gpu_index.search(query_vec, k)
    end_search = time.time()
    
    # 计算时间 (毫秒)
    search_time_ms = (end_search - start_search) * 1000
    print(f"--> Search completed in {search_time_ms:.4f} ms")

    # 7. 结果映射与展示
    print(f"\n--- Retrieval Results (Top {k}) ---")
    
    # I[0] 返回的是内部索引 (0, 1, 2...)，我们需要映射回真实 ID
    internal_indices = I[0]
    retrieved_real_ids = [ids[i] for i in internal_indices if i != -1] # 映射
    
    retrieved_texts = get_text_by_id(raw_data_path, retrieved_real_ids)

    for rank, (internal_idx, score) in enumerate(zip(internal_indices, D[0])):
        real_id = ids[internal_idx]
        content = retrieved_texts.get(real_id, "Not found")
        prefix = "✅ MATCH!" if real_id == target_real_id else "  "
        print(f"{prefix} Rank {rank+1} | Real ID: {real_id} | Score: {score:.4f}")

if __name__ == "__main__":
    main()


# 以下验证的输出结果
# (ybw_DBGenerate_2) lab@qqubmgpu:~/ybw/Generate_VectorsDB/My_M2$ python Verify_retrieval_embeddinggemma.py
# Loading vectors from /home/lab/ybw/Generate_VectorsDB/DB/vector_DB_Embeddinggemma-300m/embeddings_shard_0.bin...
# --> Loaded 4203064 vectors.
# --> Vector data size in RAM: 12.03 GB

# === Building GPU Index ===
# --> Found 5 GPUs available.
# --> Moving index to GPUs (Sharding enabled)...
# --> Added 4203064 vectors to GPU index in 0.62 seconds.

# === Loading Query Model ===

# [Test Case] Verifying Real ID: 505 (Internal Index: 100)
# Searching for text in /home/lab/ybw/Generate_VectorsDB/DB/raw_data/psgs_w100.tsv...
# --- Original Text ---
# Title: Abscess | Text: "anti-staphylococcus antibiotic such as flucloxacillin or dicloxacillin is used. The Infectious Diseases Society of America advises that the draining of an abscess is not enough...
# ---------------------
# Query: 'Given a search query, retrieve relevant passages: "anti-staphylococcus antibiotic such as flucloxaci'

# === Executing GPU Search ===
# --> Search completed in 118.3083 ms

# --- Retrieval Results (Top 5) ---
# Searching for text in /home/lab/ybw/Generate_VectorsDB/DB/raw_data/psgs_w100.tsv...
#    Rank 1 | Real ID: 10123135 | Score: 0.4280
# ✅ MATCH! Rank 2 | Real ID: 505 | Score: 0.4278
#    Rank 3 | Real ID: 10123140 | Score: 0.4199
#    Rank 4 | Real ID: 6515615 | Score: 0.3895
#    Rank 5 | Real ID: 10123130 | Score: 0.3886