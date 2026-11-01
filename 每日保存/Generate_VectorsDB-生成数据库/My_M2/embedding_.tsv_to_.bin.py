# from sentence_transformers import SentenceTransformer
# import numpy as np

# model = SentenceTransformer('/home/lab/Shared_model/all-MiniLM-L6-v2')
# sentences = [
#     "Hugging Face makes NLP easy.",
# ]
# embeddings = model.encode(sentences)
# print("Embedding shape:", embeddings.shape)
# # print(embeddings)
# vec2 = embeddings[1]
# vec3 = embeddings[2]
# cosine_similarity = np.dot(vec2, vec3) / (np.linalg.norm(vec2) * np.linalg.norm(vec3))
# print(f"Cosine similarity between sentence 2 and 3: {cosine_similarity}")


# Gemini3生成
import argparse
import os
import csv
import time
import math
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def process_and_save(model_path, data_path, output_dir, shard_id, num_shards, batch_size, chunk_size=50000):
    # 1. 准备模型
    print(f"[Shard {shard_id}] Loading model...")
    device = f"cuda:0" 
    # trust_remote_code=True 用于加载 GTE 模型
    model = SentenceTransformer(model_path, device=device, trust_remote_code=True)
    
    # 限制最大长度，防止个别超长数据撑爆显存
    model.max_seq_length = 512 
    
    # 2. 准备输出文件
    os.makedirs(output_dir, exist_ok=True)
    vec_filename = os.path.join(output_dir, f"embeddings_shard_{shard_id}.bin")
    id_filename = os.path.join(output_dir, f"ids_shard_{shard_id}.bin")
    
    # 如果由于报错导致文件残留，先删除，重新开始
    if os.path.exists(vec_filename): os.remove(vec_filename)
    if os.path.exists(id_filename): os.remove(id_filename)

    print(f"[Shard {shard_id}] Reading data from {data_path}...")
    
    # 计算总块数用于进度条
    total_rows = 21015325
    total_chunks = math.ceil(total_rows / chunk_size)

    chunk_iterator = pd.read_csv(
        data_path, 
        sep='\t', 
        header=None, 
        names=['id', 'text', 'title'], 
        chunksize=chunk_size,
        quoting=csv.QUOTE_NONE
    )
    
    total_processed_rows = 0
    start_time = time.time()
    
    with open(vec_filename, "wb") as f_vec, open(id_filename, "wb") as f_id:
        # 进度条显示
        for chunk in tqdm(chunk_iterator, total=total_chunks, desc=f"GPU {shard_id}", unit="chunk"):
            
            # 1. 先强制把 ID 列转为数字。遇到 'id' 这种表头字符串，会变成 NaN (空值)
            chunk['id'] = pd.to_numeric(chunk['id'], errors='coerce')
            
            # 2. 删除 ID 为空值的行（也就自动删除了表头行）
            chunk = chunk.dropna(subset=['id']).copy()
            
            # 3. 此时剩下的全是纯数字了，安全转为整数
            chunk['id_int'] = chunk['id'].astype(int)
            
            # 筛选属于当前 GPU 的数据
            mask = (chunk['id_int'] % num_shards) == shard_id
            sub_chunk = chunk[mask]
            
            if len(sub_chunk) == 0:
                continue
                
            texts = (sub_chunk['title'].fillna('') + " " + sub_chunk['text'].fillna('')).tolist()
            ids = sub_chunk['id_int'].values
            
            try:
                # 推理
                embeddings = model.encode(
                    texts, 
                    batch_size=batch_size, 
                    show_progress_bar=False, 
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                
                # 写入硬盘
                embeddings.astype('float32').tofile(f_vec)
                ids.astype('int64').tofile(f_id)
                
                total_processed_rows += len(ids)
                
            except Exception as e:
                print(f"\n[Error] GPU {shard_id} failed at a batch. Error: {e}")
                # 如果还是报错，可以选择 continue 跳过这一批，或者直接抛出异常
                raise e

    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    
    print(f"\n[Shard {shard_id}] MISSION COMPLETE!")
    print(f"Total Vectors Generated: {total_processed_rows}")
    print(f"Time Elapsed: {elapsed_minutes:.2f} minutes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--shard_id", type=int, required=True)
    parser.add_argument("--num_shards", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    
    args = parser.parse_args()
    
    process_and_save(
        args.model_path, 
        args.data_path, 
        args.output_dir, 
        args.shard_id, 
        args.num_shards, 
        args.batch_size
    )

    