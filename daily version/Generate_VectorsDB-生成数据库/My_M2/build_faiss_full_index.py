import faiss
import numpy as np
import os
import time
import gc

def load_shard(shard_id, base_dir):
    """读取指定分片的向量"""
    vec_path = os.path.join(base_dir, f"embeddings_shard_{shard_id}.bin")
    if not os.path.exists(vec_path):
        print(f"Shard {shard_id} not found.")
        return None
    print(f"Loading Shard {shard_id} from {vec_path}...")
    # 读取 float32 原始向量
    vecs = np.fromfile(vec_path, dtype='float32').reshape(-1, 768)
    return vecs

def build_index():
    base_dir = "/home/lab/ybw/Generate_VectorsDB/DB/vector_DB_Embeddinggemma-300m" # 原始向量文件所在目录
    index_save_path = "/home/lab/ybw/Generate_VectorsDB/DB/faiss_index/gemma_300m_ivfpq.index"

    shard_ids_to_add = [0, 1, 2, 3, 4] 

    d = 768
    nlist = 16384
    m = 64 # 将768维向量压缩到64个，每个12维度
    nbits = 8 # 每个向量分量用8位表示

    # 1. 检查是否存在已有索引
    if os.path.exists(index_save_path):
        print(f"Found existing index at {index_save_path}. Loading...")
        index = faiss.read_index(index_save_path)
    else:
        print("No index found. Starting TRAINING phase...")
        
        # --- 训练阶段 (只用 Shard 0 的一部分) ---
        print("Loading Shard 0 for training...")
        train_data = load_shard(0, base_dir)
        
        # 为了加快训练速度并节省显存，我们不需要用全部 420万条数据训练
        # 随机采样 150万条就足够训练出非常好的量化器了
        num_train_sample = 1500000
        if train_data.shape[0] > num_train_sample:
            print(f"Sampling {num_train_sample} vectors for training...")
            # 随机选择索引
            indices = np.random.choice(train_data.shape[0], num_train_sample, replace=False)
            train_vectors = train_data[indices]
        else:
            train_vectors = train_data
        
        print(f"Training IndexIVFPQ (nlist={nlist}, m={m})... This may take a while.")

        # 定义量化器
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)

         # 使用 GPU 进行训练 (可选，如果你想快一点)
        # 但考虑到显存限制，CPU 训练 50万条数据也就几分钟，更稳妥
        # 这里使用 CPU 训练
        t0 = time.time()
        index.train(train_vectors)
        print(f"Training finished in {time.time()-t0:.2f} seconds.")

         # 释放训练数据内存
        del train_vectors
        del train_data
        gc.collect() # 强制垃圾回收

    # 2. 添加数据阶段 (Add)
    print("\n=== Adding Data Phase ===")
    
    # 我们使用 CPU 添加数据，虽然比 GPU 慢，但不用担心 12GB 的 Shard 把显存撑爆
    # 而且 IVFPQ 添加速度其实很快

    for shard_id in shard_ids_to_add:
        print(f"\nProcessing Shard {shard_id}...")
        
        # 检查是否需要重新加载（如果内存够大，Shard 0 可能还在内存里，但为了逻辑通用，重新加一遍）
        vecs = load_shard(shard_id, base_dir)
        
        if vecs is not None:
            t0 = time.time()
            index.add(vecs)
            t1 = time.time()
            print(f"Shard {shard_id} added in {t1-t0:.2f} seconds.")
            print(f"Current total vectors: {index.ntotal}")
            
            # 关键：加完一个分片，立即释放内存！
            del vecs
            gc.collect()
        else:
            print(f"Skipping Shard {shard_id} (file not found).")
    
     # 3. 最终保存
    print(f"\nWriting final index to {index_save_path}...")
    faiss.write_index(index, index_save_path)
    print("ALL DONE. Index is ready.")

if __name__ == "__main__":
    build_index()