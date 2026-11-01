# 此程序验证隐患
# 验证思路：从.db中选取数条数据作为作为查询，嵌入后在.index中查询，拿到faiss返回的ID，再从.db中查询对应ID的真实文本，看是否一致

import faiss
import numpy as np
import sqlite3
import os
from sentence_transformers import SentenceTransformer
import time


# 此函数是为了得到一个faiss里某条向量的ID和.db库该条向量ID正确对应的数组
def load_id_mapping(base_dir, shard_ids):
    """
    关键步骤：构建 [Faiss内部ID -> 真实ID] 的映射表
    注意：加载顺序必须和 build_index 时添加 Shard 的顺序完全一致！
    """
    print("Loading ID mapping files (The Bridge)...")
    all_real_ids = []
    
    for shard_id in shard_ids:
        id_path = os.path.join(base_dir, f"ids_shard_{shard_id}.bin")
        if os.path.exists(id_path):
            # 读取该分片的真实 ID
            ids = np.fromfile(id_path, dtype='int64')
            all_real_ids.append(ids)
            print(f"  Loaded Shard {shard_id}: {len(ids)} IDs")
        else:
            print(f"  Warning: Shard {shard_id} ID file not found!")
    
    # 将所有分片的 ID 拼成一个大数组
    # 这个数组的下标就是 Faiss 的 Internal ID
    # 这个数组的值就是 真实 ID
    master_id_map = np.concatenate(all_real_ids)
    print(f"Mapping loaded. Total IDs: {len(master_id_map)}")
    return master_id_map

# 输入.db的id数得到该id的文本
def get_text_from_sqlite(cursor, real_id):
    """从 SQLite 查文本"""
    cursor.execute("SELECT title, text FROM documents WHERE id=?", (int(real_id),))
    result = cursor.fetchone()
    if result:
        return f"Title: {result[0]} | Text: {result[1][:200]}..." # 
    return "NOT FOUND in SQLite"

def main():
    # ================= 配置 =================
    base_dir = "/home/lab/ybw/Generate_VectorsDB/DB/vector_DB_Embeddinggemma-300m"
    db_path = "/home/lab/ybw/Generate_VectorsDB/DB/DB/wiki_knowledge.db"
    index_path = "/home/lab/ybw/Generate_VectorsDB/DB/faiss_index/gemma_300m_ivfpq.index"
    model_path = "/home/lab/Shared_model/embeddinggemma-300m"
    
    # 这里的顺序必须和 build_full_index.py 里添加的顺序一样
    # 你之前是 [0, 1, 2, 3]
    shard_list = [0, 1, 2, 3, 4]
    # =======================================

    # 1. 连接数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 2. 加载 ID 映射表 (解决隐患的关键)
    # 该数组分布是[5, 10, ..., 21015325, 1, 6, ..., 21015321, 2, 7, ..., 21015322, 3, 8, ..., 21015323, 4, 9, ..., 21015324]
    master_id_map = load_id_mapping(base_dir, shard_list)  #

    # 3. 加载 Faiss 索引
    print(f"Loading Faiss index from {index_path}...")
    index = faiss.read_index(index_path)
    print(f"Index loaded. Total vectors: {index.ntotal}")

    index.nprobe = 128

    # 4. 加载模型
    print("Loading Gemma model...")
    model = SentenceTransformer(model_path, device="cuda:0", trust_remote_code=True)

    # ================= 验证环节 =================
    # 我们挑选几个特殊的 Internal ID 来测试
    # 假设 Shard 0 有 N 个数据，那么 Index 0 是 Shard 0 的第1条
    # 假设 Shard 0 结束后的下一个 Index，就是 Shard 1 的第1条
    test_indices = [
        0,              # Shard 0 的第1条
        1000000,        # Shard 0 的某一条
        5000000,        # Shard 1 的某一条 (跨越了分片边界)
        16463728
    ]

    print("\n========== START VERIFICATION ==========")

    for internal_id in test_indices:
        if internal_id >= len(master_id_map):
            continue

        # A. 获取真值
        real_id_truth = master_id_map[internal_id] # 这是我们测试的哪条数据的真实ID
        truth_text = get_text_from_sqlite(cursor, real_id_truth) # 从 SQLite 拿真实文本，为字符串
        
        print(f"\n-------------------------------------------------------")
        print(f"🎯 Target: Internal_ID {internal_id} | Real_ID {real_id_truth}")
        print(f"📝 Query Text Snippet: {truth_text.split('| Text:')[1].strip()[:50]}...")
        
        # B. 构造查询
        query_content = truth_text.split('| Text:')[1].strip()[:200]
        instruction = "Given a search query, retrieve relevant passages: "
        query_vec = model.encode([instruction + query_content], convert_to_numpy=True, normalize_embeddings=True)

        # C. Faiss 检索 Top 10
        # 扩大搜索范围到 10
        D, I = index.search(query_vec, k=10)
        
        found_internal_ids = I[0]   # 这是查询回10个最相似向量的faiss内部ID
        found_scores = D[0]         # 这是查询回10个最相似向量的得分
        
        # 检查是否命中
        is_found = False
        print(f"🔍 Top 10 Results:")
        
        for rank, (f_int_id, score) in enumerate(zip(found_internal_ids, found_scores)):
            if f_int_id == -1: continue # Faiss padding
            
            # 映射回真实 ID
            f_real_id = master_id_map[f_int_id]  # 这是查询回的10个最相似的向量的真实ID
            
            # 简单检查一下标题
            retrieved_text_preview = get_text_from_sqlite(cursor, f_real_id)
            title_preview = retrieved_text_preview.split("|")[0].strip()
            
            marker = "  "
            if f_real_id == real_id_truth:  # 查询回来的10个最相似的向量ID，如果对应的真实ID和测试的真实ID一致，那么就说明找到了
                marker = "✅"
                is_found = True
            
            print(f"   {marker} Rank {rank+1}: Real_ID {f_real_id} (Score: {score:.4f}) -> {title_preview}")

        if is_found:
            print(f"\n✨ SUCCESS! Found original document in Top 10.")
        else:
            print(f"\n⚠️  Not in Top 10. (This might happen due to IVFPQ compression loss)")
            # 这里可以打印一下真值对应的文本，人工确认一下 Rank 1 是不是语义上其实也是对的
            print(f"   Truth was: {truth_text[:100]}...")

    conn.close()

if __name__ == "__main__":
    main()