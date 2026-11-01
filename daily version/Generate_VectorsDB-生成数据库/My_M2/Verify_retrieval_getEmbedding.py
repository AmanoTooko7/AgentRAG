# 来着Gemini3生成代码,11.20可运行,这里是验证gte-multilingual-base嵌入是否能返回正确的向量
# 执行环境是ybw_DBGenerate
import os
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import csv

def load_vectors(vec_path, id_path):
    print(f"Loading vectors from {vec_path}...")
    # 注意：这里必须和生成时的数据类型严格一致 (float32 和 int64)
    # GTE-base 的维度是 768
    vecs = np.fromfile(vec_path, dtype='float32').reshape(-1, 768)
    ids = np.fromfile(id_path, dtype='int64')
    print(f"--> Loaded {vecs.shape[0]} vectors.")
    return vecs, ids

def get_text_by_id(tsv_path, target_ids):
    """
    去 TSV 大文件里捞出指定 ID 的文本。
    为了省内存，我们分块读取。
    """
    print(f"Searching for text in {tsv_path}...")
    found_map = {}
    target_set = set(target_ids)
    
    # chunksize 设置大一点，读取快一些
    reader = pd.read_csv(
        tsv_path, 
        sep='\t', 
        header=None, 
        names=['id', 'text', 'title'], 
        chunksize=100000,
        quoting=csv.QUOTE_NONE
    )
    
    for chunk in reader:
        # 数据清洗：把 ID 转成数字
        chunk['id'] = pd.to_numeric(chunk['id'], errors='coerce')
        chunk = chunk.dropna(subset=['id'])
        chunk['id'] = chunk['id'].astype(int)
        
        # 看看当前块里有没有我们要找的 ID
        # isin 是 pandas 的高效查找
        matched = chunk[chunk['id'].isin(target_set)]
        
        for _, row in matched.iterrows():
            # 格式化一下：Title + Text
            content = f"Title: {row['title']} | Text: {row['text']}"
            found_map[row['id']] = content
            
        # 如果找齐了，就提前退出
        if len(found_map) == len(target_set):
            break
            
    return found_map

def main():
    # ================= 配置路径 =================
    base_dir = "/home/lab/ybw/Generate_VectorsDB/DB/vector_DB_Embeddinggemma-3000m"
    vec_file = os.path.join(base_dir, "embeddings_shard_0.bin")
    id_file = os.path.join(base_dir, "ids_shard_0.bin")
    
    raw_data_path = "/home/lab/ybw/Generate_VectorsDB/DB/raw_data/psgs_w100.tsv"
    # model_path = "/home/lab/Shared_model/gte-multilingual-base/converted_to_bin"
    model_path = "/home/lab/Shared_model/embeddinggemma-300m"
    # ===========================================

    # 1. 加载向量和 ID
    vecs, ids = load_vectors(vec_file, id_file)

    # 2. 构建 Faiss 索引
    print("Building Faiss index...")
    # IndexFlatIP 表示：精确搜索 (Flat) + 内积 (Inner Product)
    # 因为 GTE 输出的向量已经归一化了，所以 内积 等价于 余弦相似度
    quantizer = faiss.IndexFlatIP(768)
    
    # IndexIDMap 让我们可以使用原来的 ID (比如 TSV 里的 ID)，而不是 Faiss 自动生成的 0,1,2...
    index = faiss.IndexIDMap(quantizer)
    index.add_with_ids(vecs, ids)
    print(f"--> Index built. Total vectors: {index.ntotal}")

    # 3. 加载模型 (用于把查询变成向量)
    print("Loading model...")
    model = SentenceTransformer(model_path, device="cuda:0", trust_remote_code=True)
    
    # ================= 测试环节 =================
    
    # 我们从加载的 ID 里随便挑一个（比如第 100 个），作为“标准答案”
    # 这样能保证这个数据肯定在库里
    test_idx = 100 
    target_id = ids[test_idx]
    
    print(f"\n[Test Case] We picked ID {target_id} from the database to test.")
    
    # 去 TSV 里查出这个 ID 的原文
    text_map = get_text_by_id(raw_data_path, [target_id])
    if target_id not in text_map:
        print("Error: Could not find the text in TSV (maybe parsing issue).")
        return
    
    ground_truth_text = text_map[target_id]
    print(f"--- Original Document Text ---\n{ground_truth_text}\n------------------------------")
    
    # 构造一个“模糊查询”
    # 既然是验证，我们可以直接用原文的一部分去搜，看能不能搜到自己
    # 这里我们截取前 50 个字符作为查询
    query_text = ground_truth_text.split('| Text:')[1].strip()[:50] 
    print(f"--- Constructing Query (first 50 chars) ---\nQuery: '{query_text}'")
    
    # 4. 向量化查询
    query_vec = model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
    
    # 5. 检索 (Top 5)
    k = 5
    D, I = index.search(query_vec, k)
    
    print(f"\n--- Retrieval Results (Top {k}) ---")
    
    # 找到检索回来的 ID 的文本
    retrieved_ids = I[0]
    retrieved_texts = get_text_by_id(raw_data_path, retrieved_ids)
    
    for rank, (result_id, score) in enumerate(zip(retrieved_ids, D[0])):
        content = retrieved_texts.get(result_id, "[Text not found in TSV chunk scan]")
        prefix = "✅ MATCH!" if result_id == target_id else "  "
        print(f"{prefix} Rank {rank+1} | ID: {result_id} | Score: {score:.4f}")
        print(f"      Content: {content[:100]}...") # 只打印前100个字防止刷屏

if __name__ == "__main__":
    main()