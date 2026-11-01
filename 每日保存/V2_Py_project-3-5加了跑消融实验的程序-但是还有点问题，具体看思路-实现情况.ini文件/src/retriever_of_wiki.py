# 此文件检索wikipedia dump2018知识库，来自meta2020那篇DPR的论文
# 2025.1.9加入BM25实现混合检索
import os
import numpy as np
import faiss
import sqlite3
from typing import List
from langchain_core.documents import Document
# from langchain_core.tools import Tool
from langchain.tools.retriever import create_retriever_tool 
from sentence_transformers import SentenceTransformer
from src.constants import BOLD_ON, BOLD_OFF
import bm25s
import Stemmer 
BASE_DIR = "/home/lab/ybw/Generate_VectorsDB/DB/vector_DB_Embeddinggemma-300m"
DB_PATH = "/home/lab/ybw/Generate_VectorsDB/DB/DB/wiki_knowledge.db"
INDEX_PATH = "/home/lab/ybw/Generate_VectorsDB/DB/faiss_index/gemma_300m_ivfpq.index"
MODEL_PATH = "/home/lab/Shared_model/embeddinggemma-300m"
BM25_INDEX_PATH = "/home/lab/ybw/Generate_VectorsDB/DB/bm25_index-2" # 这是
SHARD_LIST = [0, 1, 2, 3, 4]  

class WikiRetriever:
    """封装Faiss + SQLite"""
    def __init__(self):
        print("--- 🚀 Initializing WikiRetriever (Heavy Load) ---")

        # 得到ID映射表
        self.master_id_map = self._load_id_mapping(BASE_DIR, SHARD_LIST)

        #加载Faiss索引
        # print(f"Loading Faiss index from {INDEX_PATH}...")
        self.index = faiss.read_index(INDEX_PATH)
        self.index.nprobe = 128
        # print(f"Index loaded. Total vectors:{self.index.ntotal}")

        #加载embedding模型
        # print(f"Loadign Gemma model from {MODEL_PATH}...")
        self.model = SentenceTransformer(MODEL_PATH, device="cuda:4", trust_remote_code=True)

        # 加载BM25索引
        print(f"Loading BM25 index from {BM25_INDEX_PATH}...")
        self.bm25_retriever = bm25s.BM25.load(BM25_INDEX_PATH, load_corpus=False)
        self.stemmer = Stemmer.Stemmer("english")
        print("BM25 index loaded successfully.")


        #连接SQLite数据库(注意：SQLite 连接不能跨线程共享，这里初始化一个只读路径) 
        # 在 invoke 时我们会单独建立连接或者确保持有连接
        self.db_path = DB_PATH

    def _load_id_mapping(self, base_dir, shard_ids):
        print("Loadding ID mapping files...")
        all_real_ids = []
        for shard_id in shard_ids:
            id_path = os.path.join(base_dir, f"ids_shard_{shard_id}.bin")
            if os.path.exists(id_path):
                ids = np.fromfile(id_path, dtype='int64')
                all_real_ids.append(ids)
            else:
                print(f"⚠️ Warning: Shard {shard_id} ID file not found!")
        
        master_map = np.concatenate(all_real_ids)
        print(f"Mapping loaded. Total IDs: {len(master_map)}") 
        return master_map
    
    def _get_text_from_sqlite(self, real_ids: List[int]) -> List[tuple]:
        """批量从 SQLite 获取文本，返回 [(title, text), ...]"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        results = []
        # 这里的 real_ids 是一个列表，为了安全和速度，我们还是用循环查或者 IN 查询 
        # 考虑到只有 Top-K (比如5-10个)，循环查也非常快(ms级)
        for rid in real_ids:
            cursor.execute("SELECT title, text FROM documents WHERE id=?" ,(int(rid),))
            row = cursor.fetchone()
            if row:
                results.append(row)
            else:
                results.append(("Unknowm Title", "Content not found in DB."))
        conn.close()
        return results
    
    def invoke(self, query:str, config=None) ->List[Document]:
        """
        这是 LangChain 调用的标准接口。
        输入：查询字符串
        输出：Document 对象列表
        """
        print(f"{BOLD_ON}\n🔎 [WikiRetriever] Searching for: '{query}{BOLD_OFF}'")

        #构造查询向量
        instruction = "Given a search query, retrieve relevant passages:"
        query_content = query[:512]
        final_query = instruction + query_content
        print(f"查询向量是：{final_query}")

        query_vec = self.model.encode([final_query], convert_to_numpy=True ,normalize_embeddings=True)

        # Faiss 检索 (Top 5)
        k_dense = 200
        D, I = self.index.search(query_vec, k_dense) # 返回5个向量和得分

        found_internal_ids = I[0]
        # found_scores = D[0]

        # 映射 Dense ID
        dense_real_ids = []
        for internal_id in found_internal_ids:
            if internal_id != -1 and internal_id < len(self.master_id_map):
                real_id = self.master_id_map[internal_id]
                dense_real_ids.append(real_id)
        
        # ================= B. 稀疏检索 (Sparse Retrieval) =================
        # 分词
        query_tokens = bm25s.tokenize([query], stopwords="en", stemmer=self.stemmer)
        
        # BM25 检索 (Top 100)
        k_sparse = 200
        # results.indices 返回的是索引 ID (0, 1, 2...)
        # 因为你的 jsonl 是按顺序生成的 (id=0, id=1...)，且 Faiss 也是按顺序的
        # 所以 BM25 的 index 直接对应真实 ID (或者需要 +1，取决于你的 jsonl id 是从0还是1开始)
        # 根据你之前的 head -n 2 psgs_w100.jsonl，id 是 "0", "1"...
        # 而 Faiss 里的 ID 映射表 (ids_shard_*.bin) 存的是真实 ID (可能是 1, 2...)
        # ⚠️ 关键假设：我们假设 BM25 index (0) == Faiss index (0) -> 对应 Real ID (master_id_map[0])
        
        bm25_results = self.bm25_retriever.retrieve(query_tokens, k=k_sparse)
        sparse_indices = bm25_results.documents[0] # 这里的 index 是 0-based 的行号
        
        # 映射 Sparse ID
        sparse_real_ids = []
        for idx in sparse_indices:
            if idx < len(self.master_id_map):
                # 使用相同的映射表，确保 Dense 和 Sparse 指向同一个文档
                real_id = idx + 1
                sparse_real_ids.append(real_id)

        # ================= C. 混合 (Merge & Deduplicate) =================
        # 合并 ID 并去重
        all_candidate_ids = set()
        all_candidate_ids.update(dense_real_ids)
        all_candidate_ids.update(sparse_real_ids)
        
        final_ids = list(all_candidate_ids)
        print(f"📊 Retrieval Stats: Dense({len(dense_real_ids)}) + Sparse({len(sparse_real_ids)}) -> Merged({len(final_ids)})")

        # ================= D. 查库 & 返回 =================
        docs_data = self._get_text_from_sqlite(final_ids)

        documents = []
        for (title, text), rid in zip(docs_data, final_ids):
            page_content = f"Titile: {title}\nContent: {text}"
            
            # 注意：混合检索后，score 暂时设为 0 或不设，因为 Dense 和 BM25 分数不可直接比较
            # 真正的排序工作交给后面的 Re-ranker (BGE)
            doc = Document(
                page_content=page_content,
                meta={"id": int(rid), "source": "Wikipedia"}
            )
            documents.append(doc)

        print(f"📚 [WikiRetriever] 返回 {len(documents)} docs (Sent to Re-ranker).")
        if documents:
            print(f"🧾 Top-1 Preview: {documents[0].page_content[:200]}...")
        
        return documents

def build_retriever_tool():
    # 实例化检索器
    custom_retriever = WikiRetriever()

    retriever_tool = create_retriever_tool(
        custom_retriever,
        "retriever_wikipedia_knowledge",
        "Search and return relevant information from the entire Wikipedia database (21M documents). Use this for any general knowledge questions.",
    )

    print("Wiki Retriever Tool bulit successfully")
    return retriever_tool