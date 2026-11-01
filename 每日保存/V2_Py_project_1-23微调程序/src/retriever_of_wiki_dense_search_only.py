# 此文件仅仅是密集检索，也就是只有gemma300m模型的检索
import os
import numpy as np
import faiss
import sqlite3
from typing import List
from langchain_core.documents import Document
from langchain.tools.retriever import create_retriever_tool
from sentence_transformers import SentenceTransformer
from src.constants import BOLD_ON, BOLD_OFF
BASE_DIR = "/home/lab/ybw/Generate_VectorsDB/DB/vector_DB_Embeddinggemma-300m"
DB_PATH = "/home/lab/ybw/Generate_VectorsDB/DB/DB/wiki_knowledge.db"
INDEX_PATH = "/home/lab/ybw/Generate_VectorsDB/DB/faiss_index/gemma_300m_ivfpq.index"
MODEL_PATH = "/home/lab/Shared_model/embeddinggemma-300m"
SHARD_LIST = [0, 1, 2, 3, 4]  

gpu_id = os.environ.get("V2_TARGET_DEVICE", "0")
device_str = f"cuda:{gpu_id}"

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
        self.model = SentenceTransformer(MODEL_PATH, device=device_str, trust_remote_code=True)

        #连接SQLite数据库(注意：SQLite 连接不能跨线程共享，这里初始化一个只读路径) 
        # 在 invoke 时我们会单独建立连接或者确保持有连接
        self.db_path = DB_PATH

    def _load_id_mapping(self, base_dir, shard_ids):
        # print("Loadding ID mapping files...")
        all_real_ids = []
        for shard_id in shard_ids:
            id_path = os.path.join(base_dir, f"ids_shard_{shard_id}.bin")
            if os.path.exists(id_path):
                ids = np.fromfile(id_path, dtype='int64')
                all_real_ids.append(ids)
            else:
                print(f"⚠️ Warning: Shard {shard_id} ID file not found!")
        
        master_map = np.concatenate(all_real_ids)
        # print(f"Mapping loaded. Total IDs: {len(master_map)}") 
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
        # print(f"查询向量是：{final_query}")

        query_vec = self.model.encode([final_query], convert_to_numpy=True ,normalize_embeddings=True)

        # Faiss 检索 (Top 5)
        k = 100
        D, I = self.index.search(query_vec, k) # 返回5个向量和得分

        found_internal_ids = I[0]
        found_scores = D[0]

        #映射ID
        real_ids = []
        valid_scores = []
        for internal_id, score in zip(found_internal_ids, found_scores):
            if internal_id != -1 and internal_id < len(self.master_id_map):
                real_id = self.master_id_map[internal_id]
                real_ids.append(real_id)
                valid_scores.append(score)
        
        # 查SQLite
        docs_data = self._get_text_from_sqlite(real_ids)

        # 封装成langchain Document
        documents = []
        for (title, text), score, rid in zip(docs_data, valid_scores, real_ids):
            # 将title和text组合成page_content,方便LLM阅读
            page_content = f"Titile: {title}\nContent: {text}"

            doc = Document(
                page_content=page_content,
                meta={"id": int(rid), "score": float(score), "source":"Wikiprdia"}
            )
            documents.append(doc)

        # print(f"📚 [WikiRetriever] 返回 {len(documents)} docs.")
        # if documents:
        #     print(f"🧾 Top-1 Preview: {documents[0].page_content[:200]}...")
        
        return documents

def build_retriever_tool():
    # 实例化我们自定义的检索器
    custom_retriever = WikiRetriever()

    retriever_tool = create_retriever_tool(
        custom_retriever,
        "retriever_wikipedia_knowledge",
        "Search and return relevant information from the entire Wikipedia database (21M documents). Use this for any general knowledge questions.",
    )

    # print("Wiki Retriever Tool bulit successfully")
    return retriever_tool


