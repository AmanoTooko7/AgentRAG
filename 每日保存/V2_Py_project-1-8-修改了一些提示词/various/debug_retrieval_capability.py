# 此程序用于测试直接在SQL进行检索相关内容 + 用gemma嵌入query，返回结果经过re-reanker后的文档块

import sqlite3
import torch
import numpy as np
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.retriever_of_wiki import WikiRetriever
from src.re_ranker import get_ranker_score

DB_PATH = "/home/lab/ybw/Generate_VectorsDB/DB/DB/wiki_knowledge.db"

TEST_QUERY = "Who is the father-in-law of Basilina?"
SQL_KEYWORD = "Basilina" 

def debug_pipeline():
    print(f"🔍 [Debug] 正在诊断 Query: '{TEST_QUERY}'")
    print("="*50)

    # 路径 A: 模拟 V2 完整检索流程 (Gemma + Faiss + Re-ranker)
    retriever = WikiRetriever()
    print("   -> 正在进行 Gemma 向量检索 (Top-100)...")
    instruction = "Given a search query, retrieve relevant passages:"
    final_query = instruction + TEST_QUERY

    query_vec = retriever.model.encode([final_query], convert_to_numpy=True ,normalize_embeddings=True)

    k = 500
    D, I = retriever.index.search(query_vec, k) # 返回向量和每个向量的得分
    found_internal_ids = I[0]
    real_ids = []
    for internal_id in found_internal_ids:
        if internal_id != -1 and internal_id < len(retriever.master_id_map):
            real_ids.append(retriever.master_id_map[internal_id])
    docs_data = retriever._get_text_from_sqlite(real_ids)

    # 格式化为字符串列表供 Re-ranker 使用
    # 注意：这里要模拟 nodes.py 里的格式
    raw_docs = [f"Titile: {t}\nContent: {c}" for t, c in docs_data]
    print(f"   -> 初筛完成，获取了 {len(raw_docs)} 个文档。")
    if raw_docs:
        print(f"   -> Top-1 初筛结果: {raw_docs[0][:100]}...")
    
    print("   -> 正在进行 BGE 重排序 (Top-3)...")
    reranked_docs = get_ranker_score(TEST_QUERY, raw_docs)

    print(f"\n✅ [路径 A 结果] V2 最终检索到的 Top-3:")
    for i, doc in enumerate(reranked_docs):
        print(f"   [{i+1}] {doc[:]}") # 只打印前150字符


# 路径 B: SQL 暴力搜索 (模拟上帝视角/BM25)
    print("\n" + "="*50)
    print(f"🚀 [路径 B] 正在运行 SQL 关键词匹配 (Keyword: '{SQL_KEYWORD}')...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT title, text FROM documents WHERE title LIKE ? OR text LIKE ? LIMIT 3", # 现在数据库返回文档数 
                   (f'%{SQL_KEYWORD}%', f'%{SQL_KEYWORD}%'))
    sql_results = cursor.fetchall()
    conn.close()

    if sql_results:
        print(f"✅ [路径 B 结果] 数据库里确实有关于 '{SQL_KEYWORD}' 的文档 ({len(sql_results)} examples found):")
        for i, (title, text) in enumerate(sql_results):
            print(f"   [{i+1}] Title: {title}")
            print(f"       Content: {text[:]}...")
    else:
        print(f"❌ [路径 B 结果] 数据库里根本没有 '{SQL_KEYWORD}'！非战之罪。")

    print("\n" + "="*50)

    #结论分析
    print("📊 [诊断结论]")

    # 检查路径 A 的结果里是否包含关键词
    path_a_success = any(SQL_KEYWORD.lower() in doc.lower() for doc in reranked_docs)

    if not sql_results:
        print("   -> 💀 数据库缺失：Wiki 2018 Dump 里没有这个人，神仙也搜不到。建议换题或在论文里标注数据缺陷。")
    elif sql_results and not path_a_success:
        print("   -> 📉 语义漂移 (Semantic Drift)：数据库里有，但 Gemma 没捞回来。")
        print("   -> 💡 建议：这是引入 BM25 (稀疏检索) 的绝佳理由。")
    else:
        print("   -> 🎉 成功：V2 检索到了相关文档。")

if __name__ == "__main__":
    debug_pipeline()