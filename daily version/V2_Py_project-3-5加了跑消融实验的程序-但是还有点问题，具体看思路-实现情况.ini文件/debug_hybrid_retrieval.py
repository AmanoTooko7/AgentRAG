# 这个测试混合检索是否成功

import sys
import os

# 添加项目根目录到路径
sys.path.append(os.getcwd())

from src.retriever_of_wiki import WikiRetriever
import bm25s

# 测试问题 (Basilina 是之前的痛点)
TEST_QUERY = "Who is the father-in-law of Basilina?"
KEYWORD = "Basilina"

def debug_hybrid():
    print("🚀 初始化混合检索器...")
    retriever = WikiRetriever()
    
    print(f"\n🔍 [诊断] 测试 Query: '{TEST_QUERY}'")
    print("="*60)

    # -------------------------------------------------
    # 1. 诊断 BM25 (Sparse)
    # --------------------------------,-----------------
    print("\n🧪 [测试 1] 单独测试 BM25 (Sparse)...")
    query_tokens = bm25s.tokenize([TEST_QUERY], stopwords="en", stemmer=retriever.stemmer)
    bm25_results = retriever.bm25_retriever.retrieve(query_tokens, k=10)
    sparse_indices = bm25_results.documents[0]
    
    print(f"   -> BM25 返回了 {len(sparse_indices)} 个索引。")
    
    # 检查 ID 映射是否正确 (取第一个看看)
    first_idx = sparse_indices[0]
    real_id = first_idx + 1
    print(f"   -> Top-1 Index: {first_idx} => Real ID: {real_id}")
    
    # 查库验证内容
    docs = retriever._get_text_from_sqlite([real_id])
    if docs:
        title, text = docs[0]
        print(f"   -> Top-1 文档标题: {title}")
        if KEYWORD.lower() in text.lower() or KEYWORD.lower() in title.lower():
            print(f"   ✅ BM25 成功命中关键词 '{KEYWORD}'！")
        else:
            print(f"   ❌ BM25 未命中关键词 (可能是 ID 映射错位，或者分词问题)。")
    
    # -------------------------------------------------
    # 2. 诊断 混合检索 (Hybrid)
    # -------------------------------------------------
    print("\n🧪 [测试 2] 测试完整混合检索 (invoke)...")
    final_docs = retriever.invoke(TEST_QUERY)
    
    # 统计命中率
    hit_count = 0
    for doc in final_docs:
        if KEYWORD.lower() in doc.page_content.lower():
            hit_count += 1
            
    print(f"\n📊 [最终统计]")
    print(f"   -> 总共返回文档数: {len(final_docs)}")
    print(f"   -> 包含 '{KEYWORD}' 的文档数: {hit_count}")
    
    if hit_count > 0:
        print(f"🎉 混合检索成功！BM25 补足了 Dense 的短板。")
    else:
        print(f"⚠️ 依然没搜到。请检查 BM25 索引构建是否正确。")

if __name__ == "__main__":
    debug_hybrid()