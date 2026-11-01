# 此文件用于加载数据库和构建检索器
# 此文件检索的数据库是翁丽莲的博客

import pickle
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.tools.retriever import create_retriever_tool

#这是可用的，但是下面那个输出的更形象
# def build_retriever_tool():
#     """Bulid a retriever tool from a pre-saved document spilts file.
#     """
#     print("--- 🛠️ Building Retriever Tool ---")

#     db_path = "/home/lab/ybw/V2_Py_project/DB/LilianWeng_Three_blogs_DB.pkl"
#     with open(db_path, "rb") as f:
#         doc_splits = pickle.load(f)
    
#     local_model_path = "/home/lab/Shared_model/all-MiniLM-L6-v2"
#     embedding_model = HuggingFaceEmbeddings(
#         model_name = local_model_path,
#         model_kwargs = {'device': 'cuda:1'}
#     )

#     # 构建向量数据库
#     vectorstore = InMemoryVectorStore.from_documents(
#         documents=doc_splits,
#         embedding=embedding_model
#     )
#     retriever = vectorstore.as_retriever()

#     # 构建检索工具
#     retriever_tool = create_retriever_tool(
#         retriever,
#         "retrieve_blog_posts",
#         "Search and return information about Lilian Weng blog posts.",
#     )

#     print("✅检索器工具构建完成")
#     return retriever_tool



def build_retriever_tool():
    """Build a retriever tool from a pre-saved document splits file."""
    print("--- 🛠️ Building Retriever Tool ---")

    db_path = "/home/lab/ybw/V2_Py_project/DB/LilianWeng_Three_blogs_DB.pkl"
    with open(db_path, "rb") as f:
        doc_splits = pickle.load(f)

    local_model_path = "/home/lab/Shared_model/all-MiniLM-L6-v2"
    embedding_model = HuggingFaceEmbeddings(
        model_name=local_model_path,
        model_kwargs={"device": "cuda:1"},
    )

    # 构建向量数据库
    vectorstore = InMemoryVectorStore.from_documents(
        documents=doc_splits,
        embedding=embedding_model,
    )

    base_retriever = vectorstore.as_retriever()

    # ✅ 调试包装类：支持 config 参数
    class DebugRetriever:
        def __init__(self, retriever):
            self.retriever = retriever

        def invoke(self, query: str, config=None):
            print(f"\n🔎 [Retriever] 接收到的 query: {query}")
            results = self.retriever.invoke(query, config=config)
            print(f"📚 [Retriever] 检索返回 {len(results)} 个文档块\n")
            if results:
                snippet = results[0].page_content[:300].replace("\n", " ")
                print(f"🧾 [Retriever] Top-1 文档片段: {snippet}...\n")
            return results

    retriever = DebugRetriever(base_retriever)

    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_blog_posts",
        "Search and return information about Lilian Weng blog posts.",
    )

    print("✅ 检索器工具构建完成")
    return retriever_tool
