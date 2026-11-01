# 此文件用于加载数据库和构建检索器

import pickle
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.tools.retriever import create_retriever_tool

def build_retriever_tool():
    """Bulid a retriever tool from a pre-saved document spilts file.
    """
    print("--- 🛠️ Building Retriever Tool ---")

    db_path = "/home/lab/ybw/V2_Py_project/DB/LilianWeng_Three_blogs_DB.pkl"
    with open(db_path, "rb") as f:
        doc_splits = pickle.load(f)
    
    local_model_path = "/home/lab/Shared_model/all-MiniLM-L6-v2"
    embedding_model = HuggingFaceEmbeddings(
        model_name = local_model_path,
        model_kwargs = {'device': 'cuda:1'}
    )

    # 构建向量数据库
    vectorstore = InMemoryVectorStore.from_documents(
        documents=doc_splits,
        embedding=embedding_model
    )
    retriever = vectorstore.as_retriever()

    # 构建检索工具
    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_blog_posts",
        "Search and return information about Lilian Weng blog posts.",
    )

    print("✅检索器工具构建完成")
    return retriever_tool