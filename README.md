**The documentation has not yet been finalised and will be updated subsequently.**

<img width="3264" height="1693" alt="整体架构图" src="https://github.com/user-attachments/assets/3ffeee3d-0cdb-4117-aff2-aefa158536a4" />

## Basic
Environment: 6* RTX 3080 memory 10G; cuda12.8;ubuntu.
V2_Py_project/environment provide .yaml and requirements.txt, you can try which more convenient to config env.

This project is answering Single question running by main-singleQ.py, you should config your `LLM-API` / `LangSmith-key` in nodes.py and `main-singleQ.py`. For running through this pipeline you should config following this:
1. RAG Database are Lilian Weng's blog and 2018 wikipedia dump, (but wiki dump's embedding database too large so i haven't upload), you can comment `from src.retriever_of_wiki import build_retriever_tool` use `from src.retriever import build_retriever_tool` have a try.
2. blog's retriever use dense retrieval method, wiki's retriever use hybrid search(dense retriever + BM25), for blog's retriever you should config `local_model_path` in `V2_Py_project\src\retriever.py`
3. For re-ranker you should config `RERANKER_PATH` in `V2_Py_project\src\re_ranker.py`
4. Regarding the models mentioned in second and third points above you can download [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2); [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)

## Sundry
1. LLM can use ollama or vllm for local runing, our vllm version is `0.14.1`;
2. Wiki's retriever use [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m)
3. Our system ofen use `llama3-8b`/`qwen3-8b` so sometime encounter `No tool calls generated` in `Query generator agent`, so within the `query_generator_node` function, code can be rewritten to force a database lookup for each query generated.




