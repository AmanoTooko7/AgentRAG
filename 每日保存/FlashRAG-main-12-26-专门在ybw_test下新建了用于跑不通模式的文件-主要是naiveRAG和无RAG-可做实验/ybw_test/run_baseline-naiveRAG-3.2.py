# 跑naiveRAG的实验
import sys
import os
V2_PROJECT_PATH = "/home/lab/ybw/V2_Py_project"
sys.path.append(V2_PROJECT_PATH)
import time
import argparse
# import torch
from langsmith import traceable
from flashrag.generator.openai_generator import OpenaiGenerator
from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.prompt import PromptTemplate
# 导入官方内置的 Pipeline
from flashrag.pipeline import (
    BasicPipeline, SequentialPipeline, ConditionalPipeline, AdaptivePipeline, # pipeline 
    SuRePipeline, REPLUGPipeline ,       # branching-pipeline
    IterativePipeline, SelfRAGPipeline, FLAREPipeline, SelfAskPipeline, IRCOTPipeline, RQRAGPipeline, # Loop
    # reasoning-pipeline
)
from src_of_flashRAG.retriever_of_wiki import WikiRetriever
from src_of_flashRAG.re_ranker import get_ranker_score


# ————————————用来加入LangSmith的前置程序——————————————————
original_generate = OpenaiGenerator.generate
@traceable(run_type="llm", name="FlashRAG_Generator")
def traced_generate(self, input_list):
    return original_generate(self, input_list)

OpenaiGenerator.generate = traced_generate
print("✅ 已成功注入 LangSmith Tracing 到 FlashRAG Generator")
# ————————————用来加入LangSmith的前置程序——————————————————


def setup_langsmith(project_name="V2跑实验"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    if "LANGCHAIN_API_KEY" not in os.environ:
        os.environ["LANGCHAIN_API_KEY"] = "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"
        # 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
    os.environ["LANGCHAIN_PROJECT"] = project_name

# ================= 这里自定义检索器-采用和我的V2相同的检索器 =================
class CustomRetrieverWrapper:
    """
    适配器：将 V2 的 WikiRetriever 封装成 FlashRAG 可用的格式。
    确保 Baseline 使用与 V2 完全一致的检索和重排序逻辑。
    """
    def __init__(self, config):
        self.config = config
        print("🚀 初始化 V2 同款检索器 (Gemma + Faiss + SQLite)...")
        # 初始化你的检索器 (它会自动加载模型和索引)
        self.v2_retriever = WikiRetriever()
        self.use_reranker = config['use_reranker']
        
    def batch_search(self, query_list):
        """
        FlashRAG 要求的接口。
        输入: query_list (list of str)
        输出: list of list of dict [{'contents': ..., 'id': ..., 'score': ...}]
        """
        batch_results = []
        
        # FlashRAG 通常是 batch 进来的，我们需要循环处理
        for query in query_list:
            # 1. 调用 V2 检索器 (返回 LangChain Document 对象列表)
            # 注意：WikiRetriever.invoke 内部已经包含了 Top-K (200) 的逻辑
            docs = self.v2_retriever.invoke(query)
            
            # 提取文档内容列表供 Re-ranker 使用
            # 注意：你的 invoke 返回的 page_content 已经是 "Title: ...\nContent: ..." 格式
            raw_doc_contents = [d.page_content for d in docs]
            
            final_docs = []
            
            # 2. 执行重排序 (如果配置开启)
            if self.use_reranker and raw_doc_contents:
                # print(f"   -> 正在重排序 {len(raw_doc_contents)} 个文档...")
                # get_ranker_score 返回的是重排后的 top-k 字符串列表
                reranked_texts = get_ranker_score(query, raw_doc_contents)
                
                # 封装成 FlashRAG 格式
                for text in reranked_texts:
                    final_docs.append({
                        "id": "unknown", # 重排后丢失原始ID，不过生成阶段通常不需要ID
                        "contents": text,
                        "score": 0.99 # 假分数
                    })
            else:
                # 如果不重排，直接使用检索结果
                for d in docs:
                    final_docs.append({
                        "id": str(d.metadata.get('id', 'unknown')),
                        "contents": d.page_content,
                        "score": d.metadata.get('score', 0.0)
                    })
            
            batch_results.append(final_docs)
            
        return batch_results

    # 为了兼容某些 Pipeline 可能调用的 save_cache 方法，留空即可
    def _save_cache(self):
        pass

# ================= 3. DummyRetriever (Zero-shot 用) =================
class DummyRetriever:
    def __init__(self, config=None):
        pass
    def batch_search(self, query_list):
        return [[] for _ in query_list]
    def _save_cache(self):
        pass

# ================= 4. Prompt 生成函数 =================
def get_adaptive_prompt(config, method, dataset_name):
    if method == 'zero_shot':
        sys_prompt = "Answer the question based on your own knowledge. "
    else:
        sys_prompt = "Answer the question based on the given documents. "

    if 'fever' in dataset_name:
        format_instruction = "Determine if the claim is supported. You MUST output exactly one word: SUPPORTS, REFUTES, or NOT ENOUGH INFO."
    elif 'strategy' in dataset_name:
        format_instruction = "Answer with Yes or No. You MUST output exactly one word: Yes or No."
    else: 
        format_instruction = "Answer concisely. Output ONLY the entity or short phrase. Do not output full sentences."

    final_sys_prompt = sys_prompt + format_instruction

    if method == 'zero_shot':
        return PromptTemplate(config=config, system_prompt=final_sys_prompt, user_prompt="Question: {question}\nAnswer:")
    else:
        return PromptTemplate(config=config, system_prompt=final_sys_prompt + "\n\nThe following are given documents:\n{reference}", user_prompt="Question: {question}\nAnswer:")

parser = argparse.ArgumentParser(description="Run Baseline RAG experiments")
parser.add_argument("--method", type=str, required=True, 
                    choices=['zero_shot', 'naive', 'selfask', 'replug', 'selfrag', 'recomp', 'adaptiverag'], 
                    help="Baseline method")
parser.add_argument("--dataset", type=str, required=True, help="'nq','triviaqa','hotpotqa', '2wikimultihopqa', 'fever', 'strategyqa',")
parser.add_argument("--model", type=str, default="qwen3-8b", help="Generator model key (for API)")
parser.add_argument("--num", type=int, default=3, help="Number of samples")
parser.add_argument("--gpu", type=str, default="0", help="GPU ID")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

DASHSCOPE_API_KEY = "sk-757aab3d1cdf47d2896bfef1e45efa1d"   
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

MODEL_MAP = {
        "qwen3-8b": "qwen3-8b",
        "qwen3-32b": "qwen3-32b",
        "qwen3-next-80b-a3b-instruct": "qwen3-next-80b-a3b-instruct"
}
generator_path = MODEL_MAP.get(args.model, "qwen3-8b")

def main():
    setup_langsmith(f"{args.method}_{args.model}_{args.dataset}_{args.model}")
    print(f"Lanuch baseline: method=[{args.method}] | model=[{args.model}] | dataset=[{args.dataset}]")
    DATA_DIR = "/home/lab/ybw/FlashRAG-main/datasets_flashRAG"
    SAVE_DIR = f"/home/lab/ybw/FlashRAG-main/ybw_test/results_baselines/{args.method}/{args.dataset}"

    # 检索的嵌入模型
    RETRIEVER_PATH = "/home/lab/Shared_model/embeddinggemma-300m"
    INDEX_PATH = "/home/lab/ybw/Generate_VectorsDB/DB/faiss_index/gemma_300m_ivfpq.index"
    CORPUS_PATH = "/home/lab/ybw/Generate_VectorsDB/DB/raw_data/psgs_w100.jsonl"

    config_dict = {
        "device": "cuda",
        "gpu_id": args.gpu,
        "data_dir": DATA_DIR, 
        "dataset_name": args.dataset,
        "split": ["dev"],
        "test_sample_num": args.num,
        "random_sample": True,
        "seed": 2026,
         
        #Gemma和FAISS
        "retrieval_method": "e5",
        "use_sentence_transformer": True,
        "model_path": RETRIEVER_PATH,
        "index_path": INDEX_PATH,
        "corpus_path": CORPUS_PATH,
        "retrieval_topk": 200,   # <--------------------
        "use_reranker": True,
        "rerank_model_path": "/home/lab/Shared_model/bge-reranker-v2-m3",
        "rerank_topk": 3,
        "use_fp16": True,
        "instruction": "Given a search query, retrieve relevant passages:",

        # 大模型
        "framework": "openai",
        "generator_model": generator_path,
        "generator_model_path": generator_path,
        "openai_setting": {
            "api_key": DASHSCOPE_API_KEY,
            "base_url": DASHSCOPE_BASE_URL,
        },
        "generation_params": {
            "temperature": 0.1, "max_tokens": 1024,
            "extra_body": {
                "enable_search": False,   # 关闭联网搜索（保证 RAG 纯净度）
                "enable_thinking": False  # 关闭思考模式（解决 400 报错）
            }
        },

        "metrics": ["em", "f1", "acc"],
        "save_dir": SAVE_DIR,
        "save_intermediate_data": True
    }

    # zero-shot (无RAG)
    if args.method == 'zero_shot':
        # 关键修正：给一个假名字骗过 Config 初始化，但在下面调用 naive_run
        config_dict['retrieval_method'] = 'bm25' 
        config_dict['use_reranker'] = False
        # 既然不检索，不需要加载索引，节省资源
        config_dict['index_path'] = None 
        config_dict['corpus_path'] = None
    if args.method == 'recomp':
        config_dict['refiner_name'] = 'recomp_abstractive_nq'
        config_dict['refiner_model_path'] = 'fangyuan/nq_abstractive_compressor'
    
    if args.method == 'adaptiverag':
        config_dict['judger_name'] = 'adaptive'


    print("⚙️ 加载配置...")
    config = Config(config_dict=config_dict)
    print("📥 加载数据集...")
    dataset = get_dataset(config)["dev"]
    custom_prompt = get_adaptive_prompt(config, args.method, args.dataset)
    print(f"🛠️ 初始化 Pipeline: {args.method.upper()}...")

    custom_retriever = None
    if args.method != 'zero_shot':
        custom_retriever = CustomRetrieverWrapper(config)

    if args.method == 'zero_shot':
        # 必须传入 DummyRetriever 绕过初始化检查
        pipeline = SequentialPipeline(config, retriever=DummyRetriever())
    elif args.method == 'naive':
        pipeline = SequentialPipeline(config, prompt_template=custom_prompt, retriever=custom_retriever)
    elif args.method == 'selfask':
        pipeline = SelfAskPipeline(config, prompt_template=custom_prompt, retriever=custom_retriever)
    elif args.method == 'replug':
        pipeline = REPLUGPipeline(config, prompt_template=custom_prompt, retriever=custom_retriever)
    elif args.method == 'selfrag':
        pipeline = SelfRAGPipeline(config, prompt_template=custom_prompt, retriever=custom_retriever)    # Self-RAG 需要特殊的 token 处理，FlashRAG 已内置
    elif args.method == 'recomp':
        pipeline = SequentialPipeline(config, prompt_template=custom_prompt, retriever=custom_retriever)
    elif args.method == 'adaptiverag':
        pipeline = AdaptivePipeline(config, prompt_template=custom_prompt, retriever=custom_retriever) 

        
    print("🏃 运行...")
    start_time = time.time()
    
    if args.method == 'zero_shot':
        # 显式调用 naive_run (不检索，直接生成)
        pipeline.naive_run(dataset, do_eval=True)
    else:
        # 其他方法正常运行 (检索 + 生成)
        pipeline.run(dataset, do_eval=True)
    
    end_time = time.time()
    print(f"⏱️ 耗时: {end_time - start_time:.2f} 秒")
    print(f"📊 结果已保存至: {SAVE_DIR}")

if __name__ == "__main__":
    main()
