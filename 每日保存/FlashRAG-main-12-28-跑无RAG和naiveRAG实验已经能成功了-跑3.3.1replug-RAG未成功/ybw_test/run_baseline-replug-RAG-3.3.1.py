# 来自run_baseline-naiveRAG-3.2.py，跑self-rag--无法跑该方法，需要在提供的微调后的模型上跑
# 以下跑replug方法
#3.3.1是来自run_baseline-replug-RAG-3.3.py的第二个版本
import sys
import os
import time
import argparse
from tqdm import tqdm
import openai
from langsmith import traceable

# 1. 路径设置
V2_PROJECT_PATH = "/home/lab/ybw/V2_Py_project"
sys.path.append(V2_PROJECT_PATH)

from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.prompt import PromptTemplate
from flashrag.pipeline import BasicPipeline # 继承基础类，重写逻辑
from flashrag.generator.openai_generator import OpenaiGenerator
from src_of_flashRAG.retriever_of_wiki import WikiRetriever
from src_of_flashRAG.re_ranker import get_ranker_score

# ==============================================================================
# 🛡️ 魔法补丁：给 OpenaiGenerator 穿上防弹衣
# ==============================================================================
original_generate = OpenaiGenerator.generate

def safe_generate(self, input_list):
    try:
        return original_generate(self, input_list)
    except openai.BadRequestError as e:
        error_msg = str(e)
        if "data_inspection_failed" in error_msg or "inappropriate content" in error_msg:
            print(f"\n⚠️ [风控拦截] 阿里百炼拒绝生成: {error_msg[:50]}...")
            return ["CONTENT_FILTERED"] * len(input_list)
        else:
            raise e
    except Exception as e:
        print(f"\n❌ [未知生成错误] {e}")
        return ["GENERATION_ERROR"] * len(input_list)

OpenaiGenerator.generate = safe_generate
print("✅ 已注入风控异常捕获补丁 (Safe Generator)")

# 注入 LangSmith
@traceable(run_type="llm", name="FlashRAG_Generator")
def traced_generate(self, input_list):
    return safe_generate(self, input_list)

OpenaiGenerator.generate = traced_generate
print("✅ 已成功注入 LangSmith Tracing")

def setup_langsmith(project_name):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    if "LANGCHAIN_API_KEY" not in os.environ:
        os.environ["LANGCHAIN_API_KEY"] = "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"
    os.environ["LANGCHAIN_PROJECT"] = project_name

# ================= 自定义检索器 =================
class CustomRetrieverWrapper:
    def __init__(self, config):
        self.config = config
        print("🚀 初始化 V2 同款检索器 (Gemma + Faiss + SQLite)...")
        self.v2_retriever = WikiRetriever()
        self.use_reranker = config['use_reranker']
        
    def batch_search(self, query_list):
        batch_results = []
        for query in query_list:
            docs = self.v2_retriever.invoke(query)
            raw_doc_contents = [d.page_content for d in docs]
            final_docs = []
            
            # 这里我们不仅返回内容，还尽量保留 Re-ranker 的顺序信息
            if self.use_reranker and raw_doc_contents:
                # get_ranker_score 返回的是按分数降序排列的文本列表
                reranked_texts = get_ranker_score(query, raw_doc_contents)
                for idx, text in enumerate(reranked_texts):
                    # 我们人为给一个分数，排名越靠前分数越高，模拟 Logits
                    fake_score = 1.0 - (idx * 0.01) 
                    final_docs.append({"id": "unknown", "contents": text, "score": fake_score})
            else:
                for d in docs:
                    final_docs.append({
                        "id": str(d.metadata.get('id', 'unknown')),
                        "contents": d.page_content,
                        "score": d.metadata.get('score', 0.0)
                    })
            batch_results.append(final_docs)
        return batch_results
    
    def _save_cache(self): pass

# ================= 🔥 API 版 REPLUG Pipeline (核心修改) =================
class API_REPLUGPipeline(BasicPipeline):
    """
    适配 API 模型的 REPLUG 变体。
    原理：
    1. 检索 Top-K 文档。
    2. 分别将 (Query + Doc_i) 喂给模型，生成 K 个答案。
    3. 选择 Re-ranker 分数最高的文档对应的答案作为最终输出。
    (这是在无法获取 API Logits 情况下的最佳近似策略)
    """
    def __init__(self, config, prompt_template=None, retriever=None):
        super().__init__(config, prompt_template)
        self.retriever = retriever
        # 初始化生成器 (OpenAI Generator)
        from flashrag.utils import get_generator
        self.generator = get_generator(config)

    def run(self, dataset, do_eval=True):
        input_query = dataset.question
        # 1. 检索
        retrieval_results = self.retriever.batch_search(input_query)
        dataset.update_output("retrieval_result", retrieval_results)
        
        pred_answer_list = []
        
        print(f"🔥 正在运行 API-REPLUG (Best-of-K 策略)...")
        for i, (q, docs) in enumerate(zip(tqdm(input_query), retrieval_results)):
            # 取出 Top-K 文档 (这里取配置里的 rerank_topk，比如 3)
            top_docs = docs[:self.config['rerank_topk']]
            
            if not top_docs:
                pred_answer_list.append("Unknown")
                continue

            # 策略：因为我们信任 Re-ranker (BGE-M3)，且无法获取 API 的 PPL
            # 我们直接取 Re-ranker 排名第一的文档进行生成
            # 这等价于 REPLUG 中 P(Doc|Query) 最高的文档
            
            # 构造 Prompt (只用 Top-1 文档)
            # 注意：这里和 Naive RAG 的区别是，Naive RAG 会把 Top-3 拼在一起
            # REPLUG (Ensemble) 的逻辑是单独处理。这里我们取 Top-1 单独处理。
            best_doc = top_docs[0]
            
            # 构造单文档 Prompt
            # 我们需要临时构造一个只包含单文档的 retrieval_result 结构
            single_doc_result = [best_doc]
            
            prompt = self.prompt_template.get_string(
                question=q, 
                retrieval_result=single_doc_result
            )
            
            try:
                response = self.generator.generate([prompt])[0]
            except:
                response = "Error"
                
            pred_answer_list.append(response)

        dataset.update_output("pred", pred_answer_list)
        dataset = self.evaluate(dataset, do_eval=do_eval)
        return dataset

# ================= Prompt 生成 =================
def get_adaptive_prompt(config, method, dataset_name):
    sys_prompt = "Answer the question based on the given documents. "
    if 'fever' in dataset_name:
        format_instruction = "Determine if the claim is supported. You MUST output exactly one word: SUPPORTS, REFUTES, or NOT ENOUGH INFO."
    elif 'strategy' in dataset_name:
        format_instruction = "Answer with Yes or No. You MUST output exactly one word: Yes or No."
    else: 
        format_instruction = "Answer concisely. Output ONLY the entity or short phrase. Do not output full sentences."

    final_sys_prompt = sys_prompt + format_instruction
    
    return PromptTemplate(
        config=config, 
        system_prompt=final_sys_prompt + "\n\nThe following are given documents:\n{reference}", 
        user_prompt="Question: {question}\nAnswer:"
    )

# ================= 主程序 =================
parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, required=True, choices=['replug'], help="Baseline method")
parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
parser.add_argument("--model", type=str, default="qwen3-8b", help="Generator model key")
parser.add_argument("--num", type=int, default=300, help="Number of samples")
parser.add_argument("--gpu", type=str, default="0", help="GPU ID")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["V2_TARGET_DEVICE"] = args.gpu 

DASHSCOPE_API_KEY = "sk-757aab3d1cdf47d2896bfef1e45efa1d"   
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

MODEL_MAP = {
        "qwen3-8b": "qwen3-8b",
        "qwen3-32b": "qwen3-32b",
        "qwen3-next-80b-a3b-instruct": "qwen3-next-80b-a3b-instruct"
}
generator_path = MODEL_MAP.get(args.model, "qwen3-8b")

def main():
    setup_langsmith(f"REPLUG_{args.model}_{args.dataset}")
    print(f"🧪 Launch REPLUG (API Variant): model=[{args.model}] | dataset=[{args.dataset}]")
    
    DATA_DIR = "/home/lab/ybw/FlashRAG-main/datasets_flashRAG"
    SAVE_DIR = f"/home/lab/ybw/FlashRAG-main/ybw_test/results_baselines/replug/{args.dataset}"

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
         
        # 检索配置
        "retrieval_method": "e5",
        "use_sentence_transformer": True,
        "model_path": RETRIEVER_PATH,
        "index_path": INDEX_PATH,
        "corpus_path": CORPUS_PATH,
        "retrieval_topk": 200,
        "use_reranker": True,
        "rerank_model_path": "/home/lab/Shared_model/bge-reranker-v2-m3",
        "rerank_topk": 3, # 最终只取 Top-1 给 REPLUG 生成
        "use_fp16": True,
        "instruction": "Given a search query, retrieve relevant passages:",

        # 生成器配置
        "framework": "openai",
        "generator_model": generator_path,
        "generator_model_path": generator_path,
        "openai_setting": {
            "api_key": DASHSCOPE_API_KEY,
            "base_url": DASHSCOPE_BASE_URL,
        },
        "generation_params": {
            "temperature": 0.1, 
            "max_tokens": 1024,
            "extra_body": {"enable_search": False, "enable_thinking": False}
        },

        "metrics": ["em", "f1", "acc"],
        "save_dir": SAVE_DIR,
        "save_intermediate_data": True
    }

    if args.dataset == "strategyqa":
        config_dict["split"] = ["train"]

    print("⚙️ 加载配置...")
    config = Config(config_dict=config_dict)
    
    print("📥 加载数据集...")
    dataset = get_dataset(config)[config_dict["split"][0]]
    
    custom_prompt = get_adaptive_prompt(config, "replug", args.dataset)
    print(f"🛠️ 初始化 Pipeline: API-REPLUG...")

    custom_retriever = CustomRetrieverWrapper(config)
    
    # ✅ 使用自定义的 API_REPLUGPipeline
    if args.method == 'replug':
        pipeline = API_REPLUGPipeline(config, prompt_template=custom_prompt, retriever=custom_retriever)

    print("🏃 运行 REPLUG...")
    start_time = time.time()
    
    pipeline.run(dataset, do_eval=True)
    
    end_time = time.time()
    print(f"⏱️ 耗时: {end_time - start_time:.2f} 秒")
    print(f"📊 结果已保存至: {SAVE_DIR}")

if __name__ == "__main__":
    main()