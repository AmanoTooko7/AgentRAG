# gemini生成，目的是1.跑Naive RAG2.数据对齐检查万一 Faiss 索引和 JSONL 没对齐（比如 Faiss 指向“亚里士多德”，JSONL 里却是“阿凡达”）
#运行环境是FlashRAG_Lab
# 运行该文件的目录是/home/lab！！！！！

import sys
import os
import time
# # 将 V2 项目的根目录加入 Python 搜索路径
V2_PROJECT_PATH = "/home/lab/ybw/V2_Py_project"
sys.path.append(V2_PROJECT_PATH)


import torch
from tqdm import tqdm
from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import BasicPipeline
# from src.graph import build_graph
from src_of_flashRAG.graph import build_graph

BOLD_ON = "\033[1m"    # 开启加粗
BOLD_OFF = "\033[0m"   # 恢复默认样式

# ================= 2. 定义 V2 Pipeline =================
class MyV2Pipeline(BasicPipeline):
    def __init__(self, config):
        super().__init__(config)
        print("🚀 [MyV2Pipeline] 正在初始化 V2 LangGraph...")
        self.graph = build_graph()
        
    def run(self, dataset):
        input_questions = dataset.question
        pred_answer_list = []

        # 判断当前跑的是哪个数据集
        dataset_name = self.config['dataset_name']
        if 'fever' in dataset_name:
            ds_type = 'fever'
        elif 'strategy' in dataset_name:
            ds_type = 'strategyqa'
        else:
            ds_type = 'qa'

        print(f"🔥 [MyV2Pipeline] 开始推理 {len(input_questions)} 条数据...")
        
        for question in tqdm(input_questions):
            # 构造 V2 输入
            inputs = {
                "original_question": question,
                "dataset_type": ds_type, # ✅ 传入类型
                "plan": [], "history": [], "messages": [],
                "current_sub_question": "",
                "plan_archive": [],
                "reflection": {},
                "num_cycles": 0,
                "re_strategist_call_count": 0
            }
            
            try:
                # 调用 V2 Graph
                final_state = self.graph.invoke(inputs, config={"recursion_limit": 50}) #这里返回的是V2所有chunk的内容
                raw_answer  = final_state.get("final_answer", "") # 得到最终答案
                
                import re
                match = re.search(r"<answer>(.*?)</answer>", raw_answer, re.DOTALL)
                if match:
                    clean_answer = match.group(1).strip()
                else:
                    clean_answer = raw_answer # 提取失败则用原样
                
                # print(f"   🤖 Raw: {raw_answer[:50]}... -> Clean: {clean_answer}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                final_answer = "Error"
            
            pred_answer_list.append(clean_answer)
            
        # 更新结果并评估
        dataset.update_output("pred", pred_answer_list)
        dataset = self.evaluate(dataset, do_eval=True)
        return dataset

# ================= 3. 主程序配置 =================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 强行把工作目录切换到脚本所在目录
    os.chdir(script_dir)
    print(f"📂 工作目录已切换至: {os.getcwd()}")

    start_time = time.time()
    # 你的数据集根目录
    DATA_DIR = "/home/lab/ybw/FlashRAG-main/KILT_datasets"
    
    # 想要跑的数据集名称 (文件夹名)
    # 'hotpotqa', 'nq', 'fever', 'triviaqa'
    TARGET_DATASET = "fever" 

    config_dict = {  #见https://github.com/RUC-NLPIR/FlashRAG/blob/main/docs/original_docs/configuration.md
        "device": "cuda:0",
        "framework": "openai",

        # --- 数据集路径配置 (关键修正) ---
        "data_dir": DATA_DIR,
        "dataset_name": TARGET_DATASET,
        "split": ["val"],  # 对应文件名 val.jsonl
        
        # --- 测试样本数 ---
        "test_sample_num": 50, # ⚠️ 测试数据条数，没问题了改成 None (跑全量)
        "random_sample": True,
        
        # --- 结果保存 ---
        # "save_dir": f"./results_v2_{TARGET_DATASET}",
        "save_dir": f"/home/lab/ybw/FlashRAG-main/ybw_test/results_v2_{TARGET_DATASET}",
        
        # --- 占位符 (骗过 Config 检查，实际 V2 内部自己处理) ---
        "retrieval_method": "V2_Gemma_IVFPQ", 
        "retrieval_topk": 100, # 对应 V2 内部逻辑
        "use_reranker": True,
        "rerank_model_name": "bge-reranker-v2-m3",
        "rerank_topk": 3, 

        "generator_model": "openai",
        "generator_model_path": "deepseek-chat",
        "openai_api_key": "sk-hOiSoNaP9k5qXSw1MjE9EmaNB3oAwMB9qxr4rZN3zkPq80ch", # 填你的 CloseAI Key
        "openai_api_base": "https://api.openai-proxy.org/v1", # 填 CloseAI URL
        "metrics": ["em", "f1", "acc", "precision", "recall"], 
    }
    
    print("⚙️ 加载配置...")
    config = Config(config_dict=config_dict)
    
    print(f"📥 从 {DATA_DIR}/{TARGET_DATASET} 加载数据集...")
    data_splits = get_dataset(config)
    
    # 获取 'val' 集 (对应 split 配置)
    test_data = data_splits["val"] 
    print(f"✅ 数据集加载成功，共 {len(test_data)} 条")
    print(f"   {BOLD_ON}评估数据: {test_data[0].question}{BOLD_OFF}")
    
    print("🛠️ 初始化 Pipeline...")
    pipeline = MyV2Pipeline(config)
    
    print("🏃 运行...—————————————————————————————————————————————————————————————")
    pipeline.run(test_data)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"⏱️  {elapsed_time:.2f} 秒")

if __name__ == "__main__":
    main()
