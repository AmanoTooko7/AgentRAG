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
from src_of_flashRAG.nodes_of_flashRAG import init_model 


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
        elif '2wiki' in dataset_name:
            ds_type = '2wikimultihopqa'
        elif 'trivia' in dataset_name:
            ds_type = 'triviaqa'
        elif 'hotpot' in dataset_name:
            ds_type = 'hotpotqa'
        elif 'nq' in dataset_name:
            ds_type = 'nq'
        else:
            ds_type = 'nq'

        print(f"🔥 [MyV2Pipeline] 开始推理 {len(input_questions)} 条数据...")
        
        for question in tqdm(input_questions):
            # 构造 V2 输入
            inputs = {
                "original_question": question,
                "dataset_type": ds_type, # 此参数决定根据不同的数据集模型输出不同格式的答案
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

    init_model("qwen3-8b")
    start_time = time.time()
    # 你的数据集根目录
    # DATA_DIR = "/home/lab/ybw/FlashRAG-main/KILT_datasets"
    DATA_DIR = "/home/lab/ybw/FlashRAG-main/datasets_flashRAG"
    
    # 'hotpotqa', 'nq', 'fever', 'triviaqa'，'strategyqa'，'2wikimultihopqa'
    TARGET_DATASET = "2wikimultihopqa" 

    config_dict = {  #见https://github.com/RUC-NLPIR/FlashRAG/blob/main/docs/original_docs/configuration.md
        "device": "cuda:0",
        "framework": "openai",

        # --- 数据集路径配置 (关键修正) ---
        "data_dir": DATA_DIR,
        "dataset_name": TARGET_DATASET,
        "split": ["dev"],  # 对应文件名 dev.jsonl,strategyqa从原本的train.jsonl改为了dev.jsonl
        
        # --- 测试样本数 ---
        "test_sample_num": 5, # ⚠️ 测试数据条数
        "random_sample": True,
        
        # --- 结果保存 ---
        # "save_dir": f"./results_v2_{TARGET_DATASET}",
        "save_dir": f"/home/lab/ybw/FlashRAG-main/ybw_test/results_v2_{TARGET_DATASET}",
        
        # --- 占位符 (骗过 Config 检查，实际 V2 内部自己处理) ---
        "retrieval_method": "V2_Gemma_IVFPQ", 
        "retrieval_topk": 200, # 对应 V2 内部逻辑
        "use_reranker": True,
        "rerank_model_name": "bge-reranker-v2-m3",
        "rerank_topk": 3, 

        "generator_model": "openai",
        "generator_model_path": "deepseek-chat",
        "openai_api_key": "placeholder",  # 见/home/lab/ybw/V2_Py_project/src_of_flashRAG/nodes_of_flashRAG.py
        "openai_api_base": "placeholder", # 见/home/lab/ybw/V2_Py_project/src_of_flashRAG/nodes_of_flashRAG.py
        "metrics": ["em", "f1", "acc"], 
        "seed": 2026
    }
    
    print("⚙️ 加载配置...")
    config = Config(config_dict=config_dict)
    
    print(f"📥 从 {DATA_DIR}/{TARGET_DATASET} 加载数据集...")
    data_splits = get_dataset(config)
    
    # 获取 'val' 集 (对应 split 配置)
    test_data = data_splits["dev"] 
    print(f"✅ 数据集加载成功，共 {len(test_data)} 条")
    print(f"   {BOLD_ON}评估数据: {test_data[0].question}{BOLD_OFF}")
    
    print("🛠️ 初始化 Pipeline...")
    pipeline = MyV2Pipeline(config)
    
    print(f"{BOLD_ON}🏃 运行...—————————————————————————————————————————————————————————————{BOLD_OFF}")
    pipeline.run(test_data)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"⏱️  {elapsed_time:.2f} 秒")

if __name__ == "__main__":
    main()
