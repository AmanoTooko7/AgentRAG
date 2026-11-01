# 目的是1.跑Naive RAG2.数据对齐检查万一 Faiss 索引和 JSONL 没对齐（比如 Faiss 指向“亚里士多德”，JSONL 里却是“阿凡达”）
# 运行环境是ybw_AgentRAG-2
# 运行该文件的目录是/home/lab！！！！！
# 这里和run_ours_rag.py的区别是使用命令行进行批量跑实验
# 命令行代码：cd /home/lab/ybw/FlashRAG-main/ybw_test   conda activate ybw_AgentRAG
#            python run_ours_rag_CLI-2.py --model "qwen3-8b" --dataset "nq" --num 3 --gpu "4"

import sys
import os
import time
import argparse

parser = argparse.ArgumentParser(description="Run V2 RAG experiment")

# 注意--model无实际作用，这里只是用来后续保存结果的文件夹名，实际设置每个节点的model在nodes_of_flashRAG.py函数里
parser.add_argument("--model", type=str, required=True, help="Model key (e.g., llama3-8b)")
parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., nq, triviaqa, hotpotqa, 2wikimultihopqa)")
parser.add_argument("--num", type=int, default=3, help="Number of samples to test")
parser.add_argument("--gpu", type=str, default="5", help="GPU ID")
args = parser.parse_args()

os.environ["V2_TARGET_DEVICE"] = str(args.gpu)
print(f"🔧 [System] 已设置 V2_TARGET_DEVICE = cuda:{args.gpu}")

# # 将 V2 项目的根目录加入 Python 搜索路径
V2_PROJECT_PATH = "/home/lab/ybw/V2_Py_project"
sys.path.append(V2_PROJECT_PATH)


# import torch
import json
from tqdm import tqdm
from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import BasicPipeline
# from src.graph import build_graph
from src_of_flashRAG.graph import build_graph
# from src_of_flashRAG.nodes_of_flashRAG import init_model 


BOLD_ON = "\033[1m"    # 开启加粗
BOLD_OFF = "\033[0m"   # 恢复默认样式

def setup_langsmith(project_name="V2跑实验"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    if "LANGCHAIN_API_KEY" not in os.environ:
        os.environ["LANGCHAIN_API_KEY"] = "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"
        # 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
    os.environ["LANGCHAIN_PROJECT"] = project_name

#此类用于保存终端所有输出
class DualLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8") # 使用追加模式

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # 确保实时写入

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# ================= 2. 定义 V2 Pipeline =================
class MyV2Pipeline(BasicPipeline):
    def __init__(self, config):
        super().__init__(config)
        print("🚀 [MyV2Pipeline] 正在初始化 V2 LangGraph...")
        self.graph = build_graph()
        
    def run(self, dataset):
        input_questions = dataset.question
        
        # 1. 定义中间结果保存路径
        save_dir = self.config['save_dir']
        # os.makedirs(save_dir, exist_ok=True)
        intermediate_file = os.path.join(save_dir, "intermediate_results.jsonl")
        

        processed_results = []
        if os.path.exists(intermediate_file):
            print(f"🔄 检测到中间文件: {intermediate_file}，正在加载已完成进度...")
            with open(intermediate_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        processed_results.append(json.loads(line))
                    except:
                        pass
            print(f"✅ 已完成 {len(processed_results)} 条，将跳过这些数据。")
        
        # 初始化预测列表 (填入已跑过的结果)
        # 注意：FlashRAG 最终评估需要列表长度与数据集一致，所以我们要把已跑过的先填进去
        pred_answer_list = [item['pred'] for item in processed_results]
        
        # 确定开始索引
        start_index = len(processed_results)

        # 判断当前跑的是哪个数据集
        dataset_name = self.config['dataset_name']
        if 'fever' in dataset_name: ds_type = 'fever'
        elif 'strategy' in dataset_name: ds_type = 'strategyqa'
        elif '2wiki' in dataset_name: ds_type = '2wikimultihopqa'
        elif 'trivia' in dataset_name: ds_type = 'triviaqa'
        elif 'hotpot' in dataset_name: ds_type = 'hotpotqa'
        elif 'nq' in dataset_name: ds_type = 'nq'
        # else: ds_type = 'nq'

        print(f"🔥 [MyV2Pipeline] 开始推理 (总数: {len(input_questions)}, 剩余: {len(input_questions) - start_index})...")
        
        for i in tqdm(range(start_index, len(input_questions)), file=sys.stdout, ascii=False, ncols=100):
            question = input_questions[i]
            
            # 构造 V2 输入
            inputs = {
                "original_question": question,
                "dataset_type": ds_type, 
                "plan": [], "history": [], "messages": [],
                "current_sub_question": "", "plan_archive": [],
                "reflection": {}, "num_cycles": 0,
                "re_strategist_call_count": 0
            }
            
            clean_answer = "Error"
            try:
                # 调用 V2 Graph
                final_state = self.graph.invoke(inputs, config={"recursion_limit": 60})
                raw_answer  = final_state.get("final_answer", "")
                
                import re
                match = re.search(r"<answer>\s*(.*?)\s*</answer>", raw_answer, re.DOTALL | re.IGNORECASE)
                if match:
                    clean_answer = match.group(1).strip()
                else:
                    clean_answer = raw_answer
                clean_answer = re.sub(r"^(\*\*|__)?(Answer|answer):?(\*\*|__)?\s*", "", clean_answer)
                
            except Exception as e:
                print(f"\n❌ Error processing sample {i}: {e}")
                clean_answer = "Error"
            
            # 加入内存列表
            pred_answer_list.append(clean_answer)
            
            # 4. 【实时保存逻辑】每跑完一条，立刻写入文件
            current_result = {
                "id": dataset.id[i], # 获取当前样本ID
                "question": question,
                "pred": clean_answer,
                "gold": dataset.golden_answers[i]
            }
            
            # 使用 'a' (append) 追加写入
            with open(intermediate_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(current_result, ensure_ascii=False) + "\n")
            
        # 5. 最终评估
        # 循环结束后，pred_answer_list 包含了所有结果（旧的+新的）
        dataset.update_output("pred", pred_answer_list)
        dataset = self.evaluate(dataset, do_eval=True)
        return dataset

# ================= 3. 主程序配置 =================

def main():
    # setup_langsmith(f"V2_{args.model}_{args.dataset}")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)# 强行把工作目录切换到脚本所在目录
    print(f"{BOLD_ON}🧪 启动实验: 模型=[{args.model}] | 数据集=[{args.dataset}] | 数量=[{args.num}]{BOLD_OFF}")

    # init_model(args.model)
    start_time = time.time()
    # 数据集根目录
    # DATA_DIR = "/home/lab/ybw/FlashRAG-main/KILT_datasets"
    DATA_DIR = "/home/lab/ybw/FlashRAG-main/datasets_flashRAG"
    
    BASE_SAVE_DIR = f"/home/lab/ybw/FlashRAG-main/ybw_test/results_v2/{args.model}/{args.dataset}"
    # 以下是数据集跑到一半因网路问题中断，在原有基础上继续跑而使用的地址
    # RESUME_DIR  = f"/home/lab/ybw/FlashRAG-main/ybw_test/results_v2/llama3-8b/hotpotqa/hotpotqa_2026_01_17_17_40_experiment"


    config_dict = {  #见https://github.com/RUC-NLPIR/FlashRAG/blob/main/docs/original_docs/configuration.md
        "device": f"cuda:{args.gpu}",
        "framework": "openai",

        # --- 数据集路径配置 (关键修正) ---
        "data_dir": DATA_DIR,
        "dataset_name": args.dataset,
        "split": ["dev"],  # 对应文件名 dev.jsonl,strategyqa从原本的train.jsonl改为了dev.jsonl
        
        # --- 测试样本数 ---
        "test_sample_num": args.num, # ⚠️ 测试数据条数.300-500条差不多
        "random_sample": True, # 是否随机采样
        
        # --- 结果保存 ---
        "save_dir": BASE_SAVE_DIR , #从零开始跑实验时用BASE_SAVE_DIR，从一半中断后继续跑实验时用RESUME_DIR
        
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
        "seed":777  # 以前都使用2026,777
    }
    
    print("⚙️ 加载配置...")
    config = Config(config_dict=config_dict)

    # config.final_config['save_dir'] = RESUME_DIR   # 从一半中断后继续跑实验时用RESUME_DIR
    
    REAL_SAVE_DIR = config['save_dir']
    print(f"📂 实验结果将保存至: {REAL_SAVE_DIR}")
    log_file_path = os.path.join(REAL_SAVE_DIR, "console_output.log") # 保存终端输出
    sys.stdout = DualLogger(log_file_path)

    print(f"📥 加载数据集...")
    data_splits = get_dataset(config)
    
    test_data = data_splits["dev"] 
    print(f"✅ 数据集加载成功，共 {len(test_data)} 条")
    print(f"   {BOLD_ON}评估数据: {test_data[0].question}{BOLD_OFF}")
    
    print("🛠️ 初始化 Pipeline...")
    pipeline = MyV2Pipeline(config)
    
    print(f"{BOLD_ON}🏃 运行...——————————————————————————————————————————————{BOLD_OFF}")
    pipeline.run(test_data)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"⏱️  {elapsed_time:.2f} 秒")
    print(f"📊 评估结果已保存到: {config['save_dir']}")

if __name__ == "__main__":
    main()
