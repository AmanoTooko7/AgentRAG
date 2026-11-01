# 用来跑批量实验的脚本 ，在AgentRAG构建3窗口讨论，以下代码全由gemini生成，现在放在这，还没看
import json
import os
import time
from tqdm import tqdm
from datasets import load_dataset
from src.graph import build_graph

# ================= 配置区域 =================
DATASET_NAME = "hotpot_qa"  # 可选: 'hotpot_qa', 'natural_questions', 'fever' 等
SPLIT = "validation"        # 验证集
SAMPLE_NUM = 5              # ⚠️ 先设为 5 个测试一下，没问题了再改成 300
OUTPUT_FILE = f"results/v2_output_{DATASET_NAME}.json"

# ================= 数据加载函数 =================
def load_eval_data(dataset_name, sample_num):
    """加载数据集并统一格式"""
    print(f"📥 Loading {dataset_name}...")
    
    data_list = []
    
    # 针对不同数据集的处理逻辑 (根据 HuggingFace 的格式)
    if dataset_name == "hotpot_qa":
        # HotpotQA (distractor)
        dataset = load_dataset("hotpot_qa", "distractor", split=SPLIT, trust_remote_code=True)
        # 随机采样
        dataset = dataset.shuffle(seed=42).select(range(sample_num))
        for item in dataset:
            data_list.append({
                "id": item["id"],
                "question": item["question"],
                "golden_answers": [item["answer"]] # HotpotQA 答案通常是字符串
            })
            
    elif dataset_name == "natural_questions":
        dataset = load_dataset("natural_questions", split=SPLIT, trust_remote_code=True)
        dataset = dataset.shuffle(seed=42).select(range(sample_num))
        for item in dataset:
            # NQ 的答案比较复杂，可能有多个 valid answers
            anns = item["annotations"]
            golds = [sa for a in anns for sa in a["short_answers"]] # 简化处理，实际需根据NQ格式细调
            # 这里为了演示，先取 long answer 里的文本或者简化处理，建议用 FlashRAG 的加载器
            # 为了简单，我们可以先跳过 NQ 的手动加载，后面教你用 FlashRAG 加载
            pass 

    # ⚠️ 强烈建议：为了省事，后面直接复用 FlashRAG 的加载代码
    # 这里为了让你先跑通 HotpotQA，先这样写
    
    return data_list

# ================= 主推理循环 =================
def main():
    # 1. 准备输出目录
    os.makedirs("results", exist_ok=True)
    
    # 2. 加载数据
    test_data = load_eval_data(DATASET_NAME, SAMPLE_NUM)
    print(f"✅ Loaded {len(test_data)} samples.")

    # 3. 构建图 (编译一次即可)
    app = build_graph()
    
    results = []
    
    # 4. 循环推理
    print("🚀 Starting Batch Inference...")
    for item in tqdm(test_data):
        question = item["question"]
        
        # 构造输入状态
        inputs = {
            "original_question": question,
            "plan": [],
            "history": [],
            "messages": [],
            "current_sub_question": "",
            "plan_archive": [],
            "reflection": {},
            "num_cycles": 0,
            "re_strategist_call_count": 0
        }
        
        try:
            # 运行图
            final_state = app.invoke(inputs)
            final_answer = final_state.get("final_answer", "Error: No answer generated")
            
            # 记录结果
            result_item = {
                "id": item["id"],
                "question": question,
                "golden_answers": item["golden_answers"],
                "response": final_answer, # V2 的输出
                # 可选：记录中间过程用于分析
                "steps": len(final_state.get("history", [])),
                "replan_count": final_state.get("re_strategist_call_count", 0) - 1
            }
            results.append(result_item)
            
        except Exception as e:
            print(f"\n❌ Error processing id {item['id']}: {e}")
            # 即使报错也要记录，防止中断
            results.append({
                "id": item["id"],
                "question": question,
                "golden_answers": item["golden_answers"],
                "response": "ERROR_DURING_INFERENCE"
            })

        # 5. 实时保存 (每跑完一个存一次，防止跑了一半崩了全没了)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\n🎉 Inference Complete! Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()