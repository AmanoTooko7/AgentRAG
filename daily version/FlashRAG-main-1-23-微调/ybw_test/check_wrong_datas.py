import os
import json
import re
import string
from collections import defaultdict

# ================= 配置区域 =================
# 1. 实验结果根目录
BASE_DIR = "/home/lab/ybw/FlashRAG-main/ybw_test/results_v2/llama3-8b/2wikimultihopqa"

# 2. 🎯 关键修改：指定你想分析的文件夹名称 (支持部分匹配)
# 只有包含这些关键词的文件夹才会被分析
# 如果留空 []，则分析所有文件夹
TARGET_FOLDERS = [
    "2wikimultihopqa_2026_01_14_00_26_experiment",
    "2wikimultihopqa_2026_01_16_19_09_experiment",
    "2wikimultihopqa_2026_01_17_17_40_experiment",
]

# 3. 输出目录
OUTPUT_DIR = "/home/lab/ybw/FlashRAG-main/ybw_test/Error_Analysis/2wikimultihopqa"

# ================= 辅助函数 (保持不变) =================
def normalize_answer(s):
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def is_correct(pred, gold_list):
    pred_norm = normalize_answer(pred)
    for gold in gold_list:
        if pred_norm == normalize_answer(gold):
            return True
    return False

# ================= 主逻辑 =================
def analyze_errors():
    print(f"📂 正在分析目录: {BASE_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 找到并过滤文件
    experiment_files = []
    for root, dirs, files in os.walk(BASE_DIR):
        if "intermediate_results.jsonl" in files:
            folder_name = os.path.basename(root)
            
            # 🔥 过滤逻辑
            if TARGET_FOLDERS:
                # 检查当前文件夹是否在目标列表中
                is_target = False
                for target in TARGET_FOLDERS:
                    if target in folder_name:
                        is_target = True
                        break
                if not is_target:
                    continue # 跳过不匹配的文件夹

            file_path = os.path.join(root, "intermediate_results.jsonl")
            experiment_files.append(file_path)
    
    # 检查是否找到了文件
    if not experiment_files:
        print("❌ 未找到匹配的实验文件！请检查 TARGET_FOLDERS 配置。")
        return

    print(f"🔍 选中了 {len(experiment_files)} 个实验文件进行分析:")
    for f in experiment_files:
        print(f"   ✅ {os.path.basename(os.path.dirname(f))}")

    # 2. 统计错误 (逻辑不变)
    error_counts = defaultdict(int)
    
    for file_path in experiment_files:
        print(f"\n正在读取: {os.path.basename(os.path.dirname(file_path))} ...")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    doc_id = item['id']
                    pred = item.get('pred', "")
                    gold_list = item.get('gold', [])
                    
                    if not is_correct(pred, gold_list):
                        error_counts[doc_id] += 1
                except Exception as e:
                    print(f"   ⚠️ 解析错误: {e}")

    # 3. 分类统计与输出
    categories = defaultdict(list)
    for doc_id, count in error_counts.items():
        categories[count].append(doc_id)

    print("\n" + "="*50)
    print("📊 错误频率统计报告")
    print("="*50)
    
    sorted_counts = sorted(categories.keys(), reverse=True)
    dataset_name = os.path.basename(BASE_DIR)
    
    for count in sorted_counts:
        ids = categories[count]
        print(f"\n🔴 错误 {count} 次的样本 ({len(ids)} 个):")
        print(f"   {', '.join(ids[:20])} ...")
        
        filename = f"{dataset_name}_errors_{count}_times.txt"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        with open(filepath, "w") as f:
            f.write("\n".join(ids))
            
    print(f"\n✅ 分析完成！文件已保存到 {OUTPUT_DIR}")

if __name__ == "__main__":
    analyze_errors()