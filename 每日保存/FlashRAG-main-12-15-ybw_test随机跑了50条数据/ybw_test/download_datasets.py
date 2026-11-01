# 下面为ChatGPT生成下载KILTNQ, HotpotQA, TriviaQA, and FEVER四个数据集的valida的代码
# 运行环境为FlashRAG_Lab

from datasets import load_dataset
import json
import os
from tqdm import tqdm

# 保存路径
BASE = "/home/lab/ybw/FlashRAG-main/KILT_datasets"

# 四个数据集的 KILT 名称
DATASETS = {
    "nq": "nq",
    "hotpotqa": "hotpotqa",
    "triviaqa": "triviaqa_support_only",
    "fever": "fever",
}

def extract_answers(kilt_output):
    """从 KILT 的 output 字段抽取 golden_answers 列表"""
    answers = []
    for item in kilt_output:
        if "answer" in item and item["answer"] != "":
            answers.append(item["answer"])
    return answers


for name, subset in DATASETS.items():
    print(f"\n========== Processing {name} ==========")

    # 加载 validation split
    ds = load_dataset("facebook/kilt_tasks", subset, split="validation")

    save_path = os.path.join(BASE, name, "val.jsonl")

    with open(save_path, "w", encoding="utf-8") as f:
        for idx, item in tqdm(enumerate(ds), total=len(ds)):

            # ❗ 用自定义编号替换原 id
            new_id = f"test_{idx}"

            question = item["input"]
            outputs = item["output"]
            golden_answers = extract_answers(outputs)

            new_obj = {
                "id": new_id,
                "question": question,
                "golden_answers": golden_answers
            }

            f.write(json.dumps(new_obj, ensure_ascii=False) + "\n")

    print(f"Saved → {save_path}")

print("\nAll datasets processed successfully.")
