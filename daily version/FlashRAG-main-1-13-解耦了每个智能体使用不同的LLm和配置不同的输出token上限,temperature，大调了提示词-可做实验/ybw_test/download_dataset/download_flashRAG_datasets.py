# 这里是通过FlashRAG框架，gemini生成
# 下载https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import hf_hub_download

# ================= 配置区域 =================
# 1. 你的目标根目录
TARGET_DIR = "/home/lab/ybw/FlashRAG-main/datasets_flashRAG"

# 2. HuggingFace 仓库 ID
REPO_ID = "RUC-NLPIR/FlashRAG_datasets"

# 3. 你需要下载的数据集及其对应的文件名
# 格式: "数据集文件夹名": "仓库里的文件名"
DATASETS_TO_DOWNLOAD = {
    "nq": "dev.jsonl",              # NQ
    "triviaqa": "dev.jsonl",        # TriviaQA
    "hotpotqa": "dev.jsonl",        # HotpotQA
    "fever": "dev.jsonl",           # FEVER
    "2wikimultihopqa": "dev.jsonl", # 2WikiMultiHopQA
    "strategyqa": "train.jsonl"     # StrategyQA (只有 train，我们就下这个)
}

# ================= 下载逻辑 =================
def download_data():
    print(f"🚀 开始下载数据集到: {TARGET_DIR}")
    
    # 设置国内镜像，防止网络超时
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    for dataset_name, filename in DATASETS_TO_DOWNLOAD.items():
        print(f"\n📥 正在处理: {dataset_name} ...")
        
        # 1. 构造本地存放路径 (例如 .../datasets_flashRAG/nq)
        local_dataset_dir = os.path.join(TARGET_DIR, dataset_name)
        os.makedirs(local_dataset_dir, exist_ok=True)
        
        # 2. 构造仓库中的文件路径 (例如 nq/dev.jsonl)
        # 注意：RUC-NLPIR 仓库的文件结构是 "数据集名/文件名"
        repo_filename = f"{dataset_name}/{filename}"
        
        try:
            # 3. 调用 HuggingFace 官方 API 下载
            # local_dir 指定下载到哪里
            # local_dir_use_symlinks=False 确保下载的是实体文件，不是快捷方式
            file_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=repo_filename,
                local_dir=TARGET_DIR,
                local_dir_use_symlinks=False,
                repo_type="dataset"
            )
            print(f"✅ 成功! 文件已保存: {file_path}")
            
        except Exception as e:
            print(f"❌ 下载失败 {dataset_name}: {e}")
            print(f"   请检查该数据集在 HuggingFace 上是否有 {filename} 文件")

if __name__ == "__main__":
    download_data()