# 以下程序用于下载HF的模型

import os
from huggingface_hub import snapshot_download

# ================= 配置区域 =================
TARGET_DIR = "/home/lab/HDD_data_slow/models"
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

MODELS_TO_DOWNLOAD = {
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "llama3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.1-70b-instruct": "meta-llama/Llama-3.1-70B-Instruct",
}

# ================= 下载逻辑 =================
def download_all():
    print(f"🚀 准备将模型下载到: {TARGET_DIR}")
    
    for name, repo_id in MODELS_TO_DOWNLOAD.items():
        print(f"\n⬇️  正在下载: {name} ({repo_id}) ...")
        try:
            local_path = snapshot_download(
                repo_id=repo_id,
                local_dir=os.path.join(TARGET_DIR, name),
                local_dir_use_symlinks=False,
                
                #  开启断点续传
                resume_download=True, 
                max_workers=2, 
                
                # 🔥 关键优化 3: 只下载 safetensors，忽略原始 pth 文件
                ignore_patterns=["original/*", "*.pth", "*.pt"],
                allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"]
            )
            print(f"✅ {name} 下载完成！")
            
        except Exception as e:
            print(f"❌ {name} 下载失败: {e}")
            # 如果是网络波动，脚本结束后你可以直接再次运行，它会接着下

if __name__ == "__main__":
    download_all()