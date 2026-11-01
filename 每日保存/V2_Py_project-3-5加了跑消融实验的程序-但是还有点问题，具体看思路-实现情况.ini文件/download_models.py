# 以下程序用于下载HF的模型

import os
from huggingface_hub import snapshot_download

# ================= 基础环境设置 =================

# 使用国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 关键：关闭 XetHub（避免 cas-bridge 走 AWS）
os.environ["HF_HUB_DISABLE_XET"] = "1"

# 可选：增加超时时间（防止大文件中断）
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"

TARGET_DIR = "/home/lab/HDD_data_slow/models2"

MODELS_TO_DOWNLOAD = {
    "llama3-8b-instruct": "allura-forge/Llama-3.3-8B-Instruct",
}

# ================= 下载逻辑 =================

def download_all():
    print(f"🚀 准备将模型下载到: {TARGET_DIR}")

    for name, repo_id in MODELS_TO_DOWNLOAD.items():
        print(f"\n⬇️  正在下载: {name} ({repo_id}) ...")

        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=os.path.join(TARGET_DIR, name),

                # 并发不要过高，保证稳定
                max_workers=4,

                # 只下载必要文件
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.model",
                    "*.txt"
                ]
            )

            print(f"✅ {name} 下载完成！")

        except Exception as e:
            print(f"❌ {name} 下载失败: {e}")
            print("⚠️  可重新运行脚本，会自动断点续传")

if __name__ == "__main__":
    download_all()