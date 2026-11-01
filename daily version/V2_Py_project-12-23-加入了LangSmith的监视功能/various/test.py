import os
from huggingface_hub import snapshot_download

# 指定保存路径
local_dir = "/home/lab/Shared_model/bge-reranker-v2-m3"
os.makedirs(local_dir, exist_ok=True)

print(f"🚀 开始下载 Re-ranker 模型到 {local_dir} ...")

snapshot_download(
    repo_id="BAAI/bge-reranker-v2-m3",
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # 确保下载的是真实文件而不是链接
    resume_download=True
)

print("✅ 下载完成！")