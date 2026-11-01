# 以下是下载gte-multilingual嵌入模型
# import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# from transformers import AutoModel, AutoTokenizer

# local_dir = "/home/lab/Shared_model/gte-multilingual-base"
# model = AutoModel.from_pretrained("Alibaba-NLP/gte-multilingual-base",
#                                    cache_dir=local_dir,
#                                    trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-multilingual-base", 
#                                           cache_dir=local_dir,
#                                           trust_remote_code=True)

# print(f"Model and tokenizer saved to {local_dir}")


# 以下是将下载的.safetesnor文件转换为.bin文件
# import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# from transformers import AutoModel, AutoTokenizer

# # 模型名称（直接从网上读）
# model_name = "Alibaba-NLP/gte-multilingual-base"

# # 保存路径
# save_dir = "/home/lab/Shared_model/gte-multilingual-base/converted_to_bin"

# print(">>> 正在从在线仓库加载 tokenizer 与模型（自动识别 safetensors）...")
# tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)

# model = AutoModel.from_pretrained(model_name,trust_remote_code=True)

# print(">>> 在线模型加载成功！")
# print(f">>> 正在保存模型为 bin 到目录：{save_dir}")
# model.save_pretrained(
#     save_dir,
#     safe_serialization=False        # 强制生成 pytorch_model.bin
# )

# tokenizer.save_pretrained(save_dir)
# print(">>> bin 模型保存完成！")


# 以下是下载embeddinggemma-300m嵌入模型
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import snapshot_download

my_token = "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"

local_dir = "/home/lab/Shared_model/embeddinggemma-300m"

print(f"开始下载 google/embeddinggemma-300m 到 {local_dir} ...")

try:
    snapshot_download(
        repo_id="google/embeddinggemma-300m",
        local_dir=local_dir,
        token=my_token,
        # 忽略一些不必要的文件以节省时间，比如 pytorch_model.bin (因为有 model.safetensors 了)
        # 但为了保险起见，通常默认全部下载即可，safetensors 是现在的标准
        ignore_patterns=["*.h5", "*.ot", "*.msgpack"], 
        local_dir_use_symlinks=False  # 确保下载的是真实文件而不是快捷方式
    )
    print("下载完成！")
except Exception as e:
    print(f"下载失败: {e}")
    print("请检查：\n1. 你的 Token 是否正确？\n2. 你是否在 HuggingFace 网页上点击了同意该模型的协议？")