# 以下程序用于使用本地模型模块测试,用的是ybw_AgentRAG环境
# 这个也得用三张卡，2张会报错 CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`
# 3张卡依旧非常慢，不考虑使用本地部署了
# 2025-12-17 15:58


# 下面使用vllm库看是否能跑成功，环境使用的是ybw_AgentRAG-2环境
#能跑成功，使用4张卡8.6+8.6+8.6+8.6G显存，终端回复速度与使用transformers差不多
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import time

# ================= 配置区域 =================
# 指向你刚刚启动的 vLLM 服务
VLLM_API_KEY = "EMPTY"
VLLM_BASE_URL = "http://localhost:8000/v1"

# 必须和你启动 vLLM 时指定的 --model 参数完全一致
MODEL_NAME = "/home/lab/HDD_data_slow/models/llama3.1-8b-instruct"

print(f"🚀 Connecting to Local vLLM: {VLLM_BASE_URL}")
print(f"📦 Model: {MODEL_NAME}")

# 初始化 LangChain 客户端
llm = ChatOpenAI(
    model=MODEL_NAME,
    openai_api_key=VLLM_API_KEY,
    openai_api_base=VLLM_BASE_URL,
    temperature=0.7,
    max_tokens=2048
)

# ================= 交互测试循环 =================
print("\n✨ 连接成功！开始对话测试...")
print("--------------------------------------------------")

while True:
    try:
        user_input = input("\n👤 User: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("👋 程序结束")
            break
        if not user_input: continue
        
        # 计时
        start_time = time.time()
        
        print("🤖 AI Thinking...", end="", flush=True)
        
        # 调用模型
        response = llm.invoke([HumanMessage(content=user_input)])
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 打印结果
        print(f"\r🤖 AI: {response.content}")
        print(f"⏱️  耗时: {duration:.2f}s")

    except KeyboardInterrupt:
        print("\n👋 用户强制中断")
        break
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        print("💡 提示: 请检查 vLLM 服务是否正在运行，端口是否为 8000")