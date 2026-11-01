# 这个用于解决llama3-8b回答问题时出现以下问题
# ❌ Error processing sample 1: 1 validation error for Reflection
#   Input should be an object [type=model_type, input_value=3.1, input_type=float]
#     For further information visit https://errors.pydantic.dev/2.12/v/model_type
# 当出现以上问题时加入重试

import time
from langchain_core.runnables import Runnable

def safe_invoke_llm(
    runnable: Runnable,  # 已经配置好的模型对象 (可以是 .with_structured_output, .bind_tools 或 原始 model)
    prompt_input: list,  # 传给 invoke 的输入
    fallback_return,     # 如果所有重试都失败，返回的默认值 (兜底对象)
    node_name: str = "Unknown Node",  # 用于日志打印的节点名称
    max_retries: int = 3,             # 最大重试次数
    retry_delay: int = 2              #  重试间隔秒数
):
    for attempt in range(max_retries):
        try:
            result = runnable.invoke(prompt_input)
            if result is None:
                    raise ValueError("Model returned None")
            return result
        except Exception as e:
            print(f"⚠️ [{node_name} Error] Attempt {attempt + 1}/{max_retries} failed: {e}")
            
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print(f"❌ [{node_name}] All retries failed. Using Fallback.")
                return fallback_return