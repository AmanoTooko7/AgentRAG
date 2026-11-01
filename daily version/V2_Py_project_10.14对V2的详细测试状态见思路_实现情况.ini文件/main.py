import os
import getpass
import pprint

# 从我们自己的模块中导入
from src.graph import build_graph

# --- (可选) 设置LangSmith ---
# def setup_langsmith():
#     os.environ["LANGCHAIN_TRACING_V2"] = "true"
#     if "LANGCHAIN_API_KEY" not in os.environ:
#         # 请在这里直接粘贴您的API Key，或者保持getpass
#         os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("Enter your LangSmith API Key: ")
#     os.environ["LANGCHAIN_PROJECT"] = "V1_Python工程文件测试"

# --- 主执行函数 ---
def run():
    # (可选) 启用LangSmith
    # setup_langsmith()

    # 构建可执行的图
    graph = build_graph()

    # 准备输入问题
    multi_hop_question = "Why might a language model trained with RLHF achieve high " \
    "rewards by 'modifying test cases' instead of solving tasks correctly? Please explain " \
    "using both flaws of the reward function and the definition of reward hacking."

    initial_input = {
        "original_question": multi_hop_question,
        "plan": [],
        "history": [],
        "messages": [],
        "current_sub_question": "", # 初始化为空字符串
        "plan_archive": [],
        "reflection":{},
        
        "num_cycles": 0,
        "re_strategist_call_count": 0
    }

    print(f"\n🚀 Launching Agent for question: '{multi_hop_question}'\n")

    i = 0
    # 使用 stream() 运行图并打印每一步的更新
    for chunk in graph.stream(initial_input):
        i += 1
        print(f"——————————————————————————————— chunk START {i} ——————————————————————————————————")
        print("Chunk: ", chunk, "\n") 
        # chunk的键是节点名，值是该节点返回的state更新
        for node_name, state_update in chunk.items():
            print(f"--- 🎬 Update from Node: '{node_name}' ---")
            pprint.pprint(state_update)
            print("\n")
        print(f"————————————————————————————————— chunk END {i} ———————————————————————————————————\n\n")
    print("🏁 Agent run complete!")


if __name__ == "__main__":
    run()


