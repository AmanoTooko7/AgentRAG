import os
import getpass
import pprint

from src.graph import build_graph
# from src.logger import enable_global_logging

# --- (可选) 设置LangSmith ---
def setup_langsmith():
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    if "LANGCHAIN_API_KEY" not in os.environ:
        os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("Enter your LangSmith API Key: ")
        # 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
    os.environ["LANGCHAIN_PROJECT"] = "V2完整版本测试"

# --- 主执行函数 ---
def run():
    # (可选) 启用LangSmith
    # setup_langsmith()

    # 构建可执行的图
    # enable_global_logging("/home/lab/ybw/V2_Py_project/results/run.log")
    graph = build_graph()

    # 准备输入问题
    # multi_hop_question = "Why might a language model trained with RLHF achieve high " \
    # "rewards by 'modifying test cases' instead of solving tasks correctly? Please explain " \
    # "using both flaws of the reward function and the definition of reward hacking."
    # multi_hop_question = "Can a shark play poker?"
    multi_hop_question = "Did Aristotle use a laptop?"

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
        print(f"————————————————————————————————— chunk END {i} ———————————————————————————————————\n\n")
    print("🏁 Agent run complete!")


if __name__ == "__main__":
    run()


