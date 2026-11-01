import os
import pprint

from src.graph import build_graph
from src.nodes import BOLD_ON, BOLD_OFF , init_model
# from src.nodes import BOLD_ON, BOLD_OFF 

from src.constants import BOLD_ON, BOLD_OFF



def setup_langsmith():
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    if "LANGCHAIN_API_KEY" not in os.environ:
        os.environ["LANGCHAIN_API_KEY"] = "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"
        # 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
    os.environ["LANGCHAIN_PROJECT"] = "V2完整版本测试"

#
def pretty_print_chunk(chunk):
    """
    智能打印 chunk 信息。
    如果是 retrieve 节点返回的超长文档，自动折叠显示，避免刷屏。
    """
    # 检查是否是 retrieve 节点的输出
    if "retrieve" in chunk and "messages" in chunk["retrieve"]:
        messages = chunk["retrieve"]["messages"]
        if messages and len(messages) > 0:
            tool_msg = messages[0]
            content = tool_msg.content
            
            # 估算文档数量 (根据你的格式 Titile: 来数)
            doc_count = content.count("Titile:")
            
            # 截取前 200 个字符用于预览
            preview = content[:200].replace('\n', ' ')
            
            print(f"📦 Node: [retrieve]")
            print(f"📄 Retrieved Docs: ~{doc_count} chunks")
            print(f"📝 Content Preview: \"{preview}...\"")
            print(f"🚫 (剩余 {len(content) - 200} 字符已省略显示)")
            return # 处理完毕，直接返回

    # 对于其他节点（Re_strategist, Generator 等），保持原样打印
    print(f"Chunk: {chunk}\n")


def run():

    setup_langsmith()

    # 有qwen3-8b，qwen3-32,=b，qwen3-next-80b-a3b-instruct
    init_model("llama3-8b")  #
    graph = build_graph()

    # 输入问题
    # multi_hop_question = "Why might a language model trained with RLHF achieve high " \
    # "rewards by 'modifying test cases' instead of solving tasks correctly? Please explain " \
    # "using both flaws of the reward function and the definition of reward hacking."
    # multi_hop_question = "Can a shark play poker?"
    # multi_hop_question = "Did Aristotle use a laptop?"
    # multi_hop_question = "who wrote the song photograph by ringo starr"
    multi_hop_question = "Who is the father-in-law of Basilina?"  # 2wikimultihopqa中
    # multi_hop_question = "Who is older, Annie Morton or Terry Richardson?" # hotpot
    # multi_hop_question = "Bill Berry retired through ill health as a drummer in which band?" # nq ["REM", "R. E. M.", "R. E. M", "Rem", "Rem (disambiguation)", "R E M", "REM (disambiguation)"]

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

    print(f"\n🚀 {BOLD_ON}Launching Agent for question: '{multi_hop_question}'{BOLD_OFF}\n")

    i = 0
    # 使用 stream() 运行图并打印每一步的更新
    for chunk in graph.stream(initial_input):
        i += 1
        print(f"{BOLD_ON}————————————— chunk START {i} ————————————————————{BOLD_OFF}")
        if list(chunk.keys())[0] == "Re_strategist_node":
            print(f"{BOLD_ON}initial_plan{BOLD_OFF}:")  
            [print(f"{idx+1}. {plan}") for idx, plan in enumerate(chunk['Re_strategist_node']['plan'])]         
        elif list(chunk.keys())[0] == "retrieve":
            pretty_print_chunk(chunk) 
        elif list(chunk.keys())[0] == "sub_answer_generator":
            core_data = chunk["sub_answer_generator"]["history"][0]
            print(f"{BOLD_ON}sub_question{BOLD_OFF}:", core_data["sub_question"])
            print(f"{BOLD_ON}retrieved_docs{BOLD_OFF}:", core_data["retrieved_docs"])
            print(f"{BOLD_ON}answer{BOLD_OFF}:", core_data["answer"])
        elif list(chunk.keys())[0] == "reflection_agent_node":
            print(f"{BOLD_ON}assessment{BOLD_OFF}:", chunk["reflection_agent_node"]["reflection"])
        elif list(chunk.keys())[0] == "query_generator":
            # current_sub_question = chunk["query_generator"]["current_sub_question"]
            # print(f"{BOLD_ON}current_sub_question{BOLD_OFF}: {current_sub_question}")
            # aimessage_obj = chunk["query_generator"]["messages"][0] 
            # query = aimessage_obj.tool_calls[0]["args"]["query"]
            # # if aimessage_obj.tool_calls:
            # #     query = aimessage_obj.tool_calls[0]["args"]["query"]
            # #     print(f"{BOLD_ON}query{BOLD_OFF}: {query}")
            # # else:
            # #     # 如果没调用工具，打印模型直接回复的内容，方便调试
            # #     print(f"{BOLD_ON}⚠️ Warning: Model did not call tool!{BOLD_OFF}")
            # #     print(f"Model Content: {aimessage_obj.content}")

            # print(f"{BOLD_ON}query{BOLD_OFF}: {query}")
            print(f"query_Gen chunk: {chunk}")
        elif list(chunk.keys())[0] == "final_synthesis":
            raw_answer = chunk["final_synthesis"]["final_answer"]
            clean_answer = raw_answer.replace("<answer>", "").replace("</answer>", "")
            print(f"{BOLD_ON}final_answer{BOLD_OFF}: {clean_answer}")
        # print("Chunk: ", chunk, "\n") 
        print(f"{BOLD_ON}————————————— chunk END {i} —————————————————————{BOLD_OFF}\n\n")
    print("🏁 Agent run complete!")


if __name__ == "__main__":
    run()


