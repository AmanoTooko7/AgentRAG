# 连接节点和条件边组成图
# 2025.2.28在/home/lab/ybw/V2_Py_project/src_of_flashRAG/graph.py基础上进行修改为便于做消融实验的版本，这里只对

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from src_of_flashRAG.state import GraphState
# from src_of_flashRAG.nodes_of_flashRAG_vllm import (   #这里导入的是用于做对比实验的文件
from src_of_flashRAG.nodes_of_flashRAG_vllm_for_ablation import (  # 这里是用于做消融实验的文件
    # strategist_node,
    query_generator_node,
    sub_answer_generator_node,
    final_synthesis_node,
    RETRIEVER_TOOL, 
    Re_strategist_node,
    reflection_agent_node,
    reflection_check,
)

def build_graph(mode="v2"): 
    """
    mode="v2": 完整版 (有反思)
    mode="v1": 消融版 (无反思，Plan -> Execute -> End)
    """
    workflow = StateGraph(GraphState)

    # 注册节点 (V1 和 V2 共用的)
    workflow.add_node("Re_strategist_node", Re_strategist_node)
    workflow.add_node("query_generator", query_generator_node)
    workflow.add_node("retrieve", ToolNode([RETRIEVER_TOOL]))
    workflow.add_node("sub_answer_generator", sub_answer_generator_node)
    workflow.add_node("final_synthesis", final_synthesis_node)

    if mode == "v2":
        # === V2 完整逻辑 ===
        workflow.add_node("reflection_agent_node", reflection_agent_node)
        
        workflow.set_entry_point("Re_strategist_node")
        workflow.add_edge("Re_strategist_node", "query_generator")
        workflow.add_edge("query_generator", "retrieve")
        workflow.add_edge("retrieve", "sub_answer_generator")
        workflow.add_edge("sub_answer_generator", "reflection_agent_node")
        
        workflow.add_conditional_edges(
            "reflection_agent_node",
            reflection_check,
            {
                "continue": "query_generator",
                "replan": "Re_strategist_node",
                "end": "final_synthesis",
            },
        )
    
    elif mode == "v1":
        # === V1 消融逻辑 (无反思) ===
        # 逻辑：Strategist -> Query -> Retrieve -> SubAnswer -> (Loop or End) -> Final
        # 这里需要一个简单的 Loop 判断器 (代替 Reflection)
        
        def simple_loop_check(state):
            # 如果还有未执行的步骤，继续；否则结束
            if len(state["history"]) < len(state["plan"]):
                return "continue"
            return "end"

        workflow.set_entry_point("Re_strategist_node")
        workflow.add_edge("Re_strategist_node", "query_generator")
        workflow.add_edge("query_generator", "retrieve")
        workflow.add_edge("retrieve", "sub_answer_generator")
        
        workflow.add_conditional_edges(
            "sub_answer_generator",
            simple_loop_check,
            {
                "continue": "query_generator",
                "end": "final_synthesis"
            }
        )

    workflow.add_edge("final_synthesis", END)

    graph = workflow.compile()
    print("✅ V2 Graph compiled successfully.")
    return graph


# 这是V1版本流程
# def build_graph():
#     """Builds the Plan-and-Execute agent graph."""
#     workflow = StateGraph(GraphState)

#     # 注册节点
#     workflow.add_node("strategist", strategist_node)
#     workflow.add_node("query_generator", query_generator_node)
#     workflow.add_node("retrieve", ToolNode([RETRIEVER_TOOL]))
#     workflow.add_node("sub_answer_generator", sub_answer_generator_node)
#     workflow.add_node("final_synthesis", final_synthesis_node)

#     # 设置入口点和边
#     workflow.set_entry_point("strategist")
#     workflow.add_edge("strategist", "query_generator")
#     workflow.add_edge("query_generator", "retrieve")
#     workflow.add_edge("retrieve", "sub_answer_generator")
    
#     workflow.add_conditional_edges(
#         "sub_answer_generator",
#         should_continue,
#         {
#             "continue_loop": "query_generator",
#             "end_loop": "final_synthesis",
#         },
#     )
    
#     workflow.add_edge("final_synthesis", END)

#     graph = workflow.compile()
#     print("✅ Plan-and-Execute Graph compiled successfully.")
#     return graph

