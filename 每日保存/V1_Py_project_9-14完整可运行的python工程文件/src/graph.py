# 连接节点和条件边组成图

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from src.state import GraphState
from src.nodes import (
    strategist_node,
    query_generator_node,
    sub_answer_generator_node,
    final_synthesis_node,
    should_continue,
    RETRIEVER_TOOL, 
)

def build_graph():
    """Builds the Plan-and-Execute agent graph."""
    workflow = StateGraph(GraphState)

    # 注册节点
    workflow.add_node("strategist", strategist_node)
    workflow.add_node("query_generator", query_generator_node)
    workflow.add_node("retrieve", ToolNode([RETRIEVER_TOOL]))
    workflow.add_node("sub_answer_generator", sub_answer_generator_node)
    workflow.add_node("final_synthesis", final_synthesis_node)

    # 设置入口点和边
    workflow.set_entry_point("strategist")
    workflow.add_edge("strategist", "query_generator")
    workflow.add_edge("query_generator", "retrieve")
    workflow.add_edge("retrieve", "sub_answer_generator")
    
    workflow.add_conditional_edges(
        "sub_answer_generator",
        should_continue,
        {
            "continue_loop": "query_generator",
            "end_loop": "final_synthesis",
        },
    )
    
    workflow.add_edge("final_synthesis", END)

    graph = workflow.compile()
    print("✅ Plan-and-Execute Graph compiled successfully.")
    return graph