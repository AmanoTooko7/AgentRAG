#此文件用于定义状态

from typing import List, TypedDict, Annotated
from langchain_core.messages import BaseMessage
import operator

class StepOutput(TypedDict):
    """Represents the output of a single reasoning step."""
    sub_question: str   # 这是当前的子问题
    retrieved_docs: List[str]
    answer: str         # 这是当前子问题的答案


class Reflection(TypedDict):
    """Represents the output of the Reflection Agent"""
    """
    assessment:给人类看的，它是一段自然语言对当前情况的总结和分析，也就是此处会写为什么RA模块会做出“Decision”原因
    feedback_for_planner:给Strategist'的建议，如果需要replan，则给strategist'的修改建议,若是continue/end则为空
    decision:该模块的决策输出，为“continue”，“replan”，“end”三者之一
    """
    assessment: str
    decision: str              
    feedback_for_planner: str 

class GraphState(TypedDict):

    original_question: str
    # 代表当前所需要执行的计划
    plan: List[str] 
    history: Annotated[list[StepOutput], operator.add]
    
    # query_generator对该状态进行更新，一次更新AIMessage和ToolMessage,
    # 分别表示生成的子查询(AI)和根据该查询检索到的文档(Tool)
    # final_synthesis也更新此状态
    #sub_answer仅使用该状态
    messages: Annotated[list, operator.add]  #

    # V2新增
    reflection: Reflection
    current_sub_question: str
    
    #循环状态
    num_cycles: int # 该变量是对整个系统而言的，防止无限循环
                    # 用于控制循环次数，衡量整个系统总共尝试了多少次“生成查询 -> 检索 -> 回答 -> 反思”这个核心动作
                    # 实际表示的就是回答完一个子问题后就更新+1
                    # 最终交给reflection_check判断，当大于某值就强制输出end,这个和RA模块做的任何decision无关

    # 此状态记录进入Re_Strategist模块的次数，用于strategist判断是初次计划，还是后续修改计划
    re_strategist_call_count : int 

    # 调试的状态定义
    plan_archive: Annotated[list, operator.add]# 用于保存Re_Strategist生成的所有计划，包括初始计划，重规划后的所有计划

    final_answer: str # 最终的答案，由final_synthesis模块生成

    dataset_type: str #这个是在FlashRAG框架评估新增加的状态，用于区分数据集类型

    

    

