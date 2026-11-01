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
    """
    Represents the state of our graph.
    
    Attributes:
        original_question: The original user question.
        plan: A list of steps to take.
        history: A list of StepOutputs, accumulating results.
        current_sub_question: The specific sub-question being addressed in the current loop.
        messages: A list of messages for LLM interaction.
    """
    original_question: str
    plan: List[str]
    history: Annotated[list[StepOutput], operator.add]
    current_sub_question: str
    messages: Annotated[list, operator.add]

    reflection: Reflection
    num_cycles: int # 用于控制循环次数，衡量整个系统总共尝试了多少次“生成查询 -> 检索 -> 回答 -> 反思”这个核心动作
                    # 最终交给reflection_check判断，当大于某值就强制输出end,这个和RA模块做的任何decision无关