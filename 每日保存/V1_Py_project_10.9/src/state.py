#此文件用于定义状态

from typing import List, TypedDict, Annotated
from langchain_core.messages import BaseMessage
import operator

class StepOutput(TypedDict):
    """Represents the output of a single reasoning step."""
    sub_question: str
    retrieved_docs: List[str]
    answer: str

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
    history: Annotated[list, operator.add]
    current_sub_question: str
    messages: Annotated[list, operator.add]