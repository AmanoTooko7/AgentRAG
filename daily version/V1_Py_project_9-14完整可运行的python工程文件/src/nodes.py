# 构建节点和条件边

import pprint
from typing import Literal
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing import List

from src.state import GraphState
from src.prompts import (
    STRATEGIST_PROMPT,
    SUB_ANSWER_PROMPT,
    FINAL_SYNTHESIS_PROMPT,
    QUERY_GENERATOR_PROMPT,
)
from src.retriever import build_retriever_tool


RESPONSE_MODEL = ChatOllama(model="qwen3:8b",temperature=1)
RETRIEVER_TOOL = build_retriever_tool()


class Plan(BaseModel):
    """A structured plan to answer user's question."""
    steps: List[str] = Field(
        description="A list of clear, logically sequenced, and non-overlapping question or steps that lead to the final answer"
    )

def strategist_node(state: GraphState):
    """Generates a plan."""
    print("--- 🧠 Entering Strategist Node ---")
    question = state["original_question"]
    prompt = STRATEGIST_PROMPT.format(question=question)
    
    structured_llm = RESPONSE_MODEL.with_structured_output(Plan)
    plan_object = structured_llm.invoke([{"role": "user", "content": prompt}])
    plan_steps = plan_object.steps
    
    print(f"Generated Plan: {plan_steps}")
    return {"plan": plan_steps}

def query_generator_node(state: GraphState):
    """Generates a search query for the current step."""
    print("--- 🔍 Entering Query Generator Node ---")
    plan = state["plan"]
    history = state["history"]
    current_step_index = len(history)
    current_step = plan[current_step_index]
    history_str = pprint.pformat(history)
    
    prompt = QUERY_GENERATOR_PROMPT.format(
        plan=plan, history=history_str, current_step=current_step
    )
    
    print(f"Formulating query for step {current_step_index + 1}: '{current_step}'")
    response = RESPONSE_MODEL.bind_tools([RETRIEVER_TOOL]).invoke(
        [{"role": "user", "content": prompt}]
    )
    
    return {"messages": [response], "current_sub_question": current_step}

def sub_answer_generator_node(state: GraphState):
    """Generates an answer for a single sub-question."""
    print("--- 🤖 Entering Sub-Answer Generation Node ---")
    retrieved_docs_messages = [msg for msg in state["messages"] if msg.type == "tool"]
    context = "\n\n".join([str(doc.content) for doc in retrieved_docs_messages])
    sub_question = state["current_sub_question"]
    
    prompt = SUB_ANSWER_PROMPT.format(context=context, sub_question=sub_question)
    response = RESPONSE_MODEL.invoke([{"role": "user", "content": prompt}])
    sub_answer = response.content
    
    print(f"Generated Sub-Answer: {sub_answer[:200]}...")
    step_output = {
        "sub_question": sub_question,
        "retrieved_docs": [str(doc.content) for doc in retrieved_docs_messages],
        "answer": sub_answer,
    }
    return {"history": [step_output]}

def final_synthesis_node(state: GraphState):
    """Generates the final, synthesized answer."""
    print("--- 🖋️ Entering Final Synthesis Node ---")
    original_question = state["original_question"]
    history = state["history"]
    history_str = pprint.pformat(history)
    
    prompt = FINAL_SYNTHESIS_PROMPT.format(
        original_question=original_question, history=history_str
    )
    
    response = RESPONSE_MODEL.invoke([{"role": "user", "content": prompt}])
    final_answer = response.content
    
    print(f"Generated Final Answer: {final_answer[:300]}...")
    return {"messages": [response]}

def should_continue(state: GraphState) -> Literal["continue_loop", "end_loop"]:
    """Determines whether to continue the loop."""
    print("--- 🚦 Entering Loop Controller ---")
    plan = state["plan"]
    history = state["history"]
    
    if len(history) < len(plan):
        print(f"Decision: Plan has {len(plan)} steps, {len(history)} done. Continue loop.")
        return "continue_loop"
    else:
        print(f"Decision: Plan has {len(plan)} steps, {len(history)} done. End loop.")
        return "end_loop"