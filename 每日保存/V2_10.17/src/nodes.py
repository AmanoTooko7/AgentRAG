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
    REFLECTION_PROMPT_V4,
    STRATEGIST_PROMPT_V2,
    RE_STRATEGIST_PROMPT,
    INITIAL_PLANNING_INSTRUCTIONS,
    REPLANNING_INSTRUCTIONS
)
from src.retriever import build_retriever_tool


RESPONSE_MODEL = ChatOllama(model="qwen3:8b",temperature=1)
RETRIEVER_TOOL = build_retriever_tool()


class Plan(BaseModel):
    """A structured plan to answer user's question."""
    steps: List[str] = Field(
        description="A list of clear, logically sequenced, and non-overlapping question or steps that lead to the final answer"
    )

# def strategist_node(state: GraphState):
#     """Generates a plan."""
#     print("--- 🧠 Entering Strategist Node ---")
#     question = state["original_question"]
#     prompt = STRATEGIST_PROMPT.format(question=question)
    
#     structured_llm = RESPONSE_MODEL.with_structured_output(Plan)
#     plan_object = structured_llm.invoke([{"role": "user", "content": prompt}])
#     plan_steps = plan_object.steps
    
#     print(f"Generated Plan: {plan_steps}")
#     return {"plan": plan_steps}

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

    retrieved_docs_messages = [msg for msg in state["messages"] if msg.type =="tool"] # 此句找到检索器找到内容ToolMessage
    context = "\n\n".join([str(doc.content) for doc in retrieved_docs_messages])
    sub_question = state["current_sub_question"]
    
    prompt = SUB_ANSWER_PROMPT.format(context=context, sub_question=sub_question)
    response = RESPONSE_MODEL.invoke([{"role": "user", "content": prompt}])
    sub_answer = response.content
    
    print(f"Generated Sub-Answer: {sub_answer[:200]}...")
    step_output = {
        "sub_question": sub_question, # <——————————————————————返回的内容这里应更新子query
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
  
class Reflection(BaseModel):
    """Represents the output of the Reflection Agent
    [属性名]: [类型] = Field(description="给LLM看的、关于这个属性的详细说明")
     """
    assessment: str = Field(
        escription="A detailed, natural language analysis of the progress so far. Explain WHY the latest " \
        "answer impacts (or does not impact) the remaining plan. This is for human review and context."
        )
    decision: Literal["continue", "replan", "end`"] = Field(
        description="The machine-readable final decision. Must be one of 'continue', 'replan', or 'end'."
        )
    feedback_for_planner: str = Field(
        description="Actionable guidance for the Strategist, to be provided ONLY if the decision is 'replan'. " \
        "State the key new insight and suggest the direction for the new plan."
        )

def reflection_agent_node(state: GraphState):
    """This node acts as the 'inner critic', reflecting on the progress so far
       and decide whether to continue, replan, or finish."""
    
    print("--- 🧠🧠 Entering Reflection Agent Node ---")
    original_question = state["original_question"]
    plan = state["plan"]
    history = state["history"]

    import pprint
    history_str = pprint.pformat(history)

    prompt = REFLECTION_PROMPT_V4.format(
        original_question=original_question,
        plan=plan,
        history=history_str
    )

    structured_llm = RESPONSE_MODEL.with_structured_output(Reflection)
    reflection_object = structured_llm.invoke([{"role": "user", "content": prompt}])


    print(f"Reflection Assessment: {reflection_object.assessment}")
    print(f"Reflection Decision: {reflection_object.decision}")

    return {"reflection":reflection_object.model_dump()}


#需要完成的任务是：
#1.初次规划
#2.后续规划包括增添计划+修改后续计划
def Re_strategist_node(state: GraphState):
    """
    The master strategist, capable of initial planning and replanning.
    It now also archives the old plan before overwriting it during replanning.
    """
    print("--- 🧠 Entering Re_Strategist Node ---")
    # print(f"Current state: {state}")  

    if state['re_strategist_call_count'] == 0:
        # 初次规划
        print("Mode: Initial Planning")
        prompt = RE_STRATEGIST_PROMPT.format(
            initial_planning_instructions=INITIAL_PLANNING_INSTRUCTIONS,
            replanning_instructions="",
            original_question=state['original_question'],
            history="Nohistory yet",
            feedback=""
        )
    else:
        # 计划从新规划模式
        print("Mode: Re-Planning")
        history_str = pprint.pformat(state['history'])
        # feedback = state["reflection"]["feedback_for_planner"]
        feedback = state.get("reflection", {}).get("feedback_for_planner", "No specific feedback.") #<————————————————————————————可能有问题！！！
        prompt = RE_STRATEGIST_PROMPT.format(
            initial_planning_instructions="",
            replanning_instructions=REPLANNING_INSTRUCTIONS,
            original_question=state['original_question'],
            history=history_str,
            feedback=feedback
        )

    structured_llm = RESPONSE_MODEL.with_structured_output(Plan)
    plan_object = structured_llm.invoke([{"role": "user", "content":prompt}])
    new_plan_steps = plan_object.steps   #表示Re_Strategist每运行一次生成的新计划

    print(f"Generated/Updated Plan: {new_plan_steps}")

    return{
        "plan":new_plan_steps,
        "plan_archive": [new_plan_steps], # 永远把新生成的计划版本追加到档案馆，注意该状态的定义，这里是追加而不是覆盖！！！
        "re_strategist_call_count": state['re_strategist_call_count'] + 1
    }


# 这是给V1用的，用于判断子计划执行完成没有，是结束执行还在整合子问题子答案回答最终问题
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
  

# 用于V2，该模块根据Reflection_Agent的decision判断是继续执行原计划还是重规划
def reflection_check(state: GraphState) -> Literal["continue", "replan", "end"]:
    """
    This is the central router of our V2 graph.
    It reads the decision from the Reflection Agent and determines the next path.
    It also implements a "circuit breaker" to prevent infinite loops.

    Args:
        state (GraphState): The current state of the geaph.

        Return:
            A string indicating the next node to route to: "continue", "replan", or "end".
    """
    print("--- 🚦 Entering Central Router (reflection_check) ---")

    MAX_CYCLES = 10
    if state["num_cycles"] > MAX_CYCLES:
        print(f"Decision: Circuit Breaker triggered! Max cycles ({MAX_CYCLES}) reached. Ending execution.")
        return "end"

    # 读取RA的决策
    decision = state.get("reflection", {}).get("decision", "") #<————————————————————————————可能有问题！！！
    print(f"Reflection Agent's decision: '{decision}")

    if decision == "replan":
        print("Routing to Re_strategist")
        return "replan"
    elif decision == "end":
        print("Routing to: Final Synthesis")
        return "end"
    else: # decision == "continue" or any other unexpected value
        print("Routing to: Query Generator (Continue Loop)")
        return "continue"
          

    




        