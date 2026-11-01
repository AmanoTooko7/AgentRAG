# 构建节点和条件边
import os
import pprint
# import torch
from typing import Literal
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     BitsAndBytesConfig,
#     pipeline
# )
# from langchain_huggingface import HuggingFacePipeline
# from langchain_core.output_parsers import PydanticOutputParser
# from langchain_core.exceptions import OutputParserException

from src.state import GraphState
from src.prompts import (
    SUB_ANSWER_PROMPT,
    FINAL_SYNTHESIS_PROMPT,
    QUERY_GENERATOR_PROMPT_V2,
    REFLECTION_PROMPT_V6,
    RE_STRATEGIST_INITIAL_PROMPT,
    RE_STRATEGIST_REPLAN_PROMPT,
)
# from src.retriever import build_retriever_tool # 这是检索翁丽莲的3篇博客
from src.retriever_of_wiki import build_retriever_tool
from src.re_ranker import get_ranker_score

from src.constants import BOLD_ON, BOLD_OFF

# 下面调用Close AI中的模型————————————————————————————————————————————————————————————
CloseAI_API_KEY = "sk-hOiSoNaP9k5qXSw1MjE9EmaNB3oAwMB9qxr4rZN3zkPq80ch"
CloseAI_BASE_URL = "https://api.openai-proxy.org/v1"
MODEL_MAP = {
    "deepseek": "deepseek-chat"
}
CURRENT_MODEL_KEY = "deepseek"
print(f"🚀 Loading Model via CloseAI: {MODEL_MAP[CURRENT_MODEL_KEY]}")
RESPONSE_MODEL = ChatOpenAI(
    model= MODEL_MAP[CURRENT_MODEL_KEY],
    openai_api_key=CloseAI_API_KEY,
    openai_api_base=CloseAI_BASE_URL,
    temperature=0,
    max_tokens=4096,
)

RESPONSE_MODEL = ChatOllama(model="qwen3:8b",temperature=1) # 使用chatollam
# —————————————————————————————————————————————————————————————————————————————————————
# 以下调用本地的模型
# MODEL_ROOT_DIR = "/home/lab/HDD_data_slow/models"

# LOCAL_MODEL_MAP = {
#     "qwen3-8B": "Qwen3-8B",
#     "llama3.1-8b-instruct": "llama3.1-8b-instruct",
#     "llama3.1-70b-instruct": "llama3.1-70b-instruct",
# }
# CURRENT_MODEL_KEY = "qwen3-8B"
# def get_response_model(model_key):
#     model_dir_name = LOCAL_MODEL_MAP[model_key]
#     model_path = os.path.join(MODEL_ROOT_DIR, model_dir_name)
#     print(f"Initializing Local Model gpu count: {torch.cuda.device_count()}")

#     is_large_model = "70b" in model_key.lower() or "72b" in model_key.lower()
#     if is_large_model: # 70B以上的大模型使用NF4量化
#         quantization_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_quant_type="nf4", # 使用 NF4
#             bnb_4bit_compute_dtype=torch.float16,
#             bnb_4bit_use_double_quant=True,
#         )
#         model_kwargs = {"quantization_config": quantization_config}
#     else: #8b小模型使用原精度bf16
#         model_kwargs = {"torch_dtype": torch.float16}
    
#     # 加载tokenizer
#     try:
#         tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
#     except Exception as e:
#         raise RuntimeError(f"Failed to load tokenizer from {model_path}. Error{e}")
#     # 加载模型
#     try:
#         model = AutoModelForCausalLM.from_pretrained(
#             model_path,
#             device_map="auto",  # 关键：自动分配到 6 张 3080
#             trust_remote_code=True,
#             **model_kwargs      # 传入量化或精度参数
#         )
#     except Exception as e:
#         raise RuntimeError(f"Failed to load model from {model_path}. Error: {e}")
#     pipe = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         max_new_tokens=4096,
#         temperature=0.1,     # 实验建议低温度
#         do_sample=True,
#         repetition_penalty=1.1,
#         return_full_text=False # 只返回生成的答案，不包含 Prompt
#     )
#     # 5. 封装为 LangChain 对象
#     return HuggingFacePipeline(pipeline=pipe)
    
# RESPONSE_MODEL = get_response_model(CURRENT_MODEL_KEY)
# —————————————————————————————————————————————————————————————————————————————————————

RETRIEVER_TOOL = build_retriever_tool()


class Plan(BaseModel):
    """A structured plan to answer user's question."""
    steps: List[str] = Field(
        description="A list of clear, logically sequenced, and non-overlapping question or steps that lead to the final answer"
    )

  
def query_generator_node(state: GraphState):
    """Generates a search query for the current step."""
    print("--- 🔍 Entering Query Generator Node ---")
    plan = state["plan"]
    history = state["history"][-1:]            # <------------------
    current_step_index = len(state["history"])
    current_step = plan[current_step_index]    
    history_str = pprint.pformat(history)
    
    prompt = QUERY_GENERATOR_PROMPT_V2.format(
        plan=plan, history=history_str, current_step=current_step
    )
    
    print(f"{BOLD_ON}Formulating query for step {current_step_index + 1}{BOLD_OFF}: '{current_step}'")
    response = RESPONSE_MODEL.bind_tools([RETRIEVER_TOOL]).invoke(
        [{"role": "user", "content": prompt}]
    )
    
    return {"messages": [response], "current_sub_question": current_step}

def sub_answer_generator_node(state: GraphState):
    """Generates an answer for a single sub-question."""
    print("--- 🤖 Entering Sub-Answer Generation Node ---")

    retrieved_docs_messages = [msg for msg in state["messages"] if msg.type =="tool"] # 此句找到检索器找到内容ToolMessage  # <------------------
    # context = str(retrieved_docs_messages[-1].content)  #只取最新子问题检索回的文档块
    
    raw_docs = []
    if retrieved_docs_messages:
        giant_string = str(retrieved_docs_messages[-1].content)
        raw_docs = list(filter(None, giant_string.split("\n\n")))
    sub_question = state["current_sub_question"]

    # 调用re-ranker————————————————————————————————————————————————————————————
    if raw_docs:
        print(f"Re-ranking {len(raw_docs)} docs...")
        best_docs = get_ranker_score(sub_question, raw_docs)
        context = "\n\n".join(best_docs) 
    else:
        context = "No documents found."
    # 调用re-ranker————————————————————————————————————————————————————————————
    
    prompt = SUB_ANSWER_PROMPT.format(context=context, sub_question=sub_question)
    response = RESPONSE_MODEL.invoke([{"role": "user", "content": prompt}])
    sub_answer = response.content
    
    # print(f"Generated Sub-Answer: {sub_answer[:200]}...")
    step_output = {
        "sub_question": sub_question, # <——————————————————————返回的内容这里应更新子query
        # "retrieved_docs": [str(doc.content) for doc in retrieved_docs_messages], 
        "retrieved_docs": [context] , 
        "answer": sub_answer,
    }
    return {"history": [step_output]}




# 这是V2版本用的，与下面的给FlashRAG用的不一样
def final_synthesis_node(state: GraphState):
    """Generates the final, synthesized answer."""
    print("--- 🖋️ Entering Final Synthesis Node ---")
    original_question = state["original_question"]
    history = state["history"]                       # <------------------
    history_str = pprint.pformat(history)
    
    prompt = FINAL_SYNTHESIS_PROMPT.format(
        original_question=original_question, history=history_str
    )
    final_prompt = prompt + (
        "\n\n**TASK:** Answer the complex multi-hop question.\n"
            "**FORMAT:** Output ONLY the specific entity (person, place, film, etc.) inside tags.\n"
            "**EXAMPLES:**\n"
            "Q: What is the place of birth of Paulding Farnham's wife?\n"
            "A: <answer>Ogdensburg, New York</answer>\n"
            "Q: Who is the father of the director of film Biwi-O-Biwi?\n"
            "A: <answer>H. S. Rawail</answer>\n"
            "**YOUR TURN:**\n"
    )
    response = RESPONSE_MODEL.invoke([{"role": "user", "content": final_prompt}])
    final_answer = response.content
    
    # print(f"{BOLD_ON}Generated Final Answer: {final_answer}{BOLD_OFF}")
    return {"final_answer": final_answer}
  

# # 这是给FlashRAG用的，加入了dataset_type 状态
# def final_synthesis_node(state: GraphState):
#     print("--- 🖋️ Entering Final Synthesis Node ---")
#     original_question = state["original_question"]
#     history = state["history"]
#     dataset_type = state.get("dataset_type", "qa") # 获取类型，默认 qa
    
#     history_str = pprint.pformat(history)
    
#     # 基础 Prompt
#     base_prompt = FINAL_SYNTHESIS_PROMPT.format(
#         original_question=original_question, history=history_str
#     )
    
#     # 🌟 动态追加指令 (解决问题 1 & 3)
#     if dataset_type == "fever":
#         instruction = (
#             "\n\nTASK: Verify the claim based on the evidence.\n"
#             "FORMAT: You MUST output exactly one of these labels inside tags: <answer>SUPPORTS</answer>, <answer>REFUTES</answer>, or <answer>NOT ENOUGH INFO</answer>.\n"
#             "Do not output any other text."
#         )
#     elif dataset_type == "strategyqa":
#         instruction = (
#             "\n\nTASK: Answer the question with Yes or No.\n"
#             "FORMAT: You MUST output the answer inside tags: <answer>yes</answer> or <answer>no</answer>."
#         )
#     else: # hotpotqa, nq, triviaqa
#         instruction = (
#             "\n\nTASK: Answer the question concisely.\n"
#             "FORMAT: Output ONLY the entity or short phrase inside tags. Example: <answer>Barack Obama</answer>."
#         )
        
#     final_prompt = base_prompt + instruction
    
#     response = RESPONSE_MODEL.invoke([{"role": "user", "content": final_prompt}])
#     final_answer = response.content
    
#     return {"final_answer": final_answer}



class Reflection(BaseModel):
    """Represents the output of the Reflection Agent
    [属性名]: [类型] = Field(description="给LLM看的、关于这个属性的详细说明")
     """
    assessment: str = Field(
        escription="A detailed, natural language analysis of the progress so far. Explain WHY the latest " \
        "answer impacts (or does not impact) the remaining plan. This is for human review and context."
        )
    decision: Literal["continue", "replan", "end"] = Field(
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
    # plan = state["plan"]
    history = state["history"][-1:]              # <------------------

    num_completed_steps = len(state["history"])
    remaining_plan_steps = state["plan"][num_completed_steps:]

    import pprint
    history_str = pprint.pformat(history)

    prompt = REFLECTION_PROMPT_V6.format(
        original_question=original_question,
        plan=remaining_plan_steps,
        history=history_str
    )

    structured_llm = RESPONSE_MODEL.with_structured_output(Reflection)
    reflection_object = structured_llm.invoke([{"role": "user", "content": prompt}])

    if not remaining_plan_steps and reflection_object.decision == "continue":
        reflection_object.decision = "end"
        print("⚠️ Safety Check: Plan is complete, but RA decided to continue. Overriding to 'end'.")

    print(f"Reflection Assessment: {reflection_object.assessment}")
    print(f"Reflection Decision: {reflection_object.decision}")
    print(f"Reflection feedback_for_planner: {reflection_object.feedback_for_planner}")
    # 这里还需要输出feedback_for_planner

    return {
        "reflection":reflection_object.model_dump(),
        "num_cycles": state["num_cycles"] + 1
        }

#1.初次规划
#2.后续规划包括增添计划+修改后续计划
def Re_strategist_node(state: GraphState):
    """
    The master strategist, capable of initial planning and replanning.
    Refactored to reduce redundancy and fix scope issues.
    """
    print("--- 🧠 Entering Re_Strategist Node ---")

    if state['re_strategist_call_count'] == 0:
        # --- 初次规划模式 ---
        print(f"{BOLD_ON}Mode: Initial Planning{BOLD_OFF}")
        prompt = RE_STRATEGIST_INITIAL_PROMPT.format(
            original_question=state['original_question']
        )
    else:
        # --- 重新规划模式 ---
        print("Mode: Re-Planning")
        history_str = pprint.pformat(state['history'])
        # 安全获取 feedback
        feedback = state.get("reflection", {}).get("feedback_for_planner", "No specific feedback.")
        current_plan = state["plan"] # 获取当前计划用于 Prompt
        
        prompt = RE_STRATEGIST_REPLAN_PROMPT.format(
            original_question=state['original_question'],
            plan=current_plan,
            history=history_str,
            feedback=feedback
        )

    # 2. 统一调用 LLM 
    structured_llm = RESPONSE_MODEL.with_structured_output(Plan)
    plan_object = structured_llm.invoke([{"role": "user", "content": prompt}])
    new_generated_steps = plan_object.steps

    # 3. 确定最终计划 (根据模式不同，处理逻辑不同)
    # 目的是解决当进行replan时query_generator_node出现IndexError: list index out of range问题
    if state['re_strategist_call_count'] == 0:
        # A. 初次规划：不需要拼接，LLM 生成的就是全套计划
        final_plan = new_generated_steps
    else: 
        # B. 重新规划：需要执行拼接逻辑 (解决 IndexError 的关键)
        current_plan = state["plan"] # 再次获取，确保变量存在
        num_executed = len(state["history"])
        
        # 获取旧计划中已完成的部分
        split_point = min(num_executed, len(current_plan))
        completed_steps = current_plan[:split_point]
        
        # 拼接：已完成旧步骤 + 新生成后续步骤
        final_plan = completed_steps + new_generated_steps
        
        print(f"🔄 Re-Plan Logic: Kept {len(completed_steps)} old steps, added {len(new_generated_steps)} new steps.")

    # print(f"Generated/Updated Plan: {final_plan}")

    return {
        "plan": final_plan,
        "plan_archive": [final_plan], # 追加存档
        "re_strategist_call_count": state['re_strategist_call_count'] + 1
    }


# 这是给V1用的，用于判断子计划执行完成没有，是结束执行还在整合子问题子答案回答最终问题
# def should_continue(state: GraphState) -> Literal["continue_loop", "end_loop"]:
#     """Determines whether to continue the loop."""
#     print("--- 🚦 Entering Loop Controller ---")
#     plan = state["plan"]
#     history = state["history"]
    
#     if len(history) < len(plan):
#         print(f"Decision: Plan has {len(plan)} steps, {len(history)} done. Continue loop.")
#         return "continue_loop"
#     else:
#         print(f"Decision: Plan has {len(plan)} steps, {len(history)} done. End loop.")
#         return "end_loop"
  

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
    # print(f"Reflection Agent's decision: '{decision}")
    # print("state['num_cycles']: ", state["num_cycles"])

    if decision == "replan":
        print("Routing to Re_strategist")
        return "replan"
    elif decision == "end":
        print("Routing to: Final Synthesis")
        return "end"
    else: # decision == "continue" or any other unexpected value
        print("Routing to: Query Generator (Continue Loop)")
        return "continue"
          

    




        