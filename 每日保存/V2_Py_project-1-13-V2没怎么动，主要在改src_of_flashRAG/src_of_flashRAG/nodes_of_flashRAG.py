# 构建节点和条件边
import os
import pprint
from typing import Literal
import json
# from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator
from typing import List
import time
from pydantic import ValidationError

from src_of_flashRAG.state import GraphState
from src_of_flashRAG.prompts import (
    SUB_ANSWER_PROMPT,
    FINAL_SYNTHESIS_PROMPT,
    QUERY_GENERATOR_PROMPT_V2,
    REFLECTION_PROMPT_V6,
    RE_STRATEGIST_INITIAL_PROMPT,
    RE_STRATEGIST_REPLAN_PROMPT,
)
# from src_of_flashRAG.retriever import build_retriever_tool
from src_of_flashRAG.retriever_of_wiki import build_retriever_tool # 这是混合检索的检索器
# from src_of_flashRAG.retriever_of_wiki_dense_search_only import build_retriever_tool # 这仅有稠密检索
from src_of_flashRAG.re_ranker import get_ranker_score
from src_of_flashRAG.constants import BOLD_ON, BOLD_OFF
from src_of_flashRAG.dataset_prompts import get_dataset_instruction 
# from src_of_flashRAG.utils_llm import safe_invoke_llm


# 下面调用Close AI中的模型————————————————————————————————————————————————————————————
# CloseAI_API_KEY = "sk-hOiSoNaP9k5qXSw1MjE9EmaNB3oAwMB9qxr4rZN3zkPq80ch"
# CloseAI_BASE_URL = "https://api.openai-proxy.org/v1"

# MODEL_MAP = {
#     "deepseek": "deepseek-chat"
# }
# CURRENT_MODEL_KEY = "deepseek"
# print(f"🚀 Loading Model via CloseAI: {MODEL_MAP[CURRENT_MODEL_KEY]}")

# # 初始化模型
# RESPONSE_MODEL = ChatOpenAI(
#     model= MODEL_MAP[CURRENT_MODEL_KEY],
#     openai_api_key=CloseAI_API_KEY,
#     openai_api_base=CloseAI_BASE_URL,
#     temperature=0,
#     max_tokens=4096,
# )
# —————————————————————————————————————————————————————————————————————————————————————
    # 阿里百炼平台
# DASHSCOPE_API_KEY = "sk-757aab3d1cdf47d2896bfef1e45efa1d"   
# DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# MODEL_MAP = {
#     "qwen3-8b": "qwen3-8b",
#     "qwen3-32b": "qwen3-32b",
#     "qwen3-next-80b-a3b-instruct": "qwen3-next-80b-a3b-instruct"
# }
# 这是咸鱼买的openrouter key:sk-ZdA8x8rUMCawZd9d5k2sGrpLObahFpdmVZwi0ScTSqIor5hU
# 咸鱼买的中转网站https://openrouter.fans/v1

# openrouter平台
DASHSCOPE_API_KEY = "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"   
DASHSCOPE_BASE_URL = "https://openrouter.ai/api/v1"

MODEL_MAP = {
        "llama3-8b": "meta-llama/llama-3.1-8b-instruct",
    }

NODE_MODEL_CONFIGS = {
    "Re_strategist_node": {
        "temperature": 0.1, "max_tokens": 512, "model_name": MODEL_MAP["llama3-8b"] 
    },
    "query_generator_node": {
        "temperature": 0.1, "max_tokens": 128, "model_name": MODEL_MAP["llama3-8b"] 
    },
    "sub_answer_generator_node": {
        "temperature": 0.1, "max_tokens": 300, "model_name": MODEL_MAP["llama3-8b"] 
    },
    "reflection_agent_node": {
        "temperature": 0.1, "max_tokens": 768, "model_name": MODEL_MAP["llama3-8b"] 
    },
    "final_synthesis_node": {
        "temperature": 0.6, "max_tokens": 200, "model_name": MODEL_MAP["llama3-8b"] 
    },
}

# 1. 定义一个全局字典来存储已创建的模型实例
_MODEL_INSTANCE_CACHE = {}
def get_chat_model(node_name: str):
    """根据节点名称返回配置好的 ChatOpenAI 实例"""
    if node_name in _MODEL_INSTANCE_CACHE:
        return _MODEL_INSTANCE_CACHE[node_name]

    default_config = {
        "temperature": 0.1,
        "max_tokens": 512,
        "model_name": MODEL_MAP["llama3-8b"]
    }
    
    config = NODE_MODEL_CONFIGS.get(node_name, default_config)
    
    # print(f"🚀 Loading Model for [{node_name}] | Model: {config['model_name']} | Temp: {config['temperature']}")

    new_instance = ChatOpenAI(
        model=config["model_name"],
        openai_api_key=DASHSCOPE_API_KEY,
        openai_api_base=DASHSCOPE_BASE_URL,
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
        # extra_body={ # useing qwen must use this
        #     "enable_search": False,
        #     "enable_thinking": False
        # }
    )
    _MODEL_INSTANCE_CACHE[node_name] = new_instance
    
    return new_instance


# # 下面调用阿里百炼平台和Open router平台的模型，解耦为get_chat_model()函数了
# def init_model(model_key: str):
#     """
#     由 main.py 调用，用于初始化全局的 RESPONSE_MODEL
#     """
#     global RESPONSE_MODEL
    
#     # 阿里百炼平台
#     # DASHSCOPE_API_KEY = "sk-757aab3d1cdf47d2896bfef1e45efa1d"   
#     # DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
#     # MODEL_MAP = {
#     #     "qwen3-8b": "qwen3-8b",
#     #     "qwen3-32b": "qwen3-32b",
#     #     "qwen3-next-80b-a3b-instruct": "qwen3-next-80b-a3b-instruct"
#     # }

#     # # Open router平台
#     DASHSCOPE_API_KEY = "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"   
     
#     DASHSCOPE_BASE_URL = "https://openrouter.ai/api/v1" # 这是openrouter的API地址
    
#     #Open router平台模型
#     MODEL_MAP = {
#         "llama3-8b": "meta-llama/llama-3.1-8b-instruct",
#     }

#     if model_key not in MODEL_MAP:
#         raise ValueError(f"❌ 模型 Key '{model_key}' 不在 MODEL_MAP 中！可选: {list(MODEL_MAP.keys())}")

#     model_name = MODEL_MAP[model_key]
#     # print(f"🚀 [Init] Loading Model via Alibaba DashScope: {model_name}")

#     RESPONSE_MODEL = ChatOpenAI(
#         model=model_name,
#         openai_api_key=DASHSCOPE_API_KEY,
#         openai_api_base=DASHSCOPE_BASE_URL,
#         temperature=0.1,
#         max_tokens=512,
#         # extra_body={ # useing qwen must use this
#         #     "enable_search": False,
#         #     "enable_thinking": False
#         # }
#     )


# —————————————————————————————————————————————————————————————————————————————————————
# 下面调用huggingface的API
# HF_TOKEN = "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" 

# # 2. 设置 Base URL (HF 的 OpenAI 兼容路由)
# HF_BASE_URL = "https://router.huggingface.co/v1"

# # 3. 模型名称 (必须是 HF 上的完整路径)
# MODEL_NAME = "Qwen/Qwen3-8B:nscale" 

# print(f"🚀 Loading Model via HuggingFace API: {MODEL_NAME}")

# RESPONSE_MODEL = ChatOpenAI(
#     model=MODEL_NAME,
#     openai_api_key=HF_TOKEN,
#     openai_api_base=HF_BASE_URL,
#     temperature=0,
#     max_tokens=4096,
# )
# # —————————————————————————————————————————————————————————————————————————————————————
# RESPONSE_MODEL = ChatOllama(model="qwen3:8b",temperature=1)
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
    llm = get_chat_model("query_generator_node")

    plan = state["plan"]
    history = state["history"][-1:]            # <------------------
    current_step_index = len(state["history"])
    current_step = plan[current_step_index]    
    history_str = pprint.pformat(history)
    
    prompt = QUERY_GENERATOR_PROMPT_V2.format(
        plan=plan, history=history_str, current_step=current_step
    )
    
    print(f"{BOLD_ON}Formulating query for step {current_step_index + 1}{BOLD_OFF}: '{current_step}'")
    response = llm.bind_tools([RETRIEVER_TOOL]).invoke(
        [{"role": "user", "content": prompt}]
    )
    print(f"{BOLD_ON}query_generator_node:{BOLD_OFF}\n{response}")
    
    return {"messages": [response], "current_sub_question": current_step}

def sub_answer_generator_node(state: GraphState):
    """Generates an answer for a single sub-question."""
    print("--- 🤖 Entering Sub-Answer Generation Node ---")
    llm = get_chat_model("sub_answer_generator_node")

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
    response = llm.invoke([{"role": "user", "content": prompt}])
    sub_answer = response.content
    
    # print(f"Generated Sub-Answer: {sub_answer[:200]}...")
    step_output = {
        "sub_question": sub_question, # <——————————————————————返回的内容这里应更新子query
        # "retrieved_docs": [str(doc.content) for doc in retrieved_docs_messages], 
        "retrieved_docs": [context] , 
        "answer": sub_answer,
    }
    print(f"{BOLD_ON}sub_answer_generator_node:{BOLD_OFF}\n{sub_question}\n{sub_answer}")
    return {"history": [step_output]}
  

# 这是给FlashRAG用的，加入了dataset_type 状态
def final_synthesis_node(state: GraphState):
    print("--- 🖋️ Entering Final Synthesis Node ---")
    llm = get_chat_model("final_synthesis")

    original_question = state["original_question"]
    history = state["history"]
    dataset_type = state.get("dataset_type", "qa") # 默认 qa数据集
    
    history_str = pprint.pformat(history)
    
    # 基础 Prompt
    # base_prompt = FINAL_SYNTHESIS_PROMPT.format(
    #     original_question=original_question, history=history_str
    # )
    
    instruction = get_dataset_instruction(dataset_type)
    final_prompt = FINAL_SYNTHESIS_PROMPT.format(
        original_question=original_question, 
        history=history_str,
        instruction=instruction  # <--- 填入指令
    )
        
    # final_prompt = base_prompt + instruction
    # print(f"{BOLD_ON}final_synthesis_node_final_prompt:{BOLD_OFF}\n{final_prompt}")
    
    response = llm.invoke([{"role": "user", "content": final_prompt}])
    final_answer = response.content
    print(f"{BOLD_ON}fianl_synthesis_node:{BOLD_OFF}\n{final_answer}")
    
    return {"final_answer": final_answer}



class Reflection(BaseModel):
    """Represents the output of the Reflection Agent
    [属性名]: [类型] = Field(description="给LLM看的、关于这个属性的详细说明")
     """
    assessment: str = Field(
        description="A detailed, natural language analysis of the progress so far. Explain WHY the latest " \
        "answer impacts (or does not impact) the remaining plan. This is for human review and context."
        )
    decision: Literal["continue", "replan", "end"] = Field(
        description="The machine-readable final decision. Must be one of 'continue', 'replan', or 'end'."
        )
    feedback_for_planner: str = Field(
        description="Actionable guidance for the Strategist, to be provided ONLY if the decision is 'replan'. " \
        "State the key new insight and suggest the direction for the new plan."
        "CRTICAL: Output this as a single flat string. Do Not use nested JSON objects of dictionaries."
        )
    
    @field_validator("feedback_for_planner", mode="before")
    @classmethod
    def handle_dict_feedback(cls, v):
        if isinstance(v, dict):
            return json.dumps(v, ensure_ascii=False)
        return str(v) if not isinstance(v, str) else v


def reflection_agent_node(state: GraphState):
    """This node acts as the 'inner critic', reflecting on the progress so far
       and decide whether to continue, replan, or finish."""
    print("--- 🧠🧠 Entering Reflection Agent Node ---")
    llm = get_chat_model("reflection_agent_node")

    original_question = state["original_question"]
    # plan = state["plan"]
    history = state["history"][-1:]              # <------------------

    num_completed_steps = len(state["history"])
    remaining_plan_steps = state["plan"][num_completed_steps:] # 这一步是拿到剩余的步骤

    import pprint
    history_str = pprint.pformat(history)

    prompt = REFLECTION_PROMPT_V6.format(
        original_question=original_question,
        plan=remaining_plan_steps,
        history=history_str
    )

    structured_llm = llm.with_structured_output(Reflection)

    # 以下内容解决模型不按Reflection要求返回正确格式出现error processing的问题
    reflection_object = None
    max_retries = 3 # 重试3次
    for attempt in range(max_retries):
        try:
            reflection_object = structured_llm.invoke([{"role": "user", "content": prompt}])

            # if reflection_object is None:
            #     raise ValueError("❌Model returned error format")
            break
        except Exception as e:
            print(f"⚠️[Reflection Error] Attemp {attempt + 1}/{max_retries} failed: {e}")
            
            if attempt < max_retries - 1:
                wait_time = 1
                print(f"⏳Waiting {wait_time}s before retrying...")
                time.sleep(wait_time)
            else: print("❌All retries failed. Using fallback.")

    if reflection_object is None:
        reflection_object = Reflection(
            assessment="Error parsing model output after retries. System defaulted to continue.",
            decision="continue",
            feedback_for_planner=""
        )

    if not remaining_plan_steps and reflection_object.decision == "continue":
        reflection_object.decision = "end"
        print("⚠️ Safety Check: Plan is complete, but RA decided to continue. Overriding to 'end'.")

    # print(f"reflection:{reflection_object.model_dump()}")
    print(f"{BOLD_ON}reflection_agent_node:{BOLD_OFF}\n{reflection_object.model_dump()}")
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
    llm = get_chat_model("Re_strategist_node")

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
        current_plan = state["plan"] # 获取当前计划，用于 Prompt
        
        prompt = RE_STRATEGIST_REPLAN_PROMPT.format(
            original_question=state['original_question'],
            plan=current_plan,
            history=history_str,
            feedback=feedback
        )

    # 2. 统一调用 LLM 
    structured_llm = llm.with_structured_output(Plan)
    plan_object = None
    max_retries = 3 # 重试3次
    for attempt in range(max_retries):
        try:
            plan_object = structured_llm.invoke([{"role": "user", "content": prompt}])
            break
        except Exception as e:
            print(f"⚠️[Re_strategist Error] Attemp {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else: print("❌All retries failed. Using fallback.")
    
    # 模型3次都没生成计划，初始计划就用原始问题，重规划就用空计划
    if plan_object is None:
        if state['re_strategist_call_count'] == 0:
            # A. 初次规划兜底：直接把原始问题当作唯一的步骤
            print("⚠️ Fallback: Using original question as the plan.")
            fallback_steps = [state['original_question']]
        else:
            # B. 重规划兜底：不增加新步骤 (返回空列表)
            print("⚠️ Fallback: No new steps added during replanning.")
            fallback_steps = []
            
        plan_object = Plan(steps=fallback_steps)


    new_generated_steps = plan_object.steps

    # 3. 确定最终计划 (根据模式不同，处理逻辑不同)
    # 目的是解决当进行replan时query_generator_node出现IndexError: list index out of range问题
    if state['re_strategist_call_count'] == 0:
        # A. 初次规划：不需要拼接，LLM 生成的就是全套计划
        final_plan = new_generated_steps
    else: 
        # B. 重新规划：需要执行拼接逻辑 
        current_plan = state["plan"] # 再次获取，确保变量存在
        num_executed = len(state["history"])
        
        # 获取旧计划中已完成的部分
        split_point = min(num_executed, len(current_plan))
        completed_steps = current_plan[:split_point]
        
        # 拼接：已完成旧步骤 + 新生成后续步骤
        final_plan = completed_steps + new_generated_steps
        
        # print(f"🔄 Re-Plan Logic: Kept {len(completed_steps)} old steps, added {len(new_generated_steps)} new steps.")

    print(f"{BOLD_ON}Re_strategist_node:{BOLD_OFF}\n {final_plan}")

    return {
        "plan": final_plan,
        "plan_archive": [final_plan], # 追加存档
        "re_strategist_call_count": state['re_strategist_call_count'] + 1
    }

  

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
    # print("--- 🚦 Entering Central Router (reflection_check) ---")

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
          

    




        