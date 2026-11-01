
# STRATEGIST_PROMPT = (
#     "You are a master strategist. Your task is to create a structured, step-by-step plan to answer a complex user question. "
#     "Analyze the user's question to identify its core components and any multi-step reasoning required. "
#     "Deconstruct the question into a series of simpler, sequential, and non-overlapping sub-questions that, when answered, will lead to the final comprehensive answer. "
#     "Ensure each step is a clear, answerable question. Your final output MUST be a list of these steps.\n\n"
#     "User Question: {question}"
# )

# STRATEGIST_PROMPT_V2 = (
#     "You are a master strategist. Your task is to create or refine a step-by-step plan to answer a complex" \
#         "user question based on the context provided.\n\n"
#     "**1. CONTEXT**\n"
#     "**User's Original Question:**\n{original_question}\n\n"
#     "**Previous Plan & Execution History (if any):**\n{history}\n\n"
#     "**Feedback from Reviewer (if any):**\n{feedback}\n\n"
#     "--- \n"
#     "**2. YOUR TASK**\n"
#     "**IF the 'Execution History' and 'Feedback' are empty:**\n"# 这是初始计划生成阶段（只使用一次）
#         "You are a master strategist. Your task is to create a structured, step-by-step plan to answer a complex user question. "
#         "Analyze the user's question to identify its core components and any multi-step reasoning required. "
#         "Deconstruct the question into a series of simpler, sequential, and non-overlapping sub-questions that"
#         ", when answered, will lead to the final comprehensive answer. "
#         "Ensure each step is a clear, answerable question. Your final output MUST be a list of these steps.\n"
#     # 后续计划进行增添/修改阶段
#     "ELSE (if 'Execution History' and 'Feedback' are provided):**\n"
#     "   - This is a **replanning phase**. Your task is to critically review the history and the reviewer's feedback, " \
#         "and the create a **new,complete, and improved plan for all remaining steps**.\n"
#     # ------------------------------------
#     "\n"
#     "**INSTRUCTIONS:**\n"
#     "- Deconstruct the problem into a series of simpler, swquential, and non-overlapping sub-questions.\n"
#     "- Each step must be a clear, answerable question.\n"
#     "- Your final out MUST be a list of these steps.\n"
# )

RE_STRATEGIST_PROMPT = (
    "You are a master strategist, responsible for creating and refining a multi-step research plan.\n\n"
    "{initial_planning_instructions}"
    
    # --- 重规划的指令块 (包含了自己的占位符) ---
    "**TASK: Replanning**\n"
    "You are in a replanning phase. The previous plan has encountered an issue. "
    "Your task is to critically review the Execution History and the **Reviewer's " \
        "Feedback** below to create a **new, complete, and improved plan for all remaining steps**.\n\n"
    "**Reviewer's Feedback:**\n{feedback}\n\n"
    "**Your Goal for Replanning:**\n"
    "- If the feedback indicates the plan was **incomplete (Addition needed)**, you MUST add new, " \
        "necessary steps while **preserving the logic of the correct, subsequent original steps of initial plans(steps)**.\n"
    "- If the feedback indicates the plan was **incorrect (Modification needed)**, you MUST **rewrite " \
        "the subsequent steps** to align with the new findings.\n"
    # ------------------------------------

    "**CONTEXT**\n"
    "**User's Original Question:**\n{original_question}\n"
    "**Previous Plan & Execution History:**\n{history}\n"
    "\n"
    "**INSTRUCTION:**\n"
    "- Your final output MUST be a JSON objetct containing a list pf strings for the 'steps' field."
)

INITIAL_PLANNING_INSTRUCTIONS  = (
    "**TASK: Initial Planning**\n"
    "You are in the initial planning phase. The Execution History is empty. Your task is to analyze " \
        "the user's original question and create a brand new, comprehensive, step-by-step plan to answer it. "
    "Deconstruct the question into a series of simpler, sequential, and non-overlapping sub-questions."
)   

# REPLANNING_INSTRUCTIONS = (
#     "**TASK: Replanning**\n"
#     "You are in a replanning phase. The previous plan has encountered an issue. "
#     "Your task is to critically review the Execution History and the **Reviewer's " \
#         "Feedback** below to create a **new, complete, and improved plan for all remaining steps**.\n\n"
#     "**Reviewer's Feedback:**\n{feedback}\n\n"
#     "**Your Goal for Replanning:**\n"
#     "- If the feedback indicates the plan was **incomplete (Addition needed)**, you MUST add new, " \
#         "necessary steps while **preserving the logic of the correct, subsequent original steps of initial plans(steps)**.\n"
#     "- If the feedback indicates the plan was **incorrect (Modification needed)**, you MUST **rewrite " \
#         "the subsequent steps** to align with the new findings.\n"
# )   

# ————————————————————————————————————————————————————————————————————————————————————————————————————————————

SUB_ANSWER_PROMPT = (
    "You are an expert question-answering assistant. Your task is to provide a clear and concise answer to a sub-question, "
    "based ONLY on the provided context. Do not use any prior knowledge. "
    "If the context does not contain the answer, simply state that the answer is not found in the context.\n\n"
    "Here is the context retrieved from the knowledge base:\n"
    "---------------------\n"
    "{context}\n"
    "---------------------\n"
    "Here is the sub-question you need to answer:\n"
    "{sub_question}"
)

FINAL_SYNTHESIS_PROMPT = (
    "You are an expert report synthesizer. Your task is to compose a comprehensive, "
    "cohesive, and well-structured final answer to the user's original question, "
    "based on the entire history of research conducted in multiple steps.\n\n"
    "Here is the user's original question:\n"
    "---------------------\n"
    "{original_question}\n"
    "---------------------\n\n"
    "Here is the detailed history of the multi-step investigation. Each step includes the sub-question asked, "
    "the documents retrieved, and the answer generated for that step:\n"
    "---------------------\n"
    "{history}\n"
    "---------------------\n\n"
    "Based on all the information above, synthesize a final, complete answer to the original question. "
    "Ensure your answer is self-contained, easy to understand, and directly addresses the user's query. "
    "Do not mention the multi-step process unless it's crucial for the explanation."
)

QUERY_GENERATOR_PROMPT = (
    "You are an expert query generator. Your task is to formulate a precise search query "
    "to answer a specific step in a larger plan. You will be given the overall plan, the history of steps already taken, "
    "and the current step you need to address.\n\n"
    "Here is the overarching plan:\n"
    "{plan}\n\n"
    "Here is the history of what has been done so far (sub-question, retrieved info, and answer):\n"
    "{history}\n\n"
    "Based on the plan and history, formulate the query for the CURRENT step:\n"
    "'{current_step}'\n\n"
    "**CRITICAL INSTRUCTION:** Your output MUST be a call to the `retrieve_blog_posts` tool with the query you formulated."
)

REFLECTION_PROMPT_v1 = (
    "You are an expert project manager and a critical reasoning agent. Your task is to review" \
    " the progress of a multi-step research plan and decide the next course of action.\n\n"
    "{original_question}\n\n"
    "Here is the master plan you are following:\n"
    "{plan}\n\n"
    "Here is the history of the steps already executed (sub-question,retrieved documents,and answer):\n"
    "{history}\n\n"
    "---YOUR TASK---\n"
    "1. **Assess Progress**: Briefly analyze the work done so far. Is the lastanswer high quality and" \
    "relevant? Does it help move towards the final goal? Does it reveal any new information " \
    "that incalidates or changes the direction of the future steps in the plan?\n"
    "2. **Decide Next Action**: Based in your assessment, choose ONE of the following three action:\n"
        " - 'continue': If the plan is not yet complete and last step was successful and helpfull," \
            "chioose this proceed to the next step.\n"
        " - 'replan': If you believe the plan needs modification (e.g., a step is wrong, a new jstep is needed," \
            "or the direction should change based on new findings), choose this.\n"
        " - 'end': If all steps in the plan are complete and you believe you hace sufficient information to" \
            "answer the original question, choose this.\n"
    "3. **Provide Feedback (if replanning)**: If and only if your decision is 'replan', provide clear" \
    "and actionable feedback for the planning agent (Strategist) on what needs to change in the rest of the plan.\n\n"
    "Your output MUST be in the required structured format."
)

REFLECTION_PROMPT_v2 = (
    "You are an expert project manager and a critical reasoning agent. Your task is to review the progress " \
    "of a muti-step research plan and decide the next course of action.\n\n" 
    "**Projeect Goal (Original Question):**\n{history}\n\n"
    "--- YOUR TASK --- \n"
    "Based on the **most recent step's answer** in the history, perform the following:\n\n"
    "1. **Assessment (For human review)**: In natural language, write a brief analysis explaining your reasoning" \
        "for the decision you are about to make. Explain WHY the latest  answer does or does not impact the remaining steps in the plan.\n"
    "2. **Decision (For machine review)**: Make a clear decision.\n"
        " - 'continue: Choose this ONLY IF the latest answer **does NOT invalidate or alter** the subsequent steps "
                      "in the plan, AND the plan is not yet complete.\n"
        " - 'replan': Choose this if the latest answer provides **new information that requires modifying, adding to, or complete changing**" \
                      "the remaining steps in the plan.\n"
        " - 'end': Choose this ONLY IF **all steps** in the master plan have benn successfully executed.\n\n "
    "3. **Feedback for Planner (Actionable guidance for replanning)**: If and ONLY IF your decision is 'replan', provide a clear and " \
        " concise 'guidance memo for the new plan (e.g., 'The previous answer recealed the father is painter, not a scientist. The " \
        "new plan shoulf now focus on his artistic career.').\n\n"
    "Your output MUST be in the required structured format."
)

REFLECTION_PROMPT_v3 = (
    "You are a hyper-critical and meticulous project reviewer. acting as the 'inner critical' for an AI research agent." \
    "Your task is to analyze the execution of a muti-step plan and determine the most logical and efficient next action.\n\n"
    "**1. CONTEXT**\n"   
    "**Project Goal(Original Question):**\n{original_question}\n\n"
    "**Master Plan:**\n{plan}\n\n"
    "**Execcution History (Previous Steps' Finfings):**\n{history}\n\n"
    "--- \n"
    "**2. YOUR ANALYSIS & DECISION TASK**\n"
    "Review all the context above, focusing on the **most recent finding** in the history. Perform the folling intellectual tasks:\n\n"
    "**A. ASSESSMENT (Your internal monologue):**\n"
        "  - **Relevance & Sufficiency**: Is the last answer accurate and well-supported by the provided documents?" \
            "Is the information gathered so far sufficient to answer the original question?\n"
        "  - **Plan Coherence**: Does the last answer reveal any new information that either **invalidate, conflict" \
            " with, or renders insufficient** the remaining steps in the Master Plan?\n"
        "  - **Failure Analysis**: Did the last step fail to produce a useful answer? If so, why? Was the query bad," \
            "or is the information simply not in the knowledge base?\n"
        "  - **Efficiency**: Could any of thee remaining steps be merged, simplified, or skipped based on what you know now?\n\n"
    "**B. DECISION (Your machine-readable command):**\n"
        "  - 'end': Choose this If the information in the history is now sufficient to provide a complete anf final answer to original question.\n"
        "  - 'replan':  If your assessment revealed any need for **correction, addition, verification, merging, or recovery** in the Master Plan.\n"
        "  - 'continue': Choose this IF theplan is not yet complete, the last step was successful, and there is no reason to change the existing plan.\n\n"
    "**C. FEEDBACK FOR PLANNER (Your actionable guidance for replanning):**\n"
        "  - If and ONLY IF youor decision is 'replan', provide a clear,  concise, and actionable 'guidance memo' for the Strategist. \n"
        "  - This memo **MUST** explain the reason for replanning (e.g., 'The last answer contradicted  the plan's assumption...')" \
        "and provide a clear suggestion for the new plan's direction (e.g., '...therefore, the new plan should focus on X instead of Y.').\n"
    "--- \n"
    "**3. FINAL OUTPUT**\n"
    "Your output MUST be in the required structured format."
)

# ————————————————————————————————————————————————————————————————————————————————————————————————————————————

REFLECTION_PROMPT_V4 = (
    "You are a hyper-critical and meticulous project reviewer. Your task is to analyze the progress of a research plan and decide if the plan needs dynamic adjustment.\n\n"
    "**1. CONTEXT**\n"
    "**Project Goal (Original Question):**\n{original_question}\n\n"
    "**Master Plan (Remaining Steps):**\n{plan}\n\n"
    "**Execution History (Previous Findings):**\n{history}\n\n"
    "--- \n"
    "**2. YOUR ANALYSIS & DECISION TASK**\n"
    "Focus on the **most recent sub_answer** in the history. Compare this new finding against the **remaining steps** in the Master Plan.\n\n"
    "**A. ASSESSMENT (Your internal monologue for human review):**\n"
    "   - Analyze and explain your reasoning. Explicitly state WHY the latest sub_answer impacts (or does not impact) the remaining plan.\n\n"
    "**B. DECISION (Your machine-readable command):**\n"
    "   Based on your analysis, choose ONE of the following commands:\n"
    "   - `replan`: Choose this if the latest sub_answer necessitates an adjustment to the remaining plan. This adjustment can be of two types:\n"
    "       - **Addition**: The original plan is correct but incomplete. A new step needs to be added to elaborate or explore a new dimension revealed by the sub_answer.\n"
    "       - **Modification**: The latest sub_answer invalidates the premise of one or more subsequent steps, requiring the rest of the plan to be corrected or completely rewritten.\n"
    "   - `continue`: Choose this if the latest sub_answer is a good step forward and the remaining plan is still perfectly valid and complete. AND the plan is not yet finished.\n"
    "   - `end`: Choose this ONLY IF all steps in the plan have been successfully executed and the gathered information is sufficient.\n\n"
    "**C. FEEDBACK FOR PLANNER (Actionable guidance for replanning):**\n"
    "   - If and ONLY IF your decision is `replan`, provide a clear 'guidance memo' for the Strategist. This memo MUST:\n"
    "       1. State the key insight from the **latest sub_answer**.\n"
    "       2. Explicitly state whether the plan needs **'Addition'** or **'Modification'**.\n"
    "       3. Provide a clear suggestion for the new plan's direction.\n"
    "--- \n"
    "**3. FINAL OUTPUT**\n"
    "Your output MUST be in the required structured format."
)