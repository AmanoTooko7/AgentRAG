
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

RE_STRATEGIST_INITIAL_PROMPT = (
    "You are a master strategist. Your task is to create a structured, step-by-step plan to answer a complex user question.\n"
    "You are in the initial planning phase. Analyze the user's question to identify its core components and any multi-step reasoning required.\n"
    "Deconstruct the question into a series of simpler, sequential, and non-overlapping sub-questions that, when answered, will lead to the final comprehensive answer.\n"
    "Ensure each step is a clear, answerable question.\n\n"
    "**CONTEXT**\n"
    "**User's Original Question:**\n{original_question}\n\n"
    "**INSTRUCTIONS:**\n"
    "- Your final output MUST be a JSON object containing a list of strings for the 'steps' field."
)

RE_STRATEGIST_REPLAN_PROMPT = (
    # 这个进行重新规划生成的计划中，不包括之前执行过的子计划，只有新的后续计划
    "You are performing a targeted replanning operation. Your goal is to refine the current research plan "
    "for the remaining steps with the **minimal necessary change** based on the review feedback.\n\n"

    "**--- CONTEXT FOR REPLANNING ---**\n\n"
    "1. **The User's Original Goal:**\n{original_question}\n\n"
    "2. **The Previous Plan (the one we are modifying):**\n{plan}\n\n"
    "3. **The Execution History So Far (what has been done):**\n{history}\n\n"
    "4. **The Reviewer's Feedback (WHY we are replanning):**\n{feedback}\n\n"

    "**--- YOUR TASK & INSTRUCTIONS ---**\n\n"
    "Based on the ReflectionAgent's feedback, update the current plan with **minimal change**:\n"
    "— If feedback_for_planner includes 'Addition', your output should insert only 1–2 new steps and followed by **all the original, unmodified remaining steps.** "
    "The new steps must directly address the identified gap.\n"
    "— If feedback_for_planner includes 'Modification', replace or refine the relevant existing step(s) instead of regenerating a full new plan.\n"
    "— You MUST preserve as much of the original plan’s structure as possible. The number of new steps cannot exceed 20% of the remaining plan.\n"
    "— Never regenerate the entire plan unless explicitly stated that the previous plan is invalid in its entirety.\n\n"

    "Your final output MUST be a JSON object containing a list of strings for the 'steps' field, "
    "representing the **NEW PLAN FOR ALL REMAINING STEPS ONLY.** "
    "Do NOT include steps that are already completed in the Execution History.\n\n"

    "**--- FEW-SHOT EXAMPLES (for reference) ---**\n\n"

    "### Example 1 — Addition Case (Minor Extension)\n"
    "Previous Plan:\n"
    "1. Review prior research (completed)\n"
    "2. Collect experimental data (completed)\n"
    "3. Analyze potential causes of performance degradation (incomplete)\n"
    "4. Evaluate solutions (incomplete)\n"
    "5. Summarize findings (incomplete)\n\n"
    "Feedback:\n"
    "The reflection noted that an 'ethical impact analysis' step is missing between analysis and evaluation.\n"
    "→ Action: Add one new step addressing ethics before solution evaluation.\n"
    "Expected Output:\n"
    "{ 'steps': [ "
    "'Add an ethical impact analysis of the performance issues.', "
    "'Analyze potential causes of performance degradation.', "
    "'Evaluate solutions.', "
    "'Summarize findings.' ] }\n\n"

    "### Example 2 — Modification (Reasoning Route)\n"
    "Previous Plan:\n"
    "1. Review existing technology A literature (completed)\n"
    "2. Analyze how technology A influences process B (completed)\n"
    "3. Evaluate the impact on system C (incomplete)\n"
    "4. Derive implications for D (incomplete)\n\n"
    "Feedback:\n"
    "Newly retrieved evidence shows that technology A has been replaced by a newer version A2, "
    "which fundamentally changes how process B operates. "
    "Steps 3 and 4 rely on outdated assumptions tied to A, not A2.\n"
    "→ Action: Modify the plan to realign with this new knowledge by replacing outdated steps "
    "and analyzing A2’s updated influence on B and C.\n"
    "Expected Output:\n"
    "{ 'steps': [ "
    "'Analyze how the updated technology A2 changes process B.', "
    "'Evaluate how A2-driven changes in B affect system C.', "
    "'Derive updated implications for D under the new technological context.' ] }\n\n"

    "### Example 3 — Modification (Investigation / Forensic Reasoning)\n"
    "Previous Plan:\n"
    "1. Interview key witnesses (completed)\n"
    "2. Investigate suspect A’s financial transactions (completed)\n"
    "3. Cross-check A’s presence near the crime scene (incomplete)\n"
    "4. Prepare a final report confirming A as the primary suspect (incomplete)\n\n"
    "Feedback:\n"
    "The latest sub-answer, based on newly retrieved documents, reveals that the suspicious transactions "
    "were actually conducted by suspect B using A’s identity. "
    "This discovery invalidates Steps 3 and 4, as the assumption that A is the main suspect is now false.\n"
    "→ Action: Revise the plan to redirect the investigation focus toward suspect B "
    "and reassess all prior evidence.\n"
    "Expected Output:\n"
    "{ 'steps': [ "
    "'Verify the identity misuse in the financial transaction records.', "
    "'Investigate suspect B’s connection to both A and the crime scene.', "
    "'Reassess all evidence to determine the true primary suspect.' ] }\n"
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


# ————————————————————————————————————————————————————————————————————————————————————————————————————————————


# REFLECTION_PROMPT_V5 = (
#     "You are a hyper-critical and meticulous project reviewer. "
#     "Your task is to analyze the progress of a research plan and decide if the plan needs dynamic adjustment.\n\n"

#     "**1. CONTEXT**\n"
#     "**Project Goal (Original Question):**\n{original_question}\n\n"
#     "**Master Plan (Remaining Steps):**\n{plan}\n\n"
#     "**Execution History (Previous Findings):**\n{history}\n\n"
#     "---\n"
#     "**2. YOUR ANALYSIS & DECISION TASK**\n"
#     "Focus on the **most recent sub_answer** in the history. Compare this new finding against "
#     "the **remaining steps** in the Master Plan.\n\n"

#     "**A. ASSESSMENT (Your internal monologue for human review):**\n"
#     "   - Analyze and explain your reasoning. Explicitly state WHY the latest sub_answer impacts "
#     "(or does not impact) the remaining plan.\n\n"

#     "**B. DECISION (Your machine-readable command):**\n"
#     "   Based on your analysis, choose ONE of the following commands:\n\n"

#     "   - `replan`: Choose this ONLY IF the latest sub_answer introduces a **critical finding that makes the *immediate* next step in the plan "
#     "               logically impossible or fundamentally misguided**. Do not replan for minor additions or related topics"
#     "       - **Addition**: The original plan is correct but *conceptually incomplete* — "
#     "                       a genuinely new analytical dimension or question has emerged that cannot be covered by existing steps.\n"
#     "       - **Modification**: The latest sub_answer invalidates one or more assumptions in the current plan, "
#     "                       requiring those steps to be rewritten.\n"
#     "   ⚠️ Only choose `replan` if the latest finding reveals **new analytical dimensions or invalidates assumptions**, "
#     "not merely strengthens logical connections or adds detail within existing steps.\n\n"

#     "   - `continue`: Choose this if the latest sub_answer is a good step forward and the remaining plan "
#     "is still perfectly valid and complete, AND the plan is not yet finished.\n"
#     "   - `end`: Choose this ONLY IF all steps in the plan have been successfully executed and "
#     "the gathered information is sufficient.\n\n"

#     "**C. FEEDBACK FOR PLANNER (Actionable guidance for replanning):**\n"
#     "   - If and ONLY IF your decision is `replan`, provide a clear 'guidance memo' for the Strategist. "
#     "This memo MUST:\n"
#     "       1. State the key insight from the **latest sub_answer**.\n"
#     "       2. Explicitly state whether the plan needs **'Addition'** or **'Modification'**.\n"
#     "       3. Provide a clear suggestion for the new plan's direction.\n"
#     "---\n"
#     "**3. FINAL OUTPUT**\n"
#     "Your output MUST be in the required structured format.\n"
# )

REFLECTION_PROMPT_V6 = (
    "You are a Reflection Agent responsible for assessing whether the current sub-answer "
    "requires adjusting the overall plan.\n\n"
    "You MUST NOT trigger a replan unless **one of the following strict conditions** is met:\n"
    "1. The sub-answer reveals a factual contradiction or logical inconsistency in the existing plan "
    "   (e.g., later steps depend on a false or invalid premise).\n"
    "2. The sub-answer exposes a major unplanned dimension that fundamentally changes the "
    "   research scope (e.g., a new paradigm, a new objective, or a new data source that invalidates current assumptions).\n"
    "3. The plan’s current remaining steps are no longer sufficient to reach the final goal "
    "   because the sub-answer introduces a new requirement directly necessary for completion.\n\n"
    "Minor discoveries, new examples, or analytical refinements are **NOT valid reasons** to trigger replanning. "
    "In such cases, return decision = 'continue' and include a short feedback note such as "
    "'Potential future extension: mitigation analysis.'\n\n"
    
    "**1. CONTEXT**\n"
    "**Project Goal (Original Question):**\n{original_question}\n\n"
    "**Master Plan (Remaining Steps):**\n{plan}\n\n"
    "**Execution History (Previous Findings):**\n{history}\n\n"
    "---\n"
    "**2. YOUR ANALYSIS & DECISION TASK**\n"
    "Focus primarily on the **most recent sub_answer** in the execution history. "
    "Compare its implications against the **remaining steps** in the Master Plan.\n\n"

    "**A. ASSESSMENT (Your reasoning for human interpretability):**\n"
    " - Explain your reasoning process clearly.\n"
    " - Identify whether the latest sub_answer:\n"
    "     • introduces a *new analytical dimension* not covered in the plan (→ Addition), or\n"
    "     • invalidates assumptions in the plan (→ Modification), or\n"
    "     • merely reinforces existing steps (→ Continue).\n\n"

    "**B. DECISION (Machine-readable command):**\n"
    "Choose ONE of the following actions:\n"
    "   - `replan`: ONLY IF one of the above strict conditions is satisfied.\n"
    "       • **Addition** — The plan is correct but *conceptually incomplete*;  a new analytical dimension or requirement"
    "               emerged that cannot be handled by the current plan.\n"
    "       • **Modification** — The plan is *partly invalidated* because one or more later steps depend on a false assumption or outdated premise.\n"
    "   - `continue`: Choose this in **most cases** — "
    "       The new finding fits naturally within the current reasoning path, supports existing steps, and does not alter the fundamental logic of the plan. "
    "       This should be your default decision unless the sub_answer clearly meets one of the strict replan conditions above.\n"
    "   - `end`: All steps completed; no further work needed.\n\n"

    "⚠️ You must be *extremely conservative* when triggering `replan`. "
    "Do NOT replan for minor additions, refinements, or stylistic expansions. "
    "Trigger only when the *research trajectory itself* must change.\n\n"

    "**C. FEEDBACK FOR PLANNER (Structured memo for Re-Strategist):**\n"
    "   If and ONLY IF you decide `replan`, provide a concise, structured memo including:\n"
    "   1. The key insight from the latest sub_answer that triggered replanning.\n"
    "   2. Explicitly state whether the needed change is an **'Addition'** or a **'Modification'**.\n"
    "   3. A clear suggestion for what kind of new or revised steps are needed.\n\n"
    "---\n"
    "**3. FINAL OUTPUT FORMAT**\n"
    "Your output MUST strictly follow this structure:\n"
    "{\n"
    "  'assessment': <detailed reasoning>,\n"
    "  'decision': 'continue' | 'replan' | 'end',\n"
    "  'feedback_for_planner': <structured memo if decision == 'replan'>\n"
    "}\n"
)
