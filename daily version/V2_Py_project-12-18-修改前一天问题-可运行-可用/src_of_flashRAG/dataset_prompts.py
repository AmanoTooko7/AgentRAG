# src_of_flashRAG/dataset_prompts.py

def get_dataset_instruction(dataset_type: str) -> str:
    """
    根据数据集类型返回特定的指令和 Few-shot 示例。
    """
    
    # ================= 1. NQ (Natural Questions) =================
    # 特点：短事实，通常是实体名、日期、数字。
    if dataset_type == "nq":
        return (
            "\n\n**TASK:** Answer the question concisely based on the retrieved information.\n"
            "**FORMAT:** Output ONLY the key entity, date, or short phrase inside tags. Do NOT write a full sentence.\n"
            "**EXAMPLES:**\n"
            "Q: who sings does he love me with reba\n"
            "A: <answer>Linda Davis</answer>\n"
            "Q: when does the new my hero academia movie come out\n"
            "A: <answer>July 5, 2018</answer>\n"
            "**YOUR TURN:**\n"
        )

    # ================= 2. TriviaQA =================
    # 特点：常识/冷门知识，答案通常是实体，有时会有别名。
    elif dataset_type == "triviaqa":
        return (
            "\n\n**TASK:** Answer the trivia question concisely.\n"
            "**FORMAT:** Output ONLY the answer entity inside tags. Avoid extra words like 'The answer is'.\n"
            "**EXAMPLES:**\n"
            "Q: What is the capital of Kenya?\n"
            "A: <answer>Nairobi</answer>\n"
            "Q: Sarah FitzGerald has been a 90s world champion in which sport?\n"
            "A: <answer>Squash</answer>\n"
            "**YOUR TURN:**\n"
        )

    # ================= 3. HotpotQA =================
    # 特点：多跳推理，答案通常是实体或 Yes/No。
    elif dataset_type == "hotpotqa":
        return (
            "\n\n**TASK:** Answer the multi-hop reasoning question based on the evidence.\n"
            "**FORMAT:** Output ONLY the final answer entity or 'yes'/'no' inside tags.\n"
            "**EXAMPLES:**\n"
            "Q: Are both Dictyosperma, and Huernia described as a genus?\n"
            "A: <answer>yes</answer>\n"
            "Q: What occupation was shared by David Yates and Pietro Germi?\n"
            "A: <answer>director</answer>\n"
            "**YOUR TURN:**\n"
        )

    # ================= 4. 2WikiMultiHopQA =================
    # 特点：结构化多跳，涉及两个实体关系，答案通常是人名、地名、电影名。
    elif dataset_type == "2wikimultihopqa":
        return (
            "\n\n**TASK:** Answer the complex multi-hop question.\n"
            "**FORMAT:** Output ONLY the specific entity (person, place, film, etc.) inside tags.\n"
            "**EXAMPLES:**\n"
            "Q: What is the place of birth of Paulding Farnham's wife?\n"
            "A: <answer>Ogdensburg, New York</answer>\n"
            "Q: Who is the father of the director of film Biwi-O-Biwi?\n"
            "A: <answer>H. S. Rawail</answer>\n"
            "**YOUR TURN:**\n"
        )

    # ================= 5. FEVER =================
    # 特点：事实验证，三分类 (SUPPORTS, REFUTES, NOT ENOUGH INFO)。
    # 注意：FlashRAG 的评估脚本通常期望全大写或特定格式，这里我们严格对齐。
    elif dataset_type == "fever":
        return (
            "\n\n**TASK:** Verify the claim based on the evidence.\n"
            "**FORMAT:** You MUST output exactly one of these labels inside tags: <answer>SUPPORTS</answer>, <answer>REFUTES</answer>, or <answer>NOT ENOUGH INFO</answer>.\n"
            "**EXAMPLES:**\n"
            "Claim: Fox 2000 Pictures released the film Soul Food.\n"
            "A: <answer>SUPPORTS</answer>\n"
            "Claim: Savages was exclusively a German film.\n"
            "A: <answer>REFUTES</answer>\n"
            "**YOUR TURN:**\n"
        )

    # ================= 6. StrategyQA =================
    # 特点：隐式推理，二分类 (yes/no)。
    elif dataset_type == "strategyqa":
        return (
            "\n\n**TASK:** Answer the question with Yes or No based on implicit reasoning.\n"
            "**FORMAT:** You MUST output the answer inside tags: <answer>yes</answer> or <answer>no</answer>.\n"
            "**EXAMPLES:**\n"
            "Q: Can you buy Casio products at Petco?\n"
            "A: <answer>no</answer>\n"
            "Q: Did land owners elect their rulers in the Kingdom of Hungary?\n"
            "A: <answer>no</answer>\n"
            "**YOUR TURN:**\n"
        )

    # ================= Default =================
    else:
        return (
            "\n\n**TASK:** Answer the question concisely.\n"
            "**FORMAT:** Output ONLY the entity or short phrase inside tags. Example: <answer>Barack Obama</answer>."
        )