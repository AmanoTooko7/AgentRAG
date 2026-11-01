# src_of_flashRAG/dataset_prompts.py

def get_dataset_instruction(dataset_type: str) -> str:
    """
    根据数据集类型返回特定的指令和 Few-shot 示例。
    """
    
    # ================= 1. NQ (Natural Questions) =================
    # 特点：短事实，通常是实体名、日期、数字。
    if dataset_type == "nq":
        return (
            "\n\n**TASK:** Answer the question concisely.\n"
            "**FORMAT:** Output ONLY the answer strbe an Entity, Date, Number, or a Short Phrase. Do NOT write a full sentence.\n"
            
            "**STRATEGY:**\n"
            "1. **Minimalist:** Remove unnecessary context. (e.g., instead of 'Super Bowl XLIII, Feb 1, 2009', just output 'Feb 1, 2009').\n"
            "2. **Guessing:** If the exact answer is not found, make an educated guess based on the context. Do NOT output 'Not found' or 'Error'.\n"
            "3. **Lists:** If asked for a list (e.g., 'top 10'), provide a comma-separated list of the most prominent entities.\n"
            "4. **Qualifiers:** For quantity questions, include necessary qualifiers (e.g., 'a minimum of five').\n\n"
            
            "**EXAMPLES:**\n"
            "Q: who played young king t'chaka in black panther\n"
            "A: <answer>Atandwa Kani</answer>\n"
            "Q: when did transformers the last knight come out\n"
            "A: <answer>June 21, 2017</answer>\n"
            "Q: how many games to get premier league medal\n"
            "A: <answer>a minimum of five</answer>\n"
            "Q: who has the rdsp been created to benefit\n"
            "A: <answer>individuals with disabilities</answer>\n"
            "Q: when did bruce springsteen play the super bowl\n"
            "A: <answer>Feb 1, 2009</answer>\n"  
            "**YOUR TURN:**\n"
        )

    # ================= 2. TriviaQA =================
    # 特点：常识/冷门知识，答案通常是实体，有时会有别名。
    elif dataset_type == "triviaqa":
        return (
            "\n\n**TASK:** Answer the trivia question concisely.\n"
            "**FORMAT:** Output ONLY the answer string inside tags. The answer can be a Name, Number, Title, or a short Phrase. Do NOT write a full sentence.\n"
            
            "**STRATEGY:**\n"
            "1. **Categorization:** For 'What is X?' questions, output the specific category or type (e.g., 'Grape', 'Missile'). Do NOT define it.\n"
            "2. **Phrase Completion:** For 'It takes a village to...' type questions, output the completing phrase only.\n"
            "3. **Double Questions:** If the question asks two things (e.g., 'Which sign... and what is it named after?'), answer both separated by a hyphen (e.g., 'Libra - Scales').\n"
            "4. **Minimalist:** Remove articles (a, an, the) and extra context. Just the core entity.\n\n"
            
            "**EXAMPLES:**\n"
            "Q: Which nation was the first to ratify the United Nations charter in 1945?\n"
            "A: <answer>Nicaragua</answer>\n"
            "Q: It takes a village to what?\n"
            "A: <answer>raise a child</answer>\n"
            "Q: What is a Thompson seedless?\n"
            "A: <answer>Grape</answer>\n"  # 新增例子：解决定义题
            "Q: Hellfire, Stinger, and Tomahawk are all types of what?\n"
            "A: <answer>Missiles</answer>\n" # 新增例子：解决分类题
            "Q: What is the only sign of the zodiac not named after a living creature, and what is it named after?\n"
            "A: <answer>Libra - Scales</answer>\n" # 新增例子：解决双重问题
            "**YOUR TURN:**\n"
        )

    # ================= 3. HotpotQA =================
    # 特点：多跳推理，答案通常是实体或 Yes/No。
    elif dataset_type == "hotpotqa":
        return (
            "\n\n**TASK:** Answer the multi-hop reasoning question based on the evidence.\n"
            "**FORMAT:** Output ONLY the final answer string inside tags. Do NOT write a full sentence. Just the Entity, Date, Number, or Yes/No.\n"
            
            "**STRATEGY:**\n"
            "1. **Minimalist:** Remove all context words. (e.g., instead of 'The answer is 1998', just output '1998').\n"
            "2. **Comparison:** For 'Which A or B?' questions, output the exact name of the winner. Do NOT explain why.\n"
            "3. **Guessing:** If the exact answer is not found, make an educated guess based on the context. Do NOT output 'Not found' or 'Error'.\n"
            "4. **Numbers:** Use Arabic numerals for counts (e.g., '7' instead of 'seven'), unless it's a title.\n\n"
            "**EXAMPLES:**\n"
            "Q: Are both Dictyosperma, and Huernia described as a genus?\n"
            "A: <answer>yes</answer>\n"
            "Q: Which dog's ancestors include Gordon and Irish Setters: the Manchester Terrier or the Scotch Collie?\n"
            "A: <answer>Scotch Collie</answer>\n"
            "Q: The Dukes of Hazzard was inspired by the 1975 film starring whom?\n"
            "A: <answer>James Mitchum</answer>\n"
            "Q: In which year was the King who made the 1925 Birthday Honours born?\n"
            "A: <answer>1865</answer>\n"
            "Q: Over how many centuries were the 'dwelling place of the dead' built?\n"
            "A: <answer>3 centuries</answer>\n" # 改成数字+单位
            "Q: Brahmarshi Vishwamitra is a 1991 film produced & directed by a man that served as Chief Minister for how many years?\n"
            "A: <answer>7</answer>\n" # 新增数字例子
            "**YOUR TURN:**\n"
        )

    # ================= 4. 2WikiMultiHopQA =================
    # 特点：结构化多跳，涉及两个实体关系，答案通常是人名、地名、电影名。
    elif dataset_type == "2wikimultihopqa":
        return (
            "\n\n**TASK:** Answer the complex multi-hop question.\n"
            "**FORMAT:** Output ONLY the specific entity (person, place, film, etc.) inside tags.\n"
            
            "**FALLBACK STRATEGIES:**\n"
            "1. If the exact answer is not found, make an educated guess based on the entity's nationality or family name, but prioritize retrieved evidence.\n"
            "2. For comparison questions (e.g., who is older, which is earlier), you MUST output one of the two options provided in the question. Do NOT output 'Unknown'.\n"
            "3. For Yes/No questions (e.g., are they from the same country?), if evidence is missing, guess 'no' (or 'yes'). Do NOT output 'Unknown'.\n\n"
            
            "**EXAMPLES:**\n"
            # 普通多跳实体题
            "Q: What is the place of birth of Paulding Farnham's wife?\n"
            "A: <answer>Ogdensburg, New York</answer>\n"
            
            # 比较题 
            "Q: Which film came out earlier, Aanaval Mothiram or Heroes Of The Street?\n"
            "A: <answer>Heroes of the Street</answer>\n"
            
            # Yes/No 题
            "Q: Are both directors of films The Slayer and Moscow Heat from the same country?\n"
            "A: <answer>yes</answer>\n"
            
            "**YOUR TURN:**\n"
        )

    # ================= 5. FEVER =================
    # 特点：事实验证，三分类 (SUPPORTS, REFUTES, NOT ENOUGH INFO)。
    # 注意：FlashRAG 的评估脚本通常期望全大写或特定格式，
    elif dataset_type == "fever":
        return (
            "\n\n**TASK:** Verify whether the retrieved evidence supports or refutes the user's claim.\n"
            "**FORMAT:** Output exactly one of the following labels inside tags: <answer>SUPPORTS</answer>, <answer>REFUTES</answer>, or <answer>NOT ENOUGH INFO</answer>.\n"
            
            "**STRATEGY:**\n"
            "1. **SUPPORTS:** The evidence confirms the claim is true, or the claim is a reasonable inference from the evidence.\n"
            "2. **REFUTES:** The evidence contradicts the claim. Look for mismatches in:\n"
            "   - **Dates/Numbers:** (e.g., Claim says '1999', Evidence says '2001').\n"
            "   - **Entity Types:** (e.g., Claim says 'Movie', Evidence says 'Book').\n"
            "   - **Negation:** (e.g., Claim says 'did not', Evidence says 'did').\n"
            "3. **Avoid NEI:** Only select 'NOT ENOUGH INFO' if the evidence is completely unrelated or missing. If you can infer the answer, do so.\n\n"
            
            "**EXAMPLES:**\n"
            "Claim: Scream 2 is a slasher film.\n"
            "A: <answer>SUPPORTS</answer>\n"
            "Claim: Magic Johnson did not play for the Lakers.\n"
            "A: <answer>REFUTES</answer>\n"
            "Claim: Mohra is a truck.\n"
            "A: <answer>REFUTES</answer>\n"
            "Claim: Ann Richards was the Governor of Texas for one year.\n"
            "A: <answer>REFUTES</answer>\n" # 新增：时间错误例子
            "Claim: The Hunger Games is a thought process.\n"
            "A: <answer>REFUTES</answer>\n" # 新增：类型错误例子
            "**YOUR TURN:**\n"
        )

    # ================= 6. StrategyQA =================
    # 特点：隐式推理，二分类 (yes/no)。
    elif dataset_type == "strategyqa":
        return (
            "\n\n**TASK:** Answer the question with Yes or No based on implicit multi-step reasoning.\n"
            "**FORMAT:** You MUST output the answer inside tags: <answer>yes</answer> or <answer>no</answer>.\n"
            
            "**STRATEGY:**\n"
            "1. **Step-by-Step:** Break down the question. If A implies B, and B implies C, then A implies C.\n"
            "2. **World Knowledge:** Use common sense for physical properties (e.g., size, weight, material) if not explicitly retrieved.\n"
            "3. **Math/Logic:** For questions involving numbers, time, or capacity, perform a rough calculation before answering.\n"
            "4. **Negation:** Pay close attention to words like 'not', 'unlikely', 'impossible'. Double negation means positive.\n\n"
            
            "**EXAMPLES:**\n"
            "Q: Can you buy Casio products at Petco?\n"
            "A: <answer>no</answer>\n"
            "Q: Would a student of the class of 2017 have amnesia about 9/11?\n"
            "A: <answer>yes</answer>\n"
            "Q: Can the city of Miami fit inside Uppsala?\n"
            "A: <answer>no</answer>\n"
            "Q: Is 500GB enough to save 10 hours of Netflix 4K shows?\n"
            "A: <answer>no</answer>\n" # 新增：数学计算例子
            "Q: Is it impossible to lick your elbow?\n"
            "A: <answer>yes</answer>\n" # 新增：否定/常识例子
            "**YOUR TURN:**\n"
        )

    # ================= Default =================
    else:
        return (
            "\n\n**TASK:** Answer the question concisely.\n"
            "**FORMAT:** Output ONLY the entity or short phrase inside tags. Example: <answer>Barack Obama</answer>."
        )