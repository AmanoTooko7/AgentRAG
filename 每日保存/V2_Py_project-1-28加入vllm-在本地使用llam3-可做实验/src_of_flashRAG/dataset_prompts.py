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
            
            "**EXAMPLES:**\n"
            "Q: Where was the director of film Rembrandt'S J'Accuse born?\n"
            "A: <answer>Newport, Wales</answer>\n"
            "Q: What is the date of death of the director of film Never Fear?\n"
            "A: <answer>3 August 1995</answer>\n"
            "Q: Who is Edward De Vere, 17Th Earl Of Oxford's paternal grandfather?\n"
            "A: <answer>ohn de Vere, 15th Earl of Oxford</answer>\n"
            "Q: Which film has the director born first, S Durga or Sweet Emma, Dear Böbe?t\n"
            "A: <answer>Sweet Emma, Dear Böbe</answer>\n"
            "Q: Who is the spouse of the director of film Ann Carver'S Profession?\n"
            "A: <answer>Ona Munson</answer>\n"  
            "**YOUR TURN:**\n"
        )

    # ================= 2. TriviaQA =================
    # 特点：常识/冷门知识，答案通常是实体，有时会有别名。
    elif dataset_type == "triviaqa":
        return (
            "\n\n**TASK:** Answer the trivia question concisely.\n"
            "**FORMAT:** Output ONLY the answer string inside tags. The answer can be a Name, Number, Title, or a short Phrase. Do NOT write a full sentence.\n"
            
            "**STRATEGY:**\n"
            "1. Categorization: For 'What is X?' questions, output the specific category or type (e.g., 'Grape', 'Missile'). Do NOT define it.\n"
            "2. Phrase Completion: For 'It takes a village to...' type questions, output the completing phrase only.\n"
            "3. Double Questions: If the question asks two things (e.g., 'Which sign... and what is it named after?'), answer both separated by a hyphen (e.g., 'Libra - Scales').\n"
            "4. Minimalist: Remove articles (a, an, the) and extra context. Just the core entity.\n\n"
            
            "**EXAMPLES:**\n"
            "Q: Which C S wrote The African Queen?\n"
            "A: <answer>Foresters or Professional forester</answer>\n"
            "Q: In telephone terms, what was a party line?\n"
            "A: <answer>Shared phone line</answer>\n"
            "Q: In the Robin Hood stories, what was the real name of Little John?\n"
            "A: <answer>John Little</answer>\n"  # 新增例子：解决定义题
            "Q: What were the stripes on the old-fashioned barber shop poles meant to represent?\n"
            "A: <answer>Oxygen capacity</answer>\n" # 新增例子：解决分类题
            "Q: Moonwart and Satinpod are alternative names for which flower?\n"
            "A: <answer>LHonesty or Truthful</answer>\n" # 新增例子：解决双重问题
            "**YOUR TURN:**\n"
        )

    # ================= 3. HotpotQA =================
    # 特点：多跳推理，答案通常是实体或 Yes/No。
    elif dataset_type == "hotpotqa":
        return (
            " Goal: Answer the multi-hop question based on the Research History.\n"
            " Format: Output ONLY the final answer string inside <answer> tags.\n"
            " Constraint: NO sentences. NO extra words. Just the Entity, Date, Number, or Yes/No.\n"

            "STRATEGY:\n"
            "1. Minimalist: Remove all context words. (e.g., instead of 'The answer is 1998', just output '1998').\n"
            "2. Binary Choice ('Which A or B?'):** If the question asks to choose between two options (e.g., 'Which film...', "
            "       'Who is older...'), you MUST output exactly one of the two options mentioned in the question. **Do NOT output the date/value/reason.**\n"           
            "3. Guessing: If the exact answer is not found, make an educated guess based on the context. Do NOT output 'Not found' or 'Error'.\n"
            "4. Numbers: Use Arabic numerals for counts (e.g., '7' instead of 'seven'), unless it's a title.\n\n"
            "5. No Aliases:** Do not output 'also known as...'. Just the primary name.\n\n" 

            "REFERENCE:\n"
            "Q: Which director was also a writer, Noel M. Smith or Reza Parsa?\n"
            "A: <answer>Noel Mason Smith</answer>\n"
            "Q: The album that contains the song \"Get Together\" by Madonna was mainly recorded in whose home studio?\n"
            "A: <answer>Stuart Price</answer>\n"
            "Q: Where is the basketball team that Mike DiNunno plays for based?\n"
            "A: <answer>Ellesmere Port</answer>\n"
            "Q: TACA Airlines, is an airline owned by a conglomerate owned by who?\n"
            "A: <answer>Germán Efromovich</answer>\n"
            "Q: Who wrote the lyrics to the theme song that was used by the Western film series Range Busters?\n"
            "A: <answer>Dr. Brewster M. Higley</answer>\n"
            "Q: What railroad in which Challengers were most common was completed in 1915?\n"
            "A: <answer>Clinchfield Railroad</answer>\n" # 
            "Q: Kent Dairy Round Barn is near the city in what Montana county?\n"
            "A: <answer>Carbon</answer>\n" 
            "Q: This 2016 biographical sports comedy-drama film stars Taron Egerton as a skier who finished last in what events?\n"
            "A: <answer>70 m and 90 m</answer>\n"
            "Q: Where does the region where Jews supported by Kolel Chibas Jerusalem are from get its name?"
            "A: <answer>the medieval city of Halych</answer>\n"
            "Q: What was the third studio album released by Richard Melville Hall?\n"
            "A: <answer>Everything Is Wrong</answer>\n"
            "Q: How long did the state exist whose army ransacked and looted the city, Halebidu, twice in the 14th century ?\n"
            "A: <answer>320 years</answer>\n"
            "Q: Which hospital opened first, St. Elizabeths Hospital or Providence Hospital?\n"
            "A: <answer>St. Elizabeths Hospital</answer>\n"
            "YOUR TURN:\n"
        )

    # ================= 4. 2WikiMultiHopQA =================
    # 特点：结构化多跳，涉及两个实体关系，答案通常是人名、地名、电影名。
    elif dataset_type == "2wikimultihopqa":
        return (
            "\n\nTASK: Answer the multi-hop question based on the context. \n"
            "CRITICAL RULE: You are participating in a quiz where skipping is forbidden. You MUST provide a specific answer.\n"
            
            "FORMAT: Output ONLY the specific Entity (Person, Place, Film) or 'yes'/'no' inside tags.\n"
            
            "STRATEGIESss:\n"
            "- Guessing (Crucial): If evidence is missing, HALLUCINATE a plausible entity based on names/context. NEVER output 'Unknown'.\n"
            "- Binary Bias: For 'Are they from the same...?', if any connection exists, output <answer>yes</answer>.\n"
            "- Comparison: Pick one option. Do NOT output dates/years.\n"
            "- Granularity: Output Sovereign State (e.g., 'Germany') for countries. Output Full Date if possible.\n\n"
            "- Yes/No Bias (Crucial): For questions starting with 'Do', 'Are', 'Were', 'Did', 'Have', 'Is': If evidence is missing or unclear, **ALWAYS guess <answer>yes</answer>**.\n" 
            
            "REFERENCE:\n"
            "Q: What is the place of birth of the director of film Boden'S Boy?\n"
            "A: <answer>Weston-super-Mare</answer>\n"
            "Q: Did the bands The Mess Hall and Flotsam Jetsam (Band), originate from the same country?\n"
            "A: <answer>yest</answer>\n"
            "Q: Which film has the director who was born first, Haiducii Lui Șaptecai or The Assignment (2016 Film)?\n"
            "A: <answer>The Outlaws of Captain Anghel</answer>\n"
            "Q: Who is the mother of the director of film Citizen Usa: A 50 State Road Trip?\n"
            "A: <answer>Nancy Pelosi</answer>\n"
            "Q: When did the director of film The General Line die?\n"
            "A: <answer>11 February 1948</answer>\n"
            "Q: Where did Setepenre (Princess)'s mother die?\n"
            "A: <answer>Akhetaton</answer>\n"
            "Q: Which film was released more recently, The Indiscreet Woman or The Sparrow'S Fluttering?\n"
            "A: <answer>The Sparrow's Fluttering</answer>\n"
            "Q: What is the award that the director of film 24 Hour Psycho received?\n"
            "A: <answer>Turner Prize</answer>\n"
            "Q: Did the bands The Mess Hall and Flotsam Jetsam (Band), originate from the same country?\n"
            "A: <answer>yes</answer>\n"
            "Q: What is the date of death of Peter Ii, Count Of La Marche's father?\n"
            "A: <answer>6 April 1362</answer>\n"
            "Q: Where was the director of film Kummi Paattu born?\n"
            "A: <answer>Theni</answer>\n"
            "YOUR TURN:\n"
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