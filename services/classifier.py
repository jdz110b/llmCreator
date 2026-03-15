"""语料分类服务"""
from services.llm_service import LLMService


# ========== 默认 Prompt 模板 ==========

DEFAULT_CLASSIFY_SUBJ_OBJ_PROMPT = """你是一个专业的语料分类专家。请判断以下问题是"客观题"还是"主观题"。

判断标准：
- 客观题：有明确、唯一的标准答案，如事实性问题、数学计算、选择题等
- 主观题：答案因人而异，需要个人观点、分析或创造性回答

请以 JSON 格式返回结果：
{
    "type": "objective" 或 "subjective",
    "reason": "简要说明判断理由"
}

问题：{question}"""

DEFAULT_CLASSIFY_DIFFICULTY_PROMPT = """你是一个专业的语料难度评估专家。请评估以下问题的难度等级。

难度等级标准：
- L1: 基础知识，大多数人可以直接回答
- L2: 需要一定专业知识或思考才能回答
- L3: 需要深度专业知识、复杂推理或多步骤思考

请以 JSON 格式返回结果：
{
    "difficulty": "L1" 或 "L2" 或 "L3",
    "reason": "简要说明判断理由"
}

问题：{question}"""

DEFAULT_CLASSIFY_CATEGORY_PROMPT = """你是一个专业的语料分类专家。请将以下问题归类到给定的分类中。

可选分类：{categories}

请以 JSON 格式返回结果：
{
    "category": "所属分类",
    "reason": "简要说明分类理由"
}

问题：{question}"""

DEFAULT_QUALITY_EVAL_PROMPT = """你是一个专业的语料质量评估专家。请从多个维度评估以下问题的语料质量。

评估维度：
- clarity（清晰度）：问题表述是否清晰明确、无歧义，1-10分
- completeness（完整性）：问题信息是否完整，能否被正确理解和回答，1-10分
- normalization（规范性）：语法是否正确、用词是否规范、格式是否标准，1-10分
- value（价值性）：问题是否有实际价值、是否值得收录到语料库，1-10分

请以 JSON 格式返回结果：
{
    "quality_score": 综合质量评分(1-10的数字),
    "clarity": 清晰度分数(1-10),
    "completeness": 完整性分数(1-10),
    "normalization": 规范性分数(1-10),
    "value": 价值性分数(1-10),
    "quality_label": "high" 或 "medium" 或 "low",
    "suggestion": "改进建议（如有）"
}

问题：{question}"""

DEFAULT_DOMAIN_CLASSIFY_PROMPT = """你是一个专业的语料领域分类专家。请将以下问题归类到最合适的领域/场景分类中。

可选领域分类：
- 技术/编程：软件开发、编程语言、算法、IT运维等
- 科学/学术：数学、物理、化学、生物、天文等自然科学及学术研究
- 教育/学习：教学方法、考试备考、学习技巧、课程相关
- 医疗/健康：疾病诊断、药物、健康养生、心理健康
- 法律/法规：法律条文、法规解读、法律咨询、合规问题
- 金融/商业：投资理财、财务会计、商业模式、经济分析
- 生活/日常：衣食住行、家居生活、旅游出行、购物消费
- 文化/娱乐：文学、艺术、音乐、影视、游戏、体育
- 社会/时政：社会热点、政策解读、国际关系、历史事件
- 其他：不属于以上分类的内容

请以 JSON 格式返回结果：
{
    "domain": "所属领域分类",
    "sub_domain": "更细分的子领域（可选）",
    "reason": "简要说明分类理由"
}

问题：{question}"""

DEFAULT_INTENT_CLASSIFY_PROMPT = """你是一个专业的用户意图识别专家。请分析以下问题的用户意图类型。

意图分类：
- internal_information_query（内部信息查询）：用户想了解的事实、概念或知识点属于大语言模型训练数据中已涵盖的通用知识，模型本身即可直接答复
- external_information_query（外部信息查询）：用户想了解的信息需要结合外部知识库、实时数据、私有数据或专有系统才能准确答复，模型自身训练数据不足以回答
- task_execution（任务执行）：用户需要完成某个具体操作或任务，如写代码、翻译、计算等
- opinion_consultation（观点咨询）：用户希望获得建议、推荐或专业意见
- creative_generation（创意生成）：用户需要生成创意内容，如写作、起名、设计方案等
- chat_conversation（闲聊对话）：日常寒暄、情感交流、无明确目的的对话
- reasoning_analysis（推理分析）：需要逻辑推理、数据分析、对比评估等深度思考
- instruction_following（指令遵循）：按特定格式或规则输出，如角色扮演、格式转换等

判断"内部"与"外部"信息查询的参考标准：
- 内部：常识性知识、公开的百科知识、通用学科知识、广泛已知的历史事件等
- 外部：涉及特定组织/企业的内部信息、实时行情/天气/新闻、需要查数据库或API的问题、私有业务数据等

请以 JSON 格式返回结果：
{
    "intent": "意图类型的英文标识",
    "intent_cn": "意图类型的中文名称",
    "confidence": 置信度(0.0-1.0),
    "reason": "简要说明判断理由"
}

问题：{question}"""

DEFAULT_GENERATE_ANSWER_PROMPT = """你是一个专业的题目解答专家。请为以下客观题提供准确的标准答案。

要求：
1. 答案必须准确、简洁
2. 如有必要可附带简短解释

请以 JSON 格式返回结果：
{
    "answer": "标准答案",
    "explanation": "简要解释（可选）"
}

问题：{question}"""


SYSTEM_PROMPT = "你是一个专业的语料评测助手，负责对问题进行分类、评估和解答。请严格按照要求的 JSON 格式输出结果。"


DEFAULT_COMBINED_CLASSIFY_PROMPT = """你是一个专业的语料评测专家。请对以下问题进行全方位的分析和分类。

请从以下维度进行评估：

1. 主观/客观判断：
   - objective（客观题）：有明确、唯一的标准答案
   - subjective（主观题）：答案因人而异

2. 难度等级：
   - L1: 基础知识，大多数人可以直接回答
   - L2: 需要一定专业知识或思考才能回答
   - L3: 需要深度专业知识、复杂推理或多步骤思考

3. 语料质量（1-10分）：
   - clarity（清晰度）、completeness（完整性）、normalization（规范性）、value（价值性）

4. 领域分类：技术/编程、科学/学术、教育/学习、医疗/健康、法律/法规、金融/商业、生活/日常、文化/娱乐、社会/时政、其他

5. 意图识别：internal_information_query（内部信息查询，模型自身知识可答复）、external_information_query（外部信息查询，需结合外部知识/实时数据/私有数据）、task_execution（任务执行）、opinion_consultation（观点咨询）、creative_generation（创意生成）、chat_conversation（闲聊对话）、reasoning_analysis（推理分析）、instruction_following（指令遵循）

6. 如果是客观题，请同时提供标准答案。

{extra_instructions}

请以 JSON 格式返回结果：
{
    "subj_obj": "objective" 或 "subjective",
    "difficulty": "L1" 或 "L2" 或 "L3",
    "quality_score": 综合质量评分(1-10),
    "quality_label": "high" 或 "medium" 或 "low",
    "domain": "所属领域",
    "sub_domain": "子领域",
    "intent": "意图英文标识",
    "intent_cn": "意图中文名称",
    "intent_confidence": 置信度(0.0-1.0),
    "objective_answer": "客观题的标准答案（主观题留空字符串）",
    "category": "自定义分类结果（无自定义分类时留空字符串）"
}

问题：{question}"""


def classify_combined(llm: LLMService, question: str, categories: str = '', custom_prompt: str = None):
    """一次调用完成所有维度的分类"""
    prompt = custom_prompt or DEFAULT_COMBINED_CLASSIFY_PROMPT

    extra = ''
    if categories:
        extra = f'7. 自定义分类：请将问题归类到以下类别之一：{categories}'

    user_prompt = prompt.replace('{question}', question).replace('{extra_instructions}', extra)
    result = llm.chat_json(SYSTEM_PROMPT, user_prompt)
    return result


def classify_subjective_objective(llm: LLMService, question: str, custom_prompt: str = None):
    """判断问题是主观题还是客观题"""
    prompt = custom_prompt or DEFAULT_CLASSIFY_SUBJ_OBJ_PROMPT
    user_prompt = prompt.replace('{question}', question)
    result = llm.chat_json(SYSTEM_PROMPT, user_prompt)
    return result


def classify_difficulty(llm: LLMService, question: str, custom_prompt: str = None):
    """评估问题难度"""
    prompt = custom_prompt or DEFAULT_CLASSIFY_DIFFICULTY_PROMPT
    user_prompt = prompt.replace('{question}', question)
    result = llm.chat_json(SYSTEM_PROMPT, user_prompt)
    return result


def classify_category(llm: LLMService, question: str, categories: str, custom_prompt: str = None):
    """自定义分类"""
    prompt = custom_prompt or DEFAULT_CLASSIFY_CATEGORY_PROMPT
    user_prompt = prompt.replace('{question}', question).replace('{categories}', categories)
    result = llm.chat_json(SYSTEM_PROMPT, user_prompt)
    return result


def generate_objective_answer(llm: LLMService, question: str, custom_prompt: str = None):
    """为客观题生成标准答案"""
    prompt = custom_prompt or DEFAULT_GENERATE_ANSWER_PROMPT
    user_prompt = prompt.replace('{question}', question)
    result = llm.chat_json(SYSTEM_PROMPT, user_prompt)
    return result


def evaluate_quality(llm: LLMService, question: str, custom_prompt: str = None):
    """评估语料质量"""
    prompt = custom_prompt or DEFAULT_QUALITY_EVAL_PROMPT
    user_prompt = prompt.replace('{question}', question)
    result = llm.chat_json(SYSTEM_PROMPT, user_prompt)
    return result


def classify_domain(llm: LLMService, question: str, custom_prompt: str = None):
    """领域/场景分类"""
    prompt = custom_prompt or DEFAULT_DOMAIN_CLASSIFY_PROMPT
    user_prompt = prompt.replace('{question}', question)
    result = llm.chat_json(SYSTEM_PROMPT, user_prompt)
    return result


def classify_intent(llm: LLMService, question: str, custom_prompt: str = None):
    """意图识别分类"""
    prompt = custom_prompt or DEFAULT_INTENT_CLASSIFY_PROMPT
    user_prompt = prompt.replace('{question}', question)
    result = llm.chat_json(SYSTEM_PROMPT, user_prompt)
    return result
