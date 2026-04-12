"""答案相似度对比服务"""
from services.llm_service import LLMService


SYSTEM_PROMPT = "你是一个专业的文本语义对比专家，负责对比两段答案的核心观点相似程度。请严格按照要求的 JSON 格式输出结果。"

DEFAULT_SIMILARITY_COMPARE_PROMPT = """你是一个专业的语义对比专家。请对比以下同一问题下的两个答案，评估它们核心观点的相似程度。

评分标准（0-100分）：
- 核心观点覆盖度（40%）：两个答案的核心论点/结论是否一致
- 关键事实一致性（30%）：涉及的具体数据、事实、步骤是否相同
- 推理逻辑相似度（20%）：论证过程和逻辑链是否相似
- 表述角度差异（10%）：是否仅为措辞/顺序不同，核心含义相同

评分范围：
- 80-100：核心观点高度一致（high）
- 40-79：部分观点一致，存在明显差异（medium）
- 0-39：核心观点差异较大（low）

请以 JSON 格式返回结果：
{
    "similarity_score": 相似度分数(0-100的整数),
    "similarity_label": "high" 或 "medium" 或 "low",
    "key_differences": "关键差异点（简要列出1-3条主要差异）",
    "reason": "综合评判依据"
}

问题：{question}
答案1：{answer1}
答案2：{answer2}"""


def compare_similarity(llm: LLMService, question: str, answer1: str, answer2: str, custom_prompt: str = None):
    """对比两个答案的核心观点相似度"""
    prompt = custom_prompt or DEFAULT_SIMILARITY_COMPARE_PROMPT
    user_prompt = prompt.replace('{question}', question).replace('{answer1}', answer1).replace('{answer2}', answer2)
    result = llm.chat_json(SYSTEM_PROMPT, user_prompt)
    return result
