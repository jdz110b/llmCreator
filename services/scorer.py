"""Answer 打分服务"""
import re
from services.llm_service import LLMService


def clean_answer(answer: str) -> str:
    """清理答案文本：移除 URL 和 RASP 格式内容"""
    if not answer:
        return ""
    # 移除 http/https 开头的 URL
    answer = re.sub(r'https?://\S+', '', answer)
    # 移除类似 rasp:// 开头的内容
    answer = re.sub(r'rasp://\S+', '', answer)
    # 移除多余的空白字符
    answer = re.sub(r'\s+', ' ', answer).strip()
    return answer


DEFAULT_SCORE_PROMPT = """你是一个专业的答案评估专家。请对以下问答对中的答案进行评分。

评分标准（1-10分）：
- 准确性（40%）：答案是否正确、事实是否准确
- 完整性（30%）：答案是否全面，是否覆盖了问题的关键点
- 清晰度（20%）：答案是否表达清晰、逻辑连贯
- 相关性（10%）：答案是否与问题高度相关

请以 JSON 格式返回结果：
{
    "score": 评分(1-10的数字),
    "accuracy": 准确性分数(1-10),
    "completeness": 完整性分数(1-10),
    "clarity": 清晰度分数(1-10),
    "relevance": 相关性分数(1-10),
    "reason": "综合评价说明"
}

问题：{question}
答案：{answer}"""

SYSTEM_PROMPT = "你是一个专业的语料评测助手，负责对答案进行客观、公正的评分。请严格按照要求的 JSON 格式输出结果。"


def score_answer(llm: LLMService, question: str, answer: str, custom_prompt: str = None):
    """对 QA 对中的 Answer 进行打分"""
    # 预处理答案：移除 URL 和 RASP 格式内容
    clean_ans = clean_answer(answer)
    prompt = custom_prompt or DEFAULT_SCORE_PROMPT
    user_prompt = prompt.replace('{question}', question).replace('{answer}', clean_ans)
    result = llm.chat_json(SYSTEM_PROMPT, user_prompt)
    return result
