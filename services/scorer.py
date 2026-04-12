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


def clean_answer_v2(answer: str) -> str:
    """增强版答案清洗：移除各类链接、图片、视频、多媒体标签"""
    if not answer:
        return ""
    # 1. Markdown 图片 ![alt](url)
    answer = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', answer)
    # 2. Markdown 链接 [text](url) -> 保留 text
    answer = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', answer)
    # 3. HTML img 标签
    answer = re.sub(r'<img[^>]*/?>', '', answer, flags=re.IGNORECASE)
    # 4. HTML video 标签（含内容）
    answer = re.sub(r'<video[^>]*>.*?</video>', '', answer, flags=re.IGNORECASE | re.DOTALL)
    # 5. HTML audio 标签（含内容）
    answer = re.sub(r'<audio[^>]*>.*?</audio>', '', answer, flags=re.IGNORECASE | re.DOTALL)
    # 6. HTML iframe 标签（含内容）
    answer = re.sub(r'<iframe[^>]*>.*?</iframe>', '', answer, flags=re.IGNORECASE | re.DOTALL)
    # 7. HTML a 标签 -> 保留链接文字
    answer = re.sub(r'<a[^>]*>(.*?)</a>', r'\1', answer, flags=re.IGNORECASE | re.DOTALL)
    # 8. HTTP/HTTPS URL
    answer = re.sub(r'https?://\S+', '', answer)
    # 9. superlink:// 协议链接
    answer = re.sub(r'superlink://\S+', '', answer, flags=re.IGNORECASE)
    # 10. rasp:// 协议链接
    answer = re.sub(r'rasp://\S+', '', answer, flags=re.IGNORECASE)
    # 11. 其他自定义协议链接兜底 (如 ftp://, custom://)
    answer = re.sub(r'[a-zA-Z][a-zA-Z0-9+.\-]*://\S+', '', answer)
    # 12. 多余空白归一化
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
