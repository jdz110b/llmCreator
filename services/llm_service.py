"""大模型服务：支持 OpenAI 兼容 API 的灵活接入"""
import json
import logging
import re
import time
import requests

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self, api_url, api_key, model, proxy=None, verify_ssl=True):
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.proxy = proxy
        self.verify_ssl = verify_ssl
        
        # 构造 requests 会话
        self.session = requests.Session()
        if proxy:
            # 兼容 http 和 https 代理
            if proxy.startswith('http://') or proxy.startswith('https://'):
                self.session.proxies = {
                    'http': proxy,
                    'https': proxy
                }
            else:
                # 如果没有协议前缀，自动加上 http://
                proxy_url = f'http://{proxy}'
                self.session.proxies = {
                    'http': proxy_url,
                    'https': proxy_url
                }
        self.session.verify = verify_ssl

    def chat(self, system_prompt, user_prompt, temperature=0.3, max_tokens=2000):
        """
        调用大模型 Chat Completion API（兼容 OpenAI 格式）
        """
        # 确保 URL 以 /chat/completions 结尾
        url = self.api_url
        if not url.endswith('/chat/completions'):
            url = url.rstrip('/') + '/chat/completions'

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
        }

        payload = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            'temperature': temperature,
            'max_tokens': max_tokens,
        }

        # 构造用于日志的请求摘要（隐藏 API Key，截断过长的 prompt）
        def _log_payload():
            user_preview = user_prompt[:300] + '...' if len(user_prompt) > 300 else user_prompt
            return (f"URL={url}, 模型={self.model}, temperature={temperature}, "
                    f"max_tokens={max_tokens}, prompt(前300字符)={user_preview}")

        logger.info(f"[LLM请求] {_log_payload()}")
        start_time = time.time()

        try:
            resp = self.session.post(url, headers=headers, json=payload, timeout=120)
        except requests.exceptions.Timeout:
            elapsed = time.time() - start_time
            logger.error(f"[LLM超时] 请求超时({elapsed:.1f}s)，{_log_payload()}")
            raise ValueError(
                f"大模型 API 请求超时(>{elapsed:.0f}s)，模型: {self.model}，"
                f"请检查网络连接或尝试减少输入长度"
            )
        except requests.exceptions.ConnectionError as e:
            logger.error(f"[LLM连接失败] {str(e)}，{_log_payload()}")
            raise ValueError(
                f"无法连接到大模型 API ({url})，模型: {self.model}，错误: {str(e)}"
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"[LLM请求异常] {type(e).__name__}: {str(e)}，{_log_payload()}")
            raise ValueError(
                f"大模型 API 请求失败，模型: {self.model}，"
                f"错误: {type(e).__name__}: {str(e)}"
            )

        elapsed = time.time() - start_time
        logger.info(f"[LLM响应] HTTP {resp.status_code}, 耗时={elapsed:.1f}s, 模型={self.model}")

        # HTTP 错误状态码
        if resp.status_code != 200:
            body_preview = resp.text[:500] if resp.text else '(空)'
            logger.error(f"[LLM HTTP错误] 状态码={resp.status_code}, 响应={body_preview}，{_log_payload()}")
            raise ValueError(
                f"大模型 API 返回 HTTP {resp.status_code}，模型: {self.model}，响应: {body_preview}"
            )

        # 防御性处理：API 可能返回空响应体
        raw_text = resp.text.strip()
        if not raw_text:
            logger.error(f"[LLM空响应] HTTP {resp.status_code}, 响应体为空，{_log_payload()}")
            raise ValueError(
                f"大模型 API 返回空响应体 (HTTP {resp.status_code})，"
                f"模型: {self.model}，请检查 API 地址和模型名称是否正确"
            )

        try:
            data = resp.json()
        except (json.JSONDecodeError, ValueError) as e:
            preview = raw_text[:500] if len(raw_text) > 500 else raw_text
            logger.error(f"[LLM JSON解析失败] 响应={preview}，{_log_payload()}")
            raise ValueError(
                f"大模型 API 返回非 JSON 内容 (HTTP {resp.status_code})，"
                f"模型: {self.model}，响应内容: {preview}"
            ) from e

        # 检查 API 错误响应
        if 'error' in data and 'choices' not in data:
            err_msg = data['error']
            if isinstance(err_msg, dict):
                err_msg = err_msg.get('message', str(err_msg))
            logger.error(f"[LLM API错误] {err_msg}，{_log_payload()}")
            raise ValueError(f"大模型 API 返回错误: {err_msg}")

        if not data.get('choices'):
            resp_preview = json.dumps(data, ensure_ascii=False)[:500]
            logger.error(f"[LLM无choices] 响应={resp_preview}，{_log_payload()}")
            raise ValueError(
                f"大模型 API 返回中缺少 choices 字段，"
                f"模型: {self.model}，响应: {resp_preview}"
            )

        message = data['choices'][0]['message']
        content = message.get('content') or ''

        # DeepSeek 等模型可能将内容放在 reasoning_content 字段，content 为 None
        if not content.strip():
            reasoning = message.get('reasoning_content') or ''
            if reasoning.strip():
                content = reasoning
                logger.info(f"[LLM] content 为空，使用 reasoning_content (长度={len(reasoning)})")

        # 检查是否因触发内容过滤而返回空
        if not content.strip():
            finish_reason = data['choices'][0].get('finish_reason', '')
            logger.error(
                f"[LLM空内容] finish_reason={finish_reason}, "
                f"message字段={list(message.keys())}，{_log_payload()}"
            )
            raise ValueError(
                f"大模型返回了空的回复内容，模型: {self.model}，"
                f"finish_reason: {finish_reason}，"
                f"可能原因：内容被安全过滤、max_tokens 不足、或模型名称不正确"
            )

        logger.info(f"[LLM成功] 模型={self.model}, 耗时={elapsed:.1f}s, 回复长度={len(content)}")
        return content

    def chat_json(self, system_prompt, user_prompt, temperature=0.1, max_tokens=2000):
        """
        调用大模型并期望返回 JSON 格式。
        兼容 qwen3 等模型返回 <think> 标签、markdown 代码块等情况。
        """
        result = self.chat(system_prompt, user_prompt, temperature, max_tokens)
        return self._extract_json(result)

    @staticmethod
    def _extract_json(text):
        """从大模型的原始输出中稳健地提取 JSON 对象。"""
        if not text or not text.strip():
            raise ValueError("大模型返回了空内容，无法解析 JSON")

        # 1. 去除 <think>...</think> 标签（qwen3 等模型的推理过程）
        text = re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.DOTALL)
        text = text.strip()

        if not text:
            raise ValueError("大模型返回内容仅含推理过程（<think>标签），没有实际 JSON 输出")

        # 2. 处理 markdown 代码块 ```json ... ``` 或 ``` ... ```
        md_match = re.search(r'```(?:json)?\s*\n([\s\S]*?)\n\s*```', text)
        if md_match:
            text = md_match.group(1).strip()

        # 3. 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 4. 尝试提取第一个 JSON 对象 { ... }
        brace_start = text.find('{')
        brace_end = text.rfind('}')
        if brace_start != -1 and brace_end > brace_start:
            candidate = text[brace_start:brace_end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        # 5. 尝试提取 JSON 数组 [ ... ]
        bracket_start = text.find('[')
        bracket_end = text.rfind(']')
        if bracket_start != -1 and bracket_end > bracket_start:
            candidate = text[bracket_start:bracket_end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        # 6. 都失败，抛出详细错误
        preview = text[:200] if len(text) > 200 else text
        raise ValueError(f"无法从大模型返回内容中提取 JSON。前200字符: {preview}")
