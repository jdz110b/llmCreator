"""大模型服务：支持 OpenAI 兼容 API 的灵活接入"""
import json
import re
import requests


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

        resp = self.session.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()

        # 防御性处理：API 可能返回空响应体
        raw_text = resp.text.strip()
        if not raw_text:
            raise ValueError(
                f"大模型 API 返回空响应体 (HTTP {resp.status_code})，"
                f"模型: {self.model}，请检查 API 地址和模型名称是否正确"
            )

        try:
            data = resp.json()
        except (json.JSONDecodeError, ValueError) as e:
            preview = raw_text[:500] if len(raw_text) > 500 else raw_text
            raise ValueError(
                f"大模型 API 返回非 JSON 内容 (HTTP {resp.status_code})，"
                f"模型: {self.model}，响应内容: {preview}"
            ) from e

        # 检查 API 错误响应
        if 'error' in data and 'choices' not in data:
            err_msg = data['error']
            if isinstance(err_msg, dict):
                err_msg = err_msg.get('message', str(err_msg))
            raise ValueError(f"大模型 API 返回错误: {err_msg}")

        if not data.get('choices'):
            raise ValueError(
                f"大模型 API 返回中缺少 choices 字段，"
                f"模型: {self.model}，响应: {json.dumps(data, ensure_ascii=False)[:500]}"
            )

        content = data['choices'][0]['message']['content']
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
