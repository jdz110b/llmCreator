"""大模型服务：支持 OpenAI 兼容 API 的灵活接入"""
import json
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
        data = resp.json()

        content = data['choices'][0]['message']['content']
        return content

    def chat_json(self, system_prompt, user_prompt, temperature=0.1, max_tokens=2000):
        """
        调用大模型并期望返回 JSON 格式
        """
        result = self.chat(system_prompt, user_prompt, temperature, max_tokens)
        # 尝试从返回中提取 JSON
        result = result.strip()
        # 处理 markdown 代码块包裹的情况
        if result.startswith('```'):
            lines = result.split('\n')
            # 去掉第一行和最后一行
            json_lines = []
            in_block = False
            for line in lines:
                if line.startswith('```') and not in_block:
                    in_block = True
                    continue
                elif line.startswith('```') and in_block:
                    break
                elif in_block:
                    json_lines.append(line)
            result = '\n'.join(json_lines)

        return json.loads(result)
