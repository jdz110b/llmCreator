"""豆包网页版浏览器自动化服务"""
import time
import logging
from playwright.sync_api import sync_playwright

logger = logging.getLogger(__name__)


class DoubaoBrowser:
    """通过 Playwright 自动化控制浏览器与豆包网页版交互"""

    DOUBAO_URL = "https://www.doubao.com"

    def __init__(self, cookies_str):
        """
        初始化浏览器并注入 Cookies

        Args:
            cookies_str: 从浏览器开发者工具复制的 Cookie 字符串
                        格式如: "key1=value1; key2=value2; ..."
        """
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=True)
        cookies = self._parse_cookies(cookies_str)
        self._context = self._browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 800},
        )
        # 尝试批量添加 cookies，如果失败则逐个添加
        if cookies:
            try:
                self._context.add_cookies(cookies)
            except Exception as e:
                logger.warning(f"批量添加 Cookie 失败: {str(e)}，尝试逐个添加...")
                for cookie in cookies:
                    try:
                        self._context.add_cookies([cookie])
                    except Exception as ce:
                        logger.warning(f"跳过无效 Cookie: {cookie['name']} - {str(ce)}")
        self._page = self._context.new_page()

    @staticmethod
    def _parse_cookies(cookies_str):
        """解析 Cookie 字符串为 Playwright 需要的格式"""
        cookies = []
        if not cookies_str:
            return cookies

        # 清理前缀：去掉可能的 "cookie:" 或 "Cookie:" 前缀
        cleaned = cookies_str.strip()
        if cleaned.lower().startswith('cookie:'):
            cleaned = cleaned[7:].strip()
        elif cleaned.lower().startswith('cookie\n'):
            cleaned = cleaned[6:].strip()
        elif cleaned.lower().startswith('cookie '):
            cleaned = cleaned[6:].strip()

        for pair in cleaned.split(';'):
            pair = pair.strip()
            if '=' not in pair:
                continue
            name, value = pair.split('=', 1)
            name = name.strip()
            value = value.strip()

            # 跳过空名称的 cookie
            if not name:
                continue

            # 跳过名称中包含非法字符的 cookie（只允许 ASCII 可打印字符，不含空格、分号、等号等）
            if any(c in name for c in [' ', '\t', '\n', '\r', ';', ',']):
                logger.warning(f"跳过无效 Cookie（名称含非法字符）: {name[:20]}")
                continue

            cookies.append({
                'name': name,
                'value': value,
                'domain': '.doubao.com',
                'path': '/',
            })

        return cookies

    def query(self, question, timeout=120):
        """
        向豆包发送一个问题并获取回答

        Args:
            question: 要提问的问题文本
            timeout: 等待回答的最大超时时间（秒）

        Returns:
            str: 豆包的回答文本

        Raises:
            Exception: 查询失败时抛出异常
        """
        try:
            # 导航到豆包首页/聊天页面
            self._page.goto(self.DOUBAO_URL, wait_until="domcontentloaded", timeout=60000)

            # 等待页面关键元素加载（输入框出现说明页面已就绪）
            input_selectors = [
                'textarea[data-testid="chat_input"]',
                '#chat-input',
                'textarea.chat-input',
                'div[contenteditable="true"]',
                'textarea',
            ]

            input_el = None
            for selector in input_selectors:
                try:
                    el = self._page.locator(selector).first
                    if el.is_visible(timeout=15000):
                        input_el = el
                        break
                except Exception:
                    continue

            if not input_el:
                raise Exception("无法找到豆包输入框，请检查 Cookie 是否有效或页面是否正常加载")

            # 尝试点击"新建对话"按钮（如果存在），确保每次是新对话
            try:
                new_chat_btn = self._page.locator('[data-testid="new_chat"], button:has-text("新对话"), button:has-text("新建对话")')
                if new_chat_btn.count() > 0:
                    new_chat_btn.first.click()
                    time.sleep(1)
                    # 重新定位输入框
                    for selector in input_selectors:
                        try:
                            el = self._page.locator(selector).first
                            if el.is_visible(timeout=5000):
                                input_el = el
                                break
                        except Exception:
                            continue
            except Exception:
                pass

            # 清空并输入问题
            input_el.click()
            input_el.fill("")
            time.sleep(0.3)
            input_el.fill(question)
            time.sleep(0.5)

            # 发送问题（按 Enter 或点击发送按钮）
            send_selectors = [
                'button[data-testid="send_button"]',
                'button:has-text("发送")',
                'button.send-btn',
                'button[aria-label="发送"]',
            ]

            sent = False
            for selector in send_selectors:
                try:
                    btn = self._page.locator(selector).first
                    if btn.is_visible(timeout=2000) and btn.is_enabled():
                        btn.click()
                        sent = True
                        break
                except Exception:
                    continue

            if not sent:
                input_el.press("Enter")

            time.sleep(2)

            # 等待回答生成完成
            answer = self._wait_for_answer(timeout)
            return answer

        except Exception as e:
            logger.error(f"豆包查询失败: {str(e)}")
            # 错误恢复：尝试导航回首页，确保下一次查询可以正常进行
            try:
                self._page.goto(self.DOUBAO_URL, wait_until="domcontentloaded", timeout=30000)
            except Exception:
                pass
            raise

    def _wait_for_answer(self, timeout=120):
        """
        等待豆包回答生成完成并提取回答文本

        策略：监测回答内容是否停止变化（连续若干秒内容不变则认为完成）
        """
        start_time = time.time()
        last_content = ""
        stable_count = 0
        stable_threshold = 3  # 内容连续稳定的检查次数
        check_interval = 2  # 每次检查间隔秒数

        # 回答区域的选择器（需要根据实际 DOM 适配）
        answer_selectors = [
            '[data-testid="chat_message"]:last-child .message-content',
            '.chat-message:last-child .markdown-body',
            '.message-list > div:last-child .message-content',
            '.response-content:last-child',
            '[class*="message"]:last-child [class*="content"]',
            '[class*="answer"]:last-child',
        ]

        while time.time() - start_time < timeout:
            time.sleep(check_interval)

            current_content = ""
            for selector in answer_selectors:
                try:
                    elements = self._page.locator(selector)
                    if elements.count() > 0:
                        current_content = elements.last.inner_text(timeout=3000)
                        if current_content.strip():
                            break
                except Exception:
                    continue

            # 备选方案：尝试获取页面上所有消息块中最后一个非用户消息
            if not current_content.strip():
                try:
                    # 通用方式：获取所有聊天消息，取最后一条非用户输入的消息
                    all_messages = self._page.locator('[class*="message"]')
                    count = all_messages.count()
                    if count > 0:
                        last_msg = all_messages.nth(count - 1)
                        current_content = last_msg.inner_text(timeout=3000)
                except Exception:
                    pass

            if current_content.strip():
                if current_content == last_content:
                    stable_count += 1
                    if stable_count >= stable_threshold:
                        # 内容已稳定，再验证一下发送按钮是否恢复可用
                        try:
                            send_btn = self._page.locator('button[data-testid="send_button"], button:has-text("发送")').first
                            if send_btn.is_visible(timeout=2000) and send_btn.is_enabled():
                                return current_content.strip()
                        except Exception:
                            pass
                        # 即使按钮检测失败，内容已稳定也返回
                        return current_content.strip()
                else:
                    stable_count = 0
                    last_content = current_content

        # 超时后返回已获取的内容（如果有的话）
        if last_content.strip():
            logger.warning(f"等待超时，返回已获取的部分内容（长度: {len(last_content)}）")
            return last_content.strip()

        raise Exception(f"等待豆包回答超时（{timeout}秒），未能获取到回答内容")

    def close(self):
        """关闭浏览器资源"""
        try:
            if self._page:
                self._page.close()
            if self._context:
                self._context.close()
            if self._browser:
                self._browser.close()
            if self._playwright:
                self._playwright.stop()
        except Exception as e:
            logger.error(f"关闭浏览器资源时出错: {str(e)}")
