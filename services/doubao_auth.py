"""豆包登录认证管理器 - 使用 Playwright persistent context 管理登录状态"""
import os
import shutil
import threading
import time
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DoubaoAuthManager:
    """
    管理豆包登录生命周期的单例类。

    状态机:
        idle ──start_login()──> logging_in ──检测成功──> ready
        ready ──acquire_for_query()──> busy ──release_from_query()──> ready
        ready ──logout()──> idle
    """

    DOUBAO_URL = "https://www.doubao.com"

    def __init__(self, user_data_dir, login_timeout=300):
        self._user_data_dir = os.path.abspath(user_data_dir)
        self._login_timeout = login_timeout
        self._lock = threading.Lock()
        self._state = "idle"  # idle / logging_in / ready / busy
        self._last_login_time = None
        self._login_context = None
        self._login_playwright = None
        self._login_thread = None
        self._cancel_requested = False

        # 服务器重启恢复：检查 profile 目录是否存在
        if self._has_profile():
            self._state = "ready"
            logger.info("检测到已有豆包登录 profile，状态初始化为 ready")

    def _has_profile(self):
        """检查 user_data_dir 是否包含有效的 Chromium profile"""
        if not os.path.isdir(self._user_data_dir):
            return False
        # Chromium persistent context 会创建 Default 目录或直接在根目录写入文件
        contents = os.listdir(self._user_data_dir)
        return len(contents) > 0

    def get_user_data_dir(self):
        return self._user_data_dir

    def check_status(self):
        """返回当前认证状态"""
        return {
            "state": self._state,
            "logged_in": self._state in ("ready", "busy"),
            "last_login_time": self._last_login_time.strftime("%Y-%m-%d %H:%M:%S") if self._last_login_time else None,
        }

    def is_logged_in(self):
        return self._state in ("ready", "busy")

    def start_login(self):
        """启动登录流程，在后台线程中打开 headed 浏览器"""
        with self._lock:
            if self._state == "busy":
                return {"error": "查询进行中，请等待查询完成后再登录"}
            if self._state == "logging_in":
                return {"error": "登录已在进行中，请在弹出的浏览器窗口中完成登录"}
            self._state = "logging_in"
            self._cancel_requested = False

        self._login_thread = threading.Thread(target=self._login_worker, daemon=True)
        self._login_thread.start()
        return {"status": "login_started"}

    def _login_worker(self):
        """后台线程：打开 headed 浏览器，等待用户登录"""
        try:
            from playwright.sync_api import sync_playwright

            self._login_playwright = sync_playwright().start()
            os.makedirs(self._user_data_dir, exist_ok=True)

            self._login_context = self._login_playwright.chromium.launch_persistent_context(
                self._user_data_dir,
                headless=False,
                viewport={"width": 1280, "height": 800},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            )

            page = self._login_context.pages[0] if self._login_context.pages else self._login_context.new_page()
            page.goto(self.DOUBAO_URL, wait_until="domcontentloaded", timeout=60000)

            # 轮询检测登录成功
            start_time = time.time()
            while time.time() - start_time < self._login_timeout:
                if self._cancel_requested:
                    logger.info("登录已取消")
                    break

                time.sleep(2)

                if self._check_login_success(page):
                    with self._lock:
                        self._state = "ready"
                        self._last_login_time = datetime.now()
                    logger.info("豆包登录成功")
                    break
            else:
                # 超时
                logger.warning("豆包登录超时")
                with self._lock:
                    if self._state == "logging_in":
                        self._state = "idle"

        except Exception as e:
            logger.error(f"登录流程异常: {str(e)}")
            with self._lock:
                if self._state == "logging_in":
                    self._state = "idle"
        finally:
            self._close_login_browser()

    def _check_login_success(self, page):
        """检测用户是否已成功登录"""
        try:
            # 检查是否存在聊天输入框（登录后才会出现）
            input_selectors = [
                'textarea[data-testid="chat_input"]',
                '#chat-input',
                'textarea.chat-input',
                'div[contenteditable="true"]',
            ]
            for selector in input_selectors:
                try:
                    el = page.locator(selector).first
                    if el.is_visible(timeout=1000):
                        return True
                except Exception:
                    continue

            # 检查 cookies 中是否包含登录标识
            cookies = self._login_context.cookies()
            login_cookie_names = {'sessionid', 'ttwid', 'sid_tt', 'uid_tt'}
            found = {c['name'] for c in cookies if c.get('domain', '').endswith('.doubao.com')}
            if found & login_cookie_names:
                return True

        except Exception:
            pass
        return False

    def _close_login_browser(self):
        """安全关闭登录浏览器"""
        try:
            if self._login_context:
                self._login_context.close()
        except Exception:
            pass
        try:
            if self._login_playwright:
                self._login_playwright.stop()
        except Exception:
            pass
        self._login_context = None
        self._login_playwright = None

    def cancel_login(self):
        """取消登录流程"""
        with self._lock:
            if self._state != "logging_in":
                return {"status": "ok"}
            self._cancel_requested = True
            self._state = "idle"
        # 关闭浏览器会在 _login_worker 的 finally 中处理
        # 但也主动尝试关闭，加速退出
        self._close_login_browser()
        return {"status": "ok"}

    def logout(self):
        """退出登录，清除持久化数据"""
        with self._lock:
            if self._state == "busy":
                return {"error": "查询进行中，请等待完成后再退出"}
            if self._state == "logging_in":
                self._cancel_requested = True
                self._close_login_browser()

        # 删除 profile 目录
        if os.path.isdir(self._user_data_dir):
            try:
                shutil.rmtree(self._user_data_dir)
                logger.info("已清除豆包登录 profile")
            except Exception as e:
                logger.error(f"清除 profile 失败: {str(e)}")

        with self._lock:
            self._state = "idle"
            self._last_login_time = None
        return {"status": "ok"}

    def acquire_for_query(self):
        """查询前获取锁，将状态切换为 busy"""
        with self._lock:
            if self._state == "logging_in":
                raise RuntimeError("登录进行中，请等待登录完成后再查询")
            if self._state == "idle":
                raise RuntimeError("请先登录豆包")
            if self._state == "busy":
                raise RuntimeError("已有查询在进行中，请等待完成")
            self._state = "busy"
            return True

    def release_from_query(self):
        """查询完成后释放锁"""
        with self._lock:
            if self._state == "busy":
                self._state = "ready"
