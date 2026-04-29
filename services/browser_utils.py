"""浏览器可执行文件检测工具 - 自动查找可用的 Chromium 内核浏览器"""
import os
import logging
import platform

logger = logging.getLogger(__name__)

# 模块级缓存，确保整个进程生命周期内只检测一次
_cached_args = None

# Windows 上常见的浏览器路径
_CHROME_PATHS_WIN = [
    os.path.join(os.environ.get("PROGRAMFILES", ""), "Google", "Chrome", "Application", "chrome.exe"),
    os.path.join(os.environ.get("PROGRAMFILES(X86)", ""), "Google", "Chrome", "Application", "chrome.exe"),
    os.path.join(os.environ.get("LOCALAPPDATA", ""), "Google", "Chrome", "Application", "chrome.exe"),
]

_EDGE_PATHS_WIN = [
    os.path.join(os.environ.get("PROGRAMFILES(X86)", ""), "Microsoft", "Edge", "Application", "msedge.exe"),
    os.path.join(os.environ.get("PROGRAMFILES", ""), "Microsoft", "Edge", "Application", "msedge.exe"),
]


def get_browser_launch_args(playwright_instance):
    """
    检测系统中可用的浏览器，返回 launch_persistent_context 需要的额外参数。

    检测优先级:
        1. Playwright 内置 Chromium（返回空字典，使用默认行为）
        2. 系统 Chrome（返回 {"channel": "chrome"}）
        3. 系统 Edge（返回 {"channel": "msedge"}）
        4. 都找不到则抛出 RuntimeError

    Args:
        playwright_instance: 已启动的 Playwright 实例

    Returns:
        dict: 可直接通过 ** 展开传入 launch_persistent_context 的参数字典
    """
    global _cached_args
    if _cached_args is not None:
        return _cached_args

    # 1. 检查 Playwright 内置 Chromium
    try:
        bundled_path = playwright_instance.chromium.executable_path
        if bundled_path and os.path.isfile(bundled_path):
            logger.info(f"使用 Playwright 内置 Chromium: {bundled_path}")
            _cached_args = {}
            return _cached_args
    except Exception:
        pass

    # 2. 检查系统 Chrome
    if platform.system() == "Windows":
        for path in _CHROME_PATHS_WIN:
            if path and os.path.isfile(path):
                logger.info(f"Playwright Chromium 未安装，回退到系统 Chrome: {path}")
                _cached_args = {"channel": "chrome"}
                return _cached_args

        # 3. 检查系统 Edge
        for path in _EDGE_PATHS_WIN:
            if path and os.path.isfile(path):
                logger.info(f"Playwright Chromium 未安装，回退到系统 Edge: {path}")
                _cached_args = {"channel": "msedge"}
                return _cached_args
    else:
        # 非 Windows 平台，使用 shutil.which 检测
        import shutil
        if shutil.which("google-chrome") or shutil.which("google-chrome-stable"):
            logger.info("Playwright Chromium 未安装，回退到系统 Chrome")
            _cached_args = {"channel": "chrome"}
            return _cached_args
        if shutil.which("microsoft-edge") or shutil.which("microsoft-edge-stable"):
            logger.info("Playwright Chromium 未安装，回退到系统 Edge")
            _cached_args = {"channel": "msedge"}
            return _cached_args

    raise RuntimeError(
        "未找到可用的浏览器。请执行以下任一操作：\n"
        "1. 运行 `playwright install chromium` 安装 Playwright 内置浏览器\n"
        "2. 安装 Google Chrome 或 Microsoft Edge"
    )
