import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'csv', 'txt', 'xlsx', 'xls'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB

DATABASE_URI = f"sqlite:///{os.path.join(BASE_DIR, 'data', 'corpus.db')}"

# 豆包登录配置
DOUBAO_PROFILE_DIR = os.path.join(BASE_DIR, 'data', 'doubao_profile')
DOUBAO_LOGIN_TIMEOUT = 300  # 登录超时时间（秒）

# LLM 并行与重试配置
LLM_CONCURRENCY = 3    # 默认并行调用数
LLM_MAX_RETRIES = 1    # 默认重试次数
LLM_MAX_TOKENS = 4096  # 默认最大输出 token 数（防止 finish_reason: length）

# 默认 LLM 配置（用户可在前端覆盖）
DEFAULT_LLM_CONFIG = {
    'api_url': '',
    'api_key': '',
    'model': '',
}
