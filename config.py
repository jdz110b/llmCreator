import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'csv', 'txt'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB

DATABASE_URI = f"sqlite:///{os.path.join(BASE_DIR, 'data', 'corpus.db')}"

# 默认 LLM 配置（用户可在前端覆盖）
DEFAULT_LLM_CONFIG = {
    'api_url': '',
    'api_key': '',
    'model': '',
}
