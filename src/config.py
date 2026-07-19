import os
import secrets
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# LLM 配置
# ============================================================
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "aliyun").lower()

if LLM_PROVIDER == "aliyun":
    MODEL_NAME = os.getenv("LLM_MODEL", "qwen-plus")
    API_KEY = os.getenv("ALIYUN_API_KEY")
    BASE_URL = os.getenv("ALIYUN_BASE_URL")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v2")
elif LLM_PROVIDER == "deepseek":
    MODEL_NAME = "deepseek-chat"
    API_KEY = os.getenv("DEEPSEEK_API_KEY")
    BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v2")
else:
    raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")

if not API_KEY:
    raise ValueError(f"请设置 {LLM_PROVIDER} 对应的 API Key 到环境变量")

# ============================================================
# 数据库配置
# ============================================================
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///aifitbot.db")

# ============================================================
# JWT 认证配置
# ============================================================
JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_urlsafe(32))
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# ============================================================
# 应用配置
# ============================================================
APP_TITLE = "AIFitBot - AI 私人健身教练"
APP_VERSION = "0.2.0"

# ============================================================
# 管理员默认账号（首次启动时创建）
# ============================================================
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "123456")
