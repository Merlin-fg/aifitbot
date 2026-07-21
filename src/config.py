import os
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
JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    raise ValueError("请在 .env 中设置 JWT_SECRET（用于签发和验证用户令牌）")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# ============================================================
# 应用配置
# ============================================================
APP_TITLE = "AIFitBot - AI 私人健身教练"
APP_VERSION = "0.2.1"

# ============================================================
# RAG 配置
# ============================================================
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
RAG_K = int(os.getenv("RAG_K", "2"))            # 检索返回文档数
RAG_MAX_CHUNK = int(os.getenv("RAG_MAX_CHUNK", "150"))  # 每条引用截断字数
RAG_TOKEN_BUDGET = int(os.getenv("RAG_TOKEN_BUDGET", "4096"))  # 上下文 token 预算
RAG_HISTORY_ROUNDS = int(os.getenv("RAG_HISTORY_ROUNDS", "6"))  # 对话历史保留轮数
RAG_REJECT_THRESHOLD = float(os.getenv("RAG_REJECT_THRESHOLD", "0.01"))   # rerank 分数低于此值拒答
RAG_WEAK_THRESHOLD = float(os.getenv("RAG_WEAK_THRESHOLD", "0.18"))     # rerank 分数低于此值追加免责
RAG_FAST_REJECT = float(os.getenv("RAG_FAST_REJECT", "0.30"))          # 向量相似度低于此值直接拒答，跳过 LLM 调用
# 校准依据：无关查询 0.16-0.24，相关查询 0.46+，0.30 在中间安全区

# ============================================================
# 管理员默认账号（首次启动时创建）
# ============================================================
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
if not ADMIN_PASSWORD:
    import random, string, logging
    ADMIN_PASSWORD = ''.join(random.choices(string.ascii_letters + string.digits, k=12))
    logging.getLogger("aifitbot").warning("已自动生成管理员密码，请通过 .env 文件中的 ADMIN_PASSWORD 查看")
