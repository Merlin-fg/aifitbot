import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek").lower()

if LLM_PROVIDER == "deepseek":
    MODEL_NAME = "deepseek-chat"
    API_KEY = os.getenv("DEEPSEEK_API_KEY")
    BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
elif LLM_PROVIDER == "aliyun":
    MODEL_NAME = "qwen-plus"
    API_KEY = os.getenv("DASHSCOPE_API_KEY")
    BASE_URL = os.getenv("ALIYUN_BASE_URL")
else:
    raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")

if not API_KEY:
    raise ValueError(f"请设置 {LLM_PROVIDER} 对应的 API Key 到环境变量")