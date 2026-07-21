"""LLM 工厂——初始化 ChatOpenAI 实例，供 RAG 服务和路由层共用。"""

from langchain_openai import ChatOpenAI
from src.config import LLM_PROVIDER, MODEL_NAME, API_KEY, BASE_URL


def get_llm():
    """初始化 LLM，支持阿里云百炼（OpenAI 兼容）和 DeepSeek。"""
    if LLM_PROVIDER in ("aliyun", "deepseek"):
        return ChatOpenAI(
            model=MODEL_NAME,
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=0.7,
        )
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")
