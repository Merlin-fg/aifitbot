from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from src.config import LLM_PROVIDER, MODEL_NAME, API_KEY, BASE_URL
from src.rag import create_rag_chain
import streamlit as st


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


SYSTEM_PROMPT = """你是一位专业的私人健身教练与营养师。你有以下能力：
1. 根据用户的身体数据（身高、体重、年龄、性别、目标、可用器材）设计训练计划。
2. 提供饮食建议。
3. 解释动作要领，提醒避免受伤。
4. 如果用户询问体重相关问题，可以计算 BMI（体重kg / 身高m的平方），并给出分类（偏瘦/正常/超重/肥胖）。
请用热情、专业、鼓励的语气回答。如果用户未提供身体数据，主动询问。"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])


def get_response(llm, chat_history, user_input):
    """普通对话模式：携带历史记录。"""
    chain = prompt | llm
    response = chain.invoke({
        "chat_history": chat_history,
        "input": user_input,
    })
    return response.content


def get_chat_history():
    """Streamlit 会话中的聊天记录。"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    return st.session_state.chat_history


def get_rag_response(llm, user_input):
    """RAG 知识库增强模式（当前不含历史）。"""
    rag_chain = create_rag_chain(llm)
    return rag_chain.invoke(user_input)
