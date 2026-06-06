from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from src.config import LLM_PROVIDER, MODEL_NAME, API_KEY, BASE_URL
from src.rag import create_rag_chain   # 新增导入
import streamlit as st

def get_llm():
    if LLM_PROVIDER == "deepseek":
        return ChatDeepSeek(model=MODEL_NAME, api_key=API_KEY, base_url=BASE_URL, temperature=0.7)
    else:
        return ChatOpenAI(model=MODEL_NAME, api_key=API_KEY, base_url=BASE_URL, temperature=0.7)

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
    chain = prompt | llm
    response = chain.invoke({
        "chat_history": chat_history,
        "input": user_input
    })
    return response.content

def get_chat_history():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    return st.session_state.chat_history

# ===== 新增：RAG 回复生成 =====
def get_rag_response(llm, user_input):
    rag_chain = create_rag_chain(llm)
    # 新链返回纯文本，直接使用
    return rag_chain.invoke(user_input)