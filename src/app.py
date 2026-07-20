"""Streamlit 原型入口——早期 Web 界面的开发原型，现已由 FastAPI (main.py) 取代。

如需运行: uv run streamlit run src/app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from src.bot import get_llm, get_response, get_chat_history, get_rag_response  # 新增导入
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="AIFitBot", page_icon="💪")
st.title("AIFitBot - 你的 AI 私人健身教练")

# 初始化
if "llm" not in st.session_state:
    st.session_state.llm = get_llm()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

llm = st.session_state.llm
chat_history = st.session_state.chat_history

# 侧边栏
with st.sidebar:
    st.header("📋 我的档案")
    age = st.number_input("年龄", 10, 80, 25)
    height = st.number_input("身高 (cm)", 100, 250, 170)
    weight = st.number_input("体重 (kg)", 30, 200, 65)
    gender = st.selectbox("性别", ["男", "女"])
    goal = st.selectbox("目标", ["增肌", "减脂", "塑形", "保持健康"])
    equipment = st.multiselect("可用器材", ["哑铃", "杠铃", "弹力带", "跑步机", "自重"], default=["自重"])
    if st.button("更新档案"):
        profile_text = f"用户档案：{age}岁{gender}，{height}cm，{weight}kg，目标{goal}，可用器材：{', '.join(equipment)}。"
        st.session_state.profile = profile_text
        st.success("档案已更新！")

    st.divider()
    # 新增：知识库检索开关
    use_rag = st.checkbox("开启专业健身知识库检索（RAG）", value=False,
                          help="勾选后，我会从本地专业知识库中查找相关资料来增强回答。")

# 渲染历史消息（注意：RAG 模式下历史可能不携带，但显示上保留）
for msg in chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

user_input = st.chat_input("输入你的问题...")
if user_input:
    st.chat_message("user").write(user_input)
    context = st.session_state.get("profile", "")
    full_input = f"{context}\n{user_input}" if context else user_input

    # 记录用户消息
    human_msg = HumanMessage(content=full_input)
    chat_history.append(human_msg)

    with st.spinner("教练思考中..."):
        if use_rag:
            # 使用 RAG 模式生成回答（忽略历史）
            response_text = get_rag_response(llm, full_input)
        else:
            # 普通模式：携带历史
            response_text = get_response(llm, chat_history[:-1], full_input)

    ai_msg = AIMessage(content=response_text)
    chat_history.append(ai_msg)
    st.chat_message("assistant").write(response_text)