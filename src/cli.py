"""命令行聊天入口——无需浏览器，终端直接对话。

使用方法: uv run python -m src.cli
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bot import get_llm, SYSTEM_PROMPT
from langchain_core.messages import HumanMessage, SystemMessage

def run_cli():
    """启动命令行对话循环（输入 'exit' 退出）。"""
    print("🏋️ AIFitBot 命令行模式 (输入 'exit' 退出)")
    llm = get_llm()
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    while True:
        user_input = input("👤 你：")
        if user_input.lower() == "exit":
            break
        messages.append(HumanMessage(content=user_input))
        response = llm.invoke(messages)
        messages.append(response)
        print(f"🤖 教练：{response.content}")

if __name__ == "__main__":
    run_cli()