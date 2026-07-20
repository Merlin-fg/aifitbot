"""LangChain 工具函数——供 Agent 模式自动调用。

TODO: 目前 calculate_bmi 未集成到对话流程中，后续版本可通过 ToolService 绑定。
"""

from langchain.tools import tool

@tool
def calculate_bmi(weight_kg: float, height_m: float) -> str:
    """计算 BMI，并返回分类。输入体重(kg)和身高(m)。"""
    bmi = weight_kg / (height_m ** 2)
    if bmi < 18.5:
        category = "偏瘦"
    elif bmi < 24:
        category = "正常"
    elif bmi < 28:
        category = "超重"
    else:
        category = "肥胖"
    return f"BMI = {bmi:.1f}，属于【{category}】范围。"