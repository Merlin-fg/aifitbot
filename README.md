# 💪 AIFitBot - AI 私人健身教练（集成 RAG 专业知识库）

基于 **LangChain + RAG 检索增强生成** 的智能健身助手，支持普通对话模式与专业知识库增强模式，可根据你的身体数据提供个性化训练与饮食建议。

## 🎯 场景问题
- 健身新手缺乏系统指导，网络信息碎片化，难以辨别真伪
- 需要一个能即时回答健身疑问、具备运动解剖学与营养学知识的 AI 教练
- 本项目通过 **本地专业知识库 + 大模型检索增强（RAG）**，弥补通用大模型知识宽泛、不够具体的缺陷，回答更专业、更可靠

## ✨ 核心亮点
- **🧠 RAG 检索增强生成**：内置健身动作库、饮食指南等专业文档，利用向量检索为回答提供精确事实支撑
- **🔧 底层 LCEL 链组装**：不依赖高级封装，使用 `langchain_core` 基础组件手工搭建 RAG 链，稳定可靠，充分展示对 RAG 原理的理解
- **🔀 双模式自由切换**：普通模式依靠模型自身能力，知识库模式结合本地文档，适应不同问题复杂度
- **💻 本地嵌入模型**：使用 HuggingFace 开源嵌入模型 `all-MiniLM-L6-v2`，完全免费，无 API 依赖，隐私数据不出本机
- **🌐 多 LLM 支持**：可一键切换 DeepSeek / 阿里云百炼模型，灵活应对不同场景
- **📋 用户档案记忆**：侧边栏填写身体数据后全程复用，对话历史自动维护，体验连贯

## 🛠 技术栈
| 类别 | 技术/库 |
|------|----------|
| LLM 框架 | LangChain, LangChain-Core, LangChain-Community |
| RAG 组件 | LangChain-Chroma, Chroma, HuggingFace Embeddings |
| 嵌入模型 | `all-MiniLM-L6-v2`（本地运行） |
| 大语言模型 | DeepSeek (ChatDeepSeek) / 阿里云百炼 (OpenAI 兼容) |
| 前端 | Streamlit |
| 环境管理 | uv (Python 3.13+) |
| 配置与安全 | python-dotenv, .env 环境隔离 |

## 📦 环境安装（使用 uv）

### 1️⃣ 安装 uv（若已有请跳过）
```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows PowerShell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

2️⃣ 克隆项目并进入目录
bash
git clone https://github.com/你的用户名/aifitbot.git
cd aifitbot
3️⃣ 一键安装所有依赖
bash
uv sync
系统会自动创建虚拟环境，并根据 pyproject.toml 和 uv.lock 安装精确版本的依赖包。

4️⃣ 配置 API Key
bash
cp .env.example .env
编辑 .env 文件，填入你的大模型密钥（以 DeepSeek 为例）：

env
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
5️⃣ （可选）预生成知识库向量
为了让首次启动更快，建议提前生成向量库：

bash
uv run python -c "from src.rag import get_vectorstore; get_vectorstore()"
此命令会读取 data/ 目录下的所有 .md 知识文档，分割并计算向量后存入 chroma_db/。之后启动 Streamlit 时无需等待初始化。

🚀 启动命令
Web 界面（推荐）
bash
uv run streamlit run src/app.py
浏览器访问 http://localhost:8501，在左侧填写个人档案，并 勾选“开启专业健身知识库检索（RAG）” 以体验检索增强回答。

命令行模式（简单对话，无 RAG）
bash
uv run python -m src.cli



🔭 后续优化方向
多轮 RAG 对话历史：在 RAG 模式下融合对话历史，使回答更连贯

知识库扩展：支持 PDF、网页抓取等多源文档导入

向量库迁移：从本地 Chroma 升级到 Milvus / Pinecone 等云向量服务，支持海量知识

Agent + 工具调用：将 RAG 与 Agent 结合，让模型自动决策是调用工具（如 BMI 计算）还是检索知识库

微信机器人接入：将核心能力封装为 FastAPI 接口，对接企业微信或个人微信，实现多端触达

用户训练日志：记录每次训练反馈，使用长期记忆动态调整计划

🤝 贡献与反馈
欢迎提 Issue 或 PR 一起改进！如果你觉得这个项目对你有帮助，请给个 ⭐ Star 支持一下~