# 💪 AIFitBot — AI 私人健身教练

基于 LangChain + RAG 检索增强生成的智能健身助手。上传健身知识文档后，AI 会从知识库中检索相关内容，给出有据可查的专业回答。支持多用户、多会话、流式输出。

## 功能

| 模块 | 功能 |
|------|------|
| 🔐 用户系统 | 注册、登录、修改密码、JWT 认证 |
| 👑 权限管理 | 管理员（admin/123456）可管理知识库，普通用户仅可问答 |
| 📚 知识库管理 | 上传 MD/PDF/TXT → 自动切分向量化 → 列表/删除（仅管理员） |
| 🤖 RAG 问答 | 检索知识库 + LLM 增强生成，回答带引用溯源 |
| ⚡ 流式输出 | SSE 逐字推送，支持中断 |
| 💬 多会话 | 创建/切换/删除会话，每个会话独立历史 |
| 📝 历史持久化 | 对话记录存数据库，重新登录可找回 |
| 📊 管理仪表盘 | 用户数、文档数、会话数、消息数统计 |

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端框架 | FastAPI（异步 + SSE 流式） |
| 前端 | Jinja2 模板 + HTMX + Alpine.js（纯 Python 全栈，无需 Node.js） |
| 数据库 | SQLite + SQLModel |
| 向量数据库 | ChromaDB（本地嵌入式） |
| LLM 框架 | LangChain + LCEL |
| 大模型 | 阿里云百炼 qwen-plus（OpenAI 兼容接口） |
| 嵌入模型 | 阿里云百炼 text-embedding-v2（云端 API） |
| 认证 | JWT + bcrypt |
| 包管理 | uv（Python 3.13+） |

## 项目结构

```
aifitbot/
├── .env.example                  # 环境变量模板
├── pyproject.toml                # 依赖配置
├── CLAUDE.md                     # 产品文档
├── README.md
├── src/
│   ├── main.py                   # FastAPI 入口
│   ├── config.py                 # 配置中心
│   ├── database.py               # 数据库引擎
│   ├── bot.py                    # LLM 初始化
│   ├── rag.py                    # 阿里云 Embedding 适配器
│   ├── models/                   # 数据模型（SQLModel）
│   │   ├── user.py               #   用户
│   │   ├── session.py            #   会话
│   │   ├── message.py            #   消息
│   │   └── document.py           #   文档
│   ├── repositories/             # 数据访问层
│   │   ├── user_repo.py
│   │   ├── session_repo.py
│   │   ├── message_repo.py
│   │   ├── document_repo.py
│   │   └── vector_repo.py        #   ChromaDB 操作
│   ├── services/                 # 业务逻辑层
│   │   ├── auth_service.py       #   认证
│   │   ├── kb_service.py         #   知识库管理
│   │   └── rag_service.py        #   RAG 检索增强
│   ├── routes/                   # 路由层
│   │   ├── auth_routes.py        #   /api/auth/*
│   │   ├── kb_routes.py          #   /api/admin/kb/*
│   │   └── chat_routes.py        #   /chat/*
│   ├── middleware/               # 中间件
│   │   └── auth_middleware.py     #   JWT 验证
│   ├── templates/                # Jinja2 页面模板
│   └── static/                   # CSS
```

## 快速开始

### 1. 安装 uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows PowerShell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. 克隆并安装

```bash
git clone <你的仓库地址>
cd aifitbot
uv sync
```

### 3. 配置 API Key

```bash
cp .env.example .env
```

编辑 `.env`，填入阿里云百炼的 API Key：

```env
LLM_PROVIDER=aliyun
ALIYUN_API_KEY=sk-你的密钥
ALIYUN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-plus
EMBEDDING_MODEL=text-embedding-v2
```

### 4. 启动

```bash
uv run uvicorn src.main:app --host 127.0.0.1 --port 8000
```

浏览器打开 `http://localhost:8000`

### 5. 首次使用

1. 管理员登录：用户名 `admin`，密码 `123456`
2. 点击导航栏「知识库管理」→ 上传健身知识文档（.md / .pdf / .txt）
3. 进入「对话」页面，新建会话开始提问

## 知识库

内置 8 个健身知识分类，上传后即可使用：

| 文档 | 内容 |
|------|------|
| 力量训练完全指南 | 胸/背/肩/手臂/腿/核心动作标准、组数、易错点 |
| 健身营养完全指南 | 三大宏量素、热量管理、三餐分配、微量元素 |
| 有氧训练与心肺指南 | LISS/HIIT/心率区间、不同体型策略 |
| 运动伤病预防与康复 | 热身流程、五大关节保护、拉伸动作库 |
| 训练计划模板与周期化 | 新手/中级/进阶三套模板、周期化原则 |
| 健身常见误区与纠正 | 14 个常见错误认知、新手 FAQ |
| 拉伸与柔韧性训练 | 训练前后拉伸动作库、久坐拉伸、泡沫轴 |
| 居家健身完全指南 | 零器械自重训练、哑铃/弹力带方案 |

## 架构分层

```
routes（表现层） → services（业务层） → repositories（数据层） → models（数据模型）
```

上层可调用下层，下层不依赖上层。更换前端框架只需改 routes 层。

## 开发协作

所有技术决策由开发者列出方案并解释优劣，项目负责人最终决定。详见 [CLAUDE.md](CLAUDE.md)。
