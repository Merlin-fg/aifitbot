# AIFitBot — AI 私人健身教练

基于 **LangChain + RAG 检索增强生成** 的智能健身助手。97 个结构化知识块覆盖动作库、训练原理、营养学三大领域，支持 BM25+向量混合检索、重排序、个性化训练计划生成。

## 功能亮点

| 模块 | 功能 |
|------|------|
|  🤖 RAG 智能问答 | 97 个语义知识块 + BM25+向量 RRF 混合检索 + gte-rerank 重排序 + 引用溯源 |
|  📋 训练计划生成 | 选择目标/周期/器械/水平 → AI 生成结构化 JSON → 渲染为周表卡片 |
|  🏋️ 动作库浏览 | 43 个标准动作按胸/背/肩/手臂/臀腿/核心分类，详情可展开 |
|  👤 用户档案 | 填写身高体重目标伤病，注入所有对话和计划生成 |
|  🔐 用户系统 | 注册/登录/改密，JWT 认证，admin/user 双角色 |
|  📚 知识库管理 | 上传 MD/PDF/TXT → 语义分片 → 多 Collection 存储（仅管理员） |
|  ⚡ 流式输出 | SSE 逐 token 推送 + 中断按钮 + marked.js Markdown 渲染 |
|  💬 多会话 | 创建/切换/删除会话，历史持久化，滑动窗口 6 轮 + 自动摘要压缩 |
|  📊 管理仪表盘 | 用户/文档/会话/消息数量统计 |
|  🎨 温暖健身主题 | 珊瑚橙 + 环境光阴影 + 微动效 + 渐变装饰 |

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端框架 | FastAPI（异步 + SSE） |
| 前端 | Jinja2 + HTMX + Alpine.js + marked.js |
| 数据库 | SQLite + SQLModel |
| 向量数据库 | ChromaDB（3 分类 Collection + general 聚合） |
| LLM 框架 | LangChain + LCEL |
| 混合检索 | ChromaDB 向量检索 + rank_bm25 关键词 + RRF 融合 |
| 重排序 | 阿里云 DashScope gte-rerank |
| 大模型 | 阿里云百炼 qwen-plus |
| 嵌入模型 | 阿里云百炼 text-embedding-v2 |
| 认证 | JWT + bcrypt |
| 包管理 | uv（Python 3.13+） |

## 项目结构

```
aifitbot/
├── data/                          # 知识库源文件（3 个结构化 MD）
│   ├── actions.md                 #   43 个标准动作
│   ├── principles.md              #   28 个训练原理
│   └── nutrition.md               #   22 个营养知识
├── src/
│   ├── main.py                    # FastAPI 入口 + 页面路由
│   ├── config.py                  # 配置中心
│   ├── models/                    # 数据模型
│   │   ├── user.py                #   用户（含档案字段）
│   │   ├── session.py / message.py / document.py
│   ├── repositories/              # 数据访问层
│   │   ├── vector_repo.py         #   ChromaDB 多Collection + BM25
│   │   ├── user_repo.py / session_repo.py / message_repo.py
│   ├── services/                  # 业务逻辑层
│   │   ├── rag_service.py         #   RAG v2：改写+混合检索+重排序+摘要
│   │   ├── plan_service.py        #   训练计划生成 LCEL 链
│   │   ├── auth_service.py / kb_service.py
│   ├── routes/                    # 路由层
│   │   ├── chat_routes.py         #   对话 + 流式 SSE
│   │   ├── auth_routes.py / kb_routes.py
│   ├── templates/                 # Jinja2 模板（12 个）
│   └── static/css/style.css       # Vital 温暖健身主题
├── tests/                         # 58 个 pytest 测试
├── .githooks/pre-commit           # Git 提交门禁（测试+质量检查）
└── .env.example                   # 环境变量模板
```

## 快速开始

### 1. 环境准备

需 Python 3.13+ 和 uv：

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows PowerShell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. 安装运行

```bash
git clone https://github.com/Merlin-fg/aifitbot.git
cd aifitbot
cp .env.example .env          # 编辑 .env 填入阿里云 API Key
uv sync
uv run uvicorn src.main:app --host 127.0.0.1 --port 8000
```

### 3. 首次使用

1. 浏览器打开 `http://localhost:8000`
2. 管理员登录：`admin` / `123456`
3. 进入「知识库」上传 `data/` 下的 3 个 MD 文件
4. 填写「档案」→ 开始「对话」或生成「训练计划」

### 4. Git 提交门禁

项目配置了 pre-commit hook，需通过单元测试和质量检查才能提交：

```bash
git config core.hooksPath .githooks   # 每个 clone 执行一次
```

## 架构分层

```
routes（表现层） → services（业务层） → repositories（数据层） → models（数据模型）
```

上层可调用下层，下层不依赖上层。
