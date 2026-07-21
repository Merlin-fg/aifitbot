# AIFitBot — AI 私人健身教练

基于 **LangChain+Advanced RAG** 的智能健身问答平台。208 个结构化知识块覆盖动作库、训练原理、营养学、拉伸恢复、有氧/HIIT、自重训练、补剂、恢复策略、肌肉解剖、健身心理十大领域。向量+BM25 混合检索，端到端响应 **3 秒**。

## 功能亮点

| 模块 | 功能 |
|------|------|
| 🤖 RAG 智能问答 | 208 个语义知识块 + 向量 + BM25 关键词 + RRF 融合 + 父子切割 + 引用溯源 |
| 📊 评估驱动优化 | 30 条标注数据消融实验，移除负优化组件（多 Query -6.6pp / 重排序 -23pp），Hit@3 93.3% |
| ⚡ 三级质量门控 | 快速预检（<0.30→拒答）→ 混合检索 → 质量分级（reject/weak/ok），减少无效 LLM 调用 |
| 🏆 训练计划卡片 | AI 输出隐藏 JSON → 前端解析渲染为结构化训练计划卡片 → 回复"需要"一键导入打卡 |
| ✅ 训练打卡 | Chart.js 统计图表 + 训练日历 + 勾选完成追踪 |
| 🏋️ 动作库浏览 | 48 个标准动作按胸/背/肩/手臂/臀腿/核心分类，详细可展开 |
| 👤 用户档案 | 身高/体重/目标/器械/伤病，自动注入 RAG 对话实现个性化建议 |
| 🔐 用户系统 | 注册/登录/改密/档案编辑，JWT + bcrypt 认证，admin/user 双角色 |
| 📚 知识库管理 | 上传 MD/PDF/TXT → 父子切割（300 字子块检索 + 完整父块喂 LLM）→ 5 分类 Collection（仅管理员）|
| ⚡ 流式输出 | SSE 逐 token 推送 + 中断按钮 + marked.js Markdown 渲染 |
| 💬 多会话 | 创建/切换/删除会话，历史持久化，超 6 轮自动摘要压缩 |
| 📊 管理仪表盘 | 用户/文档/会话/消息数量统计 |

## 性能：93s → 3s 的四轮优化

| 轮次 | 手段 | 延迟 |
|------|------|------|
| 起点 | HyDE + qwen3.6-plus + 多 Query + 重排序 | **93.5s** |
| 1 | 移除 HyDE（省 1 次 LLM 调用，Hit@3 仅 -3.4pp） | 56.0s |
| 2 | 换 qwen3.7-max | 29.0s |
| 3 | RAG_K=2, chunk=150（缩减上下文） | 23.0s |
| 4 | 换 **qwen-turbo** + temperature=0.9 | **3.2s** |

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端框架 | FastAPI（异步 + SSE） |
| 前端 | Jinja2 + HTMX + Alpine.js + marked.js |
| 数据库 | SQLite（WAL 模式）+ SQLModel |
| 向量数据库 | ChromaDB embedded使用HNSW索引算法 （5 分类 Collection） |
| 检索 | 关键词分类 → 向量 + BM25 → RRF 融合（K=60） → 父子替换 |
| LLM | 阿里云百炼 **qwen-turbo**（OpenAI 兼容） |
| 嵌入模型 | 阿里云百炼 text-embedding-v2 |
| LLM 框架 | LangChain + LCEL |
| 认证 | JWT + bcrypt（Cookie httponly） |
| 统计 | Chart.js（饼图 + 折线图） |
| 测试 | pytest（58 cases, 7s）|
| 压测 | Locust（100 虚拟用户） |
| 包管理 | uv（Python 3.13+） |

## 项目结构

```
aifitbot/
├── data/                              # 知识库源文件（10 个结构化 MD）
│   ├── actions.md                     #   48 个训练动作
│   ├── principles.md                  #   46 个训练原理
│   ├── nutrition.md                   #   27 个营养知识
│   ├── stretching.md                  #   16 个拉伸恢复专题
│   ├── cardio.md                      #   13 个有氧/HIIT 专题
│   ├── bodyweight.md                  #   11 个自重训练专题
│   ├── supplements.md                 #   14 个补剂深度指南
│   ├── recovery.md                    #   11 个恢复/睡眠专题
│   ├── anatomy.md                     #   12 个肌肉解剖专题
│   └── mindset.md                     #   10 个健身心理/习惯专题
├── chroma_db/                         # ChromaDB 向量数据 + parents.json
├── src/
│   ├── main.py                        # FastAPI 入口 + 页面路由
│   ├── config.py                      # 配置中心（RAG 参数/API Key/JWT）
│   ├── bot.py                         # LLM 工厂
│   ├── database.py                    # SQLite + WAL 初始化
│   ├── rag.py                         # DashScope Embeddings 适配器
│   ├── models/                        # 数据模型
│   │   ├── user.py                    #   用户（含健身档案字段）
│   │   ├── workout.py                 #   训练计划 + 打卡记录
│   │   ├── session.py / message.py / document.py
│   ├── repositories/                  # 数据访问层
│   │   ├── vector_repo.py             #   ChromaDB 多Collection + BM25 + 父子切割
│   │   ├── workout_repo.py            #   打卡数据 + 统计查询
│   │   ├── user_repo.py / session_repo.py / message_repo.py / document_repo.py
│   ├── services/                      # 业务逻辑层
│   │   ├── rag_service.py             #   RAG v4：混合检索 + 质量门控 + 缓存 + 摘要
│   │   ├── workout_service.py         #   训练计划解析导入 + 打卡统计
│   │   ├── auth_service.py / kb_service.py
│   ├── routes/                        # 路由层
│   │   ├── chat_routes.py             #   对话 + SSE 流式 + 打卡确认拦截
│   │   ├── workout_routes.py          #   打卡页面 + API
│   │   ├── auth_routes.py / kb_routes.py
│   ├── middleware/                    # 中间件
│   │   └── auth_middleware.py         #   JWT 验证 + 角色权限
│   ├── templates/                     # Jinja2 模板
│   └── static/css/style.css           # Vital 温暖健身主题
├── scripts/                           # 评估脚本
│   ├── eval_rag.py                    #   检索层消融实验（Hit@3/MRR/Recall@3）
│   └── eval_generation.py             #   生成层 LLM-as-Judge（4 维度）
├── tests/                             # 58 个 pytest 单元测试
├── tests/loadtest/                    # Locust 压测
│   ├── locustfile.py                  #   压测场景定义
│   └── prepare_data.py                #   100 虚拟用户 + JWT token 准备
├── .githooks/pre-commit               # Git 提交门禁（测试+质量检查）
└── .env.example                       # 环境变量模板
```

## 快速开始

### 1. 环境准备

Python 3.13+ + uv：

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
uv run uvicorn src.main:app --host 127.0.0.1 --port 9000
```

### 3. 首次使用

1. 浏览器打开 `http://localhost:9000`
2. 管理员登录：`admin` / `123456`
3. 进入「知识库」上传 `data/` 下的 10 个 MD 文件（或运行 `rebuild_all()` 批量导入）
4. 填写「🏋️ 档案」（身高/体重/目标）→ 开始「对话」→ 输入"我要练胸"
5. AI 生成计划卡片后，回复"需要"一键导入打卡

### 4. 运行评估

```bash
python scripts/eval_rag.py           # 检索层消融实验
python scripts/eval_generation.py    # 生成层 LLM-as-Judge
```

### 5. 运行压测

```bash
cd tests/loadtest
python prepare_data.py               # 生成测试用户（仅首次）
locust -f locustfile.py --host http://localhost:9000
```

### 6. Git 提交门禁

```bash
git config core.hooksPath .githooks   # 每个 clone 执行一次
```

## 架构分层

```
routes（表现层） → services（业务层） → repositories（数据层） → models（数据模型）
```

上层可调用下层，下层不依赖上层。

## RAG 管道 v4

```
用户提问
    ↓
关键词分类 → 路由到对应 Collection（5 选 1）
    ↓
[Gate 1] 快速预检（向量相似度 < 0.30 → 拒答，零 LLM 调用）
    ↓
混合检索：向量 top-4 + BM25 top-4 → RRF 融合（K=60） → top-2
    ↓
_resolve_parents()：子块 → 完整父块替换
    ↓
[Gate 2] 质量分级（<0.01 reject | 0.01~0.18 weak | >0.18 ok）
    ↓
注入 System Prompt（含用户档案 + 对话历史摘要） → qwen-turbo SSE 流式生成
    ↓
前端：marked.js 渲染 + STARTJSON 解析 → 训练计划卡片
```
