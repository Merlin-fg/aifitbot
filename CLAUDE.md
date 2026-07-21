# AIFitBot — 产品文档与开发规范

## 1. 产品概述

**产品名称：** AIFitBot  
**产品定位：** AI 私人健身教练，面向健身新手的专业知识问答平台  
**目标用户：** 健身初学者、需要饮食/训练指导的普通用户  
**核心价值：** 基于专业健身知识库的 RAG 智能问答，回答有据可查，弥补通用 AI 回答宽泛、不可靠的缺陷

## 2. 运行环境

- 支持 **Windows 10/11** 和 **macOS** 桌面运行
- 用户通过浏览器访问本地服务（`http://localhost:9000`）
- 无需 Docker，通过 Python 虚拟环境直接运行
- LLM 和 Embedding 均使用云端 API（阿里云百炼平台），不依赖本地 GPU

## 3. 当前功能状态

### 3.1 已实现 ✅

| 模块 | 功能 |
|------|------|
| 用户认证 | 注册/登录/改密/档案编辑，JWT + bcrypt，admin/user 双角色，Cookie httponly |
| 知识库管理 | MD/PDF/TXT 上传/预览/删除，父子切割（300 字子块 + 完整父块），5 分类 ChromaDB Collection |
| RAG 问答 | v4 极简版：向量 + BM25 混合检索 + RRF 融合 + 三级质量门控 + 对话摘要 + 高频缓存 |
| 流式输出 | SSE 逐 token 推送，marked.js 渲染，训练计划 JSON → 前端卡片渲染，中断按钮 |
| 多会话 | 创建/切换/重命名/删除，历史消息持久化，超 6 轮自动摘要压缩 |
| 训练打卡 | AI 生成计划 → "需要"确认导入 → 勾选完成 + Chart.js 统计图表 |
| 动作库浏览 | 48 个标准动作按部位分类，详情可展开 |
| 用户档案 | 身高/体重/年龄/目标/器械/伤病 → 注入 prompt 个性化建议 |
| 管理仪表盘 | 用户/文档/会话/消息数量统计 |
| 健康检查 | `/api/health` |
| 评估体系 | 30 条标注数据消融实验（Hit@3/MRR/Recall@3）+ LLM-as-Judge 生成评估 |

### 3.2 计划但未实现

| 功能 | 优先级 |
|------|--------|
| 用户反馈（点赞/点踩） | 低 |
| 系统 Prompt 界面配置 | 低 |
| 请求频率限制 | 中（代码已有框架，未启用） |
| 会话导出为 Markdown | 低 |

## 4. 技术栈（当前运行版本）

| 模块 | 选择 | 说明 |
|------|------|------|
| LLM 模型 | **qwen-turbo**（阿里云百炼） | OpenAI 兼容接口 |
| LLM 框架 | **LangChain + LCEL** | 链式调用 + 异步流式 |
| 嵌入模型 | **text-embedding-v2**（阿里云百炼） | 云端 API，batch 25 |
| 后端框架 | **FastAPI** | 原生异步，SSE 流式输出 |
| 前端方案 | **Jinja2 + HTMX + Alpine.js** | 纯 Python 全栈，无需 Node.js |
| 数据库 | **SQLite + SQLModel**（WAL 模式） | 零配置，Pydantic 类型安全 |
| 向量数据库 | **ChromaDB**（embedded） | 本地持久化，5 Collection |
| 检索引擎 | 向量 + BM25Okapi → RRF 融合（K=60） | 中文逐字+2-gram 分词 |
| 认证方案 | **JWT**（python-jose + bcrypt） | Cookie httponly，480min 过期 |
| 包管理 | **uv** | Python 3.13+ |

## 5. 项目目录结构

```
aifitbot/
├── data/                                # 知识库源文件（10 个 MD，208 片段）
│   ├── actions.md                       #   48 个训练动作
│   ├── principles.md                    #   46 个训练原理
│   ├── nutrition.md                     #   27 个营养知识
│   ├── stretching.md                    #   16 个拉伸恢复专题
│   ├── cardio.md                        #   13 个有氧/HIIT 专题
│   ├── bodyweight.md                    #   11 个自重训练专题
│   ├── supplements.md                   #   14 个补剂深度指南
│   ├── recovery.md                      #   11 个恢复/睡眠专题
│   ├── anatomy.md                       #   12 个肌肉解剖专题
│   └── mindset.md                       #   10 个健身心理/习惯专题
├── chroma_db/                           # ChromaDB 向量数据 + parents.json（父块映射）
├── src/
│   ├── __init__.py
│   ├── config.py                        # 配置中心（RAG 参数/API 密钥/JWT/阈值）
│   ├── main.py                          # FastAPI 应用入口 + 页面路由 + lifespan
│   ├── bot.py                           # LLM 工厂（ChatOpenAI）
│   ├── rag.py                           # DashScopeEmbeddings 适配器
│   ├── database.py                      # SQLite + WAL 初始化
│   ├── models/                          # 数据模型 (SQLModel)
│   │   ├── user.py                      #   用户（含身高/体重/年龄/目标/器械/伤病档案）
│   │   ├── session.py                   #   会话模型
│   │   ├── message.py                   #   消息模型（含 references JSON）
│   │   ├── document.py                  #   文档元数据
│   │   └── workout.py                   #   训练计划 + 打卡记录
│   ├── repositories/                    # 数据访问层
│   │   ├── vector_repo.py               #   ChromaDB 多Collection + BM25 + 父子切割 + RRF
│   │   ├── user_repo.py                 #   用户 CRUD + 档案更新
│   │   ├── session_repo.py              #   会话 CRUD
│   │   ├── message_repo.py              #   消息 CRUD
│   │   ├── document_repo.py             #   文档元数据 CRUD
│   │   └── workout_repo.py              #   打卡数据 + 统计查询
│   ├── services/                        # 业务逻辑层
│   │   ├── rag_service.py               #   RAG v4：混合检索 + 质量门控 + 缓存 + 摘要压缩
│   │   ├── auth_service.py              #   认证 + JWT
│   │   ├── kb_service.py                #   知识库上传/删除编排
│   │   └── workout_service.py           #   训练计划解析导入 + 打卡统计
│   ├── routes/                          # API 路由 (FastAPI routers)
│   │   ├── auth_routes.py               #   注册/登录/改密/档案更新 API
│   │   ├── kb_routes.py                 #   知识库 CRUD API
│   │   ├── chat_routes.py               #   对话 + SSE 流式 + 打卡确认拦截
│   │   └── workout_routes.py            #   打卡页面 + API
│   ├── middleware/                      # 中间件
│   │   ├── auth_middleware.py           #   JWT 验证 + 角色权限
│   │   └── rate_limit.py               #   频率限制（框架，未启用）
│   ├── static/                          # 静态资源
│   │   └── css/style.css                #   Vital 温暖健身主题（700+ 行）
│   └── templates/                       # Jinja2 页面模板
│       ├── base.html                    #   基础布局
│       ├── home.html / login.html / register.html
│       ├── chat.html                    #   对话主界面（SSE 流式 + 计划卡片渲染）
│       ├── profile.html                 #   健身档案编辑页
│       ├── workout.html                 #   打卡页面
│       ├── exercises.html               #   动作库浏览页
│       ├── admin_kb.html                #   知识库管理页
│       └── admin_dashboard.html         #   管理仪表盘
├── scripts/                             # 评估与工具脚本
│   ├── eval_rag.py                      #   检索层消融实验（5 模式对比）
│   └── eval_generation.py               #   生成层 LLM-as-Judge（4 维度）
├── tests/                               # 单元测试（58 个用例）
├── tests/loadtest/                      # Locust 压测
│   ├── locustfile.py                    #   压测场景定义（6 个任务权重）
│   └── prepare_data.py                  #   100 虚拟用户 + JWT 准备
├── .githooks/pre-commit                 # Git 提交门禁
├── .env                                 # 环境变量（不提交）
├── .env.example                         # 环境变量模板
├── pyproject.toml                       # 项目配置与依赖
├── uv.lock
├── CLAUDE.md                            # 本文件
└── README.md
```

## 6. RAG 管道 v4 架构

```
用户提问
    ↓
[0] 关键词分类 → 路由到 5 个 Collection 之一
    ↓
[1] 快速预检: search_vector_only(k=3)
    │  若 cos_sim < 0.30 → 直接拒答（零 LLM 调用）
    ↓
[2] 混合检索: search_hybrid(k=2)
    │  ├─ 向量 top-4 + BM25 top-4 → RRF 融合(K=60) → top-2
    │  └─ _resolve_parents(): 子块 → 完整父块替换
    ↓
[3] 质量门控（基于向量余弦相似度）:
    │  < 0.01 → reject（拒答）
    │  0.01~0.18 → weak（加免责声明）
    │  > 0.18 → ok
    ↓
[4] 对话摘要压缩（超 6 轮自动触发）
    ↓
[5] 注入 System Prompt（含用户档案）→ qwen-turbo SSE 流式生成
    ↓
[6] 前端 marked.js 渲染 + STARTJSON 解析 → 训练计划卡片
```

### 关键检索参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `RAG_K` | 2 | 检索返回文档数 |
| `RAG_MAX_CHUNK` | 150 | 每条引用截断字数 |
| `RRF_K` | 60 | RRF 融合平滑常数 |
| `RAG_FAST_REJECT` | 0.30 | 快速预检拒答阈值 |
| `RAG_REJECT_THRESHOLD` | 0.01 | 质量门控拒答阈值 |
| `RAG_WEAK_THRESHOLD` | 0.18 | 质量门控弱关联阈值 |
| `RAG_HISTORY_ROUNDS` | 6 | 对话轮数上限（超出触发摘要） |

### 已移除的组件（v3→v4，评估数据驱动）

| 组件 | 原因 | Hit@3 影响 |
|------|------|------------|
| ❌ HyDE 假设文档 | 省一次 LLM 调用 | 96.7% → 93.3% (-3.4pp) |
| ❌ 多 Query 扩展 | 负优化 | 93.3% → 86.7% |
| ❌ gte-rerank-v2 重排序 | 过拟合 | 93.3% → 70.0% |

### 父子切割（Parent-Child Chunking）

```
原文档 ## 节
    ↓
父块 (Parent): 完整 ## 节 → 存 chroma_db/parents.json，喂给 LLM
子块 (Child): 300 字 + 30 字重叠 → 存 ChromaDB，用于检索
    ↓
检索命中子块 → _resolve_parents() → 按 parent_id 去重替换 → LLM 获得完整上下文
```

### 多 Collection 分类

| Collection | 来源文件 | 用途 |
|------------|----------|------|
| `aifitbot_actions` | actions.md | 训练动作检索 |
| `aifitbot_principles` | principles.md | 训练原理检索 |
| `aifitbot_nutrition` | nutrition.md | 营养饮食检索 |
| `aifitbot_stretching` | stretching.md, recovery.md | 拉伸恢复检索 |
| `aifitbot_general` | **全部 10 个文件副本** | 兜底检索 + 其他专题 |

## 7. 架构分层原则

```
routes (表现层) → services (业务层) → repositories (数据层) → models (数据模型)
```

- **routes**: 处理 HTTP 请求/响应，参数校验，权限检查。不包含业务逻辑。
- **services**: 业务逻辑、编排调用。不直接操作数据库。
- **repositories**: 封装数据库和向量库的 CRUD 操作。
- **models**: Pydantic/SQLModel 数据模型定义。

依赖规则：上层可以调用下层，下层绝不能调用上层。

## 8. 开发协作规范

### 8.1 技术决策流程

> **在整个项目开发过程中，所有技术相关决策必须遵循以下流程：**
>
> 1. 开发者(Claude) 列出 2-3 个可选方案，每个方案附带优劣势说明和推荐理由
> 2. 项目负责人(Merlin-fg) 审阅方案并做出选择
> 3. 开发者按照选择执行，不可自行决定技术方案
>
> **此规则适用于：** 框架选择、库选择、架构设计、数据库设计、API 设计、部署方案、以及任何影响项目走向的技术决策。
>
> **不适用：** Bug 修复、代码格式调整、变量命名等不影响架构的细粒度实现细节。

### 8.2 代码规范

- Python 3.13+，使用 type hints
- 所有函数包含 docstring（Google 风格）
- 使用 `logging` 模块记录关键操作（含 [TIMING] 分步性能日志）
- 敏感信息（API Key、密码）不硬编码，使用 `.env`
- 检索/RAG 组件变更前需参考评估数据（scripts/eval_rag.py）
- 每完成一个模块，必须可独立验证（运行并通过基本测试）

### 8.3 提交规范

- 一个功能模块一个 commit
- Commit message 格式：`类型: 简短描述`（如 `feat: 添加用户注册接口`）
- 类型：`feat` / `fix` / `refactor` / `docs` / `test` / `chore`

## 9. 开发阶段记录

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 1 | 项目初始化 + 技术选型确定 | ✅ 完成 |
| Phase 2 | 用户认证系统 | ✅ 完成 |
| Phase 3 | 知识库管理（管理员） | ✅ 完成 |
| Phase 4 | RAG 问答核心 | ✅ v4 极简版 |
| Phase 5 | 会话管理 | ✅ 完成 |
| Phase 6 | 前端界面 | ✅ 完成 |
| Phase 7 | 性能优化（93s→3s）+ 评估体系 | ✅ 完成 |
| Phase 8 | 测试（58 cases）+ 知识库扩充（4→10 文件） | ✅ 完成 |

---

*最后更新：2026-07-22*
