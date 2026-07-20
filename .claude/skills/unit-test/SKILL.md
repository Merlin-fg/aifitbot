---
name: unit-test
description: 为 AIFitBot 生成、执行单元测试，输出分层测试报告。Mock LLM/ChromaDB，无需 API Key。
---

# Unit Test Skill

为 AIFitBot 项目生成、执行单元测试，输出测试报告。使用分层 Mock 策略，不消耗 LLM API 额度。

## 触发

```
/unit-test              → 运行全部已有测试
/unit-test core         → 只测 models + repositories（纯逻辑，毫秒级）
/unit-test services     → 测 services（Mock LLM/ChromaDB，不花钱）
/unit-test api          → 测 API 端点（FastAPI TestClient）
```

## 执行流程

1. 确认 pytest 已安装
2. 运行 `uv run python -m pytest tests/ -v --tb=short`
3. 输出分层报告

## 报告格式

```
═══════════════════════════════════
  AIFitBot 单元测试报告
═══════════════════════════════════
层级         测试数   通过   失败
──────────────────────────────────
models        10     10     0
repositories  13     13     0
services      14     14     0
API routes     8      5     0
──────────────────────────────────
合计          45     42     0
═══════════════════════════════════
```

失败项列出具体原因和修复建议。
