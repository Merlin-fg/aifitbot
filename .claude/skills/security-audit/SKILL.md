---
name: security-audit
description: 安全审查——检查密钥泄露、SQL注入、XSS、认证缺陷等 8 类安全风险，输出分级报告。
---

# Security Audit Skill

OWASP 风格安全审查，覆盖 8 个维度，每个问题带风险等级和修复建议。

## 触发

```
/security-audit               → 审查全项目
/security-audit src/routes    → 审查指定目录
/security-audit src/main.py   → 审查指定文件
```

## 审查维度

| # | 维度 | 检查内容 |
|---|------|----------|
| 1 | 密钥泄露 | API Key、JWT Secret、密码、Token 是否硬编码在源码中 |
| 2 | SQL 注入 | 是否使用字符串拼接构建 SQL（f-string、+、%）而非参数化查询 |
| 3 | XSS | Jinja2 `|safe` 是否用于未转义的用户输入、HTML 拼接是否安全 |
| 4 | 认证缺陷 | 弱默认密码、Token 无过期、Cookie 缺少 httponly/secure/samesite |
| 5 | 文件上传 | 文件类型白名单校验、大小限制、路径穿越防护 |
| 6 | 日志泄露 | 日志中是否打印密码、Token、API Key 等敏感信息 |
| 7 | 依赖漏洞 | pyproject.toml 中依赖版本是否过旧或有已知 CVE |
| 8 | 配置安全 | .env / config 中是否有 DEBUG=True、不安全的默认值 |

## 执行流程

1. 读取目标文件
2. 按 8 个维度逐项审查
3. 每项标注：🔴严重 / 🟡警告 / 🟢建议 + 具体行号 + 修复代码
4. 输出汇总表

## 输出格式

```
═══════════════════════════════════
  AIFitBot 安全审查报告
═══════════════════════════════════

🔴 严重 (需立即修复)
  config.py:50 — ADMIN_PASSWORD 有硬编码默认值
  → 修复: ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
           if not ADMIN_PASSWORD: raise ValueError(...)

🟡 警告 (建议修复)
  main.py:128 — Cookie 缺少 secure=True
  → 修复: response.set_cookie(..., secure=True)

🟢 建议 (可选优化)
  main.py:47 — 异常处理过于宽泛

───────────────────────────────────
汇总: 2严重  4警告  5建议
═══════════════════════════════════
```
