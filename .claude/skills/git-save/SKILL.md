---
name: git-save
description: 一键存档——将当前代码变更提交并推送到 GitHub，防止代码丢失。
---

# Git Save Skill

一键保存代码到 GitHub，无需懂 Git。

## 触发

```
/git-save                    → 自动提交所有变更并推送
/git-save "修复了登录bug"    → 用自定义说明存档
```

## 执行流程

1. **检查状态** — `git status` 看有哪些文件变了
2. **展示变更** — 列出所有修改/新增/删除的文件，用中文告诉用户
3. **质量门禁** — 先检查通行证是否存在：
   - 如果 `.gate/tester.pass` 和 `.gate/quality.pass` 都存在且 `passed: true` → 继续
   - 如果缺失或过期 → 提示用户先运行 `/gitcommit-agent` 或手动调用 tester + quality-engineer
4. **确认存档说明** — 如果用户没写说明，自动生成一个（如"存档 2026-07-20"）
5. **提交 + 推送** — `git add` → `git commit`（pre-commit hook 验证通行证）→ `git push`
6. **清理通行证** — push 成功后立即删除 `.gate/tester.pass` 和 `.gate/quality.pass`
7. **报告结果** — 告诉用户推送成功，附上 GitHub 链接

## 规则

- 只提交和推送，**绝不做 rebase、reset、force push 等危险操作**
- 推送前先展示变更文件列表，让用户知道推了什么
- 如果工作区是干净的（没东西可存），直接告知，不做空提交
- Commit message 用中文，格式：`存档: <简短说明>`
- push 成功后必须删除通行证，不留旧证

## 输出示例

```
📋 检测到以下变更：
  ✏️ 修改: src/config.py
  ✏️ 修改: src/templates/chat.html
  ➕ 新增: src/services/new_service.py

📝 存档说明: 修复了登录bug

✅ 已提交并推送到 GitHub
🔗 https://github.com/Merlin-fg/aifitbot
```
