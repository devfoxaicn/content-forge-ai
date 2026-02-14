# ContentForge AI Skills 完整目录

> **版本**: v1.0
> **更新日期**: 2026-02-14
> **总计**: 30+ Skills

本文档汇总了 ContentForge AI 项目中所有可用的 Skills，包括项目本地、用户级和插件级。

---

## 📊 Skills 概览

| 来源 | 数量 | 说明 |
|------|------|------|
| **项目本地** | 13 | 存储在 `.claude/skills/` |
| **用户级** | 1 | `commit` - Git 提交 |
| **插件级** | 16+ | code-review, vercel, scientific-skills 等 |
| **总计** | **30+** | - |

---

## 📂 项目本地 Skills (13个)

### 📝 内容创作类

| Skill | 用途 | 调用方式 |
|-------|------|----------|
| **content-research-writer** | 深度研究与写作，带引用 | `/content-research-writer` |
| **writing-clearly-and-concisely** | 清晰简洁写作 (Strunk原则) | `/writing-clearly-and-concisely` |
| **copywriting** | 营销文案写作 | `/copywriting` |
| **copy-editing** | 文章编辑润色 | `/copy-editing` |
| **scriptwriting** | 中文剧本创作 | `/scriptwriting` |

### 📱 平台适配类

| Skill | 用途 | 调用方式 |
|-------|------|----------|
| **platform-adaptation** | 多平台内容适配 (小红书/公众号/知乎等) | `/platform-adaptation` |
| **social-content** | 社交媒体内容 | `/social-content` |
| **email-sequence** | 邮件序列 | `/email-sequence` |
| **x-article-publisher** | 发布到 X (Twitter) Articles | `/x-article-publisher` |

### 🔧 工具类

| Skill | 用途 | 调用方式 |
|-------|------|----------|
| **marketing-psychology** | 营销心理学 (70+心理模型) | `/marketing-psychology` |
| **notebooklm** | Google NotebookLM 查询 | `/notebooklm` |
| **tech-diagram-generator** | 技术图表生成 (Mermaid/ASCII) | `/tech-diagram-generator` |
| **daily-digest** | 一键生成AI新闻简报 | `/daily-digest` |

---

## 👤 用户级 Skills (1个)

| Skill | 用途 | 调用方式 |
|-------|------|----------|
| **commit** | 规范化 Git 提交 | `/commit` |

---

## 🔌 插件级 Skills (16+个)

### 代码审查

| Skill | 用途 | 调用方式 |
|-------|------|----------|
| **code-review:code-review** | 代码审查 PR | `/code-review:code-review` |

### 部署

| Skill | 用途 | 调用方式 |
|-------|------|----------|
| **vercel:deploy** | 部署到 Vercel | `/vercel:deploy` |
| **vercel:logs** | 查看 Vercel 日志 | `/vercel:logs` |
| **vercel:setup** | Vercel CLI 配置 | `/vercel:setup` |

### 开发辅助

| Skill | 用途 | 调用方式 |
|-------|------|----------|
| **ralph-wiggum:ralph-loop** | 启动 Ralph Wiggum 循环 | `/ralph-wiggum:ralph-loop` |
| **ralph-wiggum:cancel-ralph** | 取消循环 | `/ralph-wiggum:cancel-ralph` |
| **ralph-wiggum:help** | 帮助信息 | `/ralph-wiggum:help` |

### 科学计算 (scientific-skills)

| Skill | 用途 | 调用方式 |
|-------|------|----------|
| **scientific-skills:adaptyv** | 蛋白质实验室平台 | `/scientific-skills:adaptyv` |
| **scientific-skills:aeon** | 时间序列 ML | `/scientific-skills:aeon` |
| **scientific-skills:anndata** | 单细胞数据分析 | `/scientific-skills:anndata` |
| **scientific-skills:arboreto** | 基因调控网络推断 | `/scientific-skills:arboreto` |
| **scientific-skills:astropy** | 天文学 Python 库 | `/scientific-skills:astropy` |
| **scientific-skills:biomni** | 生物医学 AI Agent | `/scientific-skills:biomni` |
| **scientific-skills:biopython** | 分子生物学工具 | `/scientific-skills:biopython` |
| **scientific-skills:bioservices** | 生物信息学服务 | `/scientific-skills:bioservices` |
| **scientific-skills:cellxgene-census** | 单细胞图谱查询 | `/scientific-skills:cellxgene-census` |
| **scientific-skills:cirq** | Google 量子计算 | `/scientific-skills:cirq` |
| **scientific-skills:cobrapy** | 代谢建模 | `/scientific-skills:cobrapy` |
| **scientific-skills:dask** | 分布式计算 | `/scientific-skills:dask` |
| **scientific-skills:datamol** | 药物发现 | `/scientific-skills:datamol` |
| **scientific-skills:deepchem** | 分子 ML | `/scientific-skills:deepchem` |

---

## 🚀 常用 Skills 快速参考

### 内容创作

```bash
# 写技术文章
"使用 content-research-writer 写一篇关于 Transformer 的深度文章"

# 生成技术图表
"使用 tech-diagram-generator 生成 CNN 架构图"

# 编辑润色
"使用 copy-editing 优化这篇文章"
```

### 平台发布

```bash
# 适配小红书
"使用 platform-adaptation 将这篇文章改编到小红书"

# 发布到 X
"使用 x-article-publisher 发布这篇文章"
```

### 项目管理

```bash
# Git 提交
"/commit"

# 代码审查
"/code-review:code-review"

# 部署
"/vercel:deploy"
```

---

## 📁 目录结构

```
content-forge-ai/
├── .claude/
│   └── skills/                      # 项目本地 Skills
│       ├── content-research-writer/
│       │   └── SKILL.md
│       ├── copywriting/
│       │   └── SKILL.md
│       ├── copy-editing/
│       │   └── SKILL.md
│       ├── scriptwriting/
│       │   └── SKILL.md
│       ├── platform-adaptation/
│       │   └── SKILL.md
│       ├── social-content/
│       │   └── SKILL.md
│       ├── email-sequence/
│       │   └── SKILL.md
│       ├── marketing-psychology/
│       │   └── SKILL.md
│       ├── writing-clearly-and-concisely/
│       │   └── SKILL.md
│       ├── notebooklm/
│       │   └── SKILL.md
│       ├── x-article-publisher/
│       │   └── SKILL.md
│       ├── tech-diagram-generator/
│       │   └── SKILL.md
│       ├── daily-digest.md          # 单文件 Skill
│       └── README.md                # Skills 说明
│
└── docs/
    └── SKILLS_CATALOG.md            # 本文档
```

---

## 🔧 如何添加新 Skill

### 方式一：目录形式

```bash
# 创建目录
mkdir -p .claude/skills/my-skill

# 创建 SKILL.md
cat > .claude/skills/my-skill/SKILL.md << 'EOF'
---
name: my-skill
description: 技能描述
---

# My Skill

技能内容...
EOF
```

### 方式二：单文件形式

```bash
# 直接创建 .md 文件
cat > .claude/skills/my-skill.md << 'EOF'
---
name: my-skill
description: 技能描述
---

# My Skill

技能内容...
EOF
```

---

## 📚 相关文档

- [Claude Code Skills 官方文档](https://docs.anthropic.com/claude-code/skills)
- [项目 Skills README](/.claude/skills/README.md)
- [Novel-OS 小说创作系统](/.novel-os/)

---

**维护者**: ContentForge AI
**最后更新**: 2026-02-14
