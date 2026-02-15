# ContentForge AI Skills

> **世界级内容生成工厂** - 交互式引导用户选择内容风格、配图风格、目标平台，整合项目级、用户级和外部 Skills

## 🏭 核心特性

- **智能 Skills 发现**：自动扫描项目级、用户级、插件级 Skills
- **交互式选择**：8种内容风格、6种配图风格、8个目标平台
- **外部 Skills 集成**：一键安装 GitHub/skills.rest 上的 Skills
- **自动化执行**：选定的 Skills 自动协同工作

---

## 快速开始

### 交互式使用（推荐）

```
用户: 帮我写一篇关于"RAG技术深度解析"的文章

Claude: 🔍 分析话题... 检测到技术/AI类话题

        📝 推荐内容风格: 技术博客 ⭐
        📱 推荐平台: 知乎、微信公众号
        🎨 推荐配图: 暗黑极简 ⭐

📝 可用内容风格:
├─ [1] 技术博客 (tech_blog)     - 简洁直接，干货分享 ⭐推荐
├─ [2] 微信公众号 (weixin)      - 专业权威，深度分析
├─ [3] 小红书 (xiaohongshu)     - emoji丰富，轻松活泼
├─ [4] 视频脚本 (video_script)  - 口语化，适合短视频
├─ [5] 营销文案 (marketing)     - 转化驱动，销售导向
├─ [6] 学术论文 (academic)      - 严谨正式，引用规范
├─ [7] SEO内容 (seo)            - 关键词优化，搜索引擎友好
└─ [8] 邮件序列 (email)         - 专业友好，自动化流程

选择 (1-8): 1

📱 可用平台:
├─ [1] 知乎 (zhihu)            - 1图，16:9，问答干货 ⭐推荐
├─ [2] 微信公众号 (weixin)      - 1图，21:9，深度分析 ⭐推荐
├─ [3] Twitter/X (twitter)     - Thread，口语化
├─ [4] 小红书 (xiaohongshu)     - 9图，3:4，emoji风格
├─ [5] LinkedIn (linkedin)     - 1图，1.91:1，职业风
├─ [6] 抖音 (douyin)           - 1图，9:16，短视频脚本
├─ [7] 微博 (weibo)            - 9图，1:1，热点话题
└─ [8] Medium (medium)         - 1图，16:9，英文深度

选择 (多选用逗号分隔): 1,2,3

🎨 可用配图风格:
├─ [1] 暗黑极简 (dark_minimal)  - 深色背景，稀疏光点 ⭐推荐
├─ [2] 科技抽象 (tech_abstract) - 代码流，电路纹理
├─ [3] 柔和渐变 (soft_gradient) - 温暖色调，舒适感
├─ [4] 手绘插画 (illustration)  - IP形象，趣味性强
├─ [5] 专业商务 (professional)  - 简洁专业，企业感
└─ [6] 跳过配图

选择 (1-6): 1

⏳ 正在生成...
✅ 完成! 输出到: data/topic/20260214/rag-技术深度解析/
```

---

## 项目本地 Skills

### 📝 内容创作 (8种)

| Skill | 用途 | 推荐场景 |
|-------|------|---------|
| `content-research-writer` | 深度研究与写作 | 技术文章、教程、指南 |
| `copywriting` | 营销文案 | 产品推广、落地页 |
| `scriptwriting` | 剧本创作 | 视频脚本、短视频 |
| `social-content` | 社交媒体内容 | 小红书、微博 |
| `email-sequence` | 邮件序列 | Newsletter、培育邮件 |
| `writing-clearly-and-concisely` | 清晰简洁写作 | 内容润色优化 |
| `scientific-writing` *(外部)* | 学术论文 | 论文、研究报告 |
| `seo-content` *(外部)* | SEO内容 | 网站内容、搜索优化 |

### 📱 平台发布 (8个)

| 平台 | 图数 | 比例 | 风格特点 |
|------|------|------|---------|
| 小红书 | 9张 | 3:4 | emoji丰富，口语化 |
| 微信公众号 | 1张 | 21:9 | 专业权威，深度分析 |
| 知乎 | 1张 | 16:9 | 问答干货，深度分析 |
| Twitter/X | Thread | 16:9 | Thread格式，口语化 |
| 抖音 | 1张 | 9:16 | 短视频脚本 |
| LinkedIn | 1张 | 1.91:1 | 职业专业 |
| 微博 | 9张 | 1:1 | 热点话题，短平快 |
| Medium | 1张 | 16:9 | 英文深度，故事化 |

### 🎨 配图风格 (6种)

| 风格 | 特点 | 适合内容 |
|------|------|---------|
| 科技抽象 | 代码流、电路纹理 | 技术、AI、编程 |
| 暗黑极简 | 深色背景、稀疏光点 | 专业、高端、架构 |
| 柔和渐变 | 温暖色调、舒适感 | 生活、情感、故事 |
| 手绘插画 | IP形象、趣味性强 | 小红书、轻松、趣味 |
| 专业商务 | 简洁专业、企业感 | 商务、B2B、报告 |
| 3D渲染 | 立体质感、现代感 | 产品、科技、未来 |

### ⚡ 增强功能 (6种)

| 增强项 | 用途 |
|--------|------|
| 技术图表 | Mermaid/ASCII图表 |
| 文案编辑 | 语法、风格优化 |
| 心理触发器 | 营销心理学应用 |
| 事实核查 | 双阶段核查 |
| SEO优化 | 关键词、元数据 |
| 多语言翻译 | 中/英/日/韩 |

---

## 外部 Skills 集成

### 🚀 发布工具

| Skill | 来源 | 功能 | 安装命令 |
|-------|------|------|---------|
| **微信公众号发布器** | GitHub | 一键发布到公众号草稿 | `npx skill add github:iamzifei/wechat-article-publisher-skill` |
| **小红书发布器** | GitHub | 一键发布到小红书 | `npx skill add github:iamzifei/red-publisher-skill` |
| **营销技能包** | GitHub | 营销任务AI技能 | `npx skill add github:coreyhaines31/marketingskills` |
| **学术写作** | GitHub | 学术稿件写作 | `npx skills add K-Dense-AI/claude-scientific-skills --skill scientific-writing` |

### 一键安装脚本

```bash
# 安装所有外部 Skills
npx skill add github:iamzifei/wechat-article-publisher-skill
npx skill add github:iamzifei/red-publisher-skill
npx skill add github:coreyhaines31/marketingskills
npx skills add K-Dense-AI/claude-scientific-skills --skill scientific-writing
```

---

## 核心文件

```
.claude/skills/
├── skills-registry.json    # Skills 注册表（所有选项定义）
├── external-skills.json    # 外部 Skills 注册表
├── topic-creator/          # 主题创作 Skill
│   └── SKILL.md           # 交互式工作流文档
└── README.md              # 本文件
```

---

## 交互式工作流

```
1. 输入话题
   ↓
2. 扫描 skills-registry.json
   ↓
3. 智能匹配推荐（标记 ⭐）
   ↓
4. 【交互】选择内容风格 (8种)
   ↓
5. 【交互】选择目标平台 (8个)
   ↓
6. 【交互】选择配图风格 (6种)
   ↓
7. 【交互】选择增强功能 (6种)
   ↓
8. 【交互】选择发布工具 (可选)
   ↓
9. 执行工作流
   ↓
10. 输出到 data/topic/YYYYMMDD/{topic}/
```

---

## 💡 使用示例

### 交互式（推荐）

```
"帮我写一篇关于'大模型微调技术'的文章"
"生成一篇'AI Agent 开发实战'的内容"
"写一个'产品发布'的营销文案"
```

### 指定风格

```
"用小红书风格写一篇'AI绘画入门'"
"生成'营销心理学'的视频脚本"
"为我的 SaaS 产品创建邮件激活序列"
```

### CLI 命令

```bash
# 交互式触发 topic-creator
python src/main.py --mode topic --topic "RAG技术深度解析"

# 指定参数（跳过交互）
python src/main.py --mode topic \
    --topic "AI绘画入门" \
    --content-style xiaohongshu \
    --image-style illustration \
    --platforms xiaohongshu

# 多平台适配
python src/main.py --mode topic \
    --topic "AI Agent开发实战" \
    --platforms xiaohongshu,weixin,zhihu,twitter

# 技术文章 + 图表
python src/main.py --mode topic \
    --topic "Transformer架构详解" \
    --content-style tech_blog \
    --enhancements tech_diagrams,fact_check
```

---

## 智能推荐系统

根据话题关键词自动推荐：

| 话题关键词 | 推荐内容风格 | 推荐平台 | 推荐配图 |
|-----------|------------|---------|---------|
| 技术/AI/编程 | 技术博客 ⭐ | 知乎、微信、Twitter | 暗黑极简 ⭐ |
| 小红书/种草 | 小红书风格 ⭐ | 小红书 | 手绘插画 ⭐ |
| 视频/抖音 | 视频脚本 ⭐ | 抖音、小红书 | 柔和渐变 |
| 营销/产品 | 营销文案 ⭐ | 小红书、微博 | 专业商务 |
| 学术/论文 | 学术论文 ⭐ | 知乎、Medium | 暗黑极简 |

---

## Skills 发现来源

1. **项目级 Skills** - `.claude/skills/` 目录
2. **用户级 Skills** - `~/.claude/skills/` 目录
3. **插件级 Skills** - 已安装的插件
4. **外部 Skills** - GitHub / skills.rest (87,000+)

---

**配置时间**: 2026-02-14
**版本**: v3.0 (世界级内容生成工厂)
**项目路径**: /Users/z/Documents/work/content-forge-ai
