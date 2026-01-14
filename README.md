# ContentForge AI v2.5

> 🚀 AI驱动的多平台内容自动化生产工厂

> **内容工厂模式**：定时产出简报 + 批量生成系列 + 手动生成长文/社交媒体内容

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-LangGraph-green.svg)](https://python.langchain.com/)
[![100 Episodes](https://img.shields.io/badge/Episodes-100-blue.svg)](config/blog_topics_100_complete.json)

## ✨ 核心功能

**ContentForge AI** 是一个基于 LangChain/LangGraph 的智能内容生产系统，专注于高质量内容产出。

### 🎯 四大内容生成模式

#### 1️⃣ **Auto 模式** - AI热点自动追踪
- 实时获取7个数据源的AI技术热点
- 自动生成热点简报（含原始链接）
- 适合每日定时任务

#### 2️⃣ **Series 模式** - 100期技术博客系列
- 系统化生成100期技术博客
- 覆盖10大系列：LLM原理、RAG实战、Agent开发等
- 批量生成，进度追踪

#### 3️⃣ **Custom 模式** - 自定义内容生成
- 根据关键词/要求生成长文本
- 支持风格转换（技术文→科普文）
- 支持重新表述、扩写等操作

#### 4️⃣ **Refine 模式** - 多平台内容精炼
- 将已有文本精炼为小红书笔记
- 生成Twitter Thread
- 生成微信公众号HTML（可直接复制粘贴）

### 🌟 核心优势

- ✅ **内容工厂** - 专注内容产出，不包含发布功能
- ✅ **实时热点** - 基于7个数据源的真实热门内容
- ✅ **精简高效** - 移除冗余功能，专注核心内容生成
- ✅ **四模合一** - Auto + Series + Custom + Refine
- ✅ **成本可控** - 免费数据源 + GLM-4.7模型

## 🚀 快速开始

### 安装

```bash
# 1. 克隆项目
git clone https://github.com/yourusername/content-forge-ai.git
cd content-forge-ai

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置API Key
cp .env.example .env
# 编辑.env，添加 ZHIPUAI_API_KEY
```

### 运行

系统使用统一入口 `src/main.py`，支持四种模式：

---

## 📖 四种模式详细使用指南

### 1️⃣ Auto 模式 - AI热点自动追踪

**用途**：基于AI热点自动生成简报（每日定时任务）

```bash
# 设置PYTHONPATH（替换为你的实际项目路径）
export PYTHONPATH=/Users/z/Documents/work/content-forge-ai

# 运行自动模式
python src/main.py --mode auto --once

# 可选：指定topic作为文件标识
python src/main.py --mode auto --topic "AI技术"
```

**输出位置**：`data/daily/YYYYMMDD/`
- `raw/` - AI热点原始数据（JSON格式）
- `digest/` - 热点简报（Markdown + JSON）

**说明**：
- Auto 模式专注于热点追踪和简报生成
- 不生成长文本、小红书笔记、Twitter帖子
- 如需生成完整内容，请使用 Custom、Refine 或 Series 模式

**适合场景**：
- 每日定时任务（如cron每天早上3点执行）
- 自动追踪AI技术热点
- 快速了解行业动态

---

### 2️⃣ Series 模式 - 100期技术博客系列

**用途**：系统化生成100期技术博客，覆盖10大系列

```bash
# 查看生成进度
python src/main.py --mode series --progress

# 生成指定集数
python src/main.py --mode series --episode 1

# 生成整个系列（如series_1）
python src/main.py --mode series --series series_1

# 批量生成（如第1-10期）
python src/main.py --mode series --all --start 1 --end 10
```

**100期内容规划**：
- 系列1（1-10期）：LLM原理基础
- 系列2（11-18期）：RAG技术实战
- 系列3（19-26期）：Agent智能体开发
- 系列4（27-32期）：提示工程
- 系列5（33-40期）：模型部署与优化
- 系列6（41-50期）：多模态与前沿技术
- 系列7（51-60期）：AI编程与开发工具
- 系列8（61-70期）：AI数据处理与工程
- 系列9（71-85期）：AI应用场景实战
- 系列10（86-100期）：AI基础设施与架构

**输出位置**：`data/series/{系列ID}/episode_{xxx}/longform/`
- 每期生成一篇高质量长文本技术文章（9000-13000字）
- Markdown 格式，可直接用于技术博客

**适合场景**：
- 系统化学习AI技术
- 构建技术博客内容库
- 批量生成系列教程

---

### 3️⃣ Custom 模式 - 自定义内容生成

**用途**：根据关键词/要求生成长文本技术文章

```bash
# 简单关键词模式
python src/main.py --mode custom --topic "RAG技术原理与实战"

# 详细描述模式
python src/main.py --mode custom \
  --topic "RAG技术" \
  --prompt "详细介绍架构、核心组件、向量数据库选择，以及生产环境最佳实践"

# 指定参数
python src/main.py --mode custom \
  --topic "Agent开发" \
  --audience "AI工程师" \
  --words 12000 \
  --style technical  # technical/practical/tutorial
```

**输出位置**：`data/custom/YYYYMMDD_HHMMSS_{topic}/`
- `longform/` - 长文本文章
- `xiaohongshu/` - 小红书笔记
- `twitter/` - Twitter帖子

**适合场景**：
- 快速生成指定主题的技术文章
- 根据具体要求定制内容
- 构建个人技术博客内容

---

### 4️⃣ Refine 模式 - 多平台内容精炼

**用途**：将已有高质量文本精炼为可直接复制粘贴的多平台内容

```bash
# 精炼为所有平台
python src/main.py --mode refine --input article.md

# 指定目标平台
python src/main.py --mode refine --input article.md --platforms wechat xiaohongshu
```

**输出内容**：

| 平台 | 输出 | 使用方式 |
|------|------|----------|
| **微信公众号** | `wechat/article.html` | 直接复制HTML代码到公众号编辑器源码模式 |
| **小红书** | `xiaohongshu/note.md` | 直接复制内容到小红书编辑器 |
| **Twitter** | `twitter/thread.md` | 直接复制内容分条发布 |

**输出位置**：`data/refine/{source_name}/`

**适合场景**：
- 一文多发（将技术文章转换为多平台内容）
- 快速生成社交媒体内容
- 批量处理已有文章库

---

### 查看输出

```bash
# Auto模式输出
ls -la data/daily/20260107/

# Series模式输出
ls -la data/series/series_1_llm_foundation/episode_001/

# Custom模式输出
ls -la data/custom/

# Refine模式输出
ls -la data/refine/
```

**查看具体内容**：
```bash
# 查看热点简报
cat data/daily/20260107/digest/digest_*.md

# 查看专业文章
cat data/series/series_1_llm_foundation/episode_001/longform/article.md

# 查看小红书笔记
cat data/series/series_1_llm_foundation/episode_001/xiaohongshu/note.md

# 查看Twitter帖子
cat data/series/series_1_llm_foundation/episode_001/twitter/thread.md

# 查看微信公众号HTML
cat data/refine/my_article/wechat/article.html
```

---

## 🚀 部署到生产环境

### 快速部署

```bash
# 1. 克隆项目
git clone https://github.com/Ming-H/content-forge-ai.git
cd content-forge-ai

# 2. 安装依赖
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. 配置环境变量
cp .env.example .env
# 编辑.env文件，填入你的API密钥

# 4. 测试运行
python src/main.py --mode auto --once

# 5. 设置定时任务（每天早上3点执行）
crontab -e
# 添加：0 3 * * * /path/to/content-forge-ai/run_and_commit.sh
```

**高级定时任务配置**：

```bash
# 每天自动生成热点内容（默认）
0 3 * * * /path/to/content-forge-ai/run_and_commit.sh

# 或者设置为系列模式
CONTENT_FORGE_MODE=series SERIES_EPISODE=1 0 3 * * * /path/to/content-forge-ai/run_and_commit.sh
```

### 🔐 环境变量配置

创建 `.env` 文件（基于 `.env.example`）：

```bash
# 必需的密钥
ZHIPUAI_API_KEY=your_zhipuai_api_key_here    # 智谱AI密钥（获取：https://open.bigmodel.cn/）

# 可选的密钥
OPENAI_API_KEY=your_openai_api_key_here      # OpenAI密钥
GEMINI_API_KEY=your_gemini_api_key_here      # Google Gemini密钥（用于图片生成）
NEWSAPI_KEY=your_newsapi_key_here            # NewsAPI密钥
```

**获取API密钥**：
- [智谱AI](https://open.bigmodel.cn/) - 必需，支持国产大模型GLM-4.7
- [OpenAI](https://platform.openai.com/api-keys) - 可选，备用LLM提供商
- [Google AI Studio](https://makersuite.google.com/app/apikey) - 可选，用于图片生成

## 📂 输出结构

```
data/
├── daily/                    # Auto模式输出（只保存简报）
│   └── 20260113/             # 按日期分层
│       ├── raw/              # AI热点原始数据
│       └── digest/           # 热点简报
│
├── series/                   # Series模式输出（只生成长文本）
│   ├── series_1_llm_foundation/
│   │   ├── episode_001/
│   │   │   └── longform/     # 长文本文章
│   │   ├── episode_002/
│   │   ├── ...
│   │   └── series_metadata.json
│   ├── series_2_rag_technique/
│   ├── series_3_agent_development/
│   └── ... (共10个系列)
│
├── custom/                   # Custom模式输出
│   └── YYYYMMDD_HHMMSS_topic/
│       ├── longform/         # 长文本文章
│       ├── xiaohongshu/      # 小红书笔记
│       └── twitter/          # Twitter帖子
│
└── refine/                   # Refine模式输出
    └── {source_name}/
        ├── raw/              # 原始输入
        ├── wechat/           # 微信公众号HTML
        ├── xiaohongshu/      # 小红书笔记
        └── twitter/          # Twitter帖子

logs/                        # 日志按日期分层
└── 20260113/
    └── app.log
```

## 🤖 工作流程

### Auto 模式 - 每日简报自动生成
```
AI热点获取（7个数据源）
  ↓
热点汇总 → 简报生成
  ↓
保存到 data/daily/YYYYMMDD/
  ├── raw/      # 原始数据
  └── digest/   # 热点简报
```

### Series 模式 - 系列批量生成
```
读取100期配置
  ↓
长文本生成（9000-13000字技术博客）
  ↓
保存到 data/series/{系列ID}/episode_{xxx}/longform/
```

### Custom 模式 - 自定义内容生成
```
根据关键词/要求
  ↓
长文本生成 → 小红书精炼 → Twitter生成 → 标题优化 → 配图提示词
  ↓
保存到 data/custom/YYYYMMDD_HHMMSS_topic/
```

### Refine 模式 - 多平台内容精炼
```
读取已有文章
  ↓
微信公众号HTML生成
小红书笔记精炼
Twitter Thread生成
  ↓
保存到 data/refine/{source_name}/
```

## 📊 AI热点数据源

### 免费无需配置（7个）✅

| 数据源 | 内容 | 实时性 |
|--------|------|--------|
| Hacker News | 技术新闻 | ⚡⚡⚡ |
| arXiv | AI学术论文 | ⚡⚡ |
| Hugging Face | AI模型趋势 | ⚡⚡⚡ |
| Stack Overflow | 技术问答 | ⚡⚡⚡ |
| Dev.to | 开发者博客 | ⚡⚡ |
| PyPI | Python包统计 | ⚡⚡ |
| GitHub Topics | 开源项目 | ⚡⚡⚡ |

## ⚙️ 配置说明

编辑 `config/config.yaml`：

```yaml
llm:
  provider: "zhipuai"  # 或 "openai"
  zhipuai:
    model: "glm-4.7"  # 最新旗舰模型（2025年12月发布）
    # 其他可选: glm-4-flash（便宜快速）, glm-4-plus（上一代旗舰）
    max_tokens: 8000  # 支持长文本生成
    timeout: 300  # 5分钟超时

agents:
  ai_trend_analyzer:
    enabled: true
    mock_mode: false  # false=真实API, true=模拟数据

  research_agent:  # v2.2新增
    enabled: true
    max_docs_per_topic: 3

  longform_generator:
    enabled: true
    article_length: "long"  # short, medium, long - long生成9000-13000字
    technical_depth: "advanced"  # beginner, intermediate, advanced

  code_review_agent:  # v2.2新增
    enabled: true

  fact_check_agent:  # v2.2新增
    enabled: true

  xiaohongshu_refiner:
    enabled: true
    style: "professional"  # professional, casual, humorous
    content_density: "dense"  # light, medium, dense

  twitter_generator:
    enabled: true
    style: "engaging"  # engaging, professional, casual
    thread_mode: true
    max_tweets: 8  # thread最多几条推文

  quality_evaluator:
    enabled: true
    min_score: 7.0  # 质量阈值
```

## 🧪 测试

```bash
cd test

# 测试topic参数逻辑
python test_topic_logic.py

# 测试存储结构
python test_storage.py

# 测试AI热点获取
python test_ai_trends.py --topic "AI"

# 测试单个数据源
python test_ai_trends.py --source hackernews
```

## 📈 性能指标

- **热点获取**：30-90秒（11个数据源）
- **深度研究**：90-150秒（Web搜索 + LLM分析）
- **长文本生成**：5-8分钟（分阶段生成9000-13000字）
- **小红书精炼**：40-100秒
- **Twitter生成**：20-40秒
- **总耗时**：10-15分钟

**内容质量**：
- 长文本：9000-13000字专业深度分析
- 小红书：3000-3500字干货笔记
- Twitter：5-8条推文Thread

**成本**（GLM-4-plus）：
- 每次运行：¥3-5
- Token使用：30000-50000

**优化建议**：
- 使用 `glm-4-flash` 降低成本约80%
- 减少数据源（只用hackernews + arxiv）
- 禁用不需要的Agent

## ❓ 常见问题

### Q: 不指定topic，内容会重复吗？

A: 不会。每次运行获取的是**实时热点**，不同时间的热点不同。

### Q: 如何只生成简报？

A: 编辑 `config/config.yaml`：
```yaml
agents:
  longform_generator:
    enabled: false
  xiaohongshu_refiner:
    enabled: false
  twitter_generator:
    enabled: false
```

### Q: 配图提示词如何使用？

A: 打开 `data/YYYYMMDD/xiaohongshu/prompts_*.txt`，复制中文提示词，粘贴到支持中文的AI绘图工具（文心一格、通义万相、即梦AI等）。

## 📚 详细文档

- **[CLAUDE.md](CLAUDE.md)** - 开发者指南和架构说明
- **[test/README.md](test/README.md)** - 测试文件说明

## 📊 四种模式对比

| 模式 | 输入 | 输出 | 用途 |
|------|------|------|------|
| **Auto** | （自动获取热点） | 原始数据 + 简报 | 每日热点追踪 |
| **Series** | 100期配置 | 长文本技术博客 | 系统化内容库 |
| **Custom** | 关键词/主题 | 长文本 + 社交内容 | 按需生成 |
| **Refine** | 已有文章 | 多平台精炼内容 | 一文多发 |

## 🔄 版本历史

### v2.6 (2026-01-14) 🆕
- ✅ **Auto 模式优化** - 只保留热点追踪和简报生成
- ✅ **Series 模式优化** - 只生成长文本技术博客，移除社交内容
- ✅ **Custom 模式** - 根据关键词生成长文本 + 社交内容
- ✅ **Refine 模式** - 多平台内容精炼（微信公众号HTML、小红书、Twitter）
- ✅ **WechatGenerator** - 新增微信公众号生成Agent，输出可直接复制粘贴的HTML
- ✅ **四模合一** - Auto + Series + Custom + Refine 统一入口
- ✅ **存储扩展** - 新增 `data/custom/` 和 `data/refine/` 存储目录

### v2.5 (2026-01-09)
- ✅ **100期技术博客系列** - 系统化规划100期技术内容，覆盖10大系列
- ✅ **存储结构优化** - 两种模式独立存储：daily/、series/
- ✅ **SeriesStorage** - 新的系列存储管理器
- ✅ **SeriesOrchestrator** - 100期系列生成协调器
- ✅ **SeriesMetadata** - 元数据管理系统，支持进度追踪
- ✅ **StorageFactory** - 统一存储工厂模式
- ✅ 配置文件 `blog_topics_100_complete.json` - 100期完整规划

### v2.2 (2026-01-08)
- ✅ 新增ResearchAgent - Web搜索增强深度研究
- ✅ 新增CodeReviewAgent - 代码审查和质量保证
- ✅ 新增FactCheckAgent - 事实核查
- ✅ 长文本分阶段生成 - 避免超时，支持9000-13000字
- ✅ 内容质量显著提升 - 专业度+深度+详细度
- ✅ 修复章节路由Bug - 技术对比正确映射

### v2.1 (2026-01-07)
- ✅ 新增Twitter Generator Agent
- ✅ 优化专业文章为微信公众号格式
- ✅ Topic参数变为可选（系统自动从热点生成）
- ✅ 更新存储结构（5个目录）
- ✅ 日志按日期分层存储
- ✅ 测试文件统一管理

### v2.0 (2026-01-06)
- ✅ 按日期分层存储
- ✅ 热点简报Agent
- ✅ 11个AI数据源集成

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE)

---

**最后更新**：2026-01-14
**版本**：v2.6
**Made with ❤️ by ContentForge AI Team**
