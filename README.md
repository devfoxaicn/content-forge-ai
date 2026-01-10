# ContentForge AI v2.4

> 🚀 AI驱动的多平台内容自动化生产工厂

> 基于Web搜索的深度研究 + 分阶段生成，打造9000-13000字专业深度分析

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-LangGraph-green.svg)](https://python.langchain.com/)
[![100 Episodes](https://img.shields.io/badge/Episodes-100-blue.svg)](config/blog_topics_100_complete.json)

## ✨ 核心功能

**ContentForge AI** 是一个基于 LangChain/LangGraph 的智能内容生产系统，实现从AI热点追踪到多平台内容发布的全流程自动化。

### 🎯 核心能力

1. **AI热点追踪** - 11个免费数据源，实时获取AI技术热点
2. **热点简报** - 汇总当天热点，生成杂志风格简报（含原始链接）
3. **深度研究** - Web搜索增强，收集官方文档、GitHub、技术博客等资料
4. **专业文章** - 9000-13000字深度技术分析，分阶段生成避免超时
5. **小红书笔记** - 3000-3500字干货风格，含emoji和标签
6. **Twitter帖子** - Thread形式（5-8条推文），精简爆款风格
7. **质量保证** - 代码审查、事实核查、质量评估三重保障

### 🌟 核心优势

- ✅ **零人工干预** - 完全自动化，无需手动指定topic
- ✅ **实时热点** - 基于11个数据源的真实热门内容
- ✅ **多平台适配** - 一次生成，多平台分发
- ✅ **高质量内容** - 13个专业Agent协作生成，基于Web搜索的深度研究
- ✅ **成本可控** - 免费数据源 + 便宜模型选择

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

系统使用统一入口 `src/main.py`，支持两种模式：

**自动模式（基于AI热点）**：
```bash
# 设置PYTHONPATH（替换为你的实际项目路径）
export PYTHONPATH=/Users/z/Documents/work/content-forge-ai

# 运行自动模式
python src/main.py --mode auto --once

# 或指定topic作为文件标识（可选）
python src/main.py --mode auto --topic "AI技术"
```

**系列模式（100期技术博客）**：
```bash
# 查看进度
python src/main.py --mode series --progress

# 生成指定集数
python src/main.py --mode series --episode 1

# 生成整个系列
python src/main.py --mode series --series series_1

# 生成全部100期
python src/main.py --mode series --all --start 1 --end 100
```

### 查看输出

```bash
# 查看存储目录
ls -la data/daily/20260107/       # 自动模式输出
ls -la data/series/series_1_llm_foundation/  # 系列模式输出

# 查看热点简报
cat data/daily/20260107/digest/digest_*.md

# 查看专业文章
cat data/daily/20260107/longform/article_*.md

# 查看小红书笔记
cat data/daily/20260107/xiaohongshu/note_*.md

# 查看Twitter帖子
cat data/daily/20260107/twitter/twitter_*.md
```

## 📚 两种内容生成模式

### 1️⃣ 自动模式（默认）

基于11个AI数据源，自动追踪实时热点并生成内容。适合每日定时任务。

```bash
python src/main.py --mode auto --once
```

### 2️⃣ 系列模式（100期技术博客）

系统化生成100期技术博客，覆盖10大系列从LLM原理到AI基础设施的全栈内容。

```bash
# 查看进度
python src/main.py --mode series --progress

# 生成指定集数
python src/main.py --mode series --episode 1

# 生成整个系列
python src/main.py --mode series --series series_1

# 生成全部100期
python src/main.py --mode series --all --start 1 --end 100
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

**配置文件**：`config/blog_topics_100_complete.json`

### 模式对比

| 特性 | 自动模式 | 系列模式 |
|------|---------|----------|
| 触发方式 | 定时任务 | 手动执行 |
| 数据来源 | AI热点分析 | 100期预设 |
| 存储位置 | `data/daily/日期/` | `data/series/{系列ID}/episode_{xxx}/` |
| 内容特点 | 实时热点 | 系统化教程 |

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
TAVILY_API_KEY=your_tavily_api_key_here      # Tavily搜索密钥（获取：https://tavily.com/）

# 可选的密钥
OPENAI_API_KEY=your_openai_api_key_here      # OpenAI密钥
GEMINI_API_KEY=your_gemini_api_key_here      # Google Gemini密钥（用于图片生成）
NEWSAPI_KEY=your_newsapi_key_here            # NewsAPI密钥
```

**获取API密钥**：
- [智谱AI](https://open.bigmodel.cn/) - 必需，支持国产大模型
- [Tavily](https://tavily.com/) - 必需，用于Web搜索深度研究
- [OpenAI](https://platform.openai.com/api-keys) - 可选
- [Google AI Studio](https://makersuite.google.com/app/apikey) - 可选，用于图片生成

## 📂 输出结构 (v2.4优化)

```
data/
├── daily/                    # 每日热点模式
│   └── 20260107/             # 按日期分层
│       ├── raw/              # AI热点原始数据
│       ├── digest/           # 热点简报
│       ├── longform/         # 长篇文章
│       ├── xiaohongshu/      # 小红书笔记
│       └── twitter/          # Twitter帖子
│
├── series/                   # 100期技术博客系列 (v2.4新增)
│   ├── series_1_llm_foundation/
│   │   ├── episode_001/
│   │   ├── episode_002/
│   │   ├── ...
│   │   └── series_metadata.json
│   ├── series_2_rag_techniques/
│   ├── series_3_agent_development/
│   └── ... (共10个系列)
│
└── archive/                  # 归档内容 (预留)

logs/                        # 日志按日期分层
└── 20260107/
    └── app.log
```

## 🤖 工作流程

```
AI热点获取（11个数据源）
  ↓
热点汇总 → 简报生成 (digest/)
  ↓
筛选TOP 1热点
  ↓
深度研究 (ResearchAgent) - Web搜索收集官方文档、GitHub、技术博客
  ↓
长文本生成 (LongFormGeneratorAgent) - 分阶段生成9000-13000字专业文章
  ↓
质量检查：
  ├─→ CodeReviewAgent (代码审查)
  └─→ FactCheckAgent (事实核查)
  ↓
并行处理：
  ├─→ 小红书精炼 (xiaohongshu/)
  └─→ Twitter生成 (twitter/)
  ↓
标题优化 + 配图提示词 + 质量评估
  ↓
保存到 data/YYYYMMDD/
```

## 📊 AI热点数据源

### 免费无需配置（8个）✅

| 数据源 | 内容 | 实时性 |
|--------|------|--------|
| Hacker News | 技术新闻Top 30 | ⚡⚡⚡ |
| arXiv | AI学术论文 | ⚡⚡ |
| Hugging Face | AI模型趋势 | ⚡⚡⚡ |
| Stack Overflow | 技术问答 | ⚡⚡⚡ |
| Dev.to | 开发者博客 | ⚡⚡ |
| PyPI | Python包统计 | ⚡⚡ |
| GitHub Topics | 开源项目 | ⚡⚡⚡ |
| Kaggle | AI应用案例 | ⚡ |

### 可选配置（3个）

| 数据源 | 说明 | 配置方法 |
|--------|------|----------|
| Reddit | 技术讨论 | 需要`REDDIT_CLIENT_ID` |
| NewsAPI | 科技新闻 | 需要`NEWSAPI_KEY` |
| GitHub Trending | 热门项目 | 第三方API（不稳定） |

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

## 🔄 版本历史

### v2.5 (2026-01-09) 🆕
- ✅ **统一入口** - `src/main.py` 现在支持两种模式切换
- ✅ **简化部署** - 更新 `run_and_commit.sh` 支持环境变量配置模式
- ✅ **清理冗余** - 删除冗余代码和文档，简化项目结构
- ✅ **双模式支持** - 同时支持自动热点模式和100期系列模式

### v2.4 (2026-01-09)
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

**最后更新**：2026-01-09
**版本**：v2.4
**Made with ❤️ by ContentForge AI Team**
