# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**IMPORTANT**: Always run commands with PYTHONPATH set to the project root directory.

**Project Root**: `/Users/z/Documents/work/content-forge-ai` (adjust if different)

**Quick Setup**:
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy environment template
cp .env.example .env

# 3. Edit .env and add ZHIPUAI_API_KEY
# Get key from: https://open.bigmodel.cn/
```

## Quick Reference

**Essential Commands**:
```bash
# Set PYTHONPATH (required for all commands - adjust path to your project root)
export PYTHONPATH=/Users/z/Documents/work/content-forge-ai

# ========== Auto Mode (Chinese AI News Digest with Scoring) ==========
# Run once (recommended for daily AI news digest)
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python src/main.py --mode auto --once

# ========== Series Mode (Two 100-episode series: LLM + ML) ==========
# View progress (LLM series - default)
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python src/main.py --mode series --progress
# View progress (ML series)
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python src/main.py --mode series --progress --series-config config/ml_topics_100_complete.json
# Generate single episode (LLM series)
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python src/main.py --mode series --episode 1
# Generate single episode (ML series)
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python src/main.py --mode series --episode 1 --series-config config/ml_topics_100_complete.json
# Generate range
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python src/main.py --mode series --all --start 1 --end 10

# ========== Batch Generation (ML Series - Parallel Execution) ==========
# Run batch generation script (3 parallel processes)
./batch_generate_ml_series.sh

# ========== Tests ==========
cd test
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python test_ai_trends.py --source hackernews
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python test_storage.py
```

**Core Files**:
- `src/main.py` - Unified entry point (use `--mode` to switch)
- `src/auto_orchestrator.py` - LangGraph workflow orchestration (auto mode)
- `src/series_orchestrator.py` - Series mode orchestrator
- `src/state.py` - State definition (WorkflowState TypedDict)
- `src/agents/` - 20+ agent implementations (base, trend analyzers, generators, quality checkers)
- `src/utils/storage_v2.py` - Unified storage (StorageFactory)
- `src/utils/series_manager.py` - Series management tools
- `src/utils/api_config.py` - API configuration manager
- `src/utils/time_filter.py` - Time parsing utility (supports RSS/Atom/HTTP Date formats)
- `config/config.yaml` - Main config (LLM, agents, data sources) - **Note: Header shows v2.5 but actual implementation is v11.0**
- `config/blog_topics_100_complete.json` - LLM 100-episode content plan
- `config/ml_topics_100_complete.json` - ML 100-episode content plan
- `config/prompts.yaml` - Agent system prompt templates
- `docs/DATA_SOURCES.md` - Complete data source documentation (30 sources across 6 categories)
- `batch_generate_ml_series.sh` - Parallel ML episode generation (3 concurrent processes)
- `monitor_and_launch_next.sh` - Workflow monitoring with auto-launch

**Key Architecture Points**:
1. **Two-Mode Architecture** (only 2 implemented): Auto (Chinese digest), Series (200 episodes across 2 series)
2. **Dual Series Structure**: LLM Series (100 episodes) + ML Series (100 episodes)
3. **Auto Mode** (v11.0): 26 data sources → concurrent fetch → time-weighted → fact-check → content enhance → translation refine → 6-category organization → 7-dimensional scoring → 全中文简报
4. **Series Mode**: 8-agent quality pipeline with staged longform generation
5. **v9.2 Category System**: 6 categories (📚 学术前沿, 🛠️ 开发工具, 🦾 AI Agent, 💼 企业应用, 🌐 消费产品, 📰 行业资讯)
6. **Data Source Integration**: Integrated into `RealAITrendAnalyzerAgent` (NOT a separate `src/data_sources/` directory)
7. **DailyStorage**: Only creates `raw/` and `digest/` directories
8. **Immutable State Updates**: Use `{**state, **updates}` pattern
9. **Claude Code Skills**: `.claude/skills/` contains custom skills for enhanced Claude Code functionality

## Deployment Automation

**GitHub Actions** - Automated deployment (3x daily):
- **Schedule**: 6:00, 12:00, 18:00 Beijing Time (via `.github/workflows/daily-digest.yml`)
- **Workflow**: Runs auto mode → commits changes → pushes to GitHub → triggers ai-insights sync
- **Timeout**: 90 minutes (configured in workflow YAML)
- **AI Insights Sync**: Uses repository_dispatch to trigger content sync to external repo (Ming-H/ai-insights)

**Commit Message Pattern**:
```bash
# Format used by GitHub Actions and run_and_commit.sh
feat: AI内容自动生成 - YYYY-MM-DD

生成时间: HH:MM:SS (北京时间)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

**run_and_commit.sh** - Manual deployment script:
```bash
# Location: /path/to/content-forge-ai/run_and_commit.sh
# Purpose: Auto-generate content and commit to GitHub

# Script workflow:
# 1. Sets PYTHONPATH
# 2. Runs auto mode once: python src/main.py --mode auto --once
# 3. Stages data/ directory changes
# 4. Creates structured commit message with date
# 5. Pushes to remote repository
```

## Project Overview

ContentForge AI is a LangChain/LangGraph-based automated content production system that generates AI-focused content.

**Auto Mode** (v11.0):
- **Multiple Data Sources**: 26 enabled sources (TechCrunch, NewsAPI, Hacker News, MIT, OpenAI, BAIR, Microsoft Research, arXiv, MarkTechPost, KDnuggets, AI Business, The Gradient, InfoQ, Hugging Face, NewsData.io, Reddit, GitHub Trending, **AI News, The Decoder, 量子位, 机器之心, Wired AI, VentureBeat AI, Google AI Blog, DeepMind Blog, arXiv CL/CV/LG, Reddit ML/AI RSS, Towards Data Science (v10.1)**)
- **9 Agents**: Concurrent Fetch → Time Weight → Auto Fact Check → Content Enhancer → Translation Refiner → Trend Categorizer → News Scoring → World Class Digest (全中文)
- **6 Categories** (v9.2): 📚 学术前沿, 🛠️ 开发工具, 🦾 AI Agent, 💼 企业应用, 🌐 消费产品, 📰 行业资讯
- **Scoring System** (v11.0): 7-dimensional scoring (source_authority 25%, engagement 15%, freshness 25%, category_balance 10%, content_quality 15%, diversity 5%, fact_confidence 5%)
- **Time Filtering** (v9.2): No 24h restriction - prioritizes latest data by timestamp, filters only items without timestamps
- **Concurrent Fetch** (v11.0): 10x performance improvement with concurrent data fetching
- **Time Weight** (v11.0): Dynamic time-weighted scoring ensures latest content priority
- **Auto Fact Check** (v11.0): Lightweight fact-checking for Top 10 items using LLM built-in knowledge
- **Content Enhancer** (v11.0): Background and impact analysis for important news (score >= 70)
- **Translation Refiner** (v11.0): Strunk rules application for improved readability
- **Real-time Sources** (v10.0): NewsData.io (秒级更新), Reddit Stream (实时社区讨论), GitHub Trending (开发者关注)
- **Output**: `data/daily/YYYYMMDD/digest/digest_YYYYMMDD.md` (全中文, with structured JSON)

**Series Mode**:
- **Two 100-episode series**: LLM Series (episodes 1-100) + ML Series (episodes 1-100)
- **8-agent quality pipeline**: research → longform → code review → fact check → quality evaluation → consistency check → visualization → citation formatting
- **Staged longform generation** (outline → sections → summary)
- **Configurable via `--series-config` flag** to switch between LLM and ML series

## Environment Setup

**Required API Keys** (`.env`):
- `ZHIPUAI_API_KEY` - Primary LLM provider (https://open.bigmodel.cn/)

**Optional Keys** (existing):
- `TAVILY_API_KEY` - Web search (for ResearchAgent)
- `NEWSAPI_KEY` - NewsAPI.org data source
- `OPENAI_API_KEY` - Backup LLM

**Optional Keys** (NEW 2026-02-01):
- `PRODUCT_HUNT_API_KEY` - Product Hunt OAuth token (https://api.producthunt.com/v2/docs)
- `GITHUB_TOKEN` - GitHub Personal Access Token (https://github.com/settings/tokens)
- `HUGGINGFACE_TOKEN` - Hugging Face token (https://huggingface.co/settings/tokens)
- `SEMANTIC_SCHOLAR_API_KEY` - Semantic Scholar API key (https://www.semanticscholar.org/product/api)
- `OPENALEX_EMAIL` - OpenAlex email (free, recommended)
- `REDDIT_CLIENT_ID/SECRET` - Reddit API credentials (https://www.reddit.com/prefs/apps)

**Optional Keys** (NEW v10.0 - Real-time data sources):
- `NEWSDATA_IO_API_KEY` - NewsData.io real-time news API (推荐, free 200 requests/day, https://newsdata.io/register)
- `REDDIT_CLIENT_ID/SECRET` - Reddit Stream API for real-time community discussions (already listed above, same credentials)

**Dependencies**:
```bash
# Core dependencies from requirements.txt
pip install langgraph>=0.2.0 langchain>=0.3.0 langchain-openai>=0.2.0
pip install loguru pyyaml python-dotenv pydantic>=2.0.0
pip install arxiv>=2.1.0 feedparser>=6.0.10 praw>=7.7.0

# New data sources (2026-02-01)
pip install requests beautifulsoup4
```

**New Data Sources (2026-02-01)**:
The system now integrates **30 data sources** across 6 categories:
- **📚 Academic Frontier** (6): arXiv, Semantic Scholar, OpenAlex, Papers with Code, OpenReview, DBLP
- **🛠️ Dev Tools** (5): Hugging Face Hub, PyPI, npm, GitHub Releases, Framework RSS
- **🦾 AI Agent** (5): GitHub Trending, Product Hunt, Reddit AI, Hacker News, Awesome AI Agents
- **💼 Enterprise AI** (4): TechCrunch, VentureBeat, AI Business, InfoQ
- **🌐 Consumer Apps** (4): Product Hunt, a16z Top 100, Hacker News, App Stores
- **📰 Industry News** (6): NewsAPI, MIT Review, The Gradient, MarkTechPost, Stanford HAI, Accenture

See `docs/DATA_SOURCES.md` for complete API documentation and implementation details.

## Auto Mode Architecture

**Workflow** (v11.0):
```
1. ConcurrentFetchAgent (v11.0: 并发数据获取，10倍性能提升)
   - 从26个数据源并发获取热点
   - 保留所有内容（不去重、不排序）
   - 输出: trends_by_source

2. TimeWeightAgent (v11.0: 时效性智能加权)
   - 动态推荐时间权重（dynamic/linear/exponential）
   - 超过72小时新闻时效分为0
   - 1小时内新闻获得2倍加成
   - 输出: time_weighted_trends

3. AutoFactCheckAgent (v11.0: 轻量级事实核查)
   - 仅核查Top 10新闻
   - 使用LLM内置知识（无需Tavily）
   - 置信度阈值0.7
   - 输出: fact_checked_trends

4. ContentEnhancerAgent (v11.0: 内容增强)
   - 使用trafilatura提取完整内容
   - 为重要性>=70的新闻生成背景分析
   - 生成影响分析
   - 输出: enhanced_trends

5. TranslationRefinerAgent (v11.0: 翻译精炼)
   - 应用Strunk原则提升可读性
   - 术语一致性检查
   - 目标可读性分数60
   - 输出: refined_trends

6. TrendCategorizerAgent (v9.2: 6分类系统)
   - 按分类组织热点
   - 6大分类：📚 学术前沿, 🛠️ 开发工具, 🦾 AI Agent, 💼 企业应用, 🌐 消费产品, 📰 行业资讯
   - 优先最新数据（按时间戳排序）
   - Top5截取（每个分类最多5条）
   - 只过滤没有时间戳的内容（无24h限制）
   - 输出: categorized_trends

7. NewsScoringAgent (v11.0: 7维度评分)
   - 对新闻进行7维度评分
   - 智能筛选，保留高价值内容
   - 输出: scored_trends

8. WorldClassDigestAgentV9
   - 生成全中文世界顶级新闻简报
   - 翻译所有标题、描述
   - 生成核心洞察和深度观察
   - 输出: news_digest (全中文 + 结构化JSON)
```

**Data Sources** (26 enabled sources):
| 数据源 | 类型 | 内容 | 版本 |
|--------|------|------|------|
| TechCrunch AI | 新闻 | AI行业新闻RSS | - |
| NewsAPI.org | 新闻 | 全球AI新闻聚合（需API key） | - |
| Hacker News | 社区 | 科技热点讨论API | - |
| MIT Tech Review | 新闻 | MIT技术评论RSS | - |
| OpenAI Blog | 官方 | OpenAI官方动态RSS | - |
| BAIR Blog | 学术 | UC Berkeley AI研究RSS | - |
| Microsoft Research | 学术 | 微软研究院博客RSS | - |
| arXiv | 学术 | AI重大论文API | - |
| MarkTechPost | 新闻 | AI研究新闻RSS | - |
| KDnuggets | 新闻 | 数据科学权威RSS | - |
| AI Business | 新闻 | AI行业新闻RSS | - |
| The Gradient | 期刊 | AI研究期刊RSS | - |
| InfoQ AI | 技术 | 技术媒体RSS | - |
| Hugging Face | 官方 | Hugging Face官方博客RSS | - |
| **NewsData.io** ⭐ | **实时** | **秒级新闻更新（免费200次/天）** | **v10.0** |
| **Reddit Stream** ⭐ | **实时** | **社区实时讨论（r/MachineLearning等）** | **v10.0** |
| **GitHub Trending** ⭐ | **实时** | **开发者关注热点** | **v10.0** |
| **AI News** | 新闻 | 顶级AI新闻媒体（免费RSS） | **v10.1** |
| **The Decoder** | 新闻 | AI专业新闻（免费RSS） | **v10.1** |
| **量子位 (qbitai)** | 新闻 | 中文AI第一媒体（免费RSS） | **v10.1** |
| **机器之心 (jiqizhixin)** | 新闻 | 深度AI报道（免费RSS） | **v10.1** |
| **Wired AI** | 新闻 | AI专题新闻（免费RSS） | **v10.1** |
| **VentureBeat AI** | 新闻 | AI商业新闻（免费RSS） | **v10.1** |
| **Google AI Blog** | 官方 | Google AI官方动态（免费RSS） | **v10.1** |
| **DeepMind Blog** | 学术 | Google DeepMind顶级研究（免费RSS） | **v10.1** |
| **arXiv CL/CV/LG** | 学术 | NLP/CV/ML论文（免费RSS） | **v10.1** |
| **Reddit ML/AI RSS** | 社区 | ML/AI讨论社区（免费RSS） | **v10.1** |
| **Towards Data Science** | 新闻 | 数据科学文章（免费RSS） | **v10.1** |

**Scoring System** (NewsScoringAgent v11.0):
- `source_authority` (25%): 来源权威度，基于预定义评分表
- `engagement` (15%): 互动数据（点赞、评论、分享）
- `freshness` (25%): 时效性（24小时内发布加分）⬆️
- `category_balance` (10%): 确保各分类平衡
- `content_quality` (15%): 标题质量、内容完整性⬆️
- `diversity` (5%): 确保来源多样性
- `fact_confidence` (5%): 事实置信度（新增）⬆️

**Output Format**:
```markdown
# AI每日热点 · 2026年02月03日

## 💡 核心洞察
- 多智能体协作范式确立...

## 📰 深度观察
**AI产业观察：从云端竞逐到端侧重构的范式转移**

## 🔍 本期热点
### 📚 学术前沿（5条，优先最新）
#### [据报Apple研发AI可穿戴设备](链接)
**来源**：TechCrunch AI  ·  **热度**：70  ·  **评分**：82
...
```

## ML Series Architecture

The ML Series (`config/ml_topics_100_complete.json`) provides 100 episodes covering machine learning and deep learning, organized into 10 sub-series:

**ML Series Structure**:
| Sub-series | Episodes | Focus |
|------------|----------|-------|
| `ml_series_1_foundation` | 1-10 | 机器学习基础 (Math foundations, algorithms) |
| `ml_series_2_deep_learning` | 11-20 | 深度学习基础 (Neural networks, training) |
| `ml_series_3_computer_vision` | 21-30 | 计算机视觉 (CNNs, image processing) |
| `ml_series_4_nlp` | 31-40 | 自然语言处理 (Text processing, NLP basics) |
| `ml_series_5_rl` | 41-50 | 强化学习 (RL agents, policies) |
| `ml_series_6_recommendation` | 51-60 | 推荐系统 (Collaborative filtering, deep learning) |
| `ml_series_7_optimization` | 61-70 | 模型优化 (Hyperparameter tuning) |
| `ml_series_8_traditional_ml` | 71-80 | 传统机器学习 (SVM, trees, clustering) |
| `ml_series_9_feature_eng` | 81-90 | 特征工程 (Feature selection, extraction) |
| `ml_series_10_advanced` | 91-100 | 高级ML主题 (Ensemble, interpretability) |

**Storage Structure for ML Series**:
```
data/series/ML_series/
├── ml_series_1_ml_foundation/
│   ├── episode_001/
│   │   └── longform/
│   │       └── ep001_..._article.md
│   └── series_metadata.json
└── ...
```

**Path Management**: ML series use `ml_series_X` IDs with paths managed by `SeriesPathManager`. The category is automatically detected (`get_series_category()` returns "ML_series" for `ml_series_*` IDs).

## Batch Generation Scripts

**`batch_generate_ml_series.sh`** - Parallel ML episode generation:
```bash
# Run batch generation (3 parallel processes by default)
./batch_generate_ml_series.sh

# Features:
# - Configurable parallelism (PARALLELISM=3)
# - PID tracking for process management
# - Automatic retry on failure
# - Progress logging to logs/batch_generate/
# - Episode list configurable via EPISODES array
```

**`monitor_and_launch_next.sh`** - Workflow monitoring script:
```bash
# Monitors running tasks and auto-launches next episodes
# when previous ones complete
./monitor_and_launch_next.sh

# Features:
# - Checks task completion by scanning output files
# - Auto-launches next episodes from NEXT_EPISODES array
# - 30-second polling interval
# - Marks completed/failed tasks for tracking
```

## Command Reference

**Auto Mode**:
- `--mode auto --once` - 生成一次简报
- `--topic STR` - 文件命名（不影响内容）

**Series Mode**:
- `--mode series --progress` - 查看进度
- `--mode series --episode INT` - 生成指定集
- `--mode series --episode INT --series-config PATH` - 生成指定系列 (LLM/ML)
- `--mode series --all` - 生成全部
- `--start INT` - Start episode (default: 1)
- `--end INT` - End episode (default: 100)
- `--series-config PATH` - 指定配置文件 (default: `config/blog_topics_100_complete.json`)

## Architecture Overview

### Big Picture: Multi-Orchestrator Architecture

This system uses a **multi-orchestrator pattern** where each mode has its own orchestrator implementing a different content generation strategy:

```
src/main.py (CLI entry point)
    │
    ├─→ AutoContentOrchestrator (src/auto_orchestrator.py)
    │   └─→ LangGraph StateGraph workflow
    │       └─→ Agent chain: trend_analyzer → categorizer → scorer → digest
    │
    └─→ SeriesOrchestrator (src/series_orchestrator.py)
        └─→ Sequential execution with error recovery
            └─→ Agent chain: research → longform → quality check → social content
```

**Key Insight**: The LangGraph StateGraph in Auto mode vs sequential execution in Series mode represents a fundamental architectural difference - Auto mode uses graph-based state management while Series mode uses traditional sequential flows.

### Design Patterns

**1. Multi-Orchestrator Pattern**
- Two orchestrators implement different content generation strategies
- Runtime switching via strategy pattern
- Each manages its own workflow and storage

**2. Factory Pattern**
- `StorageFactory` creates storage instances based on mode
- Supports: `DailyStorage` (by date) and `SeriesStorage` (by series)

**3. Chain of Responsibility**
- LangGraph workflow implements agent chain
- Each agent processes and passes state
- Immutable state updates: `{**state, **updates}`

**4. Template Method Pattern**
- All agents inherit from `BaseAgent`
- Implement standard `execute(state: Dict) -> Dict` interface
- Unified logging, error handling, LLM calls

### Two-Orchestrator Comparison

| Feature | AutoContentOrchestrator | SeriesOrchestrator |
|---------|-------------------------|-------------------|
| **File** | `src/auto_orchestrator.py` | `src/series_orchestrator.py` |
| **Data Source** | Multiple real-time APIs | 100 preset topics |
| **Trigger** | Scheduled or manual | Manual execution |
| **Storage** | `data/daily/YYYYMMDD/` | `data/series/{id}/episode_{xxx}/` |
| **Output** | Raw data + Digest | Longform articles |
| **Workflow** | LangGraph graph execution | Sequential with error recovery |
| **State Fields** | Uses `trends_by_source` | Uses `current_topic` + `selected_ai_topic` |
| **Primary Use** | Daily trend tracking | Systematic content library |
| **Storage Format** | `YYYYMMDD/` | `series_X_name/episode_XXX/` |

### Auto Workflow Agent Chain

The LangGraph StateGraph builds a directed acyclic graph (DAG) of agents:

```python
# From src/auto_orchestrator.py:_build_workflow()
workflow = StateGraph(dict)
workflow.add_entry_point("ai_trend_analyzer")
workflow.add_edge("ai_trend_analyzer", "trend_categorizer")
workflow.add_edge("trend_categorizer", "news_scoring")
workflow.add_edge("news_scoring", "world_class_digest")
workflow.add_edge("world_class_digest", END)
```

**Data Flow**:
```
ai_trend_analyzer (multiple data sources aggregation)
  ↓ state["trends_by_source"] = {...}

trend_categorizer (v9.2: 6大分类重新组织 + 优先最新 + Top5截取)
  ↓ state["categorized_trends"] = {...}

news_scoring (v7.0: 6维度智能评分筛选)
  ↓ state["scored_trends"] = {...}

world_class_digest_v9 (生成全中文世界顶级新闻简报)
  ↓ state["news_digest"] = {...}
```

### Storage Structure

```
data/
├── daily/                     # Auto模式
│   └── YYYYMMDD/
│       ├── raw/              # 原始数据（按数据源）
│       └── digest/           # 全中文简报
│           ├── digest_YYYYMMDD.md
│           └── digest_YYYYMMDD.json
│
└── series/                    # Series模式
    └── {series_id}/
        └── episode_{xxx}/
            └── longform/     # 长文本文章
```

**Note**: Only Auto and Series modes are implemented. Custom/Refine modes documented in config.yaml are NOT available in the current codebase.

### AI Trend Data Sources (config.yaml:69-105)

**Currently Enabled (26 sources)**:
- `techcrunch_ai` - TechCrunch AI RSS
- `newsapi` - NewsAPI.org (全球AI新闻聚合)
- `hackernews` - Hacker News API
- `mit_tech_review` - MIT Technology Review RSS
- `openai_blog` - OpenAI Blog RSS
- `bair_blog` - Berkeley AI Research Blog (顶级学术)
- `microsoft_research` - Microsoft Research Blog (官方)
- `arxiv_news` - arXiv API
- `marktechpost` - MarkTechPost (AI研究新闻)
- `kdnuggets` - KDnuggets (数据科学权威)
- `ai_business` - AI Business (行业新闻)
- `the_gradient` - The Gradient (AI研究期刊)
- `infoq_ai` - InfoQ AI (技术媒体)
- `hugging_face_blog` - Hugging Face Blog (官方)
- **`newsdata_io`** ⭐ - **NewsData.io (实时新闻API，秒级更新，免费200次/天，v10.0新增)**
- **`reddit_stream`** ⭐ - **Reddit (实时社区讨论，r/MachineLearning等，v10.0新增)**
- **`github_trending`** ⭐ - **GitHub Trending (开发者关注热点，v10.0新增)**
- **`ai_news`** ⭐ - **AI News (顶级AI新闻媒体，免费RSS，v10.1新增)**
- **`the_decoder`** ⭐ - **The Decoder (AI专业新闻，免费RSS，v10.1新增)**
- **`qbitai`** ⭐ - **量子位 (中文AI第一媒体，免费RSS，v10.1新增)**
- **`jiqizhixin`** ⭐ - **机器之心 (深度AI报道，免费RSS，v10.1新增)**
- **`wired_ai_v2`** ⭐ - **Wired AI (AI专题新闻，免费RSS，v10.1新增)**
- **`venturebeat_ai_v2`** ⭐ - **VentureBeat AI (AI商业新闻，免费RSS，v10.1新增)**
- **`google_ai_blog_v2`** ⭐ - **Google AI Blog (官方AI动态，免费RSS，v10.1新增)**
- **`deepmind_blog_v2`** ⭐ - **Google DeepMind (顶级研究，免费RSS，v10.1新增)**
- **`arxiv_cl`** ⭐ - **arXiv NLP (自然语言处理论文，免费RSS，v10.1新增)**
- **`arxiv_cv`** ⭐ - **arXiv CV (计算机视觉论文，免费RSS，v10.1新增)**
- **`arxiv_lg`** ⭐ - **arXiv ML (机器学习论文，免费RSS，v10.1新增)**
- **`reddit_ml_rss`** ⭐ - **Reddit ML (机器学习社区，免费RSS，v10.1新增)**
- **`reddit_ai_rss`** ⭐ - **Reddit AI (AI讨论社区，免费RSS，v10.1新增)**
- **`towards_data_science`** ⭐ - **Towards Data Science (数据科学文章，免费RSS，v10.1新增)**

**Config Params** (`config/config.yaml:108-110`):
- `max_trends: 20` - Max trend count per source
- `min_heat_score: 60` - Minimum heat score
- `cache_ttl: 3600` - Cache TTL (seconds)

**Data Source Implementation**: Sources are integrated into `RealAITrendAnalyzerAgent` in `src/agents/ai_trend_analyzer_real.py`.

**Adding New Sources**:
```python
# Add new source logic in RealAITrendAnalyzerAgent._fetch_all_trends()
# Return format: [{"title": "...", "url": "...", ...}]
```

## Core Patterns

### Agent Implementation Pattern

All agents inherit from `BaseAgent` (`src/agents/base.py:22`), implementing `execute()`:

```python
from src.agents.base import BaseAgent
from typing import Dict, Any

class NewAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        super().__init__(config, prompts)

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Agent logic
            result = self._call_llm("Your prompt")
            return {**state, "new_field": result}
        except Exception as e:
            self.log(f"Error: {e}", "ERROR")
            return {
                **state,
                "error_message": str(e),
                "current_step": "new_agent_failed"
            }
```

**Key Methods**:
- `_call_llm(prompt: str) -> str` - Call LLM (internal)
- `log(message: str, level: str = "INFO")` - Log with loguru
- `_load_system_prompt() -> str` - Load system prompt (from `config/prompts.yaml`)
- `_init_llm() -> ChatOpenAI` - Initialize LLM (supports ZhipuAI/OpenAI)

### LLM Provider Switching

Multiple LLM providers supported via `config/config.yaml` `llm.provider`:

```yaml
llm:
  provider: "zhipuai"  # or "openai"
  zhipuai:
    model: "glm-4.7"  # Latest flagship
    # Other options: glm-4-flash (cheap fast), glm-4-plus (prev flagship)
    base_url: "https://open.bigmodel.cn/api/coding/paas/v4/"  # Coding endpoint
    thinking:
      enabled: false  # Enable deep thinking mode (GLM-4.7 exclusive)
      type: "auto"    # "auto" or "enabled"
  openai:
    model: "gpt-4o"
    base_url: "https://api.openai.com/v1"
```

**GLM-4.7 Thinking Mode** (`config/config.yaml:17-19`):
- `thinking.enabled`: Enable deep thinking mode (GLM-4.7 exclusive feature)
- `thinking.type`: "auto" (auto-trigger) or "enabled" (force enable)
- Improves reasoning quality for complex tasks

**Research Agent Options** (`config/config.yaml:95-100`):
- `search_provider`: "tavily" (default, paid service), "zhipuai" (included in annual plan, recommended), "mock" (offline)
- `max_results`: Maximum search results
- `search_depth`: "basic" or "advanced"
- `mock_mode`: Set to `true` to disable all search APIs

**API Config Management**: `src/utils/api_config.py` provides `APIConfigManager` for unified API key/endpoint management.

```python
from src.utils.api_config import get_api_config

api_config = get_api_config()
zhipu_key = api_config.get_api_key("zhipuai")
tavily_key = api_config.get_api_key("tavily")
base_url = api_config.get_endpoint("llm.zhipuai.base_url")

# Supports both env vars and config.yaml
# Priority: env vars > config.yaml > defaults
```

### Unified Storage System

`src/utils/storage_v2.py` provides unified storage based on factory pattern:

**Storage Structure**:
```
data/
├── daily/                    # Auto mode (trends + digest)
│   └── YYYYMMDD/
│       ├── raw/
│       └── digest/
│
└── series/                   # Series mode (100-episode blog series)
    └── {series_id}/
        ├── episode_{xxx}/
        │   └── longform/
        └── series_metadata.json
```

**Usage**:
```python
from src.utils.storage_v2 import StorageFactory

# Auto mode (daily trends)
daily_storage = StorageFactory.create_daily()
daily_storage.save_markdown("digest", "digest.md", content)

# Series mode (100-episode series)
series_storage = StorageFactory.create_series(
    series_id="series_1",  # 使用基础ID，内部会自动处理目录名
    episode_number=1
)
series_storage.save_article(content, title="文章标题")  # 直接保存到episode目录
series_storage.save_episode_metadata(metadata)
```

**Note**: Custom and Refine modes are defined in storage_v2.py but are NOT used by the current codebase. Only Auto and Series modes are implemented.

**Series Metadata Management**:
```python
from src.utils.series_manager import get_series_metadata, print_progress_summary

metadata = get_series_metadata("config/blog_topics_100_complete.json")
topic = metadata.get_topic_by_episode(1)
series = metadata.get_series_by_id("series_1")
metadata.update_topic_status("topic_001", "completed")
print_progress_summary()
```

### File Naming Convention (TopicFormatter)

`TopicFormatter` (`src/utils/series_manager.py:201`) provides unified file naming:

```python
from src.utils.series_manager import TopicFormatter

prefix = TopicFormatter.generate_filename_prefix(topic)
# Returns: ep001_llm_transformer_attention_mechanism

filename = TopicFormatter.generate_markdown_filename(topic, "article")
# Returns: ep001_llm_transformer_attention_mechanism_article.md

summary = TopicFormatter.format_topic_summary(topic)
# Output: ✅ Episode 001 | LLM的Transformer架构与注意力机制 [series_1]
```

### Series Path Management (SeriesPathManager)

`SeriesPathManager` (`src/utils/series_manager.py`) manages series folder naming:

```python
from src.utils.series_manager import SeriesPathManager

path = SeriesPathManager.get_series_path("series_1")
# Returns: "series_1_llm_foundation"

series_id = SeriesPathManager.get_series_id_from_path("series_1_llm_foundation")
# Returns: "series_1"
```

**Series Folder Format**: `series_X_descriptive_name` (v2.5 improvement for semantic paths)

**LLM Series List** (episodes 1-100):
- `series_1_llm_foundation` (1-10) - LLM Principles
- `series_2_rag_technique` (11-18) - RAG Practice
- `series_3_agent_development` (19-26) - Agent Development
- `series_4_prompt_engineering` (27-32) - Prompt Engineering
- `series_5_model_deployment` (33-40) - Model Deployment
- `series_6_multimodal_frontier` (41-50) - Multimodal Frontiers
- `series_7_ai_coding_tools` (51-60) - AI Coding Tools
- `series_8_ai_data_engineering` (61-70) - AI Data Engineering
- `series_9_ai_applications` (71-85) - AI Application Scenarios
- `series_10_ai_infrastructure` (86-100) - AI Infrastructure

**ML Series List** (episodes 1-100):
- `ml_series_1_ml_foundation` (1-10) - 机器学习基础
- `ml_series_2_deep_learning_foundation` (11-20) - 深度学习基础
- `ml_series_3_computer_vision` (21-30) - 计算机视觉
- `ml_series_4_natural_language_processing` (31-40) - 自然语言处理
- `ml_series_5_reinforcement_learning` (41-50) - 强化学习
- `ml_series_6_recommendation_systems` (51-60) - 推荐系统
- `ml_series_7_model_optimization` (61-70) - 模型优化
- `ml_series_8_traditional_ml` (71-80) - 传统机器学习
- `ml_series_9_feature_engineering` (81-90) - 特征工程
- `ml_series_10_advanced_ml_topics` (91-100) - 高级ML主题

**Series Path Mapping** (hardcoded): `src/utils/series_manager.py:156-179` (SeriesPathManager.SERIES_NAME_MAP)

**Important**: Adding new series requires updating both:
1. `config/blog_topics_100_complete.json` or `config/ml_topics_100_complete.json` - Add series info and topics
2. `SeriesPathManager.SERIES_NAME_MAP` - Add path mapping

**Category Detection**: `SeriesPathManager.get_series_category()` automatically detects whether a series is LLM or ML based on the `series_id` prefix:
- `series_*` → "LLM_series"
- `ml_series_*` → "ML_series"

### Claude Code Skills Directory

**`.claude/skills/`** - Custom Claude Code skills for enhanced functionality:

Skills are reusable capabilities that extend Claude Code's functionality. Each skill directory contains:
- `skill.md` - Skill definition and usage instructions
- Implementation code or configuration

**Available Skills** (from git status):
- `content-research-writer` - Research and citation assistance
- `copy-editing` - Marketing copy review and improvement
- `copywriting` - Marketing copy generation
- `email-sequence` - Email campaign automation
- `marketing-psychology` - Psychological principles for marketing
- `notebooklm` - Google NotebookLM integration
- `platform-adaptation` - Content adaptation for Chinese platforms
- `scriptwriting` - Screenplay and script writing
- `social-content` - Social media content management
- `writing-clearly-and-concisely` - Strunk's writing rules
- `x-article-publisher` - X (Twitter) Articles publishing

**Using Skills**:
```bash
# List available skills
ls .claude/skills/

# Skills are automatically loaded by Claude Code
# Invoke with /<skill-name> command in Claude Code
```

### LangGraph State Management

`WorkflowState` TypedDict (`src/state.py:61`) manages shared state between agents:

```python
from src.state import create_initial_state, update_state

state = create_initial_state(
    topic=None,  # or "AI技术" (file naming only)
    target_audience="技术从业者",
    content_type="干货分享"
)

new_state = update_state(state, {"new_field": value})
```

**LangGraph Node Wrapper** (`src/auto_orchestrator.py:171-184`):
```python
def _create_agent_node(self, agent):
    def node_function(state):
        try:
            result = agent.execute(state)
            return add_agent_to_order(result, agent.name)
        except Exception as e:
            return update_state(state, {
                "error_message": str(e),
                "current_step": f"{agent.name}_failed"
            })
    return node_function
```

**Critical State Pattern**: All agents must return a complete state dict using the immutable pattern `{**state, **updates}`. The LangGraph workflow automatically merges each node's output into the shared state.

## Workflow Execution Order

**AutoContentOrchestrator** (LangGraph mode): Workflow defined in `src/auto_orchestrator.py:_build_workflow()`

**SeriesOrchestrator** (sequential mode): Workflow in `src/series_orchestrator.py:_execute_workflow()` with safety wrappers and delays

**SeriesOrchestrator Safe Execution** (`src/series_orchestrator.py:248-262`):
```python
def _call_agent_safely(agent_name: str, state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        result = self.agents[agent_name].execute(state)
        time.sleep(2)  # Delay to avoid API rate limits
        return result
    except Exception as e:
        logger.error(f"[{agent_name}] Failed: {e}")
        time.sleep(2)
        return state  # Return original state, allow continuation
```

**Execution Order**:
- **Auto Mode** (v11.0): Concurrent fetch → Time weight → Auto fact check → Content enhancer → Translation refiner → Trend categorizer → News scoring → World class digest
- **Series Mode**: Research → Longform generation → Code review → Fact check → Quality evaluation → Consistency check → Visualization → Citation formatting

**Prompt Template System**: Each agent's system prompts stored in `config/prompts.yaml`, organized by lowercase agent class name

### Agent Dependencies

**Auto Mode Agents (v11.0)**:
| Agent | Deps On | Outputs | Description |
|-------|---------|---------|-------------|
| concurrent_fetch | - | trends_by_source | 26 data source concurrent aggregation (v11.0) |
| time_weight | trends_by_source | time_weighted_trends | Dynamic time-weighted scoring (v11.0) |
| auto_fact_check | time_weighted_trends | fact_checked_trends | Top 10 fact-checking using LLM (v11.0) |
| content_enhancer | fact_checked_trends | enhanced_trends | Background/impact analysis (v11.0) |
| translation_refiner | enhanced_trends | refined_trends | Strunk rules + terminology check (v11.0) |
| trend_categorizer | refined_trends | categorized_trends | 6-category organization (v9.2) |
| news_scoring | categorized_trends | scored_trends | 7-dimensional scoring (v11.0) |
| world_class_digest_v9 | scored_trends | news_digest | Chinese digest + JSON |

**Series Mode Agents**:
| Agent | Deps On | Outputs | Description |
|-------|---------|---------|-------------|
| research_agent | selected_ai_topic | research_data, research_summary | Web search background |
| longform_generator | selected_ai_topic, research_data | longform_article | Core content (staged) |
| code_review_agent | longform_article | code_review_result | Quality assurance |
| fact_check_agent | longform_article | fact_check_result | Fact verification |
| quality_evaluator_agent | longform_article | quality_report | Comprehensive evaluation |
| consistency_checker_agent | longform_article | consistency_report | Terminology/citation check |
| visualization_generator_agent | longform_article | mermaid_diagrams | Auto-generate diagrams |
| citation_formatter_agent | longform_article | formatted_citations | GB/T 7714-2015 format |

**Available Exported Agents** (from `src/agents/__init__.py`):
- BaseAgent
- RealAITrendAnalyzerAgent
- TrendsDigestAgent
- LongFormGeneratorAgent
- TitleOptimizerAgent
- ImageGeneratorAgent

**Note**: Many agent files exist in `src/agents/` (16 total) but are NOT exported in `__init__.py`. Agents like `XiaohongshuRefinerAgent` and `TwitterGeneratorAgent` were removed during Refine/Custom mode cleanup. To use additional agents, manually import them from their modules.

### Critical Notes

1. **Agent Import Requirement**: Only 6 agents are exported by default. To use quality assurance agents (code_review, fact_check, etc.), import directly: `from src.agents.code_review_agent import CodeReviewAgent`

2. **Longform Generator Needs Research Data**: `longform_generator` prioritizes `research_data`; if unavailable, generates based only on `selected_ai_topic`

3. **Series Mode State Field Special Handling**: Must set both `current_topic` and `selected_ai_topic` for compatibility (`src/series_orchestrator.py:_initialize_state()`)

4. **Staged Generation Avoids Timeout**: `LongFormGeneratorAgent` uses three-stage generation (outline → section expansion → summary), each stage independent LLM call (`src/agents/longform_generator.py:73`)

## Development Guide

### Adding New Agents

**AutoContentOrchestrator Mode**:
1. Create agent class (`src/agents/new_agent.py`) inheriting `BaseAgent`
2. Implement `execute(self, state: Dict[str, Any]) -> Dict[str, Any]`
3. Add config in `config/config.yaml` under `agents`
4. Initialize in `src/auto_orchestrator.py` `_init_agents()`
5. Add to LangGraph workflow in `_build_workflow()`

**SeriesOrchestrator Mode**:
1. Create agent class as above
2. Add to `agent_classes` dict in `src/series_orchestrator.py` `_init_agents()`
3. Add call logic in `_execute_workflow()`

**Important**: Agents must return complete state dict using `{**state, "new_field": value}` pattern.

### Common State Fields

| State Field | Written By | Read By | Description |
|-------------|------------|----------|-------------|
| `trends_by_source` | ai_trend_analyzer | trend_categorizer | Raw trends by source (Auto mode) |
| `categorized_trends` | trend_categorizer | news_scoring | 5-category organized trends |
| `scored_trends` | news_scoring | world_class_digest_v8 | Scored and filtered trends |
| `news_digest` | world_class_digest_v8 | - | Final Chinese digest |
| `research_data` | research_agent | longform_generator | Web search research data |
| `selected_ai_topic` | series_orchestrator | longform_generator | Selected AI topic |
| `current_topic` | series_orchestrator | - | Current topic (Series mode) |
| `longform_article` | longform_generator | quality agents | Longform article |
| `error_message` | Any agent | - | Error info |
| `current_step` | Any agent | - | Current step |
| `execution_time` | orchestrator | - | Execution time stats |
| `agent_execution_order` | orchestrator | - | Agent execution order |

**Note**: `WorkflowState` TypedDict defines possible fields, but actual usage is plain Dict. Auto mode uses `trends_by_source`→`categorized_trends`→`scored_trends` flow; Series mode uses `current_topic` and `selected_ai_topic`. State updates use immutable pattern: `{**state, **updates}`.

### Error Handling Pattern

**Agent-level Error Handling**:
```python
def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        result = self._call_llm("Your prompt")
        return {**state, "new_field": result}
    except Exception as e:
        self.log(f"Error: {e}", "ERROR")
        return {
            **state,
            "error_message": str(e),
            "current_step": "new_agent_failed"
        }
```

**Series Mode Safe Execution** (`src/series_orchestrator.py:248`):
```python
def _call_agent_safely(agent_name: str, state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        result = self.agents[agent_name].execute(state)
        time.sleep(2)
        return result
    except Exception as e:
        logger.error(f"[{agent_name}] Failed: {e}")
        time.sleep(2)
        return state  # Return original state, allow continuation
```

**Error Recovery Strategy**: Agents return original state on failure, allowing workflow to continue. Records `error_message` and `current_step` for debugging.

### Testing & Debugging

**Enable Verbose Logging**: Edit `config/config.yaml`
```yaml
logging:
  level: "DEBUG"
```

**Mock Mode Testing** (no API quota consumption):
```yaml
agents:
  ai_trend_analyzer:
    mock_mode: true
```

**Test Individual Agent**:
```bash
cd test
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python test_ai_trends.py --source hackernews
```

### Test Files

**Test Directory**: `test/`

| Test File | Purpose |
|-----------|---------|
| `test_ai_trends.py` | Test AI trend fetching from single data source |
| `test_storage.py` | Test storage system functionality |
| `test_topic_logic.py` | Test topic parameter handling across modes |
| `test_digest.py` | Test trend digest generation |
| `test_auto_topic.py` | Test auto mode topic handling |
| `test_new_sources.py` | Test new data source integration |
| `test_data_sources.py` | Test all data sources |
| `test_v9_categorization.py` | Test v9.2 6-category system |

**Test README**: `test/README.md` contains detailed documentation for test files.

**Running Tests**:
```bash
cd test
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python test_ai_trends.py --source hackernews
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python test_storage.py
```

**Test Single AI Trend Source**:
```bash
cd test
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python test_ai_trends.py --source hackernews
# Sources: hackernews, arxiv, huggingface, stackoverflow, devto, pypi, github_topics, kaggle
```

**Note**: `src/data_sources/` directory removed in v2.5; data sources integrated into `src/agents/ai_trend_analyzer_real.py`.

## Code Standards

- Use type annotations (PEP 484)
- Follow PEP 8
- Add docstrings
- Use `self.log()` instead of `print()`
- Agent `execute()` must return complete state dict
- Exception handling must update `error_message` and `current_step` fields

## Architecture Highlights

### Staged Longform Generation

`LongFormGeneratorAgent` (`src/agents/longform_generator.py:73`) uses three-stage generation to avoid timeouts:

1. **Stage 1**: Generate article outline
2. **Stage 2**: Expand sections section-by-section (loop LLM calls, optional context window)
3. **Stage 3**: Generate summary

This enables 9000-13000 word professional in-depth analysis.

### Web Search Deep Research

`ResearchAgent` uses Tavily API for web search, collecting official docs, GitHub, tech blogs. Research data stored in `state["research_data"]`.

### Quality Assurance Triple Check

1. **CodeReviewAgent**: Reviews code examples for correctness and best practices
2. **FactCheckAgent**: Fact-checks statements, technical parameters, dates/times
3. **QualityEvaluatorAgent**: Comprehensive quality assessment (score 7-10)

## Performance Optimization

### Cost Reduction

- Use `glm-4-flash` instead of `glm-4.7` (~80% cost reduction)
- Reduce data sources (e.g., keep only hackernews + arxiv)
- Lower `max_tokens` settings
- Disable unnecessary agents (set `enabled: false`)

### Speed Up Execution

- Use mock mode for development testing
- Reduce data sources

## Common Issues

**Import Error**: Ensure running from project root with PYTHONPATH set
```bash
export PYTHONPATH=/Users/z/Documents/work/content-forge-ai  # Replace with your actual path
python src/main.py --mode auto --once
```

**API Key Error**: Check environment variables
```bash
echo $ZHIPUAI_API_KEY
# or
cat .env
```

**View Logs**: Logs stored by date in `logs/YYYYMMDD/app.log`
```bash
tail -f logs/$(date +%Y%m%d)/app.log
```

**Generate Only Partial Content**: Edit `config/config.yaml` to disable unwanted agents
```yaml
agents:
  longform_generator:
    enabled: false
```

**Debug Specific Agent**: Enable mock_mode to avoid API calls
```yaml
agents:
  ai_trend_analyzer:
    mock_mode: true
```

**NewsAPI Rate Limiting**: NewsAPI has a free tier limit. If you hit rate limits:
1. Reduce `max_trends` in config.yaml
2. Remove `newsapi` from sources list
3. Or upgrade to NewsAPI paid tier

## Important Architecture Gotchas

1. **Only 2 Modes Implemented**: Auto and Series modes work. Custom and Refine modes are documented in config.yaml but NOT coded.

2. **State Field Naming Confusion**: Auto mode uses `trends_by_source`/`categorized_trends`/`scored_trends`, but older code and Series mode use `trending_topics`. These are NOT compatible.

3. **Agent Name vs State Field**: The agent is named `ai_trend_analyzer` but outputs `trends_by_source`, not `trending_topics`.

4. **DailyStorage Only Creates Two Directories**: As of v4.0+, `DailyStorage` only creates `raw/` and `digest/` subdirectories. Other directories like `longform/` will NOT be created in Auto mode.

5. **Research Agent Provider**: The `research_agent` uses `search_provider: "tavily"` by default (paid service). You can change to "zhipuai" (included in annual plan) or "mock" (offline development) in config.yaml.

6. **Series ID vs Path**: Series use two different identifiers:
   - `series_id`: "series_1" (internal ID, used in JSON config)
   - `series_path`: "series_1_llm_foundation" (folder name, used in filesystem)
   - Always use `SeriesPathManager` to convert between them.

7. **LLM Provider Base URL**: ZhipuAI uses a special coding endpoint: `https://open.bigmodel.cn/api/coding/paas/v4/` (NOT the standard API endpoint). This is configured in `config.yaml`.

8. **Agent Import Limitation**: Only 6 agents are exported in `__init__.py`. Quality agents (code_review, fact_check, etc.) must be imported directly from their modules.

9. **Version Context**: `config/config.yaml` header shows v2.5 but actual implementation is v9.2. Features were added incrementally - verify actual implementation in source code.

10. **Data Source Integration**: Data sources are integrated directly into `RealAITrendAnalyzerAgent` in `src/agents/ai_trend_analyzer_real.py`. The `src/data_sources/` directory mentioned in some documentation was removed in v2.5.

11. **Config vs Implementation Mismatch**: `config.yaml` contains agent configurations for Xiaohongshu/Twitter/WeChat agents that were removed during Refine/Custom mode cleanup. These are NOT exported in `__init__.py` and NOT available for use.

## Key File Locations

### Core Files

| File | Purpose |
|------|---------|
| `src/main.py` | Unified entry (supports --mode switching) |
| `src/auto_orchestrator.py` | LangGraph workflow orchestration (auto mode) |
| `src/series_orchestrator.py` | Series mode orchestrator |
| `src/state.py` | State definition (WorkflowState TypedDict) |
| `src/utils/storage_v2.py` | Unified storage system (StorageFactory) |
| `src/utils/series_manager.py` | Series management tools |
| `src/utils/api_config.py` | API config management (APIConfigManager) |
| `config/config.yaml` | Main config (LLM, agents, workflow) |
| `config/blog_topics_100_complete.json` | 100-episode plan |
| `config/prompts.yaml` | Agent system prompt templates |
| `.env` | Environment variables (API keys) |
| `.env.example` | Environment variable examples |
| `run_and_commit.sh` | Automated deployment script |
| `.claude/skills/` | Custom Claude Code skills directory |

### Agent Classes (src/agents/)

**Exported Agents** (available via `from src.agents import ...`):
| Agent Class | File | Purpose |
|-------------|------|---------|
| `BaseAgent` | `base.py` | Agent base class |
| `RealAITrendAnalyzerAgent` | `ai_trend_analyzer_real.py` | AI trend analysis (14 data sources) |
| `TrendsDigestAgent` | `trends_digest_agent.py` | Trend digest generation |
| `LongFormGeneratorAgent` | `longform_generator.py` | Longform generation (staged) |
| `TitleOptimizerAgent` | `title_optimizer.py` | Title optimization |
| `ImageGeneratorAgent` | `image_generator.py` | Image prompt generation |

**Auto Mode Agents** (import directly):
| Agent Class | File | Purpose |
|-------------|------|---------|
| `TrendCategorizerAgent` | `trend_categorizer_agent.py` | 6-category organization (v9.2) |
| `NewsScoringAgent` | `news_scoring_agent.py` | 6-dimensional scoring |
| `WorldClassDigestAgentV9` | `world_class_digest_agent_v8.py` | Chinese digest + JSON |

**Note**: `world_class_digest_agent_v8.py` file name is legacy - it implements v9 functionality. Check file headers for actual version.

**Series Mode Quality Agents** (import directly):
| Agent Class | File | Purpose |
|-------------|------|---------|
| `ResearchAgent` | `research_agent.py` | Web search deep research |
| `CodeReviewAgent` | `code_review_agent.py` | Code quality review |
| `FactCheckAgent` | `fact_check_agent.py` | Fact verification |
| `QualityEvaluatorAgent` | `quality_evaluator_agent.py` | Comprehensive quality assessment |
| `ConsistencyCheckerAgent` | `consistency_checker_agent.py` | Terminology/citation check |
| `VisualizationGeneratorAgent` | `visualization_generator_agent.py` | Auto-generate Mermaid diagrams |
| `CitationFormatterAgent` | `citation_formatter_agent.py` | GB/T 7714-2015 format |

**Note**: Files for Xiaohongshu/Twitter/WeChat agents exist but were removed during Refine/Custom mode cleanup and are NOT exported.

## Related Documentation

- **README.md** - Project overview and quick start
- **test/README.md** - Test file documentation

---

**Version**: v11.0 (current implementation)
**Updated**: 2026-02-10

## Version Notes

**Important Version Context**:
- **v11.0** (current, 2026-02-10): **性能与质量大幅提升** - 并发数据获取（10倍性能提升）、时效性智能加权、轻量级事实核查、内容增强、翻译精炼 - **显著提升内容质量和生成速度**
  - **ConcurrentFetchAgent**: 并发获取26个数据源，性能提升10倍
  - **TimeWeightAgent**: 动态时间权重（dynamic/linear/exponential），72小时以上时效分为0，1小时内新闻2倍加成
  - **AutoFactCheckAgent**: 轻量级事实核查Top 10新闻，使用LLM内置知识（无需Tavily）
  - **ContentEnhancerAgent**: 使用trafilatura提取完整内容，为重要性>=70的新闻生成背景和影响分析
  - **TranslationRefinerAgent**: 应用Strunk原则提升可读性，术语一致性检查，目标可读性分数60
  - **Updated Scoring Weights**: 7维度评分（新增fact_confidence 5%，freshness提升至25%，content_quality提升至15%）
- **v10.1** (2026-02-08): **新增9个免费RSS数据源** - AI News, The Decoder, 量子位, 机器之心, Wired AI, VentureBeat AI, Google AI Blog, DeepMind Blog, arXiv CL/CV/LG, Reddit ML/AI RSS, Towards Data Science
- **v10.0** (2026-02-05): **新增3个实时数据源** - NewsData.io (秒级新闻API), Reddit Stream (实时社区讨论), GitHub Trending (开发者关注热点) - **显著增强新闻实时性**
- **v9.2** (2026-02-01): 6-category system, prioritize latest data, removed 24h restriction, guarantee 30 items daily
- **v9.1** (2026-02-01): Strict 24h time filtering, enhanced time format support
- **v9.0** (2026-02-01): 5-category → 6-category重构, 新增 🦾 AI Agent 分类, 30个数据源分类映射
- **v8.1** (2026-01-31): Added ML Series (100 episodes), batch generation scripts, dual-series architecture
- **v8.0** (2026-01-28): Auto and Series modes optimized, skills integration, 3x daily GitHub Actions
- `config/config.yaml` header shows v2.5 (outdated, not updated since early development)
- Features include v7.0 innovations (NewsScoringAgent, 6-dimensional scoring) plus v8.0-v11.0 improvements
- **Dual Series Architecture**: LLM Series (100 episodes) + ML Series (100 episodes) = 200 episodes total
- **Only 2 modes implemented**: Auto and Series. Custom/Refine modes documented in config but NOT coded
- **Always verify actual implementation in source code** - documented features may differ from deployed version

## Recent Changes

This CLAUDE.md has been improved with:

1. **v11.0 Performance & Quality Enhancement** (2026-02-10):
   - **Added ConcurrentFetchAgent**: 并发获取26个数据源，性能提升10倍，可配置并发数和超时
   - **Added TimeWeightAgent**: 动态时间权重推荐（dynamic/linear/exponential），72小时以上时效分为0，1小时内新闻2倍加成
   - **Added AutoFactCheckAgent**: 轻量级事实核查Top 10新闻，使用LLM内置知识（无需Tavily），置信度阈值0.7
   - **Added ContentEnhancerAgent**: 使用trafilatura提取完整内容，为重要性>=70的新闻生成背景和影响分析
   - **Added TranslationRefinerAgent**: 应用Strunk原则提升可读性，术语一致性检查，目标可读性分数60
   - **Updated Scoring Weights**: 7维度评分（source_authority 25%, engagement 15%, freshness 25%, category_balance 10%, content_quality 15%, diversity 5%, fact_confidence 5%）
   - **Updated data sources count**: 17 → 26 sources
   - **Updated Auto Mode workflow**: 4 agents → 8 agents
   - **Updated Project Overview**: Reflects v11.0 architecture and features

2. **v10.1 Free RSS Data Sources** (2026-02-08):
   - **Added 9 free RSS sources**: AI News, The Decoder, 量子位, 机器之心, Wired AI, VentureBeat AI, Google AI Blog, DeepMind Blog, arXiv CL/CV/LG, Reddit ML/AI RSS, Towards Data Science
   - **Updated data sources count**: 17 → 26 sources

3. **v10.0 Real-time Data Sources** (2026-02-05):
   - **Added NewsData.io**: 秒级更新新闻API，免费200次/天，显著提升新闻实时性
   - **Added Reddit Stream**: 实时社区讨论监控 (r/MachineLearning, r/artificial, r/ChatGPT, r/LocalLLaMA)
   - **Added GitHub Trending**: 开发者关注热点，实时热门AI项目
   - **Updated data sources count**: 14 → 17 sources
   - **Updated .env.example**: Added NEWSDATA_IO_API_KEY configuration
   - **Updated API keys section**: Documented new real-time data source API keys

2. **v9.2 Documentation Updates** (2026-02-03):
   - **Updated Core Files section**: Removed outdated `src/data_sources/` reference, added clarity on actual v9.2 implementation
   - **Updated Key Architecture Points**: Changed 5-category → 6-category system, added data source integration clarification
   - **Updated Auto Mode documentation**: Added v9.2 time filtering changes (no 24h restriction)
   - **Updated Agent Dependencies**: Changed v8.0 → v9.2, updated categorizer description
   - **Updated Data Flow section**: Added v9.2 categorizer details
   - **Added AI Insights sync**: Documented repository_dispatch trigger to external repo
   - **Updated Important Architecture Gotchas**: Added data source integration and config mismatch warnings
   - **Updated Version Notes**: Added v9.1 context, clarified v2.5 header issue

2. **v9.2 Updates** (2026-02-01):
   - **6-Category System**: 📚 学术前沿, 🛠️ 开发工具, 🦾 AI Agent, 💼 企业应用, 🌐 消费产品, 📰 行业资讯
   - **30 Data Sources**: Integrated across 6 categories with comprehensive documentation
   - **Prioritize Latest Data**: Sort by timestamp (newest first), guarantee 30 items daily (6×5)
   - **Removed 24h Restriction**: Allow older content to fill gaps, ensure daily output quota
   - **Enhanced Time Parsing**: Support for RSS/Atom/HTTP Date formats in `time_filter.py`
   - **New Data Sources**: Added 10 new sources (Semantic Scholar, Hugging Face, PyPI, npm, etc.)

3. **v8.1 Updates** (2026-01-31):
   - **Added ML Series documentation** - 100 episodes covering ML/DL topics
   - **Added batch generation scripts** - `batch_generate_ml_series.sh` for parallel execution
   - **Added workflow monitoring** - `monitor_and_launch_next.sh` for auto-launching episodes
   - **Updated Series Path Management** - ML series paths and category detection
   - **Updated command reference** - `--series-config` flag for switching between LLM/ML series

**Recommended Actions**:
1. Use `--series-config` flag to switch between LLM and ML series
2. Use batch generation scripts for parallel ML episode generation
3. Only use Auto and Series modes - Custom/Refine are not available
4. Import quality agents directly when needed: `from src.agents.code_review_agent import CodeReviewAgent`
5. Test in mock mode first before running with live APIs
6. Verify agent availability in `src/agents/__init__.py` before use
7. Check `docs/DATA_SOURCES.md` for complete 30 data source documentation (v9.2)
8. Be aware that `config.yaml` contains configurations for removed agents (Xiaohongshu/Twitter/WeChat)
9. Enable v11.0 agents in config.yaml for better performance and quality (concurrent_fetch, time_weight, auto_fact_check, content_enhancer, translation_refiner)
