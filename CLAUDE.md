# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**IMPORTANT**: Always run commands with PYTHONPATH set to the project root directory.

**Project Root**: `/Users/z/Documents/work/content-forge-ai` (adjust if different)

## Quick Reference

**Essential Commands**:
```bash
# Set PYTHONPATH (required for all commands - adjust path to your project root)
export PYTHONPATH=/Users/z/Documents/work/content-forge-ai

# ========== Auto Mode (Chinese AI News Digest with Scoring) ==========
# Run once (recommended for daily AI news digest)
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python src/main.py --mode auto --once

# ========== Series Mode (100-episode blog series) ==========
# View progress
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python src/main.py --mode series --progress
# Generate single episode
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python src/main.py --mode series --episode 1
# Generate range
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python src/main.py --mode series --all --start 1 --end 10

# ========== Tests ==========
cd test
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python test_ai_trends.py --source hackernews
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python test_storage.py
```

**Core Files**:
- `src/main.py` - Unified entry point (use `--mode` to switch)
- `config/config.yaml` - Main config (LLM, agents, data sources)
- `config/blog_topics_100_complete.json` - 100-episode content plan
- `config/prompts.yaml` - Agent system prompt templates
- `src/utils/storage_v2.py` - Unified storage (StorageFactory)
- `src/utils/api_config.py` - API configuration manager

**Key Architecture Points**:
1. **Two-Mode Architecture**: Auto (Chinese digest with scoring), Series (100 topics)
2. **Auto Mode**: Multiple data sources → 分类组织 → 评分筛选 → 全中文简报
3. **DailyStorage**: Only creates `raw/` and `digest/` directories
4. **Immutable State Updates**: Use `{**state, **updates}` pattern
5. **Claude Code Skills**: `.claude/skills/` contains custom skills for enhanced Claude Code functionality

## Deployment Automation

**run_and_commit.sh** - Automated deployment script:
```bash
# Location: /path/to/content-forge-ai/run_and_commit.sh
# Purpose: Auto-generate content and commit to GitHub

# Script workflow:
# 1. Sets PYTHONPATH
# 2. Runs auto mode once: python src/main.py --mode auto --once
# 3. Stages data/ directory changes
# 4. Creates structured commit message with date
# 5. Pushes to remote repository

# Recommended crontab entry (daily at 3 AM):
0 3 * * * /path/to/content-forge-ai/run_and_commit.sh
```

## Project Overview

ContentForge AI is a LangChain/LangGraph-based automated content production system that generates AI-focused content.

**Auto Mode**:
- **Multiple Data Sources**: TechCrunch AI, NewsAPI.org, Hacker News, MIT Tech Review, OpenAI Blog, BAIR Blog, Microsoft Research, arXiv, MarkTechPost, KDnuggets, AI Business, The Gradient, InfoQ AI, Hugging Face Blog
- **4 Agents**: AI Trend Analyzer → Trend Categorizer → News Scoring → World Class Digest (全中文)
- **5 Categories**: 产业动态, 学术前沿, 技术创新, 产品工具, 行业应用
- **Scoring System**: 6-dimensional scoring (source_authority 30%, engagement 20%, freshness 15%, category_balance 15%, content_quality 10%, diversity 10%)
- **Output**: `data/daily/YYYYMMDD/digest/digest_YYYYMMDD.md` (全中文, with structured JSON)

**Series Mode**:
- 100-episode technical blog series (episodes 1-100)

## Environment Setup

**Required API Keys** (`.env`):
- `ZHIPUAI_API_KEY` - Primary LLM provider (https://open.bigmodel.cn/)

**Optional Keys**:
- `TAVILY_API_KEY` - Web search (for ResearchAgent)
- `NEWSAPI_KEY` - NewsAPI.org data source
- `OPENAI_API_KEY` - Backup LLM

**Dependencies**:
```bash
pip install langgraph langchain langchain-openai loguru pyyaml python-dotenv arxiv praw
```

## Auto Mode Architecture

**Workflow**:
```
1. RealAITrendAnalyzerAgent
   - 从多个数据源获取热点
   - 保留所有内容（不去重、不排序）
   - 输出: trends_by_source

2. TrendCategorizerAgent
   - 按分类组织热点
   - 5大分类：产业动态、学术前沿、技术创新、产品工具、行业应用
   - 输出: categorized_trends

3. NewsScoringAgent
   - 对新闻进行6维度评分
   - 智能筛选，保留高价值内容
   - 输出: scored_trends

4. WorldClassDigestAgent
   - 生成全中文世界顶级新闻简报
   - 翻译所有标题、描述
   - 生成核心洞察和深度观察
   - 输出: news_digest (全中文 + 结构化JSON)
```

**Data Sources** (enabled sources):
| 数据源 | 类型 | 内容 |
|--------|------|------|
| TechCrunch AI | 新闻 | AI行业新闻RSS |
| NewsAPI.org | 新闻 | 全球AI新闻聚合（需API key） |
| Hacker News | 社区 | 科技热点讨论API |
| MIT Tech Review | 新闻 | MIT技术评论RSS |
| OpenAI Blog | 官方 | OpenAI官方动态RSS |
| BAIR Blog | 学术 | UC Berkeley AI研究RSS |
| Microsoft Research | 学术 | 微软研究院博客RSS |
| arXiv | 学术 | AI重大论文API |
| MarkTechPost | 新闻 | AI研究新闻RSS |
| KDnuggets | 新闻 | 数据科学权威RSS |
| AI Business | 新闻 | AI行业新闻RSS |
| The Gradient | 期刊 | AI研究期刊RSS |
| InfoQ AI | 技术 | 技术媒体RSS |
| Hugging Face | 官方 | Hugging Face官方博客RSS |

**Scoring System** (NewsScoringAgent):
- `source_authority` (30%): 来源权威度，基于预定义评分表
- `engagement` (20%): 互动数据（点赞、评论、分享）
- `freshness` (15%): 时效性（24小时内发布加分）
- `category_balance` (15%): 确保各分类平衡
- `content_quality` (10%): 标题质量、内容完整性
- `diversity` (10%): 确保来源多样性

**Output Format**:
```markdown
# AI每日热点 · 2026年01月22日

## 💡 核心洞察
- 多智能体协作范式确立...

## 📰 深度观察
**AI产业观察：从云端竞逐到端侧重构的范式转移**

## 🔍 本期热点
### 📈 产业动态（15条，已筛选）
#### [据报Apple研发AI可穿戴设备](链接)
**来源**：TechCrunch AI  ·  **热度**：70  ·  **评分**：82
...
```

## Command Reference

**Auto Mode**:
- `--mode auto --once` - 生成一次简报
- `--topic STR` - 文件命名（不影响内容）

**Series Mode**:
- `--mode series --progress` - 查看进度
- `--mode series --episode INT` - 生成指定集
- `--mode series --all` - 生成全部
- `--start INT` - Start episode (default: 1)
- `--end INT` - End episode (default: 100)

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

trend_categorizer (按5大分类重新组织)
  ↓ state["categorized_trends"] = {...}

news_scoring (v7.0新增：6维度智能评分筛选)
  ↓ state["scored_trends"] = {...}

world_class_digest (生成全中文世界顶级新闻简报)
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
```

### AI Trend Data Sources (config.yaml:30-48)

**Currently Enabled (14 sources)**:
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

**Config Params** (`config/config.yaml:50-53`):
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
├── series/                   # Series mode (100-episode blog series)
│   └── {series_id}/
│       ├── episode_{xxx}/
│       │   └── longform/
│       └── series_metadata.json
│
├── custom/                   # Custom mode (user-defined content)
│   └── {timestamp}_{topic}/
│       ├── longform/
│       ├── xiaohongshu/
│       └── twitter/
│
└── refine/                   # Refine mode (multi-platform content)
    └── {source_name}/
        ├── raw/
        ├── wechat/
        ├── xiaohongshu/
        └── twitter/
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

# Custom mode (user-defined content)
custom_storage = StorageFactory.create_custom("20260114_120000_RAG技术")
custom_storage.save_markdown("longform", "article.md", content)

# Refine mode (multi-platform)
refine_storage = StorageFactory.create_refine("my_article")
refine_storage.save_text("wechat", "article.html", html_content)
```

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

**Complete Series List**:
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

**Series Path Mapping** (hardcoded): `src/utils/series_manager.py:156-167` (SeriesPathManager.SERIES_NAME_MAP)

**Important**: Adding new series requires updating both:
1. `config/blog_topics_100_complete.json` - Add series info and topics
2. `SeriesPathManager.SERIES_NAME_MAP` - Add path mapping

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
1. AI trend analysis → Trend digest → Content research
2. Longform generation → Code review
3. Xiaohongshu refinement → Twitter generation
4. Title optimization → Image generation
5. Quality evaluation

**Prompt Template System**: Each agent's system prompts stored in `config/prompts.yaml`, organized by lowercase agent class name

### Agent Dependencies

**Auto Mode v7.0 Agents**:
| Agent | Deps On | Outputs | Description |
|-------|---------|---------|-------------|
| ai_trend_analyzer | - | trends_by_source | 15+ data source aggregation |
| trend_categorizer | trends_by_source | categorized_trends | 5-category organization |
| news_scoring | categorized_trends | scored_trends | 6-dimensional scoring (v7.0) |
| world_class_digest | scored_trends | news_digest | Chinese digest + JSON |

**Other Agents** (Series/Custom/Refine modes):
| Agent | Deps On | Outputs | Description |
|-------|---------|---------|-------------|
| trends_digest | trending_topics | digest_content | Optional digest |
| research_agent | selected_ai_topic | research_data, research_summary | Background for longform |
| longform_generator | selected_ai_topic, research_data | longform_article | Core content |
| code_review_agent | longform_article | code_review_result | Quality assurance |
| fact_check_agent | longform_article | fact_check_result | Quality assurance |
| xiaohongshu_refiner | longform_article | xiaohongshu_note | Content adaptation |
| twitter_generator | longform_article | twitter_post | Content adaptation |
| title_optimizer | longform_article | optimized_titles | SEO optimization |
| image_generator | xiaohongshu_note or twitter_post | image_prompts | Image generation |
| quality_evaluator | All outputs | quality_report | Final evaluation |
| consistency_checker_agent | longform_article | consistency_report | Terminology/citation check |
| visualization_generator_agent | longform_article | mermaid_diagrams | Auto-generate diagrams |
| citation_formatter_agent | longform_article | formatted_citations | GB/T 7714-2015 format |

### Critical Notes

1. **Xiaohongshu/Twitter Agents Must Execute Sequentially**: Both read `longform_article` but should not run in parallel to avoid state update conflicts (`src/auto_orchestrator.py:213`)

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
| `trending_topics` | ai_trend_analyzer | trends_digest | AI trend list (Auto mode) |
| `digest_content` | trends_digest | - | Trend digest content |
| `research_data` | research_agent | longform_generator | Web search research data |
| `selected_ai_topic` | ai_trend_analyzer / series_orchestrator | longform_generator | Selected AI topic |
| `current_topic` | series_orchestrator | - | Current topic (Series mode) |
| `longform_article` | longform_generator | code_review_agent, xiaohongshu_refiner, twitter_generator | Longform article |
| `xiaohongshu_note` | xiaohongshu_refiner | - | Xiaohongshu note |
| `twitter_post` | twitter_generator | - | Twitter post |
| `error_message` | Any agent | - | Error info |
| `current_step` | Any agent | - | Current step |
| `execution_time` | orchestrator | - | Execution time stats |
| `agent_execution_order` | orchestrator | - | Agent execution order |

**Note**: `WorkflowState` TypedDict defines possible fields, but actual usage is plain Dict. Auto mode uses `trending_topics`, Series mode uses `current_topic` and `selected_ai_topic`. State updates use immutable pattern: `{**state, **updates}`.

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

1. **State Field Naming Confusion**: Auto mode uses `trends_by_source`/`categorized_trends`/`scored_trends`, but older code and Series mode use `trending_topics`. These are NOT compatible.

2. **Agent Name vs State Field**: The agent is named `ai_trend_analyzer` but outputs `trends_by_source`, not `trending_topics`.

3. **DailyStorage Only Creates Two Directories**: As of v4.0+, `DailyStorage` only creates `raw/` and `digest/` subdirectories. Other directories like `longform/` will NOT be created in Auto mode.

4. **Research Agent Provider**: The `research_agent` uses `search_provider: "tavily"` by default (paid service). You can change to "zhipuai" (included in annual plan) or "mock" (offline development) in config.yaml.

5. **Series ID vs Path**: Series use two different identifiers:
   - `series_id`: "series_1" (internal ID, used in JSON config)
   - `series_path`: "series_1_llm_foundation" (folder name, used in filesystem)
   - Always use `SeriesPathManager` to convert between them.

6. **LLM Provider Base URL**: ZhipuAI uses a special coding endpoint: `https://open.bigmodel.cn/api/coding/paas/v4/` (NOT the standard API endpoint). This is configured in `config.yaml`.

7. **Version Context**: `config/config.yaml` header shows v2.5 but documentation references v7.0 features (NewsScoringAgent, 6-dimensional scoring). Features were added incrementally - verify actual implementation in source code.

8. **Claude Code Skills**: The `.claude/skills/` directory contains custom skills that extend Claude Code functionality. These are loaded automatically when using Claude Code CLI.

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

**Auto Mode v7.0 Agents**:
| Agent Class | File | Purpose |
|-------------|------|---------|
| `BaseAgent` | `base.py` | Agent base class |
| `RealAITrendAnalyzerAgent` | `ai_trend_analyzer_real.py` | AI trend analysis (15+ data sources) |
| `TrendCategorizerAgent` | `trend_categorizer_agent.py` | 5-category organization |
| `NewsScoringAgent` | `news_scoring_agent.py` | 6-dimensional scoring (v7.0) |
| `WorldClassDigestAgent` | `world_class_digest_agent.py` | Chinese digest + JSON |

**Content Generation Agents** (Series/Custom/Refine modes):
| Agent Class | File | Purpose |
|-------------|------|---------|
| `TrendsDigestAgent` | `trends_digest_agent.py` | Trend digest generation |
| `LongFormGeneratorAgent` | `longform_generator.py` | Longform generation (staged) |
| `XiaohongshuLongRefinerAgent` | `xiaohongshu_long_refiner.py` | Xiaohongshu long note (~2000 chars) |
| `XiaohongshuShortRefinerAgent` | `xiaohongshu_short_refiner.py` | Xiaohongshu short note (800-1000 chars, viral baokuan style) |
| `TwitterGeneratorAgent` | `twitter_generator.py` | Twitter post generation |
| `WechatGeneratorAgent` | `wechat_generator.py` | WeChat HTML generation |
| `TitleOptimizerAgent` | `title_optimizer.py` | Title optimization |
| `ImageGeneratorAgent` | `image_generator.py` | Image prompt generation |
| `ResearchAgent` | `research_agent.py` | Web search deep research |

**Quality Assurance Agents**:
| Agent Class | File | Purpose |
|-------------|------|---------|
| `CodeReviewAgent` | `code_review_agent.py` | Code quality review |
| `FactCheckAgent` | `fact_check_agent.py` | Fact verification |
| `QualityEvaluatorAgent` | `quality_evaluator_agent.py` | Comprehensive quality assessment |
| `ConsistencyCheckerAgent` | `consistency_checker_agent.py` | Terminology/citation check |
| `VisualizationGeneratorAgent` | `visualization_generator_agent.py` | Auto-generate Mermaid diagrams |
| `CitationFormatterAgent` | `citation_formatter_agent.py` | GB/T 7714-2015 format |

## Related Documentation

- **README.md** - Project overview and quick start
- **test/README.md** - Test file documentation

---

**Version**: v2.5 (config.yaml) / v7.0 (feature documentation)
**Updated**: 2026-01-27

## Version Notes

**Important Version Context**:
- `config/config.yaml` shows v2.5 (file header comment)
- Feature documentation references v7.0 (NewsScoringAgent, 6-dimensional scoring)
- This reflects incremental development where features were added between config updates
- **Always verify actual implementation in source code** - documented features may differ from deployed version

## Recent Changes

This CLAUDE.md has been improved with:

1. **Added version mismatch clarification** - Explains v2.5 config vs v7.0 features
2. **Added skills directory reference** - Documents `.claude/skills/` for custom Claude Code skills
3. **Added run_and_commit.sh documentation** - Automated deployment script details
4. **Added test commands** - Test files section includes actual commands to run tests
5. **Corrected data source table** - Aligned with actual config.yaml (14 sources)
6. **Enhanced architecture overview** - Added "Big Picture" section showing multi-orchestrator pattern
7. **Added state flow annotations** - Shows exactly which state fields each agent outputs
8. **Added NewsAPI rate limiting note** - Common issue when using free tier
9. **Improved PYTHONPATH reminder** - Added note to replace with actual path

**Recommended Actions**:
1. Verify which features are actually enabled in config.yaml before use
2. Check source code implementation when behavior differs from documentation
3. Test in mock mode first before running with live APIs
