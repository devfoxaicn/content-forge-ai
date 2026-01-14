# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Reference

**Essential Commands**:
```bash
# Set PYTHONPATH (required for all commands)
export PYTHONPATH=/Users/z/Documents/work/content-forge-ai

# ========== Auto Mode (AI trends-based) ==========
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python src/main.py --mode auto --once
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python src/main.py --mode auto --topic "AI技术"

# ========== Series Mode (100-episode blog series) ==========
# View progress
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python src/main.py --mode series --progress
# Generate single episode
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python src/main.py --mode series --episode 1
# Generate entire series
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python src/main.py --mode series --series series_1
# Batch generate (auto-skips completed)
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python src/main.py --mode series --all --start 1 --end 10

# ========== Custom Mode (user-defined topics) ==========
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python src/main.py --mode custom --topic "RAG技术原理与实战"
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python src/main.py --mode custom --topic "RAG技术" --prompt "详细介绍架构和实战"

# ========== Refine Mode (multi-platform content refining) ==========
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python src/main.py --mode refine --input article.md
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python src/main.py --mode refine --input article.md --platforms wechat xiaohongshu
```

**Core Files**:
- `src/main.py` - Unified entry point (use `--mode` to switch between auto/series/custom/refine)
- `config/config.yaml` - Main config (LLM, agents, data sources)
- `config/blog_topics_100_complete.json` - 100-episode content plan
- `src/utils/series_manager.py` - Series management (SeriesMetadata, SeriesPathManager)
- `src/utils/storage_v2.py` - Unified storage (StorageFactory with 4 modes)

**Key Architecture Points**:
1. **Four-Mode Architecture**: Auto (AI trends), Series (100 topics), Custom (user-defined), Refine (multi-platform)
2. **Series Path Format**: `series_X_descriptive_name` (e.g., `series_1_llm_foundation`)
3. **Immutable State Updates**: Use `{**state, **updates}` pattern
4. **Agent Return Contract**: `execute()` must return complete state dict

## Project Overview

ContentForge AI v2.6 is a LangChain/LangGraph-based automated content production system supporting four modes:

**Core Workflow**: AI trend fetching (7 data sources) → Trend digest → Deep research (web search) → Longform generation (staged) → Quality check (code review + fact check) → Multi-platform generation (WeChat/Xiaohongshu/Twitter) → Title optimization → Image prompts → Quality evaluation

**Four Modes**:
1. **Auto Mode** - AI trend tracking and digest generation (daily automation)
2. **Series Mode** - 100-episode technical blog series (systematic content library)
3. **Custom Mode** - User-defined topic content generation (on-demand)
4. **Refine Mode** - Multi-platform content refining (WeChat HTML, Xiaohongshu, Twitter)

## Environment Setup

**Required API Keys** (`.env`):
- `ZHIPUAI_API_KEY` - Primary LLM provider (https://open.bigmodel.cn/)
- `TAVILY_API_KEY` - Web search for ResearchAgent (https://tavily.com/)

**Optional Keys**:
- `OPENAI_API_KEY` - Backup LLM
- `GEMINI_API_KEY` - Image generation
- `NEWSAPI_KEY`, `REDDIT_CLIENT_ID/SECRET` - Extended data sources

**Dependencies** (requirements.txt):
- LangChain/LangGraph - Agent framework
- langchain-openai - LLM interface (ZhipuAI compatible)
- loguru - Structured logging
- pyyaml - Config parsing
- python-dotenv - Env management
- arxiv, praw - Data source clients

## Command Reference

**Unified Entry** (`src/main.py`):

**Global Parameters**:
- `--mode {auto,series,custom,refine}` - Mode selection (default: auto)

**Auto Mode Parameters**:
- `--topic STR` - Content topic identifier (optional, for file naming only)
- `--audience STR` - Target audience (default: "技术从业者")
- `--type STR` - Content type (default: "干货分享")
- `--keywords [STR ...]` - Keyword list
- `--once` - Generate once immediately

**Series Mode Parameters**:
- `--config PATH` - Global config (default: "config/config.yaml")
- `--series-config PATH` - 100-episode config (default: "config/blog_topics_100_complete.json")
- `--episode INT` - Generate specific episode
- `--series STR` - Generate specific series (e.g., series_1)
- `--start INT` - Start episode (default: 1)
- `--end INT` - End episode (default: 100)
- `--all` - Generate all in range
- `--progress` - Show progress only

**Custom Mode Parameters**:
- `--topic STR` - Content topic/keywords (required)
- `--prompt STR` - Detailed content requirements (optional)
- `--audience STR` - Target audience (default: "技术从业者")
- `--words INT` - Target word count (optional)
- `--style {technical,practical,tutorial}` - Article style (optional)

**Refine Mode Parameters**:
- `--input PATH` - Input file path (required)
- `--platforms {wechat,xiaohongshu,twitter}` - Target platforms (default: all three)

**Important**:
- In auto mode, `--topic` is **only for file naming** - actual content is fully auto-generated from real-time AI trends
- In custom mode, `--topic` is the actual content theme to generate

## Architecture Overview

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

### Four-Orchestrator Comparison

| Feature | AutoContentOrchestrator | SeriesOrchestrator | CustomContentOrchestrator | RefineOrchestrator |
|---------|-------------------------|-------------------|--------------------------|-------------------|
| **File** | `src/auto_orchestrator.py` | `src/series_orchestrator.py` | `src/custom_orchestrator.py` | `src/refine_orchestrator.py` |
| **Data Source** | 7 real-time APIs | 100 preset topics | User-defined keywords | Input file |
| **Trigger** | Scheduled or manual | Manual execution | Manual execution | Manual execution |
| **Storage** | `data/daily/YYYYMMDD/` | `data/series/{id}/episode_{xxx}/` | `data/custom/{timestamp}_topic/` | `data/refine/{source_name}/` |
| **Output** | Raw data + Digest | Longform articles | Longform + Social content | Multi-platform content |
| **Workflow** | LangGraph graph execution | Sequential with error recovery | Sequential execution | Sequential execution |
| **State Fields** | Uses `trending_topics` | Uses `current_topic` + `selected_ai_topic` | Uses `selected_ai_topic` | Uses `longform_article` |
| **Primary Use** | Daily trend tracking | Systematic content library | On-demand content | Multi-platform publishing |

### Auto Workflow Agent Chain

```
ai_trend_analyzer (7 data sources aggregation)
  ↓
trends_digest (generate trend digest → digest/)
  ↓
research_agent (web search deep research, collect docs/GitHub/blogs)
  ↓
longform_generator (staged 9000-13000 word article → longform/)
  ↓
quality check:
  ├─→ code_review_agent (code review)
  └─→ fact_check_agent (fact checking)
  ↓
sequential execution:
  ├─→ xiaohongshu_refiner (3000-3500 word note → xiaohongshu/)
  └─→ twitter_generator (5-8 tweet thread → twitter/)
  ↓
title_optimizer (title optimization)
  ↓
image_generator (generate CN image prompts → prompts_*.txt)
  ↓
quality_evaluator (quality evaluation)
```

**Critical**: Xiaohongshu and Twitter agents must execute sequentially (not parallel) to avoid state update conflicts (`src/auto_orchestrator.py:213`)

### AI Trend Data Sources (config.yaml:30-37)

**Currently Enabled (7 sources)**:
- `producthunt` - Product Hunt RSS
- `github` - GitHub Trending
- `techcrunch_ai` - TechCrunch AI RSS
- `verge_ai` - The Verge AI RSS
- `venturebeat_ai` - VentureBeat AI RSS
- `arxiv_news` - arXiv API
- `hackernews` - Hacker News API

**Config Params** (`config/config.yaml:39-42`):
- `max_trends: 20` - Max trend count
- `min_heat_score: 60` - Minimum heat score
- `cache_ttl: 3600` - Cache TTL (seconds)

**Data Source Implementation**: In v2.5, sources are integrated into `AITrendAnalyzerAgent` in `src/agents/ai_trend_analyzer_real.py`.

**Adding New Sources**:
```python
# Add new source logic in AITrendAnalyzerAgent._fetch_all_trends()
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
  openai:
    model: "gpt-4o"
    base_url: "https://api.openai.com/v1"
```

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
    series_id="series_1_llm_foundation",
    episode_number=1
)
series_storage.save_markdown("longform", "article.md", content)
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

**LangGraph Node Wrapper** (`src/auto_orchestrator.py:270-277`):
```python
def _create_agent_node(self, agent: BaseAgent):
    def node_func(state: Dict[str, Any]) -> Dict[str, Any]:
        result = agent.execute(state)
        return add_agent_to_order(result, agent.name)
    return node_func
```

Each agent's output is merged into state via `{**state, **updates}` pattern, ensuring immutability.

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

| Agent | Deps On | Outputs | Description |
|-------|---------|---------|-------------|
| ai_trend_analyzer | - | trending_topics, selected_ai_topic | Data source |
| trends_digest | trending_topics | digest_content | Optional |
| research_agent | selected_ai_topic | research_data, research_summary | Background for longform |
| longform_generator | selected_ai_topic, research_data | longform_article | Core content |
| code_review_agent | longform_article | code_review_result | Quality assurance |
| fact_check_agent | longform_article | fact_check_result | Quality assurance |
| xiaohongshu_refiner | longform_article | xiaohongshu_note | Content adaptation |
| twitter_generator | longform_article | twitter_post | Content adaptation |
| title_optimizer | longform_article | optimized_titles | SEO optimization |
| image_generator | xiaohongshu_note or twitter_post | image_prompts | Image generation |
| quality_evaluator | All outputs | quality_report | Final evaluation |

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

| Test File | Purpose |
|-----------|---------|
| `test_ai_trends.py` | Test AI trend fetching (single source support) |
| `test_storage.py` | Test storage system |
| `test_topic_logic.py` | Test topic param handling |
| `test_digest.py` | Test trend digest generation |
| `test_auto_topic.py` | Test auto mode topic handling |
| `test_new_sources.py` | Test new data source integration |

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
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python src/main.py --once
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

### Agent Classes (src/agents/)

| Agent Class | File | Purpose |
|-------------|------|---------|
| `BaseAgent` | `base.py` | Agent base class |
| `AITrendAnalyzerAgent` | `ai_trend_analyzer_real.py` | AI trend analysis (7 data sources) |
| `TrendsDigestAgent` | `trends_digest_agent.py` | Trend digest generation |
| `LongFormGeneratorAgent` | `longform_generator.py` | Longform generation (staged) |
| `XiaohongshuRefinerAgent` | `xiaohongshu_refiner.py` | Xiaohongshu note refinement |
| `TwitterGeneratorAgent` | `twitter_generator.py` | Twitter post generation |
| `WechatGeneratorAgent` | `wechat_generator.py` | WeChat HTML generation |
| `TitleOptimizerAgent` | `title_optimizer.py` | Title optimization |
| `ImageGeneratorAgent` | `image_generator.py` | Image prompt generation |

## Related Documentation

- **README.md** - Project overview and quick start
- **test/README.md** - Test file documentation

---

**Version**: v2.6
**Updated**: 2026-01-14
