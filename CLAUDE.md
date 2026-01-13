# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

ContentForge AI v2.5 是基于 LangChain/LangGraph 的多平台内容自动化生产系统，实现从AI热点追踪到多平台内容发布的全流程自动化。

**核心架构**：使用统一入口 `src/main.py`，通过 `--mode` 参数支持两种内容生成模式：
- **auto模式** - 基于实时AI热点的全自动内容生成（默认）
- **series模式** - 100期技术博客系列生成（系统化教程内容）

**核心工作流**：AI热点获取（11个数据源）→ 热点简报 → 深度研究（Web搜索）→ 长文本生成（分阶段）→ 质量检查（代码审查+事实核查）→ 小红书/Twitter内容生成 → 标题优化 → 配图提示词 → 质量评估

## 项目设置

### 环境配置

```bash
# 1. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量
cp .env.example .env
# 编辑.env，添加必需的API密钥
```

### 依赖管理

核心依赖（requirements.txt）：
- **LangChain/LangGraph** - Agent框架和工作流编排
- **langchain-openai** - LLM接口（兼容ZhipuAI）
- **loguru** - 结构化日志
- **pyyaml** - 配置文件解析
- **python-dotenv** - 环境变量管理
- **arxiv, praw** - AI热点数据源客户端

## 运行命令

项目使用统一入口 `src/main.py`，通过 `--mode` 参数切换模式。

```bash
# 激活虚拟环境
source venv/bin/activate

# ===== 自动模式（基于AI热点） =====
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python src/main.py --mode auto --once

# ===== 系列模式（100期技术博客） =====
# 查看生成进度
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python src/main.py --mode series --progress

# 生成指定集数
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python src/main.py --mode series --episode 1

# 生成整个系列
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python src/main.py --mode series --series series_1

# 生成全部100期（自动跳过已完成的）
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python src/main.py --mode series --all --start 1 --end 100

# ===== 自动化脚本 =====
# 自动运行并提交到GitHub（通过cron调用）
./run_and_commit.sh

# ===== Cron定时任务配置 =====
# 编辑crontab
crontab -e

# 每天早上3点自动运行（自动模式）
0 3 * * * /path/to/content-forge-ai/run_and_commit.sh

# 系列模式：每天生成一期（第1-100期循环）
CONTENT_FORGE_MODE=series SERIES_EPISODE=1 0 3 * * * /path/to/content-forge-ai/run_and_commit.sh
# 注：需要手动调整SERIES_EPISODE或使用SERIES_ALL批量模式

# ===== 查看日志 =====
tail -f logs/$(date +%Y%m%d)/app.log

# ===== 测试Agent =====
cd test
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python test_storage.py
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python test_ai_trends.py --source hackernews
```

**重要**：将 PYTHONPATH 替换为你的实际项目路径。

## 架构概览

### 核心设计模式

ContentForge AI 使用多种设计模式构建可扩展的内容生成系统：

**1. 多协调器模式（Multi-Orchestrator Pattern）**
- 三种协调器实现不同的内容生成策略
- 使用策略模式允许运行时切换
- 每种协调器管理自己的工作流和存储

**2. 工厂模式（Factory Pattern）**
- `StorageFactory` 根据模式创建相应的存储实例
- 支持两种存储策略：`DailyStorage`（按日期）和 `SeriesStorage`（按系列）

**3. 责任链模式（Chain of Responsibility）**
- LangGraph工作流实现Agent链
- 每个Agent处理并传递状态
- 状态不可变更新：`{**state, **updates}`

**4. 模板方法模式（Template Method Pattern）**
- 所有Agent继承自 `BaseAgent`
- 实现标准的 `execute(state: Dict) -> Dict` 接口
- 统一的日志、错误处理、LLM调用

### 双协调器对比

| 特性 | AutoContentOrchestrator | SeriesOrchestrator |
|------|-------------------------|-------------------|
| **文件位置** | `src/auto_orchestrator.py` | `src/series_orchestrator.py` |
| **数据来源** | 11个实时API数据源 | 100期预设配置 |
| **触发方式** | 定时任务或手动 | 手动执行 |
| **存储路径** | `data/daily/YYYYMMDD/` | `data/series/{id}/episode_{xxx}/` |
| **Agent数量** | 完整13个Agent | 优化8个Agent |
| **工作流** | LangGraph图式执行 | 顺序执行with错误恢复 |
| **元数据** | 简单执行追踪 | 完整进度和状态管理 |
| **状态字段** | 使用 `trending_topics` | 使用 `current_topic` + `selected_ai_topic` |

**统一入口**：`src/main.py` 通过 `--mode` 参数选择协调器

### Auto工作流Agent链

```
ai_trend_analyzer (11个数据源聚合)
  ↓
trends_digest (生成热点简报 → digest/)
  ↓
research_agent (Web搜索深度研究，收集官方文档/GitHub/技术博客)
  ↓
longform_generator (分阶段生成9000-13000字专业文章 → longform/)
  ↓
质量检查：
  ├─→ code_review_agent (代码审查)
  └─→ fact_check_agent (事实核查)
  ↓
顺序执行：
  ├─→ xiaohongshu_refiner (3000-3500字小红书笔记 → xiaohongshu/)
  └─→ twitter_generator (5-8条推文thread → twitter/)
  ↓
title_optimizer (标题优化)
  ↓
image_generator (生成中文配图提示词 → prompts_*.txt)
  ↓
quality_evaluator (质量评估)
  ↓
publisher (可选发布)
```

**重要**：小红书和Twitter Agent顺序执行（非并行）以避免状态更新冲突（`src/auto_orchestrator.py:213`）

### 11个AI热点数据源

**免费无需配置（8个）**：Hacker News, arXiv, Hugging Face, Stack Overflow, Dev.to, PyPI, GitHub Topics, Kaggle

**可选配置（3个）**：Reddit (`REDDIT_CLIENT_ID`), NewsAPI (`NEWSAPI_KEY`), GitHub Trending (第三方API)

**数据源实现**：位于 `src/data_sources/` 目录，每个数据源是一个独立的类继承自 `BaseDataSource`

```python
from src.data_sources.base import BaseDataSource

class CustomDataSource(BaseDataSource):
    def fetch_trends(self, limit: int = 5) -> List[Dict]:
        """实现数据获取逻辑"""
        # 返回格式：[{"title": "...", "url": "...", ...}]
        pass
```

## 核心模式

### Agent实现模式

所有Agent继承自 `BaseAgent`（`src/agents/base.py:22`），实现 `execute()` 方法：

```python
from src.agents.base import BaseAgent
from typing import Dict, Any

class NewAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        super().__init__(config, prompts)

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Agent逻辑
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

**关键方法**：
- `_call_llm(prompt: str) -> str`：调用LLM（内部实现）
- `log(message: str, level: str = "INFO")`：记录日志（使用loguru）
- `_load_system_prompt() -> str`：加载系统提示词（从 `config/prompts.yaml`）
- `_init_llm() -> ChatOpenAI`：初始化LLM实例（支持ZhipuAI/OpenAI）

### LLM提供商切换

系统支持多个LLM提供商，通过 `config/config.yaml` 的 `llm.provider` 配置：

```yaml
llm:
  provider: "zhipuai"  # 或 "openai"
  zhipuai:
    model: "glm-4.7"  # 最新旗舰模型
    # 其他可选: glm-4-flash（便宜快速）, glm-4-plus（上一代旗舰）
    base_url: "https://open.bigmodel.cn/api/coding/paas/v4/"  # 编码专用端点
  openai:
    model: "gpt-4o"
    base_url: "https://api.openai.com/v1"
```

**必需环境变量**：
- `ZHIPUAI_API_KEY`：智谱AI密钥（主要LLM提供商）
- `TAVILY_API_KEY`：Tavily搜索API密钥（用于ResearchAgent深度研究）

**可选环境变量**：
- `OPENAI_API_KEY`：OpenAI密钥（备用LLM提供商）
- `GEMINI_API_KEY`：Google Gemini密钥（用于ImageAgent提示词优化）
- `REDDIT_CLIENT_ID`、`REDDIT_CLIENT_SECRET`：Reddit API（扩展数据源）
- `NEWSAPI_KEY`：NewsAPI密钥（扩展数据源）

**编码专用端点**：`config/config.yaml:11` 使用 `https://open.bigmodel.cn/api/coding/paas/v4/` 获得最强编程能力

**Thinking 深度思考模式**：GLM-4.7 支持Thinking模式（`config/config.yaml:20-22`），但目前禁用（`enabled: false`）因为参数兼容性问题。启用后可获得更强的推理能力。

**API配置管理**：系统通过 `src/utils/api_config.py` 中的 `APIConfigManager` 统一管理API密钥和端点，支持环境变量和配置文件两种方式。

```python
from src.utils.api_config import get_api_config

# 获取配置管理器实例
api_config = get_api_config()

# 获取API密钥
zhipu_key = api_config.get_api_key("zhipuai")
tavily_key = api_config.get_api_key("tavily")

# 获取API端点
base_url = api_config.get_endpoint("llm.zhipuai.base_url")

# 支持环境变量和config.yaml两种配置方式
# 优先级：环境变量 > config.yaml > 默认值
```

### 统一存储系统

项目使用 `src/utils/storage_v2.py` 中的统一存储系统，基于工厂模式：

**存储结构**：
```
data/
├── daily/                    # 每日热点模式
│   └── YYYYMMDD/
│       ├── raw/
│       ├── digest/
│       ├── longform/
│       ├── xiaohongshu/
│       └── twitter/
│
├── series/                   # 100期技术博客系列
│   └── {series_id}/
│       ├── episode_{xxx}/
│       │   ├── raw/
│       │   ├── digest/
│       │   ├── longform/
│       │   ├── xiaohongshu/
│       │   └── twitter/
│       └── series_metadata.json
│
└── archive/                  # 归档内容 (预留)
```

**使用方式**：
```python
from src.utils.storage_v2 import StorageFactory

# 1. 每日热点模式
daily_storage = StorageFactory.create_daily()
daily_storage.save_markdown("longform", "article.md", content)

# 2. 100期系列模式
series_storage = StorageFactory.create_series(
    series_id="series_1_llm_foundation",
    episode_number=1
)
series_storage.save_markdown("longform", "article.md", content)
series_storage.save_episode_metadata(metadata)
```

**系列元数据管理**：
```python
from src.utils.series_manager import get_series_metadata, print_progress_summary

# 加载元数据
metadata = get_series_metadata("config/blog_topics_100_complete.json")

# 查询话题
topic = metadata.get_topic_by_episode(1)
series = metadata.get_series_by_id("series_1")

# 更新状态
metadata.update_topic_status("topic_001", "completed")

# 查看进度
print_progress_summary()
```

### 文件命名规范（TopicFormatter）

`TopicFormatter` 类（`src/utils/series_manager.py:201`）提供统一的文件命名格式：

```python
from src.utils.series_manager import TopicFormatter

# 生成文件名前缀：ep001_llm_transformer_attention_mechanism
prefix = TopicFormatter.generate_filename_prefix(topic)

# 生成Markdown文件名：ep001_llm_transformer_attention_mechanism_article.md
filename = TopicFormatter.generate_markdown_filename(topic, "article")

# 格式化话题摘要（用于日志）
summary = TopicFormatter.format_topic_summary(topic)
# 输出: ✅ Episode 001 | LLM的Transformer架构与注意力机制 [series_1]
```

### 系列路径管理（SeriesPathManager）

`SeriesPathManager` 类（`src/utils/series_manager.py`）管理系列文件夹的命名转换：

```python
from src.utils.series_manager import SeriesPathManager

# series_id 转路径文件夹
path = SeriesPathManager.get_series_path("series_1")
# 返回: "series_1_llm_foundation"

# 路径文件夹 转 series_id
series_id = SeriesPathManager.get_series_id_from_path("series_1_llm_foundation")
# 返回: "series_1"

# 命名映射表（硬编码）
NAMING_MAP = {
    "series_1": "series_1_llm_foundation",
    "series_2": "series_2_rag_technique",
    # ... 共10个系列
}
```

**重要**：系列文件夹格式为 `series_X_descriptive_name`，这是 v2.5 版本的重要改进，确保存储路径的语义化。

### LangGraph状态管理

使用 `WorkflowState` TypedDict（`src/state.py:61`）管理Agent之间的共享状态：

```python
from src.state import create_initial_state, update_state

# 创建初始状态（topic可选，留空则自动从热点生成）
state = create_initial_state(
    topic=None,  # 或 "AI技术"（仅作为文件标识，不影响内容）
    target_audience="技术从业者",
    content_type="干货分享"
)
```

**重要**：`topic` 参数仅用于文件命名，实际内容完全基于实时AI热点自动生成。

```python
# 更新状态（immutable模式）
new_state = update_state(state, {"new_field": value})
```

**LangGraph节点包装器**（`src/auto_orchestrator.py:270-277`）：
```python
def _create_agent_node(self, agent: BaseAgent):
    """创建LangGraph节点包装器"""
    def node_func(state: Dict[str, Any]) -> Dict[str, Any]:
        result = agent.execute(state)
        # 记录执行顺序
        return add_agent_to_order(result, agent.name)
    return node_func
```

每个Agent的输出会通过 `{**state, **updates}` 模式合并到状态中，确保状态的不可变性。

## 工作流执行顺序

**AutoContentOrchestrator**（LangGraph模式）：工作流在 `src/auto_orchestrator.py:_build_workflow()` 中定义

**SeriesOrchestrator**（顺序模式）：工作流在 `src/series_orchestrator.py:_execute_workflow()` 中定义，包含安全包装和延迟机制

**SeriesOrchestrator 安全执行机制**（`src/series_orchestrator.py:248-262`）：
```python
def _call_agent_safely(agent_name: str, state: Dict[str, Any]) -> Dict[str, Any]:
    """安全调用agent，处理异常"""
    try:
        result = self.agents[agent_name].execute(state)
        time.sleep(2)  # 添加延迟避免API并发限制
        return result
    except Exception as e:
        logger.error(f"[{agent_name}] 执行失败: {e}")
        time.sleep(2)  # 失败时也添加延迟
        return state  # 返回原状态，允许继续执行
```

**执行顺序**：
1. AI热点分析 → 热点汇总 → 内容研究
2. 长文本生成 → 代码审查
3. 小红书精炼 → Twitter生成
4. 标题优化 → 图像生成
5. 质量评估

**提示词模板系统**：每个Agent的系统提示词存储在 `config/prompts.yaml` 中，按Agent类名小写组织

### Agent依赖关系

| Agent | 依赖字段 | 输出字段 | 说明 |
|-------|---------|---------|------|
| ai_trend_analyzer | - | trending_topics, selected_ai_topic | 数据源 |
| trends_digest | trending_topics | digest_content | 可选 |
| research_agent | selected_ai_topic | research_data, research_summary | 为longform提供研究背景 |
| longform_generator | selected_ai_topic, research_data | longform_article | 核心内容 |
| code_review_agent | longform_article | code_review_result | 质量保证 |
| fact_check_agent | longform_article | fact_check_result | 质量保证 |
| xiaohongshu_refiner | longform_article | xiaohongshu_note | 内容适配 |
| twitter_generator | longform_article | twitter_post | 内容适配 |
| title_optimizer | longform_article | optimized_titles | SEO优化 |
| image_generator | xiaohongshu_note or twitter_post | image_prompts | 配图生成 |
| quality_evaluator | 所有输出 | quality_report | 最终评估 |

### 重要注意事项

1. **小红书和Twitter Agent必须顺序执行**：它们都读取 `longform_article`，但不应并行运行以避免状态更新冲突（`src/auto_orchestrator.py:213`）

2. **长文本生成Agent需要研究数据**：`longform_generator` 优先使用 `research_data`，如果没有则仅基于 `selected_ai_topic` 生成

3. **系列模式的状态字段特殊处理**：需要同时设置 `current_topic` 和 `selected_ai_topic` 以确保兼容性（`src/series_orchestrator.py:_initialize_state()`）

4. **分阶段生成避免超时**：`LongFormGeneratorAgent` 使用三阶段生成（大纲→章节展开→总结），每阶段独立LLM调用（`src/agents/longform_generator.py:73`）

## 开发指南

### 添加新Agent

**AutoContentOrchestrator模式**：
1. 创建Agent类（`src/agents/new_agent.py`）继承 `BaseAgent`
2. 实现 `execute(self, state: Dict[str, Any]) -> Dict[str, Any]` 方法
3. 在 `config/config.yaml` 的 `agents` 部分添加配置
4. 在 `src/auto_orchestrator.py` 的 `_init_agents()` 中初始化
5. 在 `_build_workflow()` 中添加到LangGraph工作流

**SeriesOrchestrator模式**：
1. 同上创建Agent类
2. 在 `src/series_orchestrator.py` 的 `_init_agents()` 中添加到 `agent_classes` 字典
3. 在 `_execute_workflow()` 中添加调用逻辑

**重要**：Agent必须返回完整的状态字典，使用 `{**state, "new_field": value}` 模式更新状态

### 常用状态字段

不同Agent在状态中读写的常用字段：

| 状态字段 | 写入Agent | 读取Agent | 说明 |
|---------|----------|----------|------|
| `trending_topics` | ai_trend_analyzer | trends_digest | AI热点列表（Auto模式） |
| `digest_content` | trends_digest | - | 热点简报内容 |
| `research_data` | research_agent | longform_generator | Web搜索研究数据 |
| `selected_ai_topic` | ai_trend_analyzer / series_orchestrator | longform_generator | 选中的AI话题 |
| `current_topic` | series_orchestrator | - | 当前话题（Series模式） |
| `longform_article` | longform_generator | code_review_agent, xiaohongshu_refiner, twitter_generator | 长文本文章 |
| `code_review_result` | code_review_agent | - | 代码审查结果 |
| `fact_check_result` | fact_check_agent | - | 事实核查结果 |
| `xiaohongshu_note` | xiaohongshu_refiner | - | 小红书笔记 |
| `twitter_post` | twitter_generator | - | Twitter帖子 |
| `optimized_titles` | title_optimizer | - | 优化后的标题 |
| `image_prompts` | image_generator | - | 配图提示词 |
| `quality_report` | quality_evaluator | - | 质量评估报告 |
| `error_message` | 任何Agent | - | 错误信息 |
| `current_step` | 任何Agent | - | 当前执行步骤 |
| `execution_time` | orchestrator | - | 执行时间统计 |
| `agent_execution_order` | orchestrator | - | Agent执行顺序记录 |

**重要提示**：
- `WorkflowState` TypedDict 定义了所有可能的字段，但实际使用时是普通Dict
- Auto模式使用 `trending_topics`，Series模式使用 `current_topic` 和 `selected_ai_topic`
- 状态更新使用不可变模式：`{**state, **updates}`
- **实际使用的状态字段**比 `WorkflowState` TypedDict 定义更丰富，包括：
  - `selected_ai_topic` - 选中的AI热点（Auto模式）
  - `research_data` / `research_summary` - 研究数据
  - `longform_article` - 长文本文章
  - `code_review_result` - 代码审查结果
  - `fact_check_result` - 事实核查结果
  - `xiaohongshu_note` - 小红书笔记
  - `twitter_post` - Twitter帖子
  - `optimized_titles` - 优化后的标题
  - `image_prompts` - 配图提示词
  - `quality_report` - 质量评估报告

### 错误处理模式

**Agent级别错误处理**：
```python
def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # Agent逻辑
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

**Series模式安全执行**（`src/series_orchestrator.py:248`）：
```python
def _call_agent_safely(agent_name: str, state: Dict[str, Any]) -> Dict[str, Any]:
    """安全调用agent，处理异常"""
    try:
        result = self.agents[agent_name].execute(state)
        time.sleep(2)  # 添加延迟避免API并发限制
        return result
    except Exception as e:
        logger.error(f"[{agent_name}] 执行失败: {e}")
        time.sleep(2)  # 失败时也添加延迟
        return state  # 返回原状态，允许继续执行
```

**错误恢复策略**：
- Agent失败时返回原状态，允许工作流继续
- 记录 `error_message` 和 `current_step` 用于调试
- 使用 `retry_count` 和 `max_retries` 字段控制重试（在WorkflowState中定义）

### Agent配置模式

每个Agent在 `config/config.yaml` 中有独立配置块：
    enabled: true           # 是否启用
    mock_mode: false        # 测试模式（使用模拟数据）
    max_tokens: 2000        # Token限制
    temperature: 0.7        # 温度参数
```

### 测试调试

**启用详细日志**：编辑 `config/config.yaml`
```yaml
logging:
  level: "DEBUG"
```

**Mock模式测试**（不消耗API配额）：
```yaml
agents:
  ai_trend_analyzer:
    mock_mode: true
```

**单独测试Agent**：
```bash
cd test
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python test_ai_trends.py --source hackernews
```

### 测试文件说明

项目包含多个测试文件用于验证不同组件：

| 测试文件 | 用途 |
|---------|------|
| `test_ai_trends.py` | 测试AI热点获取（支持单数据源测试） |
| `test_storage.py` | 测试存储系统功能 |
| `test_topic_logic.py` | 测试topic参数处理逻辑 |
| `test_digest.py` | 测试热点简报生成 |
| `test_auto_topic.py` | 测试自动模式topic处理 |

**测试AI热点单个数据源**：
```bash
cd test
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python test_ai_trends.py --source hackernews
# 可选数据源: hackernews, arxiv, huggingface, stackoverflow, devto, pypi, github_topics, kaggle
```

## 代码规范

- 使用类型注解（遵循 PEP 484）
- 遵循 PEP 8
- 添加docstring
- 使用 `self.log()` 而不是 `print()`
- Agent的 `execute()` 方法必须返回完整的状态字典
- 异常处理要更新 `error_message` 和 `current_step` 字段

## 架构要点

### 分阶段长文本生成

`LongFormGeneratorAgent`（`src/agents/longform_generator.py:73`）使用三阶段生成策略避免超时：

1. **第一阶段**：生成文章大纲
2. **第二阶段**：逐节展开内容（循环调用LLM，可选上下文窗口）
3. **第三阶段**：生成总结

这种方式可以生成9000-13000字的专业深度分析。

### Web搜索深度研究

`ResearchAgent` 使用Tavily API进行Web搜索，收集官方文档、GitHub、技术博客等资料。研究数据存储在 `state["research_data"]` 中。

### 质量保证三重检查

1. **CodeReviewAgent**：审查代码示例的正确性和最佳实践
2. **FactCheckAgent**：核查事实陈述、技术参数、日期时间
3. **QualityEvaluatorAgent**：综合评估内容质量（打分7-10分）

## 性能优化

### 降低成本

- 使用 `glm-4-flash` 替代 `glm-4.7`（成本降低约80%）
- 减少数据源数量（如仅保留hackernews + arxiv）
- 降低 `max_tokens` 设置
- 禁用不需要的Agent（设置 `enabled: false`）

### 加速执行

- 使用mock模式进行开发测试
- 减少数据源数量

## 常见问题

**Import错误**：确保从项目根目录运行，并设置PYTHONPATH
```bash
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python src/main.py --once
```

**API Key错误**：检查环境变量
```bash
echo $ZHIPUAI_API_KEY
# 或
cat .env
```

**日志查看**：日志按日期分层存储在 `logs/YYYYMMDD/app.log`
```bash
tail -f logs/$(date +%Y%m%d)/app.log
```

**只生成部分内容**：编辑 `config/config.yaml` 禁用不需要的Agent
```yaml
agents:
  longform_generator:
    enabled: false
```

**调试特定Agent**：启用mock_mode避免API调用
```yaml
agents:
  ai_trend_analyzer:
    mock_mode: true
```

## 常用工作流程

### 测试单个Agent
```bash
cd test
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python test_ai_trends.py --source hackernews
```

### 生成单期系列内容
```bash
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python src/main.py --mode series --episode 1
```

### 查看系列进度
```bash
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python src/main.py --mode series --progress
```

### 批量生成系列（带跳过）
```bash
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python src/main.py --mode series --all --start 1 --end 10
```

## 关键文件位置

| 文件 | 用途 |
|------|------|
| `src/main.py` | 统一入口点 |
| `src/auto_orchestrator.py` | LangGraph工作流编排 |
| `src/series_orchestrator.py` | 系列模式协调器 |
| `src/state.py` | 状态定义 |
| `src/agents/base.py` | Agent基类 |
| `src/agents/longform_generator.py` | 长文本生成（分阶段） |
| `src/agents/ai_trend_analyzer_real.py` | AI热点分析 |
| `src/utils/storage_v2.py` | 统一存储系统 |
| `src/utils/series_manager.py` | 系列管理工具 |
| `config/config.yaml` | 主配置文件 |
| `config/blog_topics_100_complete.json` | 100期内容规划 |
| `run_and_commit.sh` | 自动化部署脚本 |

## 相关文档

- **README.md** - 项目概述和快速开始
- **test/README.md** - 测试文件说明

---

**版本**: v2.5
**更新**: 2026-01-13

## 改进总结 (2026-01-13)

本次更新改进了 CLAUDE.md 文档，主要包括：

1. **新增Agent依赖关系表** - 清晰展示各Agent之间的依赖关系和数据流
2. **新增重要注意事项** - 强调小红书/Twitter顺序执行、研究数据依赖等关键点
3. **新增常用工作流程** - 提供测试、生成、查看进度的具体命令
4. **新增关键文件位置表** - 快速定位重要文件
5. **完善状态字段说明** - 补充实际使用的额外字段
6. **新增调试技巧** - mock_mode使用、部分Agent禁用等

## 重要更新点 (2026-01-12)

### 系列存储路径命名规则
系列文件夹使用 `SeriesPathManager` 统一管理命名，格式为 `series_X_descriptive_name`：
- `series_1_llm_foundation` (LLM原理基础)
- `series_2_rag_technique` (RAG技术实战)
- `series_3_agent_development` (Agent智能体开发)
- `series_4_prompt_engineering` (提示工程)
- `series_5_model_deployment` (模型部署与优化)
- `series_6_multimodal_frontier` (多模态与前沿技术)
- `series_7_ai_coding_tools` (AI编程与开发工具)
- `series_8_ai_data_engineering` (AI数据处理与工程)
- `series_9_ai_applications` (AI应用场景实战)
- `series_10_ai_infrastructure` (AI基础设施与架构)

### 状态字段映射注意事项
在 SeriesOrchestrator 模式下，需要同时设置 `current_topic` 和 `selected_ai_topic` 字段以确保与 LongFormGeneratorAgent 兼容：
```python
state = update_state(state, {
    "current_topic": topic,
    "selected_ai_topic": {  # LongFormGeneratorAgent期望的字段
        "title": topic["title"],
        "description": topic.get("description", ""),
        "source": f"series_{series_id}_episode_{episode_number}",
        "url": "",
        "tags": topic.get("keywords", []),
        "key_points": [topic.get("description", "")]
    }
})
```

### 自动化部署脚本
`run_and_commit.sh` 支持通过环境变量配置模式：
- `CONTENT_FORGE_MODE=auto` (默认) - 基于AI热点
- `CONTENT_FORGE_MODE=series` - 100期技术博客
- `SERIES_EPISODE=1` - 指定生成第1期
- `SERIES_ALL=1 SERIES_START=1 SERIES_END=10` - 批量生成第1-10期
