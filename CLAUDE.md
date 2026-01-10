# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

ContentForge AI v2.5 是基于 LangChain/LangGraph 的多平台内容自动化生产系统，实现从AI热点追踪到多平台内容发布的全流程自动化。

**核心架构**：使用统一入口 `src/main.py`，通过 `--mode` 参数支持两种内容生成模式：
- **auto模式** - 基于实时AI热点的全自动内容生成（默认）
- **series模式** - 100期技术博客系列生成（系统化教程内容）

**核心工作流**：AI热点获取（11个数据源）→ 热点简报 → 深度研究（Web搜索）→ 长文本生成（分阶段）→ 质量检查（代码审查+事实核查）→ 小红书/Twitter内容生成 → 标题优化 → 配图提示词 → 质量评估

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

### 三种协调器对比

| 特性 | AutoContentOrchestrator | SeriesOrchestrator |
|------|-------------------------|-------------------|
| **文件位置** | `src/auto_orchestrator.py` | `src/series_orchestrator.py` |
| **数据来源** | 11个实时API数据源 | 100期预设配置 |
| **触发方式** | 定时任务或手动 | 手动执行 |
| **存储路径** | `data/daily/YYYYMMDD/` | `data/series/{id}/episode_{xxx}/` |
| **Agent数量** | 完整13个Agent | 优化8个Agent |
| **工作流** | LangGraph图式执行 | 顺序执行with错误恢复 |
| **元数据** | 简单执行追踪 | 完整进度和状态管理 |

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

**API配置管理**：系统通过 `src/utils/api_config.py` 中的 `APIConfigManager` 统一管理API密钥和端点，支持环境变量和配置文件两种方式。

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

## 工作流执行顺序

**AutoContentOrchestrator**（LangGraph模式）：工作流在 `src/auto_orchestrator.py:_build_workflow()` 中定义

**SeriesOrchestrator**（顺序模式）：工作流在 `src/series_orchestrator.py:_execute_workflow()` 中定义，包含安全包装和延迟机制

**执行顺序**：
1. AI热点分析 → 热点汇总 → 内容研究
2. 长文本生成 → 代码审查
3. 小红书精炼 → Twitter生成
4. 标题优化 → 图像生成
5. 质量评估

**提示词模板系统**：每个Agent的系统提示词存储在 `config/prompts.yaml` 中，按Agent类名小写组织

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

### Agent配置模式

每个Agent在 `config/config.yaml` 中有独立配置块：

```yaml
agents:
  agent_name:
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

## 相关文档

- **README.md** - 项目概述和快速开始
- **test/README.md** - 测试文件说明

---

**版本**: v2.5
**更新**: 2026-01-10
