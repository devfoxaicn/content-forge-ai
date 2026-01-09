# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

ContentForge AI v2.4 是基于 LangChain/LangGraph 的多平台内容自动化生产系统，实现从AI热点追踪到多平台内容发布的全流程自动化。

**核心工作流**：AI热点获取（11个数据源）→ 热点简报 → 深度研究（Web搜索）→ 长文本生成（分阶段）→ 质量检查（代码审查+事实核查）→ 并行生成小红书/Twitter内容 → 标题优化 → 配图提示词 → 质量评估

**v2.4新增**：100期技术博客系列模式 - 系统化生成100期技术内容，覆盖从LLM原理到AI基础设施的全栈知识。

## 运行命令

```bash
# 激活虚拟环境
source venv/bin/activate

# ===== 每日自动热点模式（默认） =====
# 主程序：自动工作流（推荐）
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python src/main.py --once --workflow auto

# ===== 100期技术博客系列模式（v2.4新增） =====
# 查看生成进度
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python src/series_orchestrator.py --progress

# 生成指定集数
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python src/series_orchestrator.py --episode 1

# 生成整个系列
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python src/series_orchestrator.py --series series_1

# 生成全部100期
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python src/series_orchestrator.py --all --start 1 --end 100

# ===== 数据迁移（v2.4新增） =====
# 查看迁移计划（演练模式）
python scripts/migrate_data.py --dry-run

# 执行数据迁移
python scripts/migrate_data.py

# 验证存储结构
python scripts/migrate_data.py --verify

# 查看日志
tail -f logs/$(date +%Y%m%d)/app.log

# 测试Agent
cd test
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python test_storage.py
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python test_ai_trends.py --source hackernews
```

**注意**：将 PYTHONPATH 更新为你的实际项目路径。

## 架构概览

### 三种工作流协调器

1. **ContentOrchestrator** (`src/main.py`) - 原始工作流：基于指定topic的通用内容生成
2. **AutoContentOrchestrator** (`src/auto_orchestrator.py`) - **自动工作流**：基于AI热点的全自动内容生成（推荐）
3. **SeriesOrchestrator** (`src/series_orchestrator.py`) - **系列工作流**（v2.4新增）：100期技术博客系列生成

**运行模式对比**：

| 模式 | 触发方式 | 数据来源 | 存储位置 |
|------|----------|----------|----------|
| 每日自动 | 定时任务 | AI热点分析 | `data/daily/日期/` |
| 100期系列 | 手动执行 | 100期预设 | `data/series/{系列ID}/episode_{xxx}/` |

### Auto工作流Agent链（v2.2）

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
并行处理：
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

### 11个AI热点数据源

**免费无需配置（8个）**：Hacker News, arXiv, Hugging Face, Stack Overflow, Dev.to, PyPI, GitHub Topics, Kaggle

**可选配置（3个）**：Reddit (`REDDIT_CLIENT_ID`), NewsAPI (`NEWSAPI_KEY`), GitHub Trending (第三方API)

## 核心模式

### Agent实现模式

所有Agent继承自 `BaseAgent`（`src/agents/base.py:14`），实现 `execute()` 方法：

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
- `_call_llm(prompt: str) -> str`：调用LLM（`src/agents/base.py:95`）
- `log(message: str, level: str = "INFO")`：记录日志（`src/agents/base.py:125`）
- `_load_system_prompt() -> str`：加载系统提示词（`src/agents/base.py:74`）

### LLM提供商切换

系统支持多个LLM提供商，通过 `config/config.yaml` 的 `llm.provider` 配置：

```yaml
llm:
  provider: "zhipuai"  # 或 "openai"
  zhipuai:
    model: "glm-4.7"  # 最新旗舰模型（2025年12月发布）
    # 其他可选: glm-4-flash（便宜快速）, glm-4-plus（上一代旗舰）
    base_url: "https://open.bigmodel.cn/api/paas/v4/"
  openai:
    model: "gpt-4o"
    base_url: "https://api.openai.com/v1"
```

**环境变量**：
- `ZHIPUAI_API_KEY`：智谱AI密钥（必需）
- `OPENAI_API_KEY`：OpenAI密钥（可选，使用OpenAI时）
- `TAVILY_API_KEY`：Tavily搜索API密钥（v2.2必需，用于ResearchAgent）

**编码专用端点**：`config/config.yaml:11` 使用 `https://open.bigmodel.cn/api/coding/paas/v4/` 获得最强编程能力

### 统一存储系统 (v2.4优化)

项目使用新的统一存储系统 `src/utils/storage_v2.py`，支持两种内容模式：

#### 存储结构

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
├── series/                   # 100期技术博客系列 (v2.4新增)
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

#### 使用方式

```python
from src.utils.storage_v2 import StorageFactory

# 1. 每日热点模式
daily_storage = StorageFactory.create_daily()
daily_storage.save_markdown("longform", "article.md", content)

# 2. 100期系列模式 (v2.4新增)
series_storage = StorageFactory.create_series(
    series_id="series_1_llm_foundation",
    episode_number=1
)
series_storage.save_markdown("longform", "article.md", content)
series_storage.save_episode_metadata(metadata)
```

#### 兼容旧版本

旧版本代码仍然兼容：

```python
from src.utils.storage import get_storage  # 旧版本

storage = get_storage("data")  # 使用 DailyStorage
```

#### 系列100期元数据管理 (v2.4新增)

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

### 按日期分层的存储系统（旧版本兼容）

#### DailyStorage（每日自动模式）

使用 `DailyStorage` 类（`src/utils/storage.py:13`）管理按日期分层的存储：

```python
from src.utils.storage import get_storage

storage = get_storage("data")

# 保存文件（自动创建日期目录 YYYYMMDD/）
storage.save_json("longform", "article.json", data)
storage.save_markdown("digest", "digest.md", content)
storage.save_text("xiaohongshu", "prompts.txt", text)
```

#### BatchStorage（批量生成模式，v2.3新增，已移除）

**注意**：BatchStorage已在v2.4中移除。如需批量生成内容，请使用100期系列模式。

**存储目录结构**：
```
data/
├── 20260107/                    # 每日自动生成
│   ├── raw/
│   ├── digest/
│   ├── longform/
│   ├── xiaohongshu/
│   └── twitter/
│
└── batch/                       # 批量生成（v2.3新增）
    └── 20260109_batch_ai_tools/  # {日期}_batch_{批次名}
        ├── raw/
        ├── digest/
        ├── longform/
        ├── xiaohongshu/
        ├── twitter/
        └── batch_metadata.json
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

**v2.2重要变更**：`topic` 参数仅用于文件命名（如 `article_AI技术_*.md`），实际内容完全基于实时AI热点自动生成。即使指定topic，内容也会从当天热点中选取。

```python
# 更新状态（immutable模式）
new_state = update_state(state, {"new_field": value})
```

## 工作流执行顺序

工作流执行顺序在 `src/auto_orchestrator.py:_build_workflow()` 中定义（`src/auto_orchestrator.py:177`）：

1. AI热点分析 → 热点汇总 → 内容研究
2. 长文本生成 → 代码审查
3. 小红书精炼 → Twitter生成（顺序执行，避免并发冲突）
4. 标题优化 → 图像建议 → 图像生成
5. 事实核查 → 质量评估 → 发布

**重要**：小红书和Twitter Agent顺序执行（非并行）以避免状态更新冲突（`src/auto_orchestrator.py:213`）

### 提示词模板系统

每个Agent的系统提示词存储在 `config/prompts.yaml` 中，按Agent名称组织：

```yaml
prompts:
  agent_name:  # 对应Agent类名的小写形式（如 "xiaohongshu_refiner"）
    system: "你是专业的..."
    user_template: "请根据以下内容生成..."  # 可选的用户提示词模板
```

Agent通过 `_load_system_prompt()` 方法加载提示词（`src/agents/base.py:74`）。

## 开发指南

### 添加新Agent

1. 创建Agent类（`src/agents/new_agent.py`）继承 `BaseAgent`
2. 实现 `execute(self, state: Dict[str, Any]) -> Dict[str, Any]` 方法
3. 在 `config/config.yaml` 的 `agents` 部分添加配置（参考现有Agent）
4. 在 `src/auto_orchestrator.py` 的 `_init_agents()` 中初始化
5. 在 `_build_workflow()` 中添加到工作流图（定义执行顺序）

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
    # Agent特定参数...
```

### 测试调试

**启用详细日志**：
```yaml
# config/config.yaml
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
- 使用 `self.log()` 而不是 `print()`（`src/agents/base.py:125`）
- Agent的 `execute()` 方法必须返回完整的状态字典
- 异常处理要更新 `error_message` 和 `current_step` 字段

## 架构要点

### 分阶段长文本生成（v2.2核心）

`LongFormGeneratorAgent` 使用三阶段生成策略避免超时（`src/agents/longform_generator.py:72`）：

1. **第一阶段**：生成文章大纲
2. **第二阶段**：逐节展开内容（循环调用LLM）
3. **第三阶段**：生成总结

这种方式可以生成9000-13000字的专业深度分析，而不会触发API超时。

### Web搜索深度研究（v2.2新增）

`ResearchAgent` 使用Tavily API进行Web搜索，收集：
- 官方文档和技术规格
- GitHub开源项目和代码示例
- 技术博客和深度分析文章
- 社区讨论和实践案例

研究数据存储在 `state["research_data"]` 中，供后续Agent使用。

### 质量保证三重检查（v2.2新增）

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

- 并行执行独立Agent（xiaohongshu_refiner + twitter_generator）
- 使用mock模式进行开发测试
- 减少数据源数量

## 常见问题

### Import错误

确保从项目根目录运行，并设置PYTHONPATH（替换为你的实际项目路径）：
```bash
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python src/main.py --once
```

### API Key错误

检查环境变量：
```bash
echo $ZHIPUAI_API_KEY
# 或
cat .env
```

### 日志查看

日志按日期分层存储在 `logs/YYYYMMDD/app.log`：
```bash
tail -f logs/$(date +%Y%m%d)/app.log
```

### 只生成部分内容

编辑 `config/config.yaml` 禁用不需要的Agent：
```yaml
agents:
  longform_generator:
    enabled: false  # 不生成长文章
  xiaohongshu_refiner:
    enabled: false  # 不生成小红书笔记
  twitter_generator:
    enabled: false  # 不生成Twitter帖子
```

## 相关文档

- **README.md** - 项目概述和快速开始
- **BATCH_MODE_GUIDE.md** - 批量生成模式使用指南（v2.3新增）
- **PROJECT_GUIDE.md** - 完整使用指南
- **STORAGE_QUICKREF.md** - 存储结构快速参考
- **AI_TRENDS_API_GUIDE.md** - 11个数据源详细说明
- **test/README.md** - 测试文件说明

---

**版本**: v2.4
**更新**: 2026-01-09
