# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

ContentForge AI v2.1 是基于 LangChain/LangGraph 的多平台内容自动化生产系统，实现从AI热点追踪到多平台内容发布的全流程自动化。

**核心工作流**：AI热点获取（11个数据源）→ 热点简报 → 长文本生成 → 并行生成小红书/Twitter内容 → 标题优化 → 配图提示词 → 质量评估

## 运行命令

```bash
# 激活虚拟环境
source venv/bin/activate

# 主程序：自动工作流（推荐）
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python src/main.py --once --workflow auto

# 可选：指定topic标识（不影响内容，仅用于文件命名）
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python src/main.py --once --workflow auto --topic "AI技术"

# 查看日志
tail -f logs/$(date +%Y%m%d)/app.log

# 测试Agent
cd test
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python test_storage.py
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python test_ai_trends.py --source hackernews
```

**注意**：将 PYTHONPATH 更新为你的实际项目路径。

## 架构概览

### 双工作流设计

项目包含两个独立的工作流协调器：

1. **ContentOrchestrator** (`src/main.py`) - 原始工作流：基于指定topic的通用内容生成
2. **AutoContentOrchestrator** (`src/auto_orchestrator.py`) - **自动工作流**：基于AI热点的全自动内容生成（推荐使用）

### Auto工作流Agent链

```
ai_trend_analyzer (11个数据源聚合)
  ↓
trends_digest (生成热点简报 → digest/)
  ↓
longform_generator (2500-3500字专业文章 → longform/)
  ↓
并行处理：
  ├─→ xiaohongshu_refiner (800-1000字小红书笔记 → xiaohongshu/)
  └─→ twitter_generator (5条推文thread → twitter/)
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
    model: "glm-4-plus"  # 生产环境主力
    base_url: "https://open.bigmodel.cn/api/paas/v4/"
  openai:
    model: "gpt-4o"
    base_url: "https://api.openai.com/v1"
```

**环境变量**：
- `ZHIPUAI_API_KEY`：智谱AI密钥
- `OPENAI_API_KEY`：OpenAI密钥

### 按日期分层的存储系统

使用 `DailyStorage` 类（`src/utils/storage.py:13`）管理按日期分层的存储：

```python
from src.utils.storage import get_storage

storage = get_storage("data")

# 保存文件（自动创建日期目录 YYYYMMDD/）
storage.save_json("longform", "article.json", data)
storage.save_markdown("digest", "digest.md", content)
storage.save_text("xiaohongshu", "prompts.txt", text)
```

**目录结构**：
```
data/20260107/
├── raw/                   # AI热点原始数据
├── digest/                # 热点简报
├── longform/              # 微信公众号文章
├── xiaohongshu/           # 小红书笔记 + 配图提示词
└── twitter/               # Twitter帖子 + 配图提示词
```

### LangGraph状态管理

使用 `WorkflowState` TypedDict（`src/state.py:61`）管理Agent之间的共享状态：

```python
from src.state import create_initial_state, update_state

# 创建初始状态（topic可选，留空则自动从热点生成）
state = create_initial_state(
    topic=None,  # 或 "AI技术"
    target_audience="技术从业者",
    content_type="干货分享"
)

# 更新状态（immutable模式）
new_state = update_state(state, {"new_field": value})
```

## 开发指南

### 添加新Agent

1. 创建Agent类（`src/agents/new_agent.py`）继承 `BaseAgent`
2. 实现 `execute(self, state: Dict[str, Any]) -> Dict[str, Any]` 方法
3. 在 `config/config.yaml` 的 `agents` 部分添加配置
4. 在 `src/auto_orchestrator.py` 的 `_init_agents()` 中初始化
5. 在 `_build_workflow()` 中添加到工作流图

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

## 性能优化

### 降低成本

- 使用 `glm-4-flash` 替代 `glm-4-plus`（成本降低约80%）
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
- **PROJECT_GUIDE.md** - 完整使用指南
- **STORAGE_QUICKREF.md** - 存储结构快速参考
- **AI_TRENDS_API_GUIDE.md** - 11个数据源详细说明
- **test/README.md** - 测试文件说明

---

**版本**: v2.1
**更新**: 2026-01-07
