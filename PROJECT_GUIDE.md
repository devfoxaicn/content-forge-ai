# ContentForge AI - 项目完整指南

> **AI驱动的多平台内容自动化生产工厂** v2.1

## 📖 项目概述

ContentForge AI 是一个基于 LangChain/LangGraph 的智能内容生产系统，实现从AI热点追踪到多平台内容发布的全流程自动化。

### 核心功能

1. **AI热点追踪** - 11个免费数据源，实时获取AI技术热点
2. **热点简报** - 汇总当天热点，生成杂志风格简报
3. **专业文章** - 2500-3500字，微信公众号格式
4. **小红书笔记** - 800-1000字干货风格
5. **Twitter帖子** - Thread形式（5条推文）
6. **配图提示词** - 为每个内容生成AI绘图提示词

## 🚀 快速开始

### 安装

```bash
# 1. 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置API Key
cp .env.example .env
# 编辑 .env，添加 ZHIPUAI_API_KEY

# 4. 运行
python src/main.py --once --workflow auto
```

### 运行模式

```bash
# 立即执行一次
python src/main.py --once --workflow auto

# 自动模式（定时运行）
python src/main.py --auto --workflow auto

# 指定topic（可选）
python src/main.py --once --workflow auto --topic "AI技术"
```

## 📂 存储结构

```
data/20260107/              # 按日期分层
├── raw/                   # AI热点原始数据
├── digest/                # 热点简报
├── longform/              # 微信公众号文章
├── xiaohongshu/           # 小红书笔记 + 配图提示词
└── twitter/               # Twitter帖子 + 配图提示词

logs/20260107/             # 日志按日期分层
└── app.log
```

## 🤖 工作流程

```
1. AI热点获取（11个数据源）
   - Hacker News, arXiv, Hugging Face, GitHub Topics, Stack Overflow
   - Dev.to, PyPI, Kaggle, Reddit（可选）, NewsAPI（可选）
   ↓
2. 热点汇总 → 简报生成（digest/）
   ↓
3. 筛选TOP 1热点
   ↓
4. 长文本生成（longform/）- 2500-3500字
   ↓
5. 并行处理：
   ├─→ 小红书精炼（xiaohongshu/）- 800-1000字
   └─→ Twitter生成（twitter/）- 5条推文
   ↓
6. 标题优化 + 配图提示词 + 质量评估
   ↓
7. 保存到 data/YYYYMMDD/
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

### 主配置：config/config.yaml

```yaml
llm:
  provider: "zhipuai"  # 或 "openai"
  zhipuai:
    model: "glm-4.7"  # 最新旗舰模型（2025年12月发布）
    # 其他可选: glm-4-flash（便宜快速）, glm-4-plus（上一代旗舰）

agents:
  ai_trend_analyzer:
    enabled: true
    mock_mode: false  # false=真实API, true=模拟数据

  research_agent:  # v2.2新增
    enabled: true

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
    max_tweets: 5

  quality_evaluator:
    min_score: 7.0
```

### 提示词配置：config/prompts.yaml

每个Agent的system和user提示词可以自定义。

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

- **热点获取**：12-19秒（11个数据源）
- **长文本生成**：30-45秒（GLM-4-plus）
- **小红书精炼**：15-20秒
- **Twitter生成**：10-15秒
- **总耗时**：90-120秒

**成本**（GLM-4-plus）：
- 每次运行：¥1.2-1.8
- Token使用：12000-18000

**优化建议**：
- 使用 `glm-4-flash` 降低成本
- 减少数据源（只用hackernews + arxiv）
- 禁用不需要的Agent

## ❓ 常见问题

### Q: 如何只生成简报？

编辑 `config/config.yaml`：
```yaml
agents:
  longform_generator:
    enabled: false
  xiaohongshu_refiner:
    enabled: false
  twitter_generator:
    enabled: false
```

### Q: 如何修改生成内容的字数？

```yaml
longform_generator:
  article_length: "short"  # 1500-2000字

xiaohongshu_refiner:
  content_density: "light"  # 更简洁

twitter_generator:
  max_tweets: 3  # 减少推文数量
```

### Q: 配图提示词如何使用？

1. 打开 `data/YYYYMMDD/xiaohongshu/prompts_*.txt`
2. 复制中文提示词
3. 粘贴到支持中文的AI绘图工具：
   - 文心一格
   - 通义万相
   - 即梦AI
   - 或翻译后用于Stable Diffusion

## 🎯 最佳实践

### 自动化运行

```bash
# 每天上午9点和下午6点运行
0 9,18 * * * cd /path/to/content-forge-ai && \
  source venv/bin/activate && \
  PYTHONPATH=/path/to/content-forge-ai \
  python src/main.py --once --workflow auto
```

### 质量控制

```yaml
agents:
  quality_evaluator:
    min_score: 7.5  # 提高质量阈值
```

### 内容差异化

```yaml
xiaohongshu_refiner:
  style: "casual"  # 轻松风格

twitter_generator:
  style: "engaging"  # 引人入胜
```

## 📚 相关文档

- **README.md** - 项目概述
- **CLAUDE.md** - 开发者指南
- **STORAGE_QUICKREF.md** - 存储结构快速参考
- **test/README.md** - 测试文件说明
- **TEST_REPORT_20260107.md** - 测试报告

## 🔄 版本历史

### v2.1 (2026-01-07)
- ✅ 新增Twitter Generator Agent
- ✅ 优化专业文章为微信公众号格式
- ✅ Topic参数变为可选
- ✅ 更新存储结构（5个目录）
- ✅ 日志按日期分层存储
- ✅ 测试文件统一管理

### v2.0 (2026-01-06)
- ✅ 按日期分层存储
- ✅ 热点简报Agent
- ✅ 11个AI数据源集成

## 📞 支持

- 问题反馈：GitHub Issues
- 技术讨论：查看文档
- 功能建议：欢迎Pull Request

---

**最后更新**：2026-01-07
**版本**：v2.1
**许可**：MIT License
