# ContentForge AI v2.2

> 🚀 AI驱动的多平台内容自动化生产工厂

> 基于Web搜索的深度研究 + 分阶段生成，打造9000-13000字专业深度分析

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-LangGraph-green.svg)](https://python.langchain.com/)

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

```bash
# 推荐：不指定topic，系统自动从热点生成
python src/main.py --once --workflow auto

# 或指定topic作为文件标识（可选）
python src/main.py --once --workflow auto --topic "AI技术"
```

### 查看输出

```bash
# 查看存储目录
ls -la data/20260107/

# 查看热点简报
cat data/20260107/digest/digest_*.md

# 查看专业文章
cat data/20260107/longform/article_*.md

# 查看小红书笔记
cat data/20260107/xiaohongshu/note_*.md

# 查看Twitter帖子
cat data/20260107/twitter/twitter_*.md
```

## 📂 输出结构

```
data/20260107/              # 按日期分层
├── raw/                   # AI热点原始数据
│   └── trends_auto_*.json
├── digest/                # 热点简报
│   ├── digest_*.md
│   └── digest_*.json
├── longform/              # 微信公众号文章
│   ├── article_*.md
│   └── article_*.json
├── xiaohongshu/           # 小红书笔记
│   ├── note_*.md
│   └── prompts_*.txt      # 配图提示词
└── twitter/               # Twitter帖子
    ├── twitter_*.md
    └── prompts_*.txt      # 配图提示词

logs/20260107/             # 日志按日期分层
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
    model: "glm-4-plus"  # glm-4-flash 更快更便宜

agents:
  ai_trend_analyzer:
    enabled: true
    mock_mode: false  # false=真实API, true=模拟数据

  longform_generator:
    article_length: "medium"  # short, medium, long

  xiaohongshu_refiner:
    style: "professional"  # professional, casual, humorous

  twitter_generator:
    style: "engaging"  # engaging, professional, casual
    thread_mode: true
    max_tweets: 5

  quality_evaluator:
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

- **[PROJECT_GUIDE.md](PROJECT_GUIDE.md)** - 完整项目指南
- **[CLAUDE.md](CLAUDE.md)** - 开发者指南
- **[STORAGE_QUICKREF.md](STORAGE_QUICKREF.md)** - 存储结构快速参考
- **[AI_TRENDS_API_GUIDE.md](AI_TRENDS_API_GUIDE.md)** - 数据源详细说明
- **[test/README.md](test/README.md)** - 测试文件说明

## 🔄 版本历史

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

**最后更新**：2026-01-08
**版本**：v2.2
**Made with ❤️ by ContentForge AI Team**
