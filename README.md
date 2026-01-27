# ContentForge AI

> 🚀 AI驱动的多平台内容自动化生产工厂

> **世界顶级AI新闻简报**：全中文、多数据源、智能分类

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-LangGraph-green.svg)](https://python.langchain.com/)

## ✨ 核心功能

**ContentForge AI** 是一个基于 LangChain/LangGraph 的智能内容生产系统，专注于高质量内容产出。

### 🎯 两大内容生成模式

#### 1️⃣ **Auto 模式** - 全中文AI新闻简报
- 多个全球数据源实时聚合
- 智能分类：产业动态、学术前沿、技术创新、产品工具、行业应用
- 全中文翻译：标题、描述、洞察、深度观察
- 世界顶级科技媒体写作标准

#### 2️⃣ **Series 模式** - 100期技术博客系列
- 系统化生成100期技术博客
- 覆盖10大系列：LLM原理、RAG实战、Agent开发等
- 批量生成，进度追踪

### 🌟 核心优势

- ✅ **全中文简报** - 参照36氪、虎嗅等专业科技媒体标准
- ✅ **多数据源** - NewsAPI、TechCrunch、arXiv、Hacker News等
- ✅ **智能分类** - 5大分类自动组织
- ✅ **深度观察** - 核心洞察+产业分析
- ✅ **成本可控** - GLM-4.7模型，性价比高

## 🚀 快速开始

### 安装

```bash
# 1. 克隆项目
git clone https://github.com/devfoxaicn/content-forge-ai.git
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

---

## 📖 Auto模式 - 全中文AI新闻简报

**核心特色**：
- 🌍 多个全球数据源实时聚合
- 🇨🇳 全中文翻译，专业科技媒体写作标准
- 📊 5大智能分类：产业动态、学术前沿、技术创新、产品工具、行业应用
- 💡 核心洞察 + 深度产业观察
- 🔗 每条包含标题、描述、来源链接、热度评分

```bash
# 设置PYTHONPATH（替换为你的实际项目路径）
export PYTHONPATH=/Users/z/Documents/work/content-forge-ai

# 运行自动模式
python src/main.py --mode auto --once
```

**输出位置**：`data/daily/YYYYMMDD/`
- `raw/` - 原始数据（按数据源组织的JSON）
- `digest/` - 全中文简报（Markdown + JSON）

**数据源** (多个)：

| 数据源 | 类型 | 内容 |
|--------|------|------|
| TechCrunch AI | 新闻 | AI行业新闻 |
| **NewsAPI.org** | **新闻** | **全球AI新闻聚合** |
| arXiv | 学术 | AI重大论文 |
| Hacker News | 社区 | 科技热点讨论 |
| MIT Tech Review | 新闻 | MIT技术评论 |
| OpenAI Blog | 官方 | OpenAI官方动态 |
| BAIR Blog | 学术 | UC Berkeley AI研究 |
| Microsoft Research | 学术 | 微软研究院博客 |
| MarkTechPost | 新闻 | AI研究新闻 |
| KDnuggets | 新闻 | 数据科学权威 |
| AI Business | 新闻 | AI行业新闻 |
| The Gradient | 期刊 | AI研究期刊 |
| InfoQ AI | 技术 | 技术媒体 |
| Hugging Face Blog | 官方 | Hugging Face官方博客 |

**输出格式示例**：

```markdown
# AI每日热点 · 2026年01月22日

## 💡 核心洞察
- 多智能体协作范式确立，标志着AI从单一对话迈向自主执行新阶段
- 数据泄露与合成数据困境，反映出GenAI爆发期底层安全与质量的隐忧

## 📰 深度观察
**AI产业观察：从云端竞逐到端侧重构的范式转移**
（约350字专业分析文章）

## 🔍 本期热点

### 📈 产业动态（35条）
#### [据报Apple研发AI可穿戴设备，不甘落后OpenAI](链接)
**来源**：TechCrunch AI  ·  **热度**：70
一份关于该设备的报告显示，若这款可穿戴设备最终问世，最早可能会在2027年发布。

...

### 🎓 学术前沿（15条）
...
```

---

## 📖 Series 模式 - 100期技术博客系列

```bash
# 查看生成进度
python src/main.py --mode series --progress

# 生成指定集数
python src/main.py --mode series --episode 1

# 批量生成（如第1-10期）
python src/main.py --mode series --all --start 1 --end 10
```

---

## 🚀 部署到生产环境

```bash
# 1. 克隆项目
git clone https://github.com/devfoxaicn/content-forge-ai.git
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

### 🔐 环境变量配置

创建 `.env` 文件：

```bash
# 必需的密钥
ZHIPUAI_API_KEY=your_zhipuai_api_key_here    # 智谱AI密钥（获取：https://open.bigmodel.cn/）

# 可选的密钥
NEWSAPI_KEY=your_newsapi_key_here            # NewsAPI密钥（推荐配置）
OPENAI_API_KEY=your_openai_api_key_here      # OpenAI密钥（备用）
TAVILY_API_KEY=your_tavily_api_key_here      # Web搜索（可选）
```

**获取API密钥**：
- [智谱AI](https://open.bigmodel.cn/) - 必需，支持国产大模型GLM-4.7
- [NewsAPI.org](https://newsapi.org/) - 推荐，全球新闻聚合
- [OpenAI](https://platform.openai.com/api-keys) - 可选，备用LLM提供商

---

## 📂 输出结构

```
data/
├── daily/                    # Auto模式输出
│   └── 20260122/
│       ├── raw/              # 原始数据（按数据源）
│       └── digest/           # 全中文简报
│           ├── digest_20260122.md
│           └── digest_20260122.json
│
└── series/                   # Series模式输出
    ├── LLM_series/           # LLM系列分类
    │   ├── series_1_llm_foundation/
    │   │   ├── episode_001/
    │   │   │   ├── episode_metadata.json
    │   │   │   └── ep001_标题_article.md
    │   │   ├── episode_002/
    │   │   └── series_metadata.json
    │   └── ...
    └── ML_series/            # ML系列分类
        └── ...

logs/                        # 日志按日期分层
└── YYYYMMDD/
    └── app.log
```

---

## 🤖 Auto模式工作流程

```
1. RealAITrendAnalyzerAgent
   从多个数据源获取热点
   保留所有内容（不去重、不排序截断）
   输出: trends_by_source

2. TrendCategorizerAgent
   按分类组织热点
   5大分类：产业动态、学术前沿、技术创新、产品工具、行业应用
   输出: categorized_trends

3. NewsScoringAgent
   对新闻进行6维度评分筛选
   输出: scored_trends

4. WorldClassDigestAgent
   生成全中文世界顶级新闻简报
   翻译所有标题、描述
   生成核心洞察和深度观察
   输出: news_digest (全中文)
```

---

## 📊 性能指标

- **热点获取**：30-60秒（多个数据源）
- **分类整理**：<1秒
- **全中文翻译+生成**：50-60分钟
- **总耗时**：约60分钟

**内容质量**：
- 全中文简报：500+行，专业科技媒体标准
- 核心洞察：3-5条深度判断
- 深度观察：约350字产业分析

**成本**（GLM-4.7）：
- 每次运行：约¥15-25
- Token使用：约150,000-200,000（含翻译）

**优化建议**：
- 使用 `glm-4-flash` 降低成本约80%
- 减少数据源（只用TechCrunch + arXiv + Hacker News）
- 调整翻译条数限制

---

## ❓ 常见问题

### Q: 简报是全中文的吗？

A: 是的。Auto模式生成的简报完全使用中文，包括：
- 所有标题和描述
- 核心洞察
- 深度观察分析
- 分类导语

### Q: data/daily目录为什么不会被提交到GitHub？

A: `data/daily/` 在 `.gitignore` 中，避免临时数据占用仓库空间。如需提交，可以手动添加或修改gitignore。

---

## 📚 详细文档

- **[CLAUDE.md](CLAUDE.md)** - 开发者指南和架构说明
- **[test/README.md](test/README.md)** - 测试文件说明

---

## 📊 两种模式对比

| 模式 | 输入 | 输出 | 用途 |
|------|------|------|------|
| **Auto** | （自动获取热点） | 全中文简报 | 每日AI新闻 |
| **Series** | 100期配置 | 长文本技术博客 | 系统化内容库 |

---

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE)

---

**最后更新**：2026-01-27
**Made with ❤️ by ContentForge AI Team**
