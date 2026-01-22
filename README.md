# ContentForge AI v4.0

> 🚀 AI驱动的多平台内容自动化生产工厂

> **世界顶级AI新闻简报**：全中文、多数据源、智能分类

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-LangGraph-green.svg)](https://python.langchain.com/)

## ✨ 核心功能

**ContentForge AI v4.0** 是一个基于 LangChain/LangGraph 的智能内容生产系统，专注于高质量内容产出。

### 🎯 四大内容生成模式

#### 1️⃣ **Auto 模式 v4.0** - 全中文AI新闻简报
- 8个全球数据源实时聚合
- 智能分类：产业动态、学术前沿、技术创新、产品工具、行业应用
- 全中文翻译：标题、描述、洞察、深度观察
- 世界顶级科技媒体写作标准

#### 2️⃣ **Series 模式** - 100期技术博客系列
- 系统化生成100期技术博客
- 覆盖10大系列：LLM原理、RAG实战、Agent开发等
- 批量生成，进度追踪

#### 3️⃣ **Custom 模式** - 自定义内容生成
- 根据关键词/要求生成长文本
- 支持风格转换（技术文→科普文）
- 支持重新表述、扩写等操作

#### 4️⃣ **Refine 模式** - 多平台内容精炼
- 将已有文本精炼为小红书笔记
- 生成Twitter Thread
- 生成微信公众号HTML

### 🌟 核心优势

- ✅ **全中文简报** - 参照36氪、虎嗅等专业科技媒体标准
- ✅ **8大数据源** - NewsAPI、TechCrunch、arXiv、Hacker News等
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

系统使用统一入口 `src/main.py`，支持四种模式：

---

## 📖 Auto模式 v4.0 - 全中文AI新闻简报

**新版本特色**：
- 🌍 8个全球数据源实时聚合
- 🇨🇳 全中文翻译，专业科技媒体写作标准
- 📊 5大智能分类：产业动态(35)、学术前沿(15)、技术创新(4)、产品工具(1)、行业应用
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

**数据源** (8个)：

| 数据源 | 类型 | 内容 |
|--------|------|------|
| Product Hunt | 产品 | 热门AI产品 |
| GitHub | 产品 | AI应用开源项目 |
| TechCrunch AI | 新闻 | AI行业新闻 |
| The Verge AI | 新闻 | AI技术创新 |
| VentureBeat AI | 新闻 | AI商业动态 |
| **NewsAPI.org** | **新闻** | **全球AI新闻聚合** |
| arXiv | 学术 | AI重大论文 |
| Hacker News | 社区 | 科技热点讨论 |

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

## 📖 其他模式

### Series 模式 - 100期技术博客系列

```bash
# 查看生成进度
python src/main.py --mode series --progress

# 生成指定集数
python src/main.py --mode series --episode 1

# 批量生成（如第1-10期）
python src/main.py --mode series --all --start 1 --end 10
```

### Custom 模式 - 自定义内容生成

```bash
# 简单关键词模式
python src/main.py --mode custom --topic "RAG技术原理与实战"

# 详细描述模式
python src/main.py --mode custom \
  --topic "RAG技术" \
  --prompt "详细介绍架构、核心组件、向量数据库选择，以及生产环境最佳实践"
```

### Refine 模式 - 多平台内容精炼

```bash
# 精炼为所有平台
python src/main.py --mode refine --input article.md

# 指定目标平台
python src/main.py --mode refine --input article.md --platforms wechat xiaohongshu
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
TAVILY_API_KEY=your_tavily_api_key_here      # Web搜索（Custom模式需要）
```

**获取API密钥**：
- [智谱AI](https://open.bigmodel.cn/) - 必需，支持国产大模型GLM-4.7
- [NewsAPI.org](https://newsapi.org/) - 推荐，全球新闻聚合
- [OpenAI](https://platform.openai.com/api-keys) - 可选，备用LLM提供商

---

## 📂 输出结构

```
data/
├── daily/                    # Auto模式输出（v4.0）
│   └── 20260122/
│       ├── raw/              # 原始数据（按数据源）
│       └── digest/           # 全中文简报
│           ├── digest_20260122.md
│           └── digest_20260122.json
│
├── series/                   # Series模式输出
│   ├── series_1_llm_foundation/
│   │   ├── episode_001/
│   │   │   └── longform/     # 长文本文章
│   │   ├── episode_002/
│   │   └── series_metadata.json
│   └── ...
│
├── custom/                   # Custom模式输出
│   └── YYYYMMDD_HHMMSS_topic/
│
└── refine/                   # Refine模式输出
    └── {source_name}/

logs/                        # 日志按日期分层
└── YYYYMMDD/
    └── app.log
```

---

## 🤖 Auto模式 v4.0 工作流程

```
1. AITrendAnalyzerAgent
   从8个数据源获取热点
   保留所有内容（不去重、不排序截断）
   输出: trends_by_source

2. TrendCategorizerAgent
   按分类组织热点
   5大分类：产业动态、学术前沿、技术创新、产品工具、行业应用
   输出: categorized_trends

3. WorldClassDigestAgent
   生成全中文世界顶级新闻简报
   翻译所有标题、描述
   生成核心洞察和深度观察
   输出: news_digest (全中文)
```

---

## 📊 性能指标

- **热点获取**：30-60秒（8个数据源）
- **分类整理**：<1秒
- **全中文翻译+生成**：50-60分钟（55条热点）
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

### Q: Auto模式v4.0与之前版本有什么区别？

A:
- **数据源**：从7个增加到8个（新增NewsAPI）
- **分类方式**：新增智能分类，按5大类组织
- **语言**：全部翻译为中文，符合专业科技媒体标准
- **存储**：只创建raw和digest目录

### Q: 简报是全中文的吗？

A: 是的。v4.0版本的Auto模式生成的简报完全使用中文，包括：
- 所有标题和描述
- 核心洞察
- 深度观察分析
- 分类导语

### Q: data/daily目录为什么不会被提交到GitHub？

A: `data/daily/` 在 `.gitignore` 中，避免临时数据占用仓库空间。如需提交，可以手动添加或修改gitignore。

### Q: 如何只生成简报不翻译？

A: 编辑 `src/auto_orchestrator.py`，将 `WorldClassDigestAgent` 改为 `NewsDigestAgent`（v3.0版本，英文版）。

---

## 📚 详细文档

- **[CLAUDE.md](CLAUDE.md)** - 开发者指南和架构说明
- **[test/README.md](test/README.md)** - 测试文件说明

---

## 📊 四种模式对比

| 模式 | 输入 | 输出 | 用途 |
|------|------|------|------|
| **Auto v4.0** | （自动获取热点） | 全中文简报 | 每日AI新闻 |
| **Series** | 100期配置 | 长文本技术博客 | 系统化内容库 |
| **Custom** | 关键词/主题 | 长文本 + 社交内容 | 按需生成 |
| **Refine** | 已有文章 | 多平台精炼内容 | 一文多发 |

---

## 🔄 版本历史

### v4.0 (2026-01-22) 🆕
- ✅ **NewsAPI数据源** - 新增全球AI新闻聚合
- ✅ **智能分类** - 新增TrendCategorizerAgent，按5大类组织
- ✅ **全中文简报** - 新增WorldClassDigestAgent，生成全中文世界顶级新闻简报
- ✅ **存储优化** - DailyStorage只创建raw和digest目录
- ✅ **专业写作标准** - 参照36氪、虎嗅、品玩等专业科技媒体

### v2.7 (2026-01-15)
- ✅ **100期技术博客系列完成** - 全部100期内容生成完毕

### v2.6 (2026-01-14)
- ✅ **四模合一** - Auto + Series + Custom + Refine 统一入口

### v2.5 (2026-01-09)
- ✅ **100期技术博客系列** - 系统化规划100期技术内容

---

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE)

---

**最后更新**：2026-01-22
**版本**：v4.0
**Made with ❤️ by ContentForge AI Team**
