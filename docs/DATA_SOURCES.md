# 数据源文档

本文档记录ContentForge AI集成的所有数据源，包括API信息、数据格式和使用说明。

## 数据源概览

总计 **30个数据源**，分为6大分类：
- 📚 学术前沿: 6个
- 🛠️ 开发工具: 5个
- 🦾 AI Agent: 5个
- 💼 企业应用: 4个
- 🌐 消费产品: 4个
- 📰 行业资讯: 6个

---

## 📚 学术前沿 (6个)

### 1. arXiv API
- **状态**: ✅ 已集成
- **类型**: 论文预印本
- **API**: http://export.arxiv.org/api/query
- **费用**: 免费 (3次/秒)
- **24h过滤**: ✅ 支持
- **获取方式**: Python arxiv库
- **更新频率**: 实时

### 2. Semantic Scholar API
- **状态**: 🆕 新增
- **类型**: 论文元数据
- **API**: https://api.semanticscholar.org/api-docs/
- **费用**: 免费 (100万次/天)
- **24h过滤**: ✅ 支持
- **获取方式**: REST API
- **特点**: 引用关系、影响力评分

### 3. OpenAlex API
- **状态**: 🆕 新增
- **类型**: 开放学术数据
- **API**: https://docs.openalex.org/
- **费用**: 完全免费
- **24h过滤**: ✅ 支持
- **获取方式**: REST API (POX方式)
- **特点**: 无需注册，完全开放

### 4. Papers with Code
- **状态**: 🆕 新增
- **类型**: 论文+代码实现
- **网站**: https://paperswithcode.com/
- **费用**: 免费
- **24h过滤**: ✅ 支持
- **获取方式**: 网页爬取/Semantic Scholar过滤

### 5. OpenReview API
- **状态**: 🆕 框架就绪
- **类型**: 论文评审
- **网站**: https://openreview.net/
- **费用**: 免费
- **24h过滤**: ✅ 支持

### 6. DBLP API
- **状态**: 🆕 框架就绪
- **类型**: 计算机科学文献
- **API**: https://dblp.org/faq/13501473
- **费用**: 免费
- **24h过滤**: ✅ 支持

---

## 🛠️ 开发工具 (5个)

### 7. Hugging Face Hub API
- **状态**: 🆕 新增
- **类型**: 模型/数据集
- **API**: https://huggingface.co/docs/hub/en/api
- **费用**: 免费 (可认证提高限额)
- **24h过滤**: ✅ 支持
- **获取方式**: REST API
- **特点**: ML模型和数据集

### 8. PyPI RSS
- **状态**: 🆕 新增
- **类型**: Python包
- **RSS**: https://pypi.org/rss/packages.xml
- **费用**: 免费
- **24h过滤**: ✅ 支持
- **获取方式**: RSS feed (feedparser)

### 9. npm Registry API
- **状态**: 🆕 新增
- **类型**: JavaScript/TypeScript包
- **API**: https://registry.npmjs.org/
- **费用**: 免费
- **24h过滤**: ✅ 支持
- **获取方式**: REST API + RSS

### 10. GitHub Releases API
- **状态**: 🆕 框架就绪
- **类型**: 版本发布
- **API**: https://docs.github.com/en/rest/releases
- **费用**: 免费 (可认证)
- **24h过滤**: ✅ 支持

### 11. 框架RSS
- **状态**: 🆕 框架就绪
- **类型**: 框架更新
- **来源**: PyTorch/TensorFlow官方博客
- **费用**: 免费
- **24h过滤**: ✅ 支持

---

## 🦾 AI Agent (5个)

### 12. GitHub Trending
- **状态**: 🆕 新增
- **类型**: 热门Agent项目
- **API**: 第三方服务 (Apify)
- **费用**: 免费 (GitHub搜索)
- **24h过滤**: ✅ 支持
- **获取方式**: GitHub Search API
- **注意**: 无官方Trending API

### 13. Product Hunt API
- **状态**: 🆕 新增
- **类型**: Agent产品
- **API**: https://api.producthunt.com/v2/docs
- **费用**: 免费 (OAuth)
- **24h过滤**: ✅ 支持
- **获取方式**: GraphQL API

### 14. Reddit (r/MachineLearning)
- **状态**: 🆕 新增
- **类型**: 社区讨论
- **API**: https://www.reddit.com/dev/api/
- **费用**: 免费 (可认证)
- **24h过滤**: ✅ 支持
- **子版块**: r/MachineLearning, r/artificial, r/OpenAI

### 15. Hacker News
- **状态**: ✅ 已集成
- **类型**: 社区新闻
- **API**: https://news.ycombinator.com/
- **费用**: 免费
- **24h过滤**: ✅ 支持

### 16. Awesome AI Agents
- **状态**: 🆕 框架就绪
- **类型**: 精选列表
- **来源**: GitHub
- **费用**: 免费
- **仓库**: https://github.com/slavakurilyak/awesome-ai-agents

---

## 💼 企业应用 (4个)

### 17. TechCrunch AI RSS
- **状态**: ✅ 已集成
- **类型**: 企业新闻
- **RSS**: https://techcrunch.com/category/artificial-intelligence/feed/
- **费用**: 免费
- **24h过滤**: ✅ 支持

### 18. VentureBeat AI RSS
- **状态**: 🆕 新增
- **类型**: 企业新闻
- **RSS**: https://venturebeat.com/ai/feed/
- **费用**: 免费
- **24h过滤**: ✅ 支持

### 19. AI Business RSS
- **状态**: ✅ 已集成
- **类型**: 行业资讯
- **RSS**: https://www.artificialintelligence-news.com/feed/
- **费用**: 免费
- **24h过滤**: ✅ 支持

### 20. InfoQ AI RSS
- **状态**: ✅ 已集成
- **类型**: 技术企业
- **RSS**: https://www.infoq.cn/artificialintelligence
- **费用**: 免费
- **24h过滤**: ✅ 支持

---

## 🌐 消费产品 (4个)

### 21. Product Hunt API
- **状态**: 🆕 新增
- **类型**: 产品发布
- **API**: https://api.producthunt.com/v2/docs
- **费用**: 免费 (OAuth)
- **24h过滤**: ✅ 支持

### 22. a16z Top 100
- **状态**: 🆕 框架就绪
- **类型**: 消费应用榜单
- **报告**: https://a16z.com/100-gen-ai-apps-5/
- **费用**: 免费
- **24h过滤**: ⚠️ 周报 (定期更新)

### 23. Hacker News (Show HN)
- **状态**: ✅ 已集成
- **类型**: 新产品
- **来源**: Show HN帖子
- **费用**: 免费
- **24h过滤**: ✅ 支持

### 24. App Store/Google Play
- **状态**: 🆕 待调研
- **类型**: 移动应用
- **费用**: 免费
- **24h过滤**: ✅ 支持

---

## 📰 行业资讯 (6个)

### 25. NewsAPI.org
- **状态**: ✅ 已集成
- **类型**: 新闻聚合
- **API**: https://newsapi.org/docs
- **费用**: 免费层/付费
- **24h过滤**: ✅ 支持

### 26. MIT Tech Review RSS
- **状态**: ✅ 已集成
- **类型**: 深度报道
- **RSS**: https://www.technologyreview.com/feed/
- **费用**: 免费
- **24h过滤**: ✅ 支持

### 27. The Gradient RSS
- **状态**: ✅ 已集成
- **类型**: AI期刊
- **RSS**: https://thegradient.pub/rss/
- **费用**: 免费
- **24h过滤**: ✅ 支持

### 28. MarkTechPost RSS
- **状态**: ✅ 已集成
- **类型**: AI新闻
- **RSS**: https://www.marktechpost.com/feed/
- **费用**: 免费
- **24h过滤**: ✅ 支持

### 29. Stanford HAI Report
- **状态**: 🆕 框架就绪
- **类型**: AI指数报告
- **报告**: https://hai.stanford.edu/projects/ai-index/
- **费用**: 免费
- **24h过滤**: ⚠️ 年报

### 30. Accenture Tech Vision
- **状态**: 🆕 框架就绪
- **类型**: 技术趋势
- **报告**: https://www.accenture.com/reports
- **费用**: 免费
- **24h过滤**: ⚠️ 年报

---

## 数据源分类映射

### 按分类统计

| 分类 | 已集成 | 新增 | 总计 |
|------|--------|------|------|
| 📚 学术前沿 | 1 | 5 | 6 |
| 🛠️ 开发工具 | 0 | 5 | 5 |
| 🦾 AI Agent | 1 | 4 | 5 |
| 💼 企业应用 | 3 | 1 | 4 |
| 🌐 消费产品 | 1 | 3 | 4 |
| 📰 行业资讯 | 4 | 2 | 6 |
| **合计** | **10** | **20** | **30** |

### API密钥需求

| API密钥 | 必需 | 获取地址 |
|---------|------|----------|
| ZHIPUAI_API_KEY | ✅ 必需 | https://open.bigmodel.cn/ |
| PRODUCT_HUNT_API_KEY | ⚠️ 推荐 | https://api.producthunt.com/v2/docs |
| GITHUB_TOKEN | ⚠️ 推荐 | https://github.com/settings/tokens |
| HUGGINGFACE_TOKEN | ⚠️ 推荐 | https://huggingface.co/settings/tokens |
| SEMANTIC_SCHOLAR_API_KEY | ⚠️ 可选 | https://www.semanticscholar.org/product/api |
| OPENALEX_EMAIL | ⚠️ 推荐 | 免费 (只需邮箱) |
| REDDIT_CLIENT_ID | ⚠️ 可选 | https://www.reddit.com/prefs/apps |
| NEWSAPI_KEY | ⚠️ 可选 | https://newsapi.org/ |

---

## 实现状态

### 已实现的数据源模块

```
src/data_sources/
├── __init__.py                 ✅
├── time_filter.py              ✅ (独立模块，在utils/)
├── semantic_scholar.py          ✅
├── openalex.py                  ✅
├── papers_with_code.py          ✅
├── huggingface_hub.py           ✅
├── pypi_rss.py                  ✅
├── npm_registry.py              ✅
├── github_trending.py           ✅
├── product_hunt.py              ✅
├── reddit_ai.py                 ✅
└── manager.py                   ✅ (统一管理器)
```

### 使用方式

```python
from src.data_sources.manager import create_data_source_manager
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

# 配置
config = {
    "api_keys": {
        "semantic_scholar": os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
        "openalex_email": os.getenv("OPENALEX_EMAIL"),
        "huggingface": os.getenv("HUGGINGFACE_TOKEN"),
        "github": os.getenv("GITHUB_TOKEN"),
        "product_hunt": os.getenv("PRODUCT_HUNT_API_KEY"),
        "reddit_client_id": os.getenv("REDDIT_CLIENT_ID"),
        "reddit_client_secret": os.getenv("REDDIT_CLIENT_SECRET"),
    }
}

# 创建管理器
manager = create_data_source_manager(config)

# 获取所有数据
data = manager.fetch_all_data(per_category=5)

# 按分类访问
academic_papers = data["academic_frontier"]
dev_tools = data["dev_tools"]
agent_projects = data["ai_agent"]
# ...
```

---

## 更新日志

- **2026-02-01**: 新增20个数据源，总计30个
- **2026-02-01**: 创建统一的数据源管理器
- **2026-02-01**: 更新.env.example添加新API密钥
