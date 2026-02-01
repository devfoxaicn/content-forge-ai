"""
数据源模块 - 集成各种AI相关数据源

支持的数据源分类：
- 学术前沿: arXiv, Semantic Scholar, OpenAlex, Papers with Code, OpenReview, DBLP
- 开发工具: Hugging Face, PyPI, npm, GitHub Releases, 框架RSS
- AI Agent: GitHub Trending, Product Hunt, Reddit, Hacker News
- 企业应用: TechCrunch, VentureBeat, AI Business, InfoQ
- 消费产品: Product Hunt, a16z, Hacker News, App Stores
- 行业资讯: NewsAPI, MIT Review, The Gradient, MarkTechPost, Stanford HAI, Accenture
"""

from src.utils.time_filter import TimeFilter, create_time_filter, is_within_24h, filter_last_24h

__all__ = [
    "TimeFilter",
    "create_time_filter",
    "is_within_24h",
    "filter_last_24h",
]
