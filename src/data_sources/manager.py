"""
数据源管理器 - 统一管理所有数据源
"""
from typing import Dict, List, Any, Optional
from loguru import logger

# 导入所有数据源
from src.data_sources.semantic_scholar import SemanticScholarAPI
from src.data_sources.openalex import OpenAlexAPI
from src.data_sources.papers_with_code import PapersWithCodeAPI
from src.data_sources.huggingface_hub import HuggingFaceAPI
from src.data_sources.pypi_rss import PyPIRSS
from src.data_sources.npm_registry import NpmRegistryAPI
from src.data_sources.github_trending import GitHubTrendingAPI
from src.data_sources.product_hunt import ProductHuntAPI
from src.data_sources.reddit_ai import RedditAIAPI


class DataSourceManager:
    """数据源管理器 - 统一管理所有数据源的获取"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据源管理器

        Args:
            config: 配置字典，包含API密钥等
        """
        self.config = config
        self._init_data_sources()

    def _init_data_sources(self):
        """初始化所有数据源客户端"""
        api_keys = self.config.get("api_keys", {})

        # 学术前沿
        self.semantic_scholar = SemanticScholarAPI(api_key=api_keys.get("semantic_scholar"))
        self.openalex = OpenAlexAPI(email=api_keys.get("openalex_email"))
        self.papers_with_code = PapersWithCodeAPI()

        # 开发工具
        self.huggingface = HuggingFaceAPI(api_token=api_keys.get("huggingface"))
        self.pypi = PyPIRSS()
        self.npm = NpmRegistryAPI()
        self.github_trending = GitHubTrendingAPI(api_token=api_keys.get("github"))

        # AI Agent
        self.product_hunt = ProductHuntAPI(api_token=api_keys.get("product_hunt"))
        self.reddit = RedditAIAPI(
            client_id=api_keys.get("reddit_client_id"),
            client_secret=api_keys.get("reddit_client_secret")
        )

    # ==================== 学术前沿 ====================
    def get_academic_frontier_papers(self, limit: int = 5) -> List[Dict[str, Any]]:
        """获取学术前沿论文"""
        all_papers = []

        # Semantic Scholar
        try:
            papers = self.semantic_scholar.get_ai_papers_by_category(limit_per_category=10, days_ago=1)
            all_papers.extend(papers[:2])
        except Exception as e:
            logger.warning(f"Semantic Scholar获取失败: {e}")

        # OpenAlex
        try:
            papers = self.openalex.get_ai_papers_by_concept(limit_per_concept=10, days_ago=1)
            all_papers.extend(papers[:2])
        except Exception as e:
            logger.warning(f"OpenAlex获取失败: {e}")

        # Papers with Code
        try:
            papers = self.papers_with_code.get_recent_papers(days_ago=1, limit=20)
            all_papers.extend(papers[:1])
        except Exception as e:
            logger.warning(f"Papers with Code获取失败: {e}")

        # 去重并排序
        seen_ids = set()
        unique_papers = []
        for paper in all_papers:
            if paper["id"] not in seen_ids:
                seen_ids.add(paper["id"])
                unique_papers.append(paper)

        unique_papers.sort(key=lambda x: x["published_at"], reverse=True)
        return unique_papers[:limit]

    # ==================== 开发工具 ====================
    def get_dev_tools_updates(self, limit: int = 5) -> List[Dict[str, Any]]:
        """获取开发工具更新"""
        all_tools = []

        # Hugging Face模型
        try:
            models = self.huggingface.get_recent_models(days_ago=1, limit=20)
            all_tools.extend(models[:2])
        except Exception as e:
            logger.warning(f"Hugging Face模型获取失败: {e}")

        # Hugging Face数据集
        try:
            datasets = self.huggingface.get_recent_datasets(days_ago=1, limit=10)
            all_tools.extend(datasets[:1])
        except Exception as e:
            logger.warning(f"Hugging Face数据集获取失败: {e}")

        # PyPI AI/ML包
        try:
            packages = self.pypi.get_ai_ml_packages(limit=10)
            all_tools.extend(packages[:1])
        except Exception as e:
            logger.warning(f"PyPI包获取失败: {e}")

        # npm AI/ML包
        try:
            packages = self.npm.get_ai_ml_packages(limit=10)
            all_tools.extend(packages[:1])
        except Exception as e:
            logger.warning(f"npm包获取失败: {e}")

        # 去重并排序
        seen_ids = set()
        unique_tools = []
        for tool in all_tools:
            if tool["id"] not in seen_ids:
                seen_ids.add(tool["id"])
                unique_tools.append(tool)

        unique_tools.sort(key=lambda x: x["published_at"], reverse=True)
        return unique_tools[:limit]

    # ==================== AI Agent ====================
    def get_ai_agent_projects(self, limit: int = 5) -> List[Dict[str, Any]]:
        """获取AI Agent项目和产品"""
        all_agents = []

        # GitHub Trending AI项目
        try:
            repos = self.github_trending.get_trending_ai_repos(days_ago=1, limit=30)
            # 只选择Agent相关的
            agent_repos = [r for r in repos if "agent" in r["title"].lower() or "agent" in r["description"].lower()]
            all_agents.extend(agent_repos[:2])
        except Exception as e:
            logger.warning(f"GitHub Trending获取失败: {e}")

        # Product Hunt AI产品
        try:
            products = self.product_hunt.get_ai_products(limit=20)
            # 只选择Agent相关的
            agent_products = [p for p in products if p["category"] == "ai_agent"]
            all_agents.extend(agent_products[:2])
        except Exception as e:
            logger.warning(f"Product Hunt获取失败: {e}")

        # Reddit Agent讨论
        try:
            discussions = self.reddit.get_agent_discussions(limit=20)
            all_agents.extend(discussions[:1])
        except Exception as e:
            logger.warning(f"Reddit Agent讨论获取失败: {e}")

        # Hacker News (使用现有集成)
        # ...

        # 去重并排序
        seen_ids = set()
        unique_agents = []
        for agent in all_agents:
            if agent["id"] not in seen_ids:
                seen_ids.add(agent["id"])
                unique_agents.append(agent)

        unique_agents.sort(key=lambda x: x["published_at"], reverse=True)
        return unique_agents[:limit]

    # ==================== 企业应用 ====================
    def get_enterprise_ai_news(self, limit: int = 5) -> List[Dict[str, Any]]:
        """获取企业AI新闻"""
        # 这里主要使用已有的RSS源（TechCrunch, AI Business, InfoQ）
        # VentureBeat需要新增
        logger.info("企业应用新闻使用已有RSS源")
        return []

    # ==================== 消费产品 ====================
    def get_consumer_ai_products(self, limit: int = 5) -> List[Dict[str, Any]]:
        """获取消费AI产品"""
        all_products = []

        # Product Hunt (非Agent产品)
        try:
            products = self.product_hunt.get_ai_products(limit=30)
            consumer_products = [p for p in products if p["category"] == "consumer_apps"]
            all_products.extend(consumer_products[:3])
        except Exception as e:
            logger.warning(f"Product Hunt消费产品获取失败: {e}")

        # Hacker News Show HN
        # ...

        # 去重并排序
        seen_ids = set()
        unique_products = []
        for product in all_products:
            if product["id"] not in seen_ids:
                seen_ids.add(product["id"])
                unique_products.append(product)

        unique_products.sort(key=lambda x: x["published_at"], reverse=True)
        return unique_products[:limit]

    # ==================== 行业资讯 ====================
    def get_industry_ai_news(self, limit: int = 5) -> List[Dict[str, Any]]:
        """获取行业AI新闻"""
        # 这里主要使用已有的RSS源（NewsAPI, MIT Review, The Gradient, MarkTechPost）
        logger.info("行业资讯使用已有RSS源")
        return []

    # ==================== 统一获取接口 ====================
    def fetch_all_data(self, per_category: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        获取所有分类的数据

        Args:
            per_category: 每个分类的数量限制

        Returns:
            按分类组织的数据字典
        """
        results = {
            "academic_frontier": self.get_academic_frontier_papers(limit=per_category),
            "dev_tools": self.get_dev_tools_updates(limit=per_category),
            "ai_agent": self.get_ai_agent_projects(limit=per_category),
            "enterprise_apps": self.get_enterprise_ai_news(limit=per_category),
            "consumer_apps": self.get_consumer_ai_products(limit=per_category),
            "industry_news": self.get_industry_ai_news(limit=per_category)
        }

        # 记录统计
        total_count = sum(len(items) for items in results.values())
        logger.info(f"数据获取完成: 总共 {total_count} 条")
        for category, items in results.items():
            logger.info(f"  {category}: {len(items)} 条")

        return results


def create_data_source_manager(config: Dict[str, Any]) -> DataSourceManager:
    """
    创建数据源管理器的工厂函数

    Args:
        config: 配置字典

    Returns:
        DataSourceManager实例
    """
    return DataSourceManager(config)
