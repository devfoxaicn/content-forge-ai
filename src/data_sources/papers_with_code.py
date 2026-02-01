"""
Papers with Code - 论文与代码实现数据源
网站: https://paperswithcode.com/
注意: Papers with Code目前没有公开API，使用网页爬取方式
"""
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import List, Dict, Any
from loguru import logger
from src.utils.time_filter import TimeFilter


class PapersWithCodeAPI:
    """Papers with Code 数据爬取器"""

    def __init__(self):
        """初始化Papers with Code爬取器"""
        self.base_url = "https://paperswithcode.com"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

    def get_recent_papers(
        self,
        days_ago: int = 1,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        获取最近的论文（带代码实现）

        Args:
            days_ago: 搜索最近几天的论文
            limit: 返回数量限制

        Returns:
            论文列表
        """
        # Papers with Code没有API，这里实现一个基本框架
        # 实际使用时可能需要Selenium或更复杂的爬虫

        papers = []

        # 方法1: 尝试爬取最新页面
        try:
            url = f"{self.base_url}/latest"
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # 解析论文列表（需要根据实际页面结构调整选择器）
            # 这里只是框架示例

            logger.info(f"Papers with Code: 爬取完成")
        except Exception as e:
            logger.warning(f"Papers with Code爬取失败: {e}，使用备用方案")

        # 方法2: 使用Semantic Scholar API + 代码过滤
        # 这是一个更可靠的替代方案
        from src.data_sources.semantic_scholar import SemanticScholarAPI

        semantic_api = SemanticScholarAPI()

        # 搜索包含代码实现的论文
        # 在论文标题或摘要中搜索关键词
        code_keywords = ["github", "code", "implementation", "repository"]

        papers = semantic_api.search_recent_papers(
            query="artificial intelligence",
            limit=limit * 2,  # 多获取一些用于过滤
            days_ago=days_ago
        )

        # 过滤有代码实现的论文
        papers_with_code = []
        for paper in papers:
            title = paper["title"].lower()
            desc = paper["description"].lower()

            # 检查是否包含代码相关关键词
            if any(keyword in title or keyword in desc for keyword in code_keywords):
                paper["metadata"]["has_code"] = True
                papers_with_code.append(paper)

        logger.info(f"Papers with Code: 找到 {len(papers_with_code)} 篇带代码实现的论文")
        return papers_with_code

    def get_papers_by_task(
        self,
        task: str = "image-classification",
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        按任务获取论文

        Args:
            task: 任务名称
            limit: 返回数量限制

        Returns:
            论文列表
        """
        # 这里的实现需要根据Papers with Code的实际页面结构来调整
        # 当前提供接口框架

        papers = []
        logger.info(f"Papers with Code: 按任务 {task} 搜索（功能待实现）")

        return papers

    def _normalize_paper(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """标准化论文数据格式"""
        return {
            "id": paper_data.get("paperId", ""),
            "title": paper_data.get("title", ""),
            "description": paper_data.get("abstract", "")[:500],
            "url": paper_data.get("url", ""),
            "published_at": paper_data.get("publicationDate", ""),
            "source": "Papers with Code",
            "category": "academic_frontier",
            "metadata": {
                "has_code": True,
                "github_url": paper_data.get("github_url", ""),
                "stars": paper_data.get("stars", 0),
                "framework": paper_data.get("framework", ""),
                "task": paper_data.get("task", "")
            }
        }


def create_papers_with_code_client() -> PapersWithCodeAPI:
    """
    创建Papers with Code客户端的工厂函数

    Returns:
        PapersWithCodeAPI实例
    """
    return PapersWithCodeAPI()
