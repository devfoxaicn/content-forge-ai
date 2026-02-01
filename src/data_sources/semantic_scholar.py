"""
Semantic Scholar API - 学术论文数据源
API文档: https://api.semanticscholar.org/api-docs/
免费限额: 100万次/天
"""
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from loguru import logger
from src.utils.time_filter import TimeFilter


class SemanticScholarAPI:
    """Semantic Scholar API客户端"""

    def __init__(self, api_key: Optional[str] = None):
        """
        初始化Semantic Scholar API客户端

        Args:
            api_key: API密钥（可选，有密钥可以提高限额）
        """
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.api_key = api_key
        self.headers = {}
        if api_key:
            self.headers["x-api-key"] = api_key

    def search_recent_papers(
        self,
        query: str = "artificial intelligence",
        fields: List[str] = None,
        limit: int = 100,
        days_ago: int = 1
    ) -> List[Dict[str, Any]]:
        """
        搜索最近的AI论文

        Args:
            query: 搜索关键词
            fields: 返回字段列表
            limit: 返回数量限制
            days_ago: 搜索最近几天的论文

        Returns:
            论文列表
        """
        if fields is None:
            fields = [
                "paperId",
                "title",
                "abstract",
                "authors",
                "publicationDate",
                "publicationVenue",
                "citationCount",
                "influentialCitationCount",
                "isOpenAccess",
                "openAccessPdf",
                "url",
                "year"
            ]

        # 计算时间范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_ago)

        # 构建搜索查询（添加时间过滤）
        search_query = f"{query} AND year:{start_date.year}-{end_date.year}"

        params = {
            "query": search_query,
            "fields": ",".join(fields),
            "limit": limit,
            "sort": "publicationDate:desc"
        }

        try:
            response = requests.get(
                f"{self.base_url}/paper/search",
                headers=self.headers,
                params=params,
                timeout=30
            )
            response.raise_for_status()

            data = response.json()
            papers = data.get("data", [])

            # 进一步过滤：确保是最近N天的
            time_filter = TimeFilter(hours=days_ago * 24)
            filtered_papers = []

            for paper in papers:
                pub_date_str = paper.get("publicationDate")
                if pub_date_str and time_filter.is_within_time_window(pub_date_str):
                    filtered_papers.append(self._normalize_paper(paper))

            logger.info(f"Semantic Scholar: 搜索到 {len(papers)} 篇论文，24小时内 {len(filtered_papers)} 篇")
            return filtered_papers

        except Exception as e:
            logger.error(f"Semantic Scholar API调用失败: {e}")
            return []

    def _normalize_paper(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """标准化论文数据格式"""
        return {
            "id": paper.get("paperId", ""),
            "title": paper.get("title", ""),
            "description": paper.get("abstract", "")[:500] if paper.get("abstract") else "",
            "url": paper.get("url", f"https://www.semanticscholar.org/paper/{paper.get('paperId', '')}"),
            "published_at": paper.get("publicationDate", ""),
            "source": "Semantic Scholar",
            "category": "academic_frontier",
            "metadata": {
                "authors": [author.get("name", "") for author in paper.get("authors", [])],
                "venue": paper.get("publicationVenue", ""),
                "citation_count": paper.get("citationCount", 0),
                "influential_citation_count": paper.get("influentialCitationCount", 0),
                "is_open_access": paper.get("isOpenAccess", False),
                "pdf_url": paper.get("openAccessPdf", ""),
                "year": paper.get("year")
            }
        }

    def get_ai_papers_by_category(
        self,
        categories: List[str] = None,
        limit_per_category: int = 20,
        days_ago: int = 1
    ) -> List[Dict[str, Any]]:
        """
        按分类获取AI论文

        Args:
            categories: 论文分类列表
            limit_per_category: 每个分类的数量限制
            days_ago: 搜索最近几天的论文

        Returns:
            论文列表
        """
        if categories is None:
            categories = [
                "machine learning",
                "deep learning",
                "natural language processing",
                "computer vision",
                "reinforcement learning",
                "large language models",
                "transformer",
                "diffusion models"
            ]

        all_papers = []

        for category in categories:
            papers = self.search_recent_papers(
                query=category,
                limit=limit_per_category,
                days_ago=days_ago
            )
            all_papers.extend(papers)

        # 去重（基于paperId）
        seen_ids = set()
        unique_papers = []
        for paper in all_papers:
            if paper["id"] not in seen_ids:
                seen_ids.add(paper["id"])
                unique_papers.append(paper)

        # 按发布时间排序
        unique_papers.sort(key=lambda x: x["published_at"], reverse=True)

        logger.info(f"Semantic Scholar: 总共获取 {len(unique_papers)} 篇不重复论文")
        return unique_papers


def create_semantic_scholar_client(api_key: Optional[str] = None) -> SemanticScholarAPI:
    """
    创建Semantic Scholar API客户端的工厂函数

    Args:
        api_key: API密钥（可选）

    Returns:
        SemanticScholarAPI实例
    """
    return SemanticScholarAPI(api_key=api_key)
