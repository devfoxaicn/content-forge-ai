"""
OpenAlex API - 开放学术数据源
API文档: https://docs.openalex.org/
完全免费，无需API密钥
"""
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from loguru import logger
from src.utils.time_filter import TimeFilter


class OpenAlexAPI:
    """OpenAlex API客户端 - 完全免费的开放学术数据库"""

    def __init__(self, email: Optional[str] = None):
        """
        初始化OpenAlex API客户端

        Args:
            email: 邮箱地址（可选，OpenAlex建议提供以便他们联系）
        """
        self.base_url = "https://api.openalex.org"
        self.email = email
        self.params = {}
        if email:
            self.params["mailto"] = email

    def search_recent_papers(
        self,
        query: str = "artificial intelligence",
        limit: int = 100,
        days_ago: int = 1
    ) -> List[Dict[str, Any]]:
        """
        搜索最近的AI论文

        Args:
            query: 搜索关键词
            limit: 返回数量限制
            days_ago: 搜索最近几天的论文

        Returns:
            论文列表
        """
        # 计算日期范围（从今天开始往前推N天）
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")

        # 构建过滤参数
        filters = {
            "from_publication_date": start_date,
            "to_publication_date": end_date,
            "has_fulltext": True,
            "type": "article"
        }

        params = {
            "search": query,
            "filter": ",".join([f"{k}:{v}" for k, v in filters.items()]),
            "per-page": min(limit, 200),  # OpenAlex最大200
            "sort": "publication_date:desc"
        }
        params.update(self.params)

        try:
            response = requests.get(
                f"{self.base_url}/works",
                params=params,
                timeout=30
            )
            response.raise_for_status()

            data = response.json()
            papers = data.get("results", [])

            # 标准化数据
            normalized_papers = []
            time_filter = TimeFilter(hours=days_ago * 24)

            for paper in papers:
                pub_date = paper.get("publication_date")
                if pub_date and time_filter.is_within_time_window(pub_date):
                    normalized_papers.append(self._normalize_paper(paper))

            logger.info(f"OpenAlex: 搜索到 {len(papers)} 篇论文，24小时内 {len(normalized_papers)} 篇")
            return normalized_papers

        except Exception as e:
            logger.error(f"OpenAlex API调用失败: {e}")
            return []

    def get_ai_papers_by_concept(
        self,
        concepts: List[str] = None,
        limit_per_concept: int = 20,
        days_ago: int = 1
    ) -> List[Dict[str, Any]]:
        """
        按概念获取AI论文

        Args:
            concepts: 概念列表（OpenAlex的concept ID）
            limit_per_concept: 每个概念的数量限制
            days_ago: 搜索最近几天的论文

        Returns:
            论文列表
        """
        if concepts is None:
            # 使用OpenAlex的概念ID
            concepts = [
                "C154945302",  # Machine learning
                "C144025419",  # Artificial intelligence
                "C121332964",  # Computer vision
                "C418678692",  # Deep learning
                "C2517358374", # Natural language processing
                "C81211004",   # Reinforcement learning
                "C155201202",  # Neural network
            ]

        all_papers = []

        for concept_id in concepts:
            params = {
                "filter": f"concepts.id:{concept_id},from_publication_date:{(datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')}",
                "per-page": min(limit_per_concept, 200),
                "sort": "publication_date:desc"
            }
            params.update(self.params)

            try:
                response = requests.get(
                    f"{self.base_url}/works",
                    params=params,
                    timeout=30
                )
                response.raise_for_status()

                data = response.json()
                papers = data.get("results", [])

                for paper in papers:
                    normalized = self._normalize_paper(paper)
                    if normalized:
                        all_papers.append(normalized)

            except Exception as e:
                logger.warning(f"获取概念 {concept_id} 的论文失败: {e}")
                continue

        # 去重
        seen_ids = set()
        unique_papers = []
        for paper in all_papers:
            if paper["id"] not in seen_ids:
                seen_ids.add(paper["id"])
                unique_papers.append(paper)

        unique_papers.sort(key=lambda x: x["published_at"], reverse=True)

        logger.info(f"OpenAlex: 总共获取 {len(unique_papers)} 篇不重复论文")
        return unique_papers

    def _normalize_paper(self, paper: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """标准化论文数据格式"""
        try:
            # 提取作者
            authorships = paper.get("authorships", [])
            authors = []
            for authorship in authorships[:5]:  # 只取前5个作者
                author = authorship.get("author", {})
                if author:
                    name = author.get("display_name", "")
                    if name:
                        authors.append(name)

            # 提取概念（主题标签）
            concepts = paper.get("concepts", [])
            concept_names = [c.get("display_name", "") for c in concepts if c.get("score", 0) > 0.3]

            # 提取期刊/会议信息
            primary_location = paper.get("primary_location") or {}
            source = primary_location.get("source") or {}
            venue = source.get("display_name", "")

            return {
                "id": paper.get("id", ""),
                "title": paper.get("title", ""),
                "description": paper.get("abstract", "")[:500] if paper.get("abstract") else "",
                "url": paper.get("id", ""),
                "published_at": paper.get("publication_date", ""),
                "source": "OpenAlex",
                "category": "academic_frontier",
                "metadata": {
                    "authors": authors,
                    "venue": venue,
                    "citation_count": paper.get("cited_by_count", 0),
                    "concepts": concept_names[:5],  # 前5个概念
                    "type": paper.get("type", ""),
                    "open_access": paper.get("open_access", {}).get("is_oa", False),
                    "pdf_url": paper.get("best_oa_location", {}).get("pdf_url", "")
                }
            }
        except Exception as e:
            logger.warning(f"标准化论文数据失败: {e}")
            return None


def create_openalex_client(email: Optional[str] = None) -> OpenAlexAPI:
    """
    创建OpenAlex API客户端的工厂函数

    Args:
        email: 邮箱地址（可选）

    Returns:
        OpenAlexAPI实例
    """
    return OpenAlexAPI(email=email)
