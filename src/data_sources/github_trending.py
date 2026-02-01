"""
GitHub Trending - 热门AI项目数据源
注意: GitHub没有官方Trending API，使用第三方服务或爬虫
这里使用简化实现，实际可用GitHub API + 搜索关键词
"""
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any
from loguru import logger
from src.utils.time_filter import TimeFilter


class GitHubTrendingAPI:
    """GitHub Trending数据获取器"""

    def __init__(self, api_token: str = None):
        """
        初始化GitHub Trending获取器

        Args:
            api_token: GitHub API token（可选，提高限额）
        """
        self.base_url = "https://api.github.com"
        self.headers = {"Accept": "application/vnd.github.v3+json"}
        if api_token:
            self.headers["Authorization"] = f"token {api_token}"

    def get_trending_ai_repos(
        self,
        limit: int = 50,
        days_ago: int = 1
    ) -> List[Dict[str, Any]]:
        """
        获取热门AI项目（使用GitHub API搜索）

        Args:
            limit: 返回数量限制
            days_ago: 搜索最近几天的项目

        Returns:
            项目列表
        """
        # AI/ML相关关键词
        keywords = [
            "artificial intelligence", "machine learning", "deep learning",
            "llm", "gpt", "transformer", "agent", "langchain",
            "huggingface", "pytorch", "tensorflow", "diffusion",
            "reinforcement learning", "computer vision", "nlp"
        ]

        all_repos = []

        for keyword in keywords:
            try:
                # 搜索最近创建或更新的仓库
                query = f"{keyword} language:python OR language:javascript OR language:typescript"
                params = {
                    "q": query,
                    "sort": "stars",
                    "order": "desc",
                    "per_page": 5
                }

                response = requests.get(
                    f"{self.base_url}/search/repositories",
                    headers=self.headers,
                    params=params,
                    timeout=30
                )

                if response.status_code == 200:
                    data = response.json()
                    repos = data.get("items", [])

                    for repo in repos:
                        normalized = self._normalize_repo(repo)
                        if normalized:
                            all_repos.append(normalized)

            except Exception as e:
                logger.warning(f"GitHub搜索关键词 {keyword} 失败: {e}")
                continue

        # 去重
        seen_ids = set()
        unique_repos = []
        for repo in all_repos:
            if repo["id"] not in seen_ids:
                seen_ids.add(repo["id"])
                unique_repos.append(repo)

        # 过滤最近N天的
        time_filter = TimeFilter(hours=days_ago * 24)
        filtered_repos = [repo for repo in unique_repos if time_filter.is_within_time_window(repo["published_at"])]

        # 按stars排序
        filtered_repos.sort(key=lambda x: x["metadata"]["stars"], reverse=True)
        filtered_repos = filtered_repos[:limit]

        logger.info(f"GitHub Trending: 找到 {len(filtered_repos)} 个最近24小时的热门AI项目")
        return filtered_repos

    def get_trending_by_topic(
        self,
        topic: str = "machine-learning",
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        按topic获取热门项目

        Args:
            topic: GitHub topic名称
            limit: 返回数量限制

        Returns:
            项目列表
        """
        try:
            params = {
                "q": f"topic:{topic}",
                "sort": "stars",
                "order": "desc",
                "per_page": limit
            }

            response = requests.get(
                f"{self.base_url}/search/repositories",
                headers=self.headers,
                params=params,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                repos = data.get("items", [])

                normalized_repos = []
                for repo in repos:
                    normalized = self._normalize_repo(repo)
                    if normalized:
                        normalized_repos.append(normalized)

                return normalized_repos

        except Exception as e:
            logger.error(f"GitHub按topic获取失败: {e}")

        return []

    def _normalize_repo(self, repo: Dict[str, Any]) -> Dict[str, Any]:
        """标准化仓库数据格式"""
        try:
            # 提取语言信息
            language = repo.get("language", "")

            # 判断是否是AI/ML相关
            topics = repo.get("topics", [])
            ai_keywords = ["ai", "ml", "machine-learning", "deep-learning", "llm", "agent", "nlp", "cv"]
            is_ai_related = any(topic in ai_keywords for topic in topics) or any(
                kw in repo.get("description", "").lower() for kw in ai_keywords
            )

            if not is_ai_related and language not in ["Python", "Jupyter Notebook", "TypeScript", "JavaScript"]:
                return None

            return {
                "id": repo.get("full_name", ""),
                "title": repo.get("full_name", ""),
                "description": repo.get("description", "")[:500] or "No description",
                "url": repo.get("html_url", ""),
                "published_at": repo.get("created_at", ""),
                "source": "GitHub",
                "category": "ai_agent" if "agent" in repo.get("description", "").lower() else "dev_tools",
                "metadata": {
                    "stars": repo.get("stargazers_count", 0),
                    "forks": repo.get("forks_count", 0),
                    "language": language,
                    "topics": topics[:10],
                    "license": repo.get("license", {}).get("name", "") if repo.get("license") else "",
                    "updated_at": repo.get("updated_at", ""),
                    "homepage": repo.get("homepage", ""),
                    "type": "repository"
                }
            }
        except Exception as e:
            logger.warning(f"标准化GitHub仓库数据失败: {e}")
            return None


def create_github_trending_client(api_token: str = None) -> GitHubTrendingAPI:
    """
    创建GitHub Trending客户端的工厂函数

    Args:
        api_token: GitHub API token（可选）

    Returns:
        GitHubTrendingAPI实例
    """
    return GitHubTrendingAPI(api_token=api_token)
