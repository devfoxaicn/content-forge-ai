"""
npm Registry API - JavaScript/TypeScript包数据源
API: https://registry.npmjs.org/
"""
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any
from loguru import logger
from src.utils.time_filter import TimeFilter


class NpmRegistryAPI:
    """npm Registry API客户端"""

    def __init__(self):
        """初始化npm Registry API客户端"""
        self.base_url = "https://registry.npmjs.org"

    def get_recent_packages(
        self,
        limit: int = 50,
        days_ago: int = 1
    ) -> List[Dict[str, Any]]:
        """
        获取最近的npm包（通过npm RSS）

        Args:
            limit: 返回数量限制
            days_ago: 搜索最近几天的包

        Returns:
            包列表
        """
        # npm官方RSS
        rss_url = "https://www.npmjs.com/feed/updates"

        try:
            import feedparser
            feed = feedparser.parse(rss_url)
            entries = feed.get("entries", [])

            all_packages = []
            time_filter = TimeFilter(hours=days_ago * 24)

            for entry in entries:
                package = self._normalize_package_from_rss(entry)
                if package and time_filter.is_within_time_window(package["published_at"]):
                    all_packages.append(package)

            # 按发布时间排序
            all_packages.sort(key=lambda x: x["published_at"], reverse=True)
            all_packages = all_packages[:limit]

            logger.info(f"npm: 获取到 {len(all_packages)} 个最近24小时的包")
            return all_packages

        except Exception as e:
            logger.error(f"npm RSS获取失败: {e}")
            return []

    def get_ai_ml_packages(
        self,
        keywords: List[str] = None,
        limit: int = 30
    ) -> List[Dict[str, Any]]:
        """
        获取AI/ML相关的npm包

        Args:
            keywords: 关键词列表
            limit: 返回数量限制

        Returns:
            包列表
        """
        if keywords is None:
            keywords = [
                "tensorflow", "torch", "transformers", "openai", "langchain",
                "ai", "ml", "machine-learning", "neural", "deep-learning",
                "chatgpt", "gpt", "llm", "embedding", "vector", "agent"
            ]

        all_packages = []

        try:
            import feedparser
            rss_url = "https://www.npmjs.com/feed/updates"
            feed = feedparser.parse(rss_url)
            entries = feed.get("entries", [])

            for entry in entries:
                package = self._normalize_package_from_rss(entry)
                if package:
                    # 检查包名是否包含AI/ML关键词
                    package_name = package["title"].lower()
                    if any(keyword in package_name for keyword in keywords):
                        all_packages.append(package)

        except Exception as e:
            logger.error(f"npm AI/ML包获取失败: {e}")
            return []

        # 过滤最近7天的包
        time_filter = TimeFilter(hours=7 * 24)
        filtered_packages = [pkg for pkg in all_packages if time_filter.is_within_time_window(pkg["published_at"])]

        # 按发布时间排序并限制数量
        filtered_packages.sort(key=lambda x: x["published_at"], reverse=True)
        filtered_packages = filtered_packages[:limit]

        logger.info(f"npm: 找到 {len(filtered_packages)} 个AI/ML相关包")
        return filtered_packages

    def get_package_info(self, package_name: str) -> Dict[str, Any]:
        """
        获取包的详细信息

        Args:
            package_name: 包名

        Returns:
            包信息
        """
        try:
            response = requests.get(
                f"{self.base_url}/{package_name}",
                timeout=30
            )
            response.raise_for_status()

            data = response.json()
            latest_version = data.get("dist-tags", {}).get("latest", "")

            if latest_version:
                version_info = data.get("versions", {}).get(latest_version, {})
                return self._normalize_package_info(package_name, version_info)

        except Exception as e:
            logger.warning(f"获取npm包 {package_name} 信息失败: {e}")

        return {}

    def _normalize_package_from_rss(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """从RSS条目标准化包数据"""
        try:
            title = entry.get("title", "")
            link = entry.get("link", "")
            published = entry.get("published", "") or entry.get("updated", "")
            description = entry.get("description", "")[:500]

            # npm RSS格式通常是 "package-name version"
            parts = title.split()
            package_name = parts[0] if parts else title
            version = parts[1] if len(parts) > 1 else ""

            return {
                "id": f"npm-{package_name}",
                "title": f"{package_name}@{version}" if version else package_name,
                "description": description or "JavaScript/TypeScript package",
                "url": link or f"https://www.npmjs.com/package/{package_name}",
                "published_at": published,
                "source": "npm",
                "category": "dev_tools",
                "metadata": {
                    "package_name": package_name,
                    "version": version,
                    "type": "javascript_package"
                }
            }
        except Exception as e:
            logger.warning(f"标准化npm包数据失败: {e}")
            return None

    def _normalize_package_info(self, package_name: str, info: Dict[str, Any]) -> Dict[str, Any]:
        """标准化包详细信息"""
        version = info.get("version", "")
        author = info.get("author", {}).get("name", "Unknown")
        description = info.get("description", "")[:500]
        keywords = info.get("keywords", [])
        homepage = info.get("homepage", "")
        repository = info.get("repository", {}).get("url", "")

        return {
            "id": f"npm-{package_name}",
            "title": f"{package_name}@{version}",
            "description": description,
            "url": homepage or repository or f"https://www.npmjs.com/package/{package_name}",
            "published_at": info.get("publish_time", ""),
            "source": "npm",
            "category": "dev_tools",
            "metadata": {
                "package_name": package_name,
                "version": version,
                "author": author,
                "keywords": keywords,
                "homepage": homepage,
                "repository": repository,
                "type": "javascript_package"
            }
        }


def create_npm_client() -> NpmRegistryAPI:
    """创建npm Registry API客户端的工厂函数"""
    return NpmRegistryAPI()
