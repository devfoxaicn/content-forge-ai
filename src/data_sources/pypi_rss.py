"""
PyPI RSS - Python包数据源
RSS: https://pypi.org/rss/
"""
import feedparser
from datetime import datetime, timedelta
from typing import List, Dict, Any
from loguru import logger
from src.utils.time_filter import TimeFilter


class PyPIRSS:
    """PyPI RSS数据获取器"""

    def __init__(self):
        """初始化PyPI RSS获取器"""
        self.rss_urls = {
            "packages": "https://pypi.org/rss/packages.xml",
            "updates": "https://pypi.org/rss/updates.xml"
        }

    def get_recent_packages(
        self,
        limit: int = 50,
        days_ago: int = 1
    ) -> List[Dict[str, Any]]:
        """
        获取最近的Python包

        Args:
            limit: 返回数量限制
            days_ago: 搜索最近几天的包

        Returns:
            包列表
        """
        all_packages = []

        for rss_type, url in self.rss_urls.items():
            try:
                feed = feedparser.parse(url)
                entries = feed.get("entries", [])

                for entry in entries:
                    package = self._normalize_package(entry)
                    if package:
                        all_packages.append(package)

            except Exception as e:
                logger.warning(f"PyPI RSS ({rss_type}) 获取失败: {e}")
                continue

        # 过滤最近N天的包
        time_filter = TimeFilter(hours=days_ago * 24)
        filtered_packages = [pkg for pkg in all_packages if time_filter.is_within_time_window(pkg["published_at"])]

        # 按发布时间排序
        filtered_packages.sort(key=lambda x: x["published_at"], reverse=True)

        # 限制数量
        filtered_packages = filtered_packages[:limit]

        logger.info(f"PyPI: 获取到 {len(filtered_packages)} 个最近24小时的包")
        return filtered_packages

    def get_ai_ml_packages(
        self,
        keywords: List[str] = None,
        limit: int = 30
    ) -> List[Dict[str, Any]]:
        """
        获取AI/ML相关的Python包

        Args:
            keywords: 关键词列表
            limit: 返回数量限制

        Returns:
            包列表
        """
        if keywords is None:
            keywords = [
                "tensorflow", "pytorch", "keras", "scikit-learn", "pandas",
                "numpy", "transformers", "langchain", "openai", "anthropic",
                "diffusers", "accelerate", "peft", "datasets", "evaluate",
                "huggingface", "torch", "jax", "flax", "mlx", "llama", "ollama"
            ]

        all_packages = []

        try:
            feed = feedparser.parse(self.rss_urls["updates"])
            entries = feed.get("entries", [])

            for entry in entries:
                package = self._normalize_package(entry)
                if package:
                    # 检查包名是否包含AI/ML关键词
                    package_name = package["title"].lower()
                    if any(keyword in package_name for keyword in keywords):
                        all_packages.append(package)

        except Exception as e:
            logger.error(f"PyPI AI/ML包获取失败: {e}")
            return []

        # 过滤最近7天的包
        time_filter = TimeFilter(hours=7 * 24)
        filtered_packages = [pkg for pkg in all_packages if time_filter.is_within_time_window(pkg["published_at"])]

        # 按发布时间排序并限制数量
        filtered_packages.sort(key=lambda x: x["published_at"], reverse=True)
        filtered_packages = filtered_packages[:limit]

        logger.info(f"PyPI: 找到 {len(filtered_packages)} 个AI/ML相关包")
        return filtered_packages

    def _normalize_package(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """标准化包数据格式"""
        try:
            # 从RSS entry中提取信息
            title = entry.get("title", "")
            link = entry.get("link", "")
            published = entry.get("published", "") or entry.get("updated", "")
            description = entry.get("description", "")[:500]
            author = entry.get("author", "")

            # 从标题中提取包名和版本
            # PyPI RSS格式通常是 "package-name version" 或 "package-name (version)"
            parts = title.split()
            package_name = parts[0] if parts else title
            version = parts[1].strip("()") if len(parts) > 1 else ""

            return {
                "id": f"pypi-{package_name}",
                "title": f"{package_name} {version}".strip(),
                "description": description or f"Python package by {author}",
                "url": link,
                "published_at": published,
                "source": "PyPI",
                "category": "dev_tools",
                "metadata": {
                    "package_name": package_name,
                    "version": version,
                    "author": author,
                    "type": "python_package"
                }
            }
        except Exception as e:
            logger.warning(f"标准化PyPI包数据失败: {e}")
            return None


def create_pypi_client() -> PyPIRSS:
    """创建PyPI RSS客户端的工厂函数"""
    return PyPIRSS()
