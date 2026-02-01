"""
Product Hunt API - 产品发布数据源
API文档: https://api.producthunt.com/v2/docs
需要OAuth认证
"""
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from loguru import logger
from src.utils.time_filter import TimeFilter


class ProductHuntAPI:
    """Product Hunt API客户端"""

    def __init__(self, api_token: str = None):
        """
        初始化Product Hunt API客户端

        Args:
            api_token: Product Hunt OAuth token (必需)
        """
        self.base_url = "https://api.producthunt.com/v2"
        self.headers = {}
        if api_token:
            self.headers["Authorization"] = f"Bearer {api_token}"

    def get_recent_posts(
        self,
        limit: int = 50,
        days_ago: int = 1
    ) -> List[Dict[str, Any]]:
        """
        获取最近的产品发布

        Args:
            limit: 返回数量限制
            days_ago: 搜索最近几天的产品

        Returns:
            产品列表
        """
        if not self.headers.get("Authorization"):
            logger.warning("Product Hunt API需要OAuth token")
            return []

        # GraphQL查询
        query = """
        query ($after: DateTime) {
          posts(order: RANKING, first: %d, after: $after) {
            nodes {
              id
              name
              tagline
              description
              url
              website
              createdAt
              featuredAt
              votesCount
              commentsCount
              productState
              topics {
                nodes {
                  name
                }
              }
            }
          }
        }
        """ % limit

        # 计算时间范围
        after_date = (datetime.now() - timedelta(days=days_ago)).isoformat() + "Z"

        try:
            response = requests.post(
                f"{self.base_url}/api/graphql",
                headers=self.headers,
                json={"query": query, "variables": {"after": after_date}},
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                posts = data.get("data", {}).get("posts", {}).get("nodes", [])

                filtered_posts = []
                time_filter = TimeFilter(hours=days_ago * 24)

                for post in posts:
                    normalized = self._normalize_post(post)
                    if normalized and time_filter.is_within_time_window(normalized["published_at"]):
                        filtered_posts.append(normalized)

                logger.info(f"Product Hunt: 获取到 {len(filtered_posts)} 个最近24小时的产品")
                return filtered_posts

            else:
                logger.error(f"Product Hunt API错误: {response.status_code}")

        except Exception as e:
            logger.error(f"Product Hunt API调用失败: {e}")

        return []

    def get_ai_products(
        self,
        limit: int = 30
    ) -> List[Dict[str, Any]]:
        """
        获取AI相关的产品

        Args:
            limit: 返回数量限制

        Returns:
            产品列表
        """
        if not self.headers.get("Authorization"):
            logger.warning("Product Hunt API需要OAuth token")
            return []

        # AI相关主题关键词
        ai_topics = [
            "Artificial Intelligence", "Machine Learning", "Chatbots",
            "Developer Tools", "Productivity", "Automation"
        ]

        query = """
        query {
          posts(order: RANKING, first: %d) {
            nodes {
              id
              name
              tagline
              description
              url
              createdAt
              votesCount
              topics {
                nodes {
                  name
                }
              }
            }
          }
        }
        """ % (limit * 2)

        try:
            response = requests.post(
                f"{self.base_url}/api/graphql",
                headers=self.headers,
                json={"query": query},
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                posts = data.get("data", {}).get("posts", {}).get("nodes", [])

                ai_products = []

                for post in posts:
                    # 检查话题是否包含AI相关
                    topics = [t["name"] for t in post.get("topics", {}).get("nodes", [])]
                    if any(topic in ai_topics for topic in topics):
                        # 检查标题/描述是否包含AI关键词
                        title = post.get("name", "").lower()
                        tagline = post.get("tagline", "").lower()
                        if any(kw in title or kw in tagline for kw in ["ai", "ml", "chatgpt", "gpt", "agent", "llm"]):
                            normalized = self._normalize_post(post)
                            if normalized:
                                ai_products.append(normalized)

                # 过滤最近7天的
                time_filter = TimeFilter(hours=7 * 24)
                filtered_products = [p for p in ai_products if time_filter.is_within_time_window(p["published_at"])]

                # 按votes排序
                filtered_products.sort(key=lambda x: x["metadata"]["votes"], reverse=True)
                filtered_products = filtered_products[:limit]

                logger.info(f"Product Hunt: 找到 {len(filtered_products)} 个AI相关产品")
                return filtered_products

        except Exception as e:
            logger.error(f"Product Hunt AI产品获取失败: {e}")

        return []

    def _normalize_post(self, post: Dict[str, Any]) -> Dict[str, Any]:
        """标准化产品数据格式"""
        try:
            # 提取话题
            topics = post.get("topics", {}).get("nodes", [])
            topic_names = [t["name"] for t in topics]

            # 判断产品类型（Agent/应用/工具）
            title = post.get("name", "").lower()
            tagline = post.get("tagline", "").lower()

            if "agent" in title or "agent" in tagline:
                category = "ai_agent"
            elif any(t in topic_names for t in ["Developer Tools", "Productivity"]):
                category = "dev_tools"
            else:
                category = "consumer_apps"

            return {
                "id": post.get("id", ""),
                "title": post.get("name", ""),
                "description": post.get("tagline", "") or post.get("description", "")[:500],
                "url": post.get("url", "") or post.get("website", ""),
                "published_at": post.get("createdAt", "") or post.get("featuredAt", ""),
                "source": "Product Hunt",
                "category": category,
                "metadata": {
                    "votes": post.get("votesCount", 0),
                    "comments": post.get("commentsCount", 0),
                    "topics": topic_names,
                    "product_state": post.get("productState", ""),
                    "type": "product"
                }
            }
        except Exception as e:
            logger.warning(f"标准化Product Hunt数据失败: {e}")
            return None


def create_product_hunt_client(api_token: str = None) -> ProductHuntAPI:
    """
    创建Product Hunt API客户端的工厂函数

    Args:
        api_token: Product Hunt OAuth token

    Returns:
        ProductHuntAPI实例
    """
    return ProductHuntAPI(api_token=api_token)
