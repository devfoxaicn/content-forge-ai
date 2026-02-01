"""
Reddit AI - AI社区讨论数据源
API文档: https://www.reddit.com/dev/api/
Subreddits: r/MachineLearning, r/artificial, r/OpenAI, r/singularity
"""
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from loguru import logger
from src.utils.time_filter import TimeFilter


class RedditAIAPI:
    """Reddit AI社区数据获取器"""

    def __init__(self, client_id: str = None, client_secret: str = None):
        """
        初始化Reddit API客户端

        Args:
            client_id: Reddit app client_id (可选，用于更高限额)
            client_secret: Reddit app client_secret (可选)
        """
        self.base_url = "https://www.reddit.com"
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = None

        if client_id and client_secret:
            self._authenticate()

    def _authenticate(self):
        """OAuth认证（如果有密钥）"""
        try:
            auth = requests.auth.HTTPBasicAuth(self.client_id, self.client_secret)
            data = {"grant_type": "client_credentials"}
            headers = {"User-Agent": "ContentForgeAI/1.0"}

            response = requests.post(
                f"{self.base_url}/access_token",
                auth=auth,
                data=data,
                headers=headers,
                timeout=30
            )

            if response.status_code == 200:
                self.access_token = response.json().get("access_token")

        except Exception as e:
            logger.warning(f"Reddit认证失败: {e}")

    def get_hot_posts(
        self,
        subreddit: str = "MachineLearning",
        limit: int = 50,
        days_ago: int = 1
    ) -> List[Dict[str, Any]]:
        """
        获取热门帖子

        Args:
            subreddit: 子版块名称
            limit: 返回数量限制
            days_ago: 搜索最近几天的帖子

        Returns:
            帖子列表
        """
        headers = {"User-Agent": "ContentForgeAI/1.0"}
        if self.access_token:
            headers["Authorization"] = f"bearer {self.access_token}"

        params = {
            "limit": limit,
            "sort": "hot"
        }

        try:
            response = requests.get(
                f"{self.base_url}/r/{subreddit}/hot",
                headers=headers,
                params=params,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                posts = data.get("data", {}).get("children", [])

                filtered_posts = []
                time_filter = TimeFilter(hours=days_ago * 24)

                for post_data in posts:
                    post = post_data.get("data", {})
                    normalized = self._normalize_post(post, subreddit)
                    if normalized and time_filter.is_within_time_window(normalized["published_at"]):
                        filtered_posts.append(normalized)

                logger.info(f"Reddit r/{subreddit}: 获取到 {len(filtered_posts)} 个最近24小时的帖子")
                return filtered_posts

        except Exception as e:
            logger.error(f"Reddit r/{subreddit} 获取失败: {e}")

        return []

    def get_ai_discussions(
        self,
        limit: int = 30
    ) -> List[Dict[str, Any]]:
        """
        获取AI相关讨论

        Args:
            limit: 每个subreddit的数量限制

        Returns:
            讨论列表
        """
        subreddits = ["MachineLearning", "artificial", "OpenAI", "LocalLLaMA", "singularity"]

        all_posts = []

        for subreddit in subreddits:
            posts = self.get_hot_posts(subreddit, limit, days_ago=1)
            all_posts.extend(posts)

        # 去重
        seen_ids = set()
        unique_posts = []
        for post in all_posts:
            if post["id"] not in seen_ids:
                seen_ids.add(post["id"])
                unique_posts.append(post)

        # 按score排序
        unique_posts.sort(key=lambda x: x["metadata"]["score"], reverse=True)
        unique_posts = unique_posts[:limit]

        logger.info(f"Reddit AI: 总共获取 {len(unique_posts)} 个不重复的热门讨论")
        return unique_posts

    def get_agent_discussions(
        self,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        获取AI Agent相关讨论

        Args:
            limit: 返回数量限制

        Returns:
            讨论列表
        """
        # AI Agent相关的subreddits和搜索关键词
        agent_subreddits = ["LocalLLaMA", "OpenAI", "MachineLearning"]
        agent_keywords = ["agent", "autogpt", "babyagi", "agentgpt", "langchain agent"]

        all_posts = []

        for subreddit in agent_subreddits:
            posts = self.get_hot_posts(subreddit, limit=50, days_ago=1)

            # 过滤agent相关帖子
            for post in posts:
                title = post["title"].lower()
                if any(kw in title for kw in agent_keywords):
                    post["category"] = "ai_agent"
                    all_posts.append(post)

        # 去重并排序
        seen_ids = set()
        unique_posts = []
        for post in all_posts:
            if post["id"] not in seen_ids:
                seen_ids.add(post["id"])
                unique_posts.append(post)

        unique_posts.sort(key=lambda x: x["metadata"]["score"], reverse=True)
        unique_posts = unique_posts[:limit]

        logger.info(f"Reddit AI Agent: 找到 {len(unique_posts)} 个Agent相关讨论")
        return unique_posts

    def _normalize_post(self, post: Dict[str, Any], subreddit: str) -> Dict[str, Any]:
        """标准化帖子数据格式"""
        try:
            # 提取帖子数据
            title = post.get("title", "")
            selftext = post.get("selftext", "")
            url = post.get("url", "")
            permalink = post.get("permalink", "")
            created_utc = post.get("created_utc", 0)
            score = post.get("score", 0)
            num_comments = post.get("num_comments", 0)
            author = post.get("author", "")

            # 转换时间戳
            created_at = datetime.fromtimestamp(created_utc).isoformat() if created_utc else ""

            # 判断分类
            title_lower = title.lower()
            if "agent" in title_lower or "autogpt" in title_lower:
                category = "ai_agent"
            elif any(kw in title_lower for kw in ["paper", "research", "arxiv"]):
                category = "academic_frontier"
            elif any(kw in title_lower for kw in ["tool", "library", "framework", "package"]):
                category = "dev_tools"
            else:
                category = "industry_news"

            return {
                "id": post.get("id", ""),
                "title": title,
                "description": selftext[:500] if selftext else title[:200],
                "url": url or f"https://reddit.com{permalink}",
                "published_at": created_at,
                "source": f"Reddit r/{subreddit}",
                "category": category,
                "metadata": {
                    "score": score,
                    "num_comments": num_comments,
                    "author": author,
                    "subreddit": subreddit,
                    "permalink": permalink,
                    "upvote_ratio": post.get("upvote_ratio", 0),
                    "type": "discussion"
                }
            }
        except Exception as e:
            logger.warning(f"标准化Reddit帖子失败: {e}")
            return None


def create_reddit_client(client_id: str = None, client_secret: str = None) -> RedditAIAPI:
    """
    创建Reddit API客户端的工厂函数

    Args:
        client_id: Reddit app client_id（可选）
        client_secret: Reddit app client_secret（可选）

    Returns:
        RedditAIAPI实例
    """
    return RedditAIAPI(client_id=client_id, client_secret=client_secret)
