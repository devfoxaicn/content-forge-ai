"""
真实AI热点分析Agent - 多数据源聚合版本
整合Hacker News、arXiv、GitHub Trending、Reddit等免费数据源
"""

from typing import Dict, Any, List, Optional
import json
import os
import requests
from datetime import datetime, timedelta
from src.agents.base import BaseAgent
from src.utils.storage import get_storage


class RealAITrendAnalyzerAgent(BaseAgent):
    """真实的AI热点分析Agent - 使用免费API"""

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        super().__init__(config, prompts)
        self.mock_mode = config.get("agents", {}).get("ai_trend_analyzer", {}).get("mock_mode", False)

        # 使用新的存储管理器
        self.storage = get_storage(config.get("storage", {}).get("base_dir", "data"))

        # 数据源配置
        sources_config = config.get("agents", {}).get("ai_trend_analyzer", {}).get("sources", [])
        self.sources = {
            "hackernews": "hackernews" in sources_config,
            "arxiv": "arxiv" in sources_config,
            "github_trending": "github" in sources_config,
            "reddit": "reddit" in sources_config,
            "huggingface": "huggingface" in sources_config,
            "stackoverflow": "stackoverflow" in sources_config,
            "kaggle": "kaggle" in sources_config,
            "newsapi": "newsapi" in sources_config,
            "devto": "devto" in sources_config,
            "pypi": "pypi" in sources_config,
            "github_topics": "github_topics" in sources_config
        }

        # NewsAPI配置（需要API密钥）
        self.newsapi_key = os.getenv("NEWSAPI_KEY", None)

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行AI热点分析

        Args:
            state: 当前工作流状态

        Returns:
            Dict[str, Any]: 更新后的状态
        """
        self.log(f"开始分析AI技术热点，领域: {state['topic']}")

        try:
            topic = state['topic']

            # 判断是否使用mock模式
            if self.mock_mode:
                self.log("使用Mock模式（模拟数据）")
                from src.agents.ai_trend_analyzer import AITrendAnalyzerAgent
                mock_agent = AITrendAnalyzerAgent(self.config, self.prompts)
                hot_topics = mock_agent._get_mock_ai_trends(topic)
            else:
                self.log("使用真实API模式（多数据源聚合）")
                hot_topics = self._get_real_ai_trends(topic)

            self.log(f"成功分析出 {len(hot_topics)} 个热点话题")

            # 保存热点分析结果
            self._save_trends(topic, hot_topics)

            # 选择热度最高的话题
            selected_topic = hot_topics[0]
            self.log(f"选择热点话题: {selected_topic['title']}")

            return {
                **state,
                "ai_hot_topics": hot_topics,
                "selected_ai_topic": selected_topic,
                "current_step": "ai_trend_analyzer_completed"
            }
        except Exception as e:
            self.log(f"AI热点分析失败: {str(e)}", "ERROR")
            return {
                **state,
                "error_message": f"AI热点分析失败: {str(e)}",
                "current_step": "ai_trend_analyzer_failed"
            }

    def _get_real_ai_trends(self, topic: str = None) -> List[Dict[str, Any]]:
        """
        从多个免费数据源获取真实AI热点（无需关键词过滤）

        Args:
            topic: 领域参数（已弃用，保留用于兼容性）

        Returns:
            List[Dict[str, Any]]: 热点话题列表
        """
        all_trends = []

        # 1. Hacker News
        if self.sources["hackernews"]:
            try:
                hn_trends = self._get_hacker_news_trends()
                all_trends.extend(hn_trends)
                self.log(f"Hacker News: 获取 {len(hn_trends)} 条热点")
            except Exception as e:
                self.log(f"Hacker News获取失败: {e}", "WARNING")

        # 2. arXiv论文
        if self.sources["arxiv"]:
            try:
                arxiv_trends = self._get_arxiv_papers()
                all_trends.extend(arxiv_trends)
                self.log(f"arXiv: 获取 {len(arxiv_trends)} 条热点")
            except Exception as e:
                self.log(f"arXiv获取失败: {e}", "WARNING")

        # 3. GitHub Trending
        if self.sources["github_trending"]:
            try:
                github_trends = self._get_github_trending()
                all_trends.extend(github_trends)
                self.log(f"GitHub Trending: 获取 {len(github_trends)} 条热点")
            except Exception as e:
                self.log(f"GitHub Trending获取失败: {e}", "WARNING")

        # 4. Reddit
        if self.sources["reddit"]:
            try:
                reddit_trends = self._get_reddit_trends()
                all_trends.extend(reddit_trends)
                self.log(f"Reddit: 获取 {len(reddit_trends)} 条热点")
            except Exception as e:
                self.log(f"Reddit获取失败: {e}", "WARNING")

        # 5. Hugging Face Trending Models
        if self.sources["huggingface"]:
            try:
                hf_trends = self._get_huggingface_trends()
                all_trends.extend(hf_trends)
                self.log(f"Hugging Face: 获取 {len(hf_trends)} 条热点")
            except Exception as e:
                self.log(f"Hugging Face获取失败: {e}", "WARNING")

        # 6. Stack Overflow Hot Questions
        if self.sources["stackoverflow"]:
            try:
                so_trends = self._get_stackoverflow_trends()
                all_trends.extend(so_trends)
                self.log(f"Stack Overflow: 获取 {len(so_trends)} 条热点")
            except Exception as e:
                self.log(f"Stack Overflow获取失败: {e}", "WARNING")

        # 7. Kaggle竞赛和数据集
        if self.sources["kaggle"]:
            try:
                kaggle_trends = self._get_kaggle_trends()
                all_trends.extend(kaggle_trends)
                self.log(f"Kaggle: 获取 {len(kaggle_trends)} 条热点")
            except Exception as e:
                self.log(f"Kaggle获取失败: {e}", "WARNING")

        # 8. NewsAPI科技新闻
        if self.sources["newsapi"]:
            try:
                news_trends = self._get_newsapi_trends()
                all_trends.extend(news_trends)
                self.log(f"NewsAPI: 获取 {len(news_trends)} 条热点")
            except Exception as e:
                self.log(f"NewsAPI获取失败: {e}", "WARNING")

        # 9. Dev.to开发者博客
        if self.sources["devto"]:
            try:
                devto_trends = self._get_devto_trends()
                all_trends.extend(devto_trends)
                self.log(f"Dev.to: 获取 {len(devto_trends)} 条热点")
            except Exception as e:
                self.log(f"Dev.to获取失败: {e}", "WARNING")

        # 10. PyPI热门包
        if self.sources["pypi"]:
            try:
                pypi_trends = self._get_pypi_trends()
                all_trends.extend(pypi_trends)
                self.log(f"PyPI: 获取 {len(pypi_trends)} 条热点")
            except Exception as e:
                self.log(f"PyPI获取失败: {e}", "WARNING")

        # 11. GitHub Topics（行业应用）
        if self.sources["github_topics"]:
            try:
                topics_trends = self._get_github_topics_trends()
                all_trends.extend(topics_trends)
                self.log(f"GitHub Topics: 获取 {len(topics_trends)} 条热点")
            except Exception as e:
                self.log(f"GitHub Topics获取失败: {e}", "WARNING")

        # 按综合热度评分排序
        all_trends.sort(key=lambda x: x.get("heat_score", 0), reverse=True)

        # 去重（基于标题相似度）
        all_trends = self._deduplicate_trends(all_trends)

        # 返回Top 10
        return all_trends[:10]

    def _get_hacker_news_trends(self) -> List[Dict[str, Any]]:
        """获取Hacker News热门技术话题（直接获取Top 30）"""
        try:
            # 获取热门故事ID列表
            stories_url = "https://hacker-news.firebaseio.com/v0/topstories.json"
            response = requests.get(stories_url, timeout=10)
            story_ids = response.json()[:30]  # 取前30个

            trends = []

            for story_id in story_ids:
                try:
                    # 获取故事详情
                    item_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
                    item = requests.get(item_url, timeout=5).json()

                    if not item or "url" not in item:
                        continue

                    title = item.get("title", "")

                    # 计算热度评分
                    score = item.get("score", 0)
                    comments = item.get("descendants", 0)
                    heat_score = score * 2 + comments

                    trends.append({
                        "title": title,
                        "description": item.get("text", title)[:200],
                        "url": item.get("url", ""),
                        "source": "Hacker News",
                        "timestamp": datetime.fromtimestamp(item["time"]).strftime("%Y-%m-%d %H:%M"),
                        "metrics": {
                            "upvotes": score,
                            "comments": comments
                        },
                        "heat_score": heat_score,
                        "tags": ["技术新闻", "HN"]
                    })
                except Exception as e:
                    self.log(f"获取HN故事 {story_id} 失败: {e}", "WARNING")
                    continue

            return trends
        except Exception as e:
            self.log(f"Hacker News API调用失败: {e}", "ERROR")
            return []

    def _get_arxiv_papers(self) -> List[Dict[str, Any]]:
        """获取arXiv最新AI论文（直接获取AI相关分类）"""
        try:
            import arxiv

            # 搜索AI和计算机科学相关分类
            query = "cat:cs.AI OR cat:cs.CL OR cat:cs.LG OR cat:cs.CV OR cat:cs.NE"

            # 搜索最近7天的论文
            search = arxiv.Search(
                query=query,
                max_results=20,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )

            trends = []
            cutoff_date = datetime.now() - timedelta(days=7)

            for result in search.results():
                # 检查论文发布时间（最近7天）
                pub_date = result.published.replace(tzinfo=None)
                if pub_date < cutoff_date:
                    continue

                # 计算热度评分（新论文加分）
                days_ago = (datetime.now() - pub_date).days
                heat_score = 100 - days_ago * 10  # 越新分数越高

                trends.append({
                    "title": result.title,
                    "description": result.summary[:300],
                    "url": result.entry_id,
                    "source": "arXiv",
                    "timestamp": pub_date.strftime("%Y-%m-%d"),
                    "metrics": {
                        "authors": [a.name for a in result.authors[:3]],
                        "categories": result.categories,
                        "days_ago": days_ago
                    },
                    "heat_score": heat_score,
                    "tags": result.categories[:2] + ["论文", "学术"]
                })

            return trends
        except ImportError:
            self.log("arXiv库未安装，跳过arXiv数据源。运行: pip install arxiv", "WARNING")
            return []
        except Exception as e:
            self.log(f"arXiv API调用失败: {e}", "ERROR")
            return []

    def _get_github_trending(self) -> List[Dict[str, Any]]:
        """获取GitHub Trending热门项目（所有语言）"""
        try:
            # 使用第三方GitHub Trending API（不限制语言）
            api_url = "https://github-trending-api.now.sh/repositories"
            params = {
                "since": "weekly",
                "spoken_language": "en"
            }

            response = requests.get(api_url, params=params, timeout=10)
            repos = response.json()

            trends = []

            for repo in repos[:15]:
                # 解析star数字
                stars_str = repo.get("stars", "0")
                stars = self._parse_stars(stars_str)

                # 计算热度评分
                forks = self._parse_stars(repo.get("forks", "0"))
                heat_score = stars * 0.5 + forks * 0.3

                description = repo.get("description", "")

                trends.append({
                    "title": f"{repo['author']}/{repo['name']}",
                    "description": description or "No description",
                    "url": repo["url"],
                    "source": "GitHub Trending",
                    "timestamp": datetime.now().strftime("%Y-%m-%d"),
                    "metrics": {
                        "stars": stars_str,
                        "forks": repo.get("forks", "0"),
                        "language": repo.get("language", "Unknown")
                    },
                    "heat_score": int(heat_score),
                    "tags": ["开源", repo.get("language", ""), "GitHub"]
                })

            return trends
        except Exception as e:
            self.log(f"GitHub Trending API调用失败: {e}", "ERROR")
            return []

    def _get_reddit_trends(self) -> List[Dict[str, Any]]:
        """获取Reddit热门技术讨论（科技相关Subreddit）"""
        try:
            import praw

            # 从配置读取Reddit API凭证
            reddit_config = self.config.get("agents", {}).get("ai_trend_analyzer", {}).get("reddit", {})

            client_id = reddit_config.get("client_id") or os.getenv("REDDIT_CLIENT_ID")
            client_secret = reddit_config.get("client_secret") or os.getenv("REDDIT_CLIENT_SECRET")
            user_agent = reddit_config.get("user_agent", "AI_Trend_Analyzer/1.0")

            if not client_id or not client_secret:
                self.log("Reddit API凭证未配置，跳过Reddit数据源", "WARNING")
                return []

            reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent
            )

            # 固定的科技相关Subreddit
            subreddits = ["MachineLearning", "artificial", "technology", "programming"]
            trends = []

            for sub_name in subreddits[:2]:  # 限制2个subreddit
                try:
                    subreddit = reddit.subreddit(sub_name)
                    for post in subreddit.hot(limit=5):
                        # 过滤：只取最近3天的
                        post_time = datetime.fromtimestamp(post.created_utc)
                        if (datetime.now() - post_time).days > 3:
                            continue

                        # 计算热度评分
                        upvotes = post.score
                        comments = post.num_comments
                        heat_score = upvotes + comments * 2

                        trends.append({
                            "title": post.title,
                            "description": post.selftext[:200] if hasattr(post, 'selftext') else "",
                            "url": f"https://reddit.com{post.permalink}",
                            "source": f"Reddit r/{sub_name}",
                            "timestamp": post_time.strftime("%Y-%m-%d %H:%M"),
                            "metrics": {
                                "upvotes": upvotes,
                                "comments": comments
                            },
                            "heat_score": heat_score,
                            "tags": ["社区讨论", "Reddit"]
                        })
                except Exception as e:
                    self.log(f"获取Reddit r/{sub_name}失败: {e}", "WARNING")
                    continue

            return trends
        except ImportError:
            self.log("PRAW库未安装，跳过Reddit数据源。运行: pip install praw", "WARNING")
            return []
        except Exception as e:
            self.log(f"Reddit API调用失败: {e}", "ERROR")
            return []

    def _parse_stars(self, stars_str: str) -> int:
        """解析star数字字符串"""
        if isinstance(stars_str, int):
            return stars_str

        stars_str = str(stars_str).replace(",", "").strip()

        if "k" in stars_str.lower():
            return int(float(stars_str.lower().replace("k", "")) * 1000)
        elif "m" in stars_str.lower():
            return int(float(stars_str.lower().replace("m", "")) * 1000000)
        else:
            try:
                return int(stars_str)
            except ValueError:
                return 0

    def _deduplicate_trends(self, trends: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """去重相似的热点话题"""
        seen_titles = set()
        unique_trends = []

        for trend in trends:
            title = trend.get("title", "").lower()
            # 简单去重：标题完全相同或包含关系
            is_duplicate = False
            for seen in seen_titles:
                if title == seen or title in seen or seen in title:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_trends.append(trend)
                seen_titles.add(title)

        return unique_trends

    def _save_trends(self, topic: str, trends: List[Dict[str, Any]]):
        """保存热点分析结果到raw目录"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trends_{topic}_{timestamp}.json"

            output = {
                "topic": topic,
                "timestamp": datetime.now().isoformat(),
                "data_sources": list(self.sources.keys()),
                "total_trends": len(trends),
                "trends": trends
            }

            # 使用新的存储管理器，保存到raw目录
            filepath = self.storage.save_json("raw", filename, output)

            self.log(f"热点分析已保存: {filepath}")
        except Exception as e:
            self.log(f"保存热点分析失败: {str(e)}", "WARNING")

    def _get_huggingface_trends(self) -> List[Dict[str, Any]]:
        """获取Hugging Face热门模型（直接获取热门模型）"""
        try:
            # 使用Hugging Face模型搜索API（按likes排序）
            api_url = "https://huggingface.co/api/models"

            # 搜索热门模型
            params = {
                "limit": 20,
                "sort": "likes",  # 按likes排序
                "direction": -1   # 降序
            }

            response = requests.get(api_url, params=params, timeout=15, headers={
                "User-Agent": "AI-Trend-Analyzer/1.0"
            })

            if response.status_code != 200:
                self.log(f"Hugging Face API返回错误: {response.status_code}", "ERROR")
                return []

            # 解析JSON
            try:
                data = response.json()
            except Exception as e:
                self.log(f"Hugging Face API返回格式错误: {e}", "ERROR")
                return []

            # 检查返回数据格式
            if isinstance(data, dict) and "models" in data:
                models = data["models"]
            elif isinstance(data, list):
                models = data
            else:
                self.log(f"Hugging Face API返回数据格式异常: {type(data)}", "ERROR")
                return []

            if not models or len(models) == 0:
                self.log("Hugging Face API返回空列表", "WARNING")
                return []

            trends = []

            for model in models[:15]:
                try:
                    model_id = model.get("id", model.get("modelId", ""))
                    likes = model.get("likes", 0)
                    downloads = model.get("downloads", 0)
                    pipeline = model.get("pipeline", "")

                    if not model_id:
                        continue

                    # 计算热度评分
                    heat_score = likes * 10 + downloads // 100

                    # 格式化pipeline
                    if pipeline:
                        pipeline_name = pipeline.replace('-', ' ').replace('_', ' ').title()
                    else:
                        pipeline_name = "Model"

                    trends.append({
                        "title": f"🤗 {model_id}",
                        "description": f"{pipeline_name} | {likes}👍 | {downloads}⬇️",
                        "url": f"https://huggingface.co/{model_id}",
                        "source": "Hugging Face",
                        "timestamp": datetime.now().strftime("%Y-%m-%d"),
                        "metrics": {
                            "likes": likes,
                            "downloads": downloads,
                            "pipeline": pipeline or "unknown"
                        },
                        "heat_score": heat_score,
                        "tags": ["模型", "HuggingFace", "AI"]
                    })
                except Exception as e:
                    self.log(f"处理Hugging Face模型数据失败: {e}", "WARNING")
                    continue

            self.log(f"Hugging Face成功获取 {len(trends)} 个模型")
            return trends
        except Exception as e:
            self.log(f"Hugging Face API调用失败: {e}", "ERROR")
            return []

    def _get_stackoverflow_trends(self) -> List[Dict[str, Any]]:
        """获取Stack Overflow热门技术问题"""
        try:
            # Stack Exchange API
            api_url = "https://api.stackexchange.com/2.3/questions"

            # 获取热门问题
            params = {
                "order": "desc",
                "sort": "hot",  # 按热度排序
                "site": "stackoverflow",
                "pagesize": 50  # 增加数量以获得更多候选
            }

            response = requests.get(api_url, params=params, timeout=10)
            data = response.json()

            trends = []

            if "items" not in data:
                self.log(f"Stack Overflow API返回格式异常: {data}", "WARNING")
                return []

            for item in data["items"][:30]:
                title = item.get("title", "")
                tags = item.get("tags", [])

                # 计算热度评分
                score = item.get("score", 0)
                views = item.get("view_count", 0)
                answers = item.get("answer_count", 0)
                heat_score = score * 5 + answers * 15 + views // 200

                # 获取标签和描述
                tags_str = ", ".join(tags[:5])

                # 获取问题正文（去除HTML标签）
                body = item.get("body", "")
                if body:
                    # 简单去除HTML标签
                    import re
                    body_clean = re.sub(r'<[^>]+>', '', body)[:150].replace("\n", " ")
                    description = body_clean if body_clean else f"Tags: {tags_str}"
                else:
                    description = f"Tags: {tags_str}"

                trends.append({
                    "title": title,
                    "description": description,
                    "url": item.get("link", ""),
                    "source": "Stack Overflow",
                    "timestamp": datetime.fromtimestamp(item.get("creation_date", 0)).strftime("%Y-%m-%d"),
                    "metrics": {
                        "score": score,
                        "views": views,
                        "answers": answers,
                        "tags": tags
                    },
                    "heat_score": heat_score,
                    "tags": tags[:3] + ["问答", "StackOverflow"]
                })

            self.log(f"Stack Overflow成功获取 {len(trends)} 个问题")
            return trends
        except Exception as e:
            self.log(f"Stack Overflow API调用失败: {e}", "ERROR")
            return []

    def _get_kaggle_trends(self) -> List[Dict[str, Any]]:
        """获取Kaggle竞赛和数据集（AI应用案例）"""
        try:
            # Kaggle不提供官方公开API，使用GitHub搜索替代
            # 搜索机器学习和数据科学相关项目
            search_query = "machine learning OR data science language:python"
            api_url = "https://api.github.com/search/repositories"
            params = {
                "q": search_query,
                "sort": "stars",
                "order": "desc",
                "per_page": 15
            }

            response = requests.get(api_url, params=params, timeout=15, headers={
                "Accept": "application/vnd.github.v3+json"
            })

            if response.status_code != 200:
                self.log(f"GitHub API返回错误: {response.status_code}", "WARNING")
                return []

            data = response.json()

            if "items" not in data:
                return []

            trends = []
            for item in data["items"][:15]:
                # 计算热度评分
                stars = item.get("stargazers_count", 0)
                forks = item.get("forks_count", 0)
                open_issues = item.get("open_issues_count", 0)
                heat_score = stars * 0.5 + forks * 0.3 + open_issues * 2

                trends.append({
                    "title": item.get("full_name", ""),
                    "description": item.get("description", "机器学习相关项目")[:200],
                    "url": item.get("html_url", ""),
                    "source": "ML/GitHub",
                    "timestamp": datetime.now().strftime("%Y-%m-%d"),
                    "metrics": {
                        "stars": stars,
                        "forks": forks,
                        "open_issues": open_issues,
                        "language": item.get("language", "Unknown")
                    },
                    "heat_score": heat_score,
                    "tags": ["数据竞赛", "AI应用", "开源"]
                })

            self.log(f"GitHub机器学习项目成功获取 {len(trends)} 个项目")
            return trends
        except Exception as e:
            self.log(f"Kaggle API调用失败: {e}", "ERROR")
            return []

    def _get_newsapi_trends(self) -> List[Dict[str, Any]]:
        """获取NewsAPI科技新闻（需要API密钥）"""
        try:
            if not self.newsapi_key:
                self.log("NewsAPI密钥未配置，跳过", "WARNING")
                return []

            # NewsAPI免费版每天1000次请求
            base_url = "https://newsapi.org/v2/everything"

            # 使用通用的AI技术关键词
            query = "artificial intelligence OR machine learning OR AI OR LLM OR GPT OR Claude"

            params = {
                "q": query,
                "language": "en",
                "sortBy": "popularity",
                "pageSize": 15,
                "apiKey": self.newsapi_key
            }

            response = requests.get(base_url, params=params, timeout=15)

            if response.status_code == 401:
                self.log("NewsAPI密钥无效", "WARNING")
                return []
            elif response.status_code == 429:
                self.log("NewsAPI请求超限", "WARNING")
                return []
            elif response.status_code != 200:
                self.log(f"NewsAPI返回错误: {response.status_code}", "WARNING")
                return []

            data = response.json()

            if data.get("status") != "ok":
                return []

            trends = []
            for article in data.get("articles", [])[:15]:
                if not article.get("title") or article.get("title") == "[Removed]":
                    continue

                # 计算热度评分（基于来源和时间）
                source_name = article.get("source", {}).get("name", "")
                published_at = article.get("publishedAt", "")

                # 简单的热度评分
                heat_score = 50  # 基础分

                # 时间衰减
                if published_at:
                    try:
                        pub_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                        days_ago = (datetime.now(pub_date.tzinfo) - pub_date).days
                        heat_score -= days_ago * 5
                    except:
                        pass

                trends.append({
                    "title": article.get("title", ""),
                    "description": article.get("description", article.get("content", ""))[:200],
                    "url": article.get("url", ""),
                    "source": f"NewsAPI ({source_name})",
                    "timestamp": published_at[:10] if published_at else datetime.now().strftime("%Y-%m-%d"),
                    "metrics": {
                        "source": source_name,
                        "published_at": published_at
                    },
                    "heat_score": max(heat_score, 10),
                    "tags": ["新闻", "AI资讯", "行业动态"]
                })

            self.log(f"NewsAPI成功获取 {len(trends)} 条新闻")
            return trends
        except Exception as e:
            self.log(f"NewsAPI调用失败: {e}", "ERROR")
            return []

    def _get_devto_trends(self) -> List[Dict[str, Any]]:
        """获取Dev.to开发者博客文章（热门技术文章）"""
        try:
            # Dev.to公开API，无需认证
            base_url = "https://dev.to/api/articles"

            # 使用热门技术标签
            tags = ["ai", "machinelearning", "python", "javascript", "webdev"]
            trends = []

            # 对每个标签进行搜索
            for tag in tags[:2]:  # 只取前2个标签避免过多请求
                params = {
                    "tag": tag,
                    "top": "7",  # 按热度排序
                    "per_page": 10
                }

                response = requests.get(base_url, params=params, timeout=15)

                if response.status_code != 200:
                    continue

                articles = response.json()

                if not isinstance(articles, list):
                    continue

                for article in articles[:7]:
                    # 计算热度评分
                    comments_count = article.get("comments_count", 0)
                    positive_reactions_count = article.get("positive_reactions_count", 0)
                    heat_score = comments_count * 10 + positive_reactions_count * 2 + 30

                    # 获取标签（可能是字符串列表或字典列表）
                    tag_list = article.get("tag_list", [])
                    if tag_list and isinstance(tag_list[0], dict):
                        article_tags = [t.get("name", "") for t in tag_list[:4]]
                    else:
                        article_tags = tag_list[:4] if isinstance(tag_list, list) else []

                    trends.append({
                        "title": article.get("title", ""),
                        "description": article.get("description", "")[:200],
                        "url": article.get("url", ""),
                        "source": "Dev.to",
                        "timestamp": article.get("published_at", "")[:10] if article.get("published_at") else datetime.now().strftime("%Y-%m-%d"),
                        "metrics": {
                            "comments": comments_count,
                            "reactions": positive_reactions_count,
                            "tags": article_tags
                        },
                        "heat_score": heat_score,
                        "tags": article_tags[:3] + ["开发者博客", "Dev.to"]
                    })

            # 按热度排序并去重
            trends.sort(key=lambda x: x.get("heat_score", 0), reverse=True)
            seen_titles = set()
            unique_trends = []
            for trend in trends:
                if trend["title"] not in seen_titles:
                    seen_titles.add(trend["title"])
                    unique_trends.append(trend)

            self.log(f"Dev.to成功获取 {len(unique_trends)} 篇文章")
            return unique_trends[:15]
        except Exception as e:
            self.log(f"Dev.to API调用失败: {e}", "ERROR")
            return []

    def _get_pypi_trends(self) -> List[Dict[str, Any]]:
        """获取PyPI热门Python包（热门AI和数据科学包）"""
        try:
            # 使用pypistats.org API（完全免费）
            base_url = "https://pypistats.org/api/packages"

            # 使用热门AI和数据科学包列表
            packages = [
                "langchain", "openai", "anthropic", "transformers", "torch",
                "tensorflow", "numpy", "pandas", "scikit-learn", "requests",
                "fastapi", "pytest", "matplotlib", "plotly", "gradio"
            ]
            trends = []

            for package in packages[:10]:  # 最多10个包
                try:
                    # 获取最近30天的下载统计
                    url = f"{base_url}/{package}/recent"
                    response = requests.get(url, timeout=10)

                    if response.status_code != 200:
                        continue

                    data = response.json()

                    # 获取下载量
                    last_month = data.get("data", {}).get("last_month", 0)
                    last_week = data.get("data", {}).get("last_week", 0)

                    if last_month == 0:
                        continue

                    # 计算热度评分（下载量的对数）
                    import math
                    heat_score = math.log10(max(last_month, 1)) * 20

                    # 获取包详情
                    package_url = f"https://pypi.org/pypi/{package}/json"
                    package_response = requests.get(package_url, timeout=10)

                    description = ""
                    if package_response.status_code == 200:
                        package_info = package_response.json().get("info", {})
                        description = package_info.get("summary", "")[:200]

                    trends.append({
                        "title": f"📦 {package}",
                        "description": description or f"PyPI包 - 最近30天下载量: {last_month:,}",
                        "url": f"https://pypi.org/project/{package}/",
                        "source": "PyPI",
                        "timestamp": datetime.now().strftime("%Y-%m-%d"),
                        "metrics": {
                            "last_month_downloads": last_month,
                            "last_week_downloads": last_week
                        },
                        "heat_score": heat_score,
                        "tags": ["Python", "包管理", "工具"]
                    })
                except Exception as e:
                    self.log(f"获取PyPI包 {package} 失败: {e}", "WARNING")
                    continue

            # 按热度排序
            trends.sort(key=lambda x: x.get("heat_score", 0), reverse=True)

            self.log(f"PyPI成功获取 {len(trends)} 个包")
            return trends[:15]
        except Exception as e:
            self.log(f"PyPI API调用失败: {e}", "ERROR")
            return []

    def _get_github_topics_trends(self) -> List[Dict[str, Any]]:
        """获取GitHub Topics（AI和科技热门主题）"""
        try:
            # GitHub Topics API - 获取AI和科技相关热门主题
            topics = [
                "artificial-intelligence", "machine-learning", "deep-learning",
                "llm", "generative-ai", "automation", "developer-tools"
            ]
            trends = []

            for topic_name in topics[:3]:  # 最多3个主题
                try:
                    # 搜索该主题下的热门仓库
                    api_url = "https://api.github.com/search/repositories"
                    params = {
                        "q": f"topic:{topic_name}",
                        "sort": "stars",
                        "order": "desc",
                        "per_page": 10
                    }

                    response = requests.get(api_url, params=params, timeout=15, headers={
                        "Accept": "application/vnd.github.v3+json"
                    })

                    if response.status_code != 200:
                        continue

                    data = response.json()

                    if "items" not in data:
                        continue

                    for item in data["items"][:7]:
                        # 计算热度评分
                        stars = item.get("stargazers_count", 0)
                        forks = item.get("forks_count", 0)
                        heat_score = stars * 0.5 + forks * 0.3

                        # 获取主题标签
                        item_topics = item.get("topics", [])[:5]

                        trends.append({
                            "title": item.get("full_name", ""),
                            "description": item.get("description", f"GitHub Topic: {topic_name}")[:200],
                            "url": item.get("html_url", ""),
                            "source": f"GitHub Topics ({topic_name})",
                            "timestamp": datetime.now().strftime("%Y-%m-%d"),
                            "metrics": {
                                "stars": stars,
                                "forks": forks,
                                "topics": item_topics,
                                "language": item.get("language", "Unknown")
                            },
                            "heat_score": heat_score,
                            "tags": item_topics[:3] + ["行业应用", "开源"]
                        })

                except Exception as e:
                    self.log(f"获取GitHub Topic {topic_name} 失败: {e}", "WARNING")
                    continue

            # 按热度排序并去重
            trends.sort(key=lambda x: x.get("heat_score", 0), reverse=True)
            seen_titles = set()
            unique_trends = []
            for trend in trends:
                if trend["title"] not in seen_titles:
                    seen_titles.add(trend["title"])
                    unique_trends.append(trend)

            self.log(f"GitHub Topics成功获取 {len(unique_trends)} 个项目")
            return unique_trends[:15]
        except Exception as e:
            self.log(f"GitHub Topics API调用失败: {e}", "ERROR")
            return []
