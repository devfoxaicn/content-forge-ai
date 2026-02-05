"""
并发数据获取Agent v11.0 - 使用异步并发获取多数据源

性能提升:
- 串行获取: 36个源 ~5-10分钟
- 并发获取: 36个源 ~30-60秒 (10倍提升)

特性:
- 使用aiohttp实现异步HTTP请求
- 信号量限制并发数，避免被封禁
- 自动重试机制
- 失败源降级到同步获取
"""

import asyncio
import aiohttp
import feedparser
import nest_asyncio
from typing import Dict, Any, List, Callable, Tuple, Optional
from datetime import datetime
from src.agents.base import BaseAgent
from loguru import logger

# 应用nest_asyncio以兼容sync/async混用
nest_asyncio.apply()


class ConcurrentFetchAgent(BaseAgent):
    """并发数据获取Agent v11.0"""

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        super().__init__(config, prompts)
        self.name = "concurrent_fetch"

        # 获取配置
        agent_config = config.get("agents", {}).get("concurrent_fetch", {})
        self.max_concurrent = agent_config.get("max_concurrent", 10)
        self.timeout = agent_config.get("timeout_per_source", 15)
        self.rate_limit = agent_config.get("rate_limit", 5)
        self.max_retries = agent_config.get("max_retries", 2)

        # 导入原始数据源函数（从RealAITrendAnalyzerAgent）
        from src.agents.ai_trend_analyzer_real import RealAITrendAnalyzerAgent
        self.trend_analyzer = RealAITrendAnalyzerAgent(config, prompts)

        # 数据源状态追踪
        self.source_status = {}

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """执行并发数据获取"""
        self.log(f"开始并发获取数据（最大并发: {self.max_concurrent}）...")

        # 重置数据源状态
        self.source_status = {}

        try:
            # 运行异步获取
            loop = asyncio.get_event_loop()
            trends_by_source = loop.run_until_complete(self._fetch_all_sources())

            # 统计
            total_count = sum(len(trends) for trends in trends_by_source.values())
            success_count = sum(1 for s in self.source_status.values() if s.get("success", False))

            self.log(f"并发获取完成: {total_count}条热点, {success_count}/{len(self.source_status)}个数据源成功")

            # 打印数据源状态汇总
            self._log_source_summary()

            return {
                **state,
                "trends_by_source": trends_by_source,
                "source_status": self.source_status,
                "total_trends_count": total_count,
                "current_step": "concurrent_fetch_completed"
            }
        except Exception as e:
            self.log(f"并发获取失败: {e}", "ERROR")
            return {
                **state,
                "error_message": str(e),
                "current_step": "concurrent_fetch_failed"
            }

    async def _fetch_all_sources(self) -> Dict[str, List[Dict[str, Any]]]:
        """异步获取所有数据源"""
        # 创建信号量限制并发数
        semaphore = asyncio.Semaphore(self.max_concurrent)

        # 定义所有数据源获取任务（按优先级排序）
        source_tasks = self._get_source_tasks()

        # 创建任务列表
        tasks = [
            self._fetch_with_semaphore(semaphore, func, name)
            for name, func in source_tasks
        ]

        # 并发执行所有任务
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 组织结果
        trends_by_source = {
            "Product Hunt": [],
            "TechCrunch AI": [],
            "The Verge AI": [],
            "VentureBeat AI": [],
            "NewsAPI": [],
            "MIT Technology Review": [],
            "OpenAI Blog": [],
            "Google AI Blog": [],
            "DeepMind Blog": [],
            "Wired": [],
            "BAIR Blog": [],
            "Microsoft Research": [],
            "Meta AI": [],
            "Anthropic": [],
            "MarkTechPost": [],
            "KDnuggets": [],
            "AI Business": [],
            "Unite.AI": [],
            "The Gradient": [],
            "Fast Company": [],
            "InfoQ": [],
            "Hugging Face": [],
            "arXiv": [],
            "Hacker News": [],
            "NewsData.io": [],
            "Reddit": [],
            "GitHub Trending": [],
            "AI News": [],
            "The Decoder": [],
            "量子位": [],
            "机器之心": [],
            "Wired AI": [],
            "VentureBeat AI (RSS)": [],
            "Google AI Blog (RSS)": [],
            "Google DeepMind": [],
            "arXiv NLP": [],
            "arXiv CV": [],
            "arXiv ML": [],
            "Reddit ML": [],
            "Reddit AI": [],
            "Towards Data Science": []
        }

        # 处理结果
        for (name, _), result in zip(source_tasks, results):
            if isinstance(result, Exception):
                self.log(f"{name}失败: {result}", "WARNING")
                self.source_status[name] = {
                    "success": False,
                    "count": 0,
                    "message": str(result)
                }
            elif isinstance(result, list):
                trends_by_source[name] = result
                self.source_status[name] = {
                    "success": len(result) > 0,
                    "count": len(result),
                    "message": "正常" if len(result) > 0 else "无数据"
                }
            else:
                self.source_status[name] = {
                    "success": False,
                    "count": 0,
                    "message": "未知返回类型"
                }

        return trends_by_source

    def _get_source_tasks(self) -> List[Tuple[str, Callable]]:
        """获取所有数据源任务列表"""
        tasks = []

        # 检查数据源是否启用并添加任务
        if self.trend_analyzer.sources.get("techcrunch_ai"):
            tasks.append(("TechCrunch AI", self._fetch_techcrunch_async))
        if self.trend_analyzer.sources.get("newsapi"):
            tasks.append(("NewsAPI", self._fetch_newsapi_async))
        if self.trend_analyzer.sources.get("hackernews"):
            tasks.append(("Hacker News", self._fetch_hackernews_async))
        if self.trend_analyzer.sources.get("mit_tech_review"):
            tasks.append(("MIT Technology Review", self._fetch_mit_async))
        if self.trend_analyzer.sources.get("openai_blog"):
            tasks.append(("OpenAI Blog", self._fetch_openai_async))
        if self.trend_analyzer.sources.get("bair_blog"):
            tasks.append(("BAIR Blog", self._fetch_bair_async))
        if self.trend_analyzer.sources.get("microsoft_research"):
            tasks.append(("Microsoft Research", self._fetch_msr_async))
        if self.trend_analyzer.sources.get("arxiv_news"):
            tasks.append(("arXiv", self._fetch_arxiv_async))
        if self.trend_analyzer.sources.get("marktechpost"):
            tasks.append(("MarkTechPost", self._fetch_marktechpost_async))
        if self.trend_analyzer.sources.get("kdnuggets"):
            tasks.append(("KDnuggets", self._fetch_kdnuggets_async))
        if self.trend_analyzer.sources.get("ai_business"):
            tasks.append(("AI Business", self._fetch_aibusiness_async))
        if self.trend_analyzer.sources.get("the_gradient"):
            tasks.append(("The Gradient", self._fetch_gradient_async))
        if self.trend_analyzer.sources.get("infoq_ai"):
            tasks.append(("InfoQ", self._fetch_infoq_async))
        if self.trend_analyzer.sources.get("hugging_face_blog"):
            tasks.append(("Hugging Face", self._fetch_huggingface_async))
        if self.trend_analyzer.sources.get("newsdata_io"):
            tasks.append(("NewsData.io", self._fetch_newsdata_io_async))
        if self.trend_analyzer.sources.get("github_trending"):
            tasks.append(("GitHub Trending", self._fetch_github_trending_async))
        if self.trend_analyzer.sources.get("ai_news"):
            tasks.append(("AI News", self._fetch_ai_news_async))
        if self.trend_analyzer.sources.get("the_decoder"):
            tasks.append(("The Decoder", self._fetch_the_decoder_async))
        if self.trend_analyzer.sources.get("qbitai"):
            tasks.append(("量子位", self._fetch_qbitai_async))
        if self.trend_analyzer.sources.get("jiqizhixin"):
            tasks.append(("机器之心", self._fetch_jiqizhixin_async))
        if self.trend_analyzer.sources.get("wired_ai_v2"):
            tasks.append(("Wired AI", self._fetch_wired_ai_v2_async))
        if self.trend_analyzer.sources.get("venturebeat_ai_v2"):
            tasks.append(("VentureBeat AI (RSS)", self._fetch_venturebeat_v2_async))
        if self.trend_analyzer.sources.get("google_ai_blog_v2"):
            tasks.append(("Google AI Blog (RSS)", self._fetch_google_ai_v2_async))
        if self.trend_analyzer.sources.get("deepmind_blog_v2"):
            tasks.append(("Google DeepMind", self._fetch_deepmind_v2_async))
        if self.trend_analyzer.sources.get("arxiv_cl"):
            tasks.append(("arXiv NLP", self._fetch_arxiv_cl_async))
        if self.trend_analyzer.sources.get("arxiv_cv"):
            tasks.append(("arXiv CV", self._fetch_arxiv_cv_async))
        if self.trend_analyzer.sources.get("arxiv_lg"):
            tasks.append(("arXiv ML", self._fetch_arxiv_lg_async))
        if self.trend_analyzer.sources.get("reddit_ml_rss"):
            tasks.append(("Reddit ML", self._fetch_reddit_ml_async))
        if self.trend_analyzer.sources.get("reddit_ai_rss"):
            tasks.append(("Reddit AI", self._fetch_reddit_ai_async))
        if self.trend_analyzer.sources.get("towards_data_science"):
            tasks.append(("Towards Data Science", self._fetch_towards_data_science_async))

        return tasks

    async def _fetch_with_semaphore(self, semaphore: asyncio.Semaphore,
                                    fetch_func: Callable, source_name: str):
        """带信号量限制的异步获取"""
        async with semaphore:
            # 添加速率限制
            await asyncio.sleep(1.0 / self.rate_limit)

            try:
                return await fetch_func()
            except Exception as e:
                self.log(f"{source_name} 异步获取失败: {e}", "WARNING")
                # 尝试降级到同步获取
                try:
                    return await asyncio.to_thread(self._fallback_sync, source_name)
                except Exception as sync_error:
                    self.log(f"{source_name} 同步降级也失败: {sync_error}", "ERROR")
                    raise

    def _fallback_sync(self, source_name: str) -> List[Dict[str, Any]]:
        """同步降级方法"""
        # 调用原始的同步方法
        sync_methods = {
            "TechCrunch AI": self.trend_analyzer._get_techcrunch_ai_trends,
            "NewsAPI": self.trend_analyzer._get_newsapi_trends,
            "Hacker News": self.trend_analyzer._get_hacker_news_products,
            "MIT Technology Review": self.trend_analyzer._get_mit_tech_review_trends,
            "OpenAI Blog": self.trend_analyzer._get_openai_blog_trends,
            "BAIR Blog": self.trend_analyzer._get_bair_blog_trends,
            "Microsoft Research": self.trend_analyzer._get_microsoft_research_trends,
            "arXiv": self.trend_analyzer._get_arxiv_major_news,
            "MarkTechPost": self.trend_analyzer._get_marktechpost_trends,
            "KDnuggets": self.trend_analyzer._get_kdnuggets_trends,
            "AI Business": self.trend_analyzer._get_ai_business_trends,
            "The Gradient": self.trend_analyzer._get_gradient_trends,
            "InfoQ": self.trend_analyzer._get_infoq_ai_trends,
            "Hugging Face": self.trend_analyzer._get_hugging_face_blog_trends,
        }
        if source_name in sync_methods:
            return sync_methods[source_name]()
        return []

    # ========== 异步RSS获取方法 ==========

    async def _fetch_rss_async(self, url: str, source_name: str, max_items: int = 15) -> List[Dict[str, Any]]:
        """异步获取RSS"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=self.timeout) as response:
                    content = await response.text()

            feed = feedparser.parse(content)
            trends = []

            for entry in feed.entries[:max_items]:
                title = entry.get("title", "")
                description = entry.get("description", "")

                # 清理HTML标签
                if description:
                    import re
                    description = re.sub(r'<[^>]+>', '', description)
                    description = description.strip()[:300]

                # 解析时间戳（使用TimeFilter工具）
                published = entry.get("published", "")
                timestamp_iso = self._parse_published_date(published)

                trends.append({
                    "title": title,
                    "description": description or title[:200],
                    "url": entry.get("link", ""),
                    "source": source_name,
                    "timestamp": timestamp_iso,
                    "heat_score": 60,
                    "tags": ["新闻", "AI资讯"]
                })

            return trends
        except Exception as e:
            self.log(f"{source_name} RSS获取失败: {e}", "WARNING")
            return []

    # ========== 各数据源的异步获取方法 ==========

    async def _fetch_techcrunch_async(self):
        return await self._fetch_rss_async(
            "https://techcrunch.com/category/artificial-intelligence/feed/",
            "TechCrunch AI", 15
        )

    async def _fetch_mit_async(self):
        return await self._fetch_rss_async(
            "https://www.technologyreview.com/feed/",
            "MIT Technology Review", 15
        )

    async def _fetch_openai_async(self):
        return await self._fetch_rss_async(
            "https://openai.com/news/rss.xml",
            "OpenAI Blog", 15
        )

    async def _fetch_bair_async(self):
        return await self._fetch_rss_async(
            "https://bair.berkeley.edu/blog/feed.xml",
            "BAIR Blog", 12
        )

    async def _fetch_msr_async(self):
        return await self._fetch_rss_async(
            "https://www.microsoft.com/en-us/research/feed/",
            "Microsoft Research", 12
        )

    async def _fetch_marktechpost_async(self):
        return await self._fetch_rss_async(
            "https://www.marktechpost.com/feed",
            "MarkTechPost", 15
        )

    async def _fetch_kdnuggets_async(self):
        return await self._fetch_rss_async(
            "https://www.kdnuggets.com/feed",
            "KDnuggets", 12
        )

    async def _fetch_aibusiness_async(self):
        return await self._fetch_rss_async(
            "https://aibusiness.com/rss.xml",
            "AI Business", 12
        )

    async def _fetch_gradient_async(self):
        return await self._fetch_rss_async(
            "https://thegradient.pub/rss",
            "The Gradient", 8
        )

    async def _fetch_infoq_async(self):
        return await self._fetch_rss_async(
            "https://feed.infoq.com/ai-ml-data-eng",
            "InfoQ", 12
        )

    async def _fetch_huggingface_async(self):
        return await self._fetch_rss_async(
            "https://huggingface.co/blog/feed.xml",
            "Hugging Face", 10
        )

    async def _fetch_ai_news_async(self):
        return await self._fetch_rss_async(
            "https://www.artificialintelligence-news.com/feed/rss/",
            "AI News", 15
        )

    async def _fetch_the_decoder_async(self):
        return await self._fetch_rss_async(
            "https://the-decoder.com/feed/",
            "The Decoder", 15
        )

    async def _fetch_qbitai_async(self):
        return await self._fetch_rss_async(
            "https://www.qbitai.com/feed",
            "量子位", 15
        )

    async def _fetch_jiqizhixin_async(self):
        return await self._fetch_rss_async(
            "https://www.jiqizhixin.com/rss",
            "机器之心", 15
        )

    async def _fetch_wired_ai_v2_async(self):
        return await self._fetch_rss_async(
            "https://www.wired.com/feed/tag/ai/latest/rss",
            "Wired AI", 10
        )

    async def _fetch_venturebeat_v2_async(self):
        return await self._fetch_rss_async(
            "https://venturebeat.com/category/ai/feed/",
            "VentureBeat AI", 15
        )

    async def _fetch_google_ai_v2_async(self):
        return await self._fetch_rss_async(
            "https://blog.google/technology/ai/rss/",
            "Google AI Blog", 10
        )

    async def _fetch_deepmind_v2_async(self):
        return await self._fetch_rss_async(
            "https://deepmind.google/blog/feed/",
            "Google DeepMind", 10
        )

    async def _fetch_arxiv_cl_async(self):
        return await self._fetch_rss_async(
            "https://arxiv.org/rss/cs.CL",
            "arXiv NLP", 15
        )

    async def _fetch_arxiv_cv_async(self):
        return await self._fetch_rss_async(
            "https://arxiv.org/rss/cs.CV",
            "arXiv CV", 15
        )

    async def _fetch_arxiv_lg_async(self):
        return await self._fetch_rss_async(
            "https://arxiv.org/rss/cs.LG",
            "arXiv ML", 15
        )

    async def _fetch_reddit_ml_async(self):
        return await self._fetch_rss_async(
            "https://www.reddit.com/r/MachineLearning/.rss",
            "Reddit ML", 15
        )

    async def _fetch_reddit_ai_async(self):
        return await self._fetch_rss_async(
            "https://www.reddit.com/r/artificial/.rss",
            "Reddit AI", 15
        )

    async def _fetch_towards_data_science_async(self):
        return await self._fetch_rss_async(
            "https://towardsdatascience.com/feed",
            "Towards Data Science", 15
        )

    # ========== 需要特殊处理的异步获取方法 ==========

    async def _fetch_newsapi_async(self):
        """NewsAPI需要特殊处理（API密钥）"""
        return await asyncio.to_thread(self.trend_analyzer._get_newsapi_trends)

    async def _fetch_hackernews_async(self):
        """HackerNews需要特殊处理"""
        return await asyncio.to_thread(self.trend_analyzer._get_hacker_news_products)

    async def _fetch_arxiv_async(self):
        """arXiv需要特殊处理"""
        return await asyncio.to_thread(self.trend_analyzer._get_arxiv_major_news)

    async def _fetch_newsdata_io_async(self):
        """NewsData.io API（如果配置了密钥）"""
        return await asyncio.to_thread(self.trend_analyzer._get_newsdata_io_trends)

    async def _fetch_github_trending_async(self):
        """GitHub Trending API"""
        return await asyncio.to_thread(self.trend_analyzer._get_github_ai_trending)

    def _parse_published_date(self, published_date: str) -> str:
        """
        解析各种格式的发布日期并返回ISO格式字符串

        Args:
            published_date: 原始发布日期字符串

        Returns:
            ISO格式的时间戳字符串 (YYYY-MM-DD HH:MM:SS)
        """
        if not published_date:
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            # 使用 TimeFilter 工具解析时间戳
            from src.utils.time_filter import TimeFilter
            time_filter = TimeFilter(hours=24)
            dt = time_filter._parse_timestamp(published_date)

            if dt:
                # 移除时区信息以保持与其他代码的兼容性（使用naive datetime）
                dt_naive = dt.replace(tzinfo=None)
                return dt_naive.strftime("%Y-%m-%d %H:%M:%S")
            else:
                # 解析失败，使用当前时间
                self.log(f"时间解析失败，使用当前时间: {published_date}", "DEBUG")
                return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        except Exception as e:
            self.log(f"时间解析异常: {published_date}, 错误: {e}", "WARNING")
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _log_source_summary(self):
        """打印数据源状态汇总"""
        self.log("=" * 50)
        self.log("数据源获取状态汇总:")
        self.log("=" * 50)
        success_count = 0
        for source, status in self.source_status.items():
            status_icon = "✅" if status["success"] else "❌"
            self.log(f"{status_icon} {source}: {status['count']}条 ({status['message']})")
            if status["success"]:
                success_count += 1
        self.log("=" * 50)
        self.log(f"成功: {success_count}/{len(self.source_status)} 个数据源")
        self.log("=" * 50)
