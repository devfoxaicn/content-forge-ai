"""
真实AI热点分析Agent - 产品+新闻+学术导向
聚焦热门AI产品、应用、行业新闻和重大突破
受众：广泛，非技术细节
"""

from typing import Dict, Any, List, Optional
import json
import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import feedparser
from datetime import datetime, timedelta
from src.agents.base import BaseAgent
from src.utils.storage_v2 import StorageFactory


class RealAITrendAnalyzerAgent(BaseAgent):
    """真实的AI热点分析Agent - 产品新闻学术导向"""

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        super().__init__(config, prompts)
        self.mock_mode = config.get("agents", {}).get("ai_trend_analyzer", {}).get("mock_mode", False)

        # 使用新的存储管理器
        self.storage = StorageFactory.create_daily(
            base_dir=config.get("storage", {}).get("base_dir", "data")
        )

        # 数据源配置
        sources_config = config.get("agents", {}).get("ai_trend_analyzer", {}).get("sources", [])
        self.sources = {
            # 产品类
            "producthunt": "producthunt" in sources_config,
            # 新闻类
            "techcrunch_ai": "techcrunch_ai" in sources_config,
            "verge_ai": "verge_ai" in sources_config,
            "venturebeat_ai": "venturebeat_ai" in sources_config,
            "newsapi": "newsapi" in sources_config,
            # 原有权威媒体
            "mit_tech_review": "mit_tech_review" in sources_config,
            "openai_blog": "openai_blog" in sources_config,
            "google_ai_blog": "google_ai_blog" in sources_config,
            "deepmind_blog": "deepmind_blog" in sources_config,
            "wired_ai": "wired_ai" in sources_config,
            # ========== 新增权威AI媒体源 ==========
            "bair_blog": "bair_blog" in sources_config,
            "microsoft_research": "microsoft_research" in sources_config,
            "meta_ai_blog": "meta_ai_blog" in sources_config,
            "anthropic_blog": "anthropic_blog" in sources_config,
            "marktechpost": "marktechpost" in sources_config,
            "kdnuggets": "kdnuggets" in sources_config,
            "ai_business": "ai_business" in sources_config,
            "unite_ai": "unite_ai" in sources_config,
            "the_gradient": "the_gradient" in sources_config,
            "fastcompany_ai": "fastcompany_ai" in sources_config,
            "infoq_ai": "infoq_ai" in sources_config,
            "hugging_face_blog": "hugging_face_blog" in sources_config,
            # 学术类（重大新闻）
            "arxiv_news": "arxiv_news" in sources_config,
            # 科技新闻（过滤产品类）
            "hackernews": "hackernews" in sources_config,
            # ========== 新增实时数据源 (v10.0) ==========
            "newsdata_io": "newsdata_io" in sources_config,
            "reddit_stream": "reddit_stream" in sources_config,
            "github_trending": "github_trending" in sources_config,
            # ========== 新增免费RSS数据源 (v10.1) ==========
            "ai_news": "ai_news" in sources_config,
            "the_decoder": "the_decoder" in sources_config,
            "qbitai": "qbitai" in sources_config,
            "jiqizhixin": "jiqizhixin" in sources_config,
            "wired_ai_v2": "wired_ai_v2" in sources_config,
            "venturebeat_ai_v2": "venturebeat_ai_v2" in sources_config,
            "google_ai_blog_v2": "google_ai_blog_v2" in sources_config,
            "deepmind_blog_v2": "deepmind_blog_v2" in sources_config,
            "arxiv_cl": "arxiv_cl" in sources_config,
            "arxiv_cv": "arxiv_cv" in sources_config,
            "arxiv_lg": "arxiv_lg" in sources_config,
            "reddit_ml_rss": "reddit_ml_rss" in sources_config,
            "reddit_ai_rss": "reddit_ai_rss" in sources_config,
            "towards_data_science": "towards_data_science" in sources_config,
        }

        # 获取配置
        agent_config = config.get("agents", {}).get("ai_trend_analyzer", {})
        self.max_trends = agent_config.get("max_trends", 20)
        self.min_score = agent_config.get("min_heat_score", 60)

        # 初始化分类关键词
        self._init_category_keywords()

        # 创建带重试机制的session
        self.session = self._create_retry_session()

        # 数据源状态追踪
        self.source_status = {}

    def _create_retry_session(self) -> requests.Session:
        """创建带重试机制的HTTP session"""
        session = requests.Session()

        # 配置重试策略
        retry_strategy = Retry(
            total=3,  # 总共重试3次
            backoff_factor=1,  # 重试间隔递增因子
            status_forcelist=[429, 500, 502, 503, 504],  # 需要重试的HTTP状态码
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # 设置默认超时和headers
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })

        return session

    def _safe_request(self, url: str, timeout: int = 15, params: dict = None) -> Optional[requests.Response]:
        """安全的HTTP请求，带重试和错误处理"""
        try:
            response = self.session.get(url, timeout=timeout, params=params)
            response.raise_for_status()
            return response
        except requests.exceptions.SSLError as e:
            self.log(f"SSL错误 {url}: {e}", "WARNING")
            return None
        except requests.exceptions.Timeout:
            self.log(f"请求超时 {url}", "WARNING")
            return None
        except requests.exceptions.RequestException as e:
            self.log(f"请求失败 {url}: {e}", "WARNING")
            return None
        except Exception as e:
            self.log(f"未知错误 {url}: {e}", "ERROR")
            return None

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行AI热点分析

        Args:
            state: 当前工作流状态

        Returns:
            Dict[str, Any]: 更新后的状态
        """
        # 检测是否为用户指定话题模式
        if state.get("current_step") == "user_topic_set":
            self.log("检测到用户指定话题模式，跳过AI热点分析")
            return state

        self.log(f"开始分析AI产品与科技热点，目标: {self.max_trends}个")
        # 重置数据源状态
        self.source_status = {}

        try:
            # 判断是否使用mock模式
            if self.mock_mode:
                self.log("使用Mock模式（模拟数据）")
                from src.agents.ai_trend_analyzer import AITrendAnalyzerAgent
                mock_agent = AITrendAnalyzerAgent(self.config, self.prompts)
                hot_topics = mock_agent._get_mock_ai_trends(state.get('topic', 'AI'))
            else:
                self.log("使用真实API模式（产品+新闻+学术）")
                hot_topics = self._get_real_ai_trends()

            self.log(f"成功分析出 {len(hot_topics)} 个热点话题")

            # 保存热点分析结果
            self._save_trends(hot_topics)

            # 计算总数量
            total_count = sum(len(trends) for trends in hot_topics.values())

            # 选择热度最高的话题（用于兼容旧代码）
            all_trends_flat = []
            for trends in hot_topics.values():
                all_trends_flat.extend(trends)

            if all_trends_flat:
                # 按热度排序
                all_trends_flat.sort(key=lambda x: x.get("heat_score", 0), reverse=True)
                selected_topic = all_trends_flat[0]
                self.log(f"选择热点话题: {selected_topic['title']}")
            else:
                selected_topic = {
                    "title": "AI技术发展",
                    "description": "人工智能前沿动态",
                    "url": "",
                    "source": "默认"
                }

            # 打印数据源状态汇总
            self._log_source_summary()

            return {
                **state,
                "trends_by_source": hot_topics,
                "source_status": self.source_status,  # 新增：数据源状态
                "total_trends_count": total_count,
                "ai_hot_topics": all_trends_flat[:20],  # 保留旧字段兼容
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

    def _get_real_ai_trends(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        从多个数据源获取真实AI热点（产品+新闻+学术）

        Returns:
            Dict[str, List[Dict[str, Any]]]: 按数据源组织的热点
        """
        all_trends = []

        # ===== 产品类数据源 =====

        # 1. Product Hunt - 热门AI产品
        if self.sources["producthunt"]:
            ph_trends = self._get_product_hunt_trends()
            all_trends.extend(ph_trends)
            self.source_status["Product Hunt"] = {
                "success": len(ph_trends) > 0,
                "count": len(ph_trends),
                "message": "正常" if len(ph_trends) > 0 else "无数据"
            }
        else:
            self.source_status["Product Hunt"] = {
                "success": False,
                "count": 0,
                "message": "未启用"
            }

        # ===== 新闻类数据源 =====

        # 2. TechCrunch AI
        if self.sources["techcrunch_ai"]:
            tc_trends = self._get_techcrunch_ai_trends()
            all_trends.extend(tc_trends)
            self.source_status["TechCrunch AI"] = {
                "success": len(tc_trends) > 0,
                "count": len(tc_trends),
                "message": "正常" if len(tc_trends) > 0 else "无数据"
            }
        else:
            self.source_status["TechCrunch AI"] = {
                "success": False,
                "count": 0,
                "message": "未启用"
            }

        # 3. The Verge AI
        if self.sources["verge_ai"]:
            verge_trends = self._get_verge_ai_trends()
            all_trends.extend(verge_trends)
            self.source_status["The Verge AI"] = {
                "success": len(verge_trends) > 0,
                "count": len(verge_trends),
                "message": "正常" if len(verge_trends) > 0 else "无数据"
            }
        else:
            self.source_status["The Verge AI"] = {
                "success": False,
                "count": 0,
                "message": "未启用"
            }

        # 4. VentureBeat AI
        if self.sources["venturebeat_ai"]:
            vb_trends = self._get_venturebeat_ai_trends()
            all_trends.extend(vb_trends)
            self.source_status["VentureBeat AI"] = {
                "success": len(vb_trends) > 0,
                "count": len(vb_trends),
                "message": "正常" if len(vb_trends) > 0 else "无数据"
            }
        else:
            self.source_status["VentureBeat AI"] = {
                "success": False,
                "count": 0,
                "message": "未启用"
            }

        # 5. NewsAPI.org
        if self.sources["newsapi"]:
            newsapi_trends = self._get_newsapi_trends()
            all_trends.extend(newsapi_trends)
            self.source_status["NewsAPI"] = {
                "success": len(newsapi_trends) > 0,
                "count": len(newsapi_trends),
                "message": "正常" if len(newsapi_trends) > 0 else "无数据"
            }
        else:
            self.source_status["NewsAPI"] = {
                "success": False,
                "count": 0,
                "message": "未启用"
            }

        # ===== 新增权威媒体 =====

        # 6. MIT Technology Review
        if self.sources["mit_tech_review"]:
            mit_trends = self._get_mit_tech_review_trends()
            all_trends.extend(mit_trends)
            self.source_status["MIT Technology Review"] = {
                "success": len(mit_trends) > 0,
                "count": len(mit_trends),
                "message": "正常" if len(mit_trends) > 0 else "无数据"
            }
        else:
            self.source_status["MIT Technology Review"] = {
                "success": False,
                "count": 0,
                "message": "未启用"
            }

        # 7. OpenAI Blog
        if self.sources["openai_blog"]:
            openai_trends = self._get_openai_blog_trends()
            all_trends.extend(openai_trends)
            self.source_status["OpenAI Blog"] = {
                "success": len(openai_trends) > 0,
                "count": len(openai_trends),
                "message": "正常" if len(openai_trends) > 0 else "无数据"
            }
        else:
            self.source_status["OpenAI Blog"] = {
                "success": False,
                "count": 0,
                "message": "未启用"
            }

        # 8. Google AI Blog
        if self.sources["google_ai_blog"]:
            google_trends = self._get_google_ai_blog_trends()
            all_trends.extend(google_trends)
            self.source_status["Google AI Blog"] = {
                "success": len(google_trends) > 0,
                "count": len(google_trends),
                "message": "正常" if len(google_trends) > 0 else "无数据"
            }
        else:
            self.source_status["Google AI Blog"] = {
                "success": False,
                "count": 0,
                "message": "未启用"
            }

        # 9. DeepMind Blog
        if self.sources["deepmind_blog"]:
            deepmind_trends = self._get_deepmind_blog_trends()
            all_trends.extend(deepmind_trends)
            self.source_status["DeepMind Blog"] = {
                "success": len(deepmind_trends) > 0,
                "count": len(deepmind_trends),
                "message": "正常" if len(deepmind_trends) > 0 else "无数据"
            }
        else:
            self.source_status["DeepMind Blog"] = {
                "success": False,
                "count": 0,
                "message": "未启用"
            }

        # 10. Wired AI
        if self.sources["wired_ai"]:
            wired_trends = self._get_wired_ai_trends()
            all_trends.extend(wired_trends)
            self.source_status["Wired"] = {
                "success": len(wired_trends) > 0,
                "count": len(wired_trends),
                "message": "正常" if len(wired_trends) > 0 else "无数据"
            }
        else:
            self.source_status["Wired"] = {
                "success": False,
                "count": 0,
                "message": "未启用"
            }

        # ===== 学术类数据源（重大新闻） =====

        # 11. arXiv重大论文新闻
        if self.sources["arxiv_news"]:
            arxiv_trends = self._get_arxiv_major_news()
            all_trends.extend(arxiv_trends)
            self.source_status["arXiv"] = {
                "success": len(arxiv_trends) > 0,
                "count": len(arxiv_trends),
                "message": "正常" if len(arxiv_trends) > 0 else "无数据"
            }
        else:
            self.source_status["arXiv"] = {
                "success": False,
                "count": 0,
                "message": "未启用"
            }

        # ===== 新增权威AI媒体源 =====

        # 13. BAIR Blog (Berkeley AI Research)
        if self.sources["bair_blog"]:
            bair_trends = self._get_bair_blog_trends()
            all_trends.extend(bair_trends)
            self.source_status["BAIR Blog"] = {
                "success": len(bair_trends) > 0,
                "count": len(bair_trends),
                "message": "正常" if len(bair_trends) > 0 else "无数据"
            }
        else:
            self.source_status["BAIR Blog"] = {
                "success": False,
                "count": 0,
                "message": "未启用"
            }

        # 14. Microsoft Research
        if self.sources["microsoft_research"]:
            ms_trends = self._get_microsoft_research_trends()
            all_trends.extend(ms_trends)
            self.source_status["Microsoft Research"] = {
                "success": len(ms_trends) > 0,
                "count": len(ms_trends),
                "message": "正常" if len(ms_trends) > 0 else "无数据"
            }
        else:
            self.source_status["Microsoft Research"] = {
                "success": False,
                "count": 0,
                "message": "未启用"
            }

        # 15. Meta AI Blog
        if self.sources["meta_ai_blog"]:
            meta_trends = self._get_meta_ai_blog_trends()
            all_trends.extend(meta_trends)
            self.source_status["Meta AI"] = {
                "success": len(meta_trends) > 0,
                "count": len(meta_trends),
                "message": "正常" if len(meta_trends) > 0 else "无数据"
            }
        else:
            self.source_status["Meta AI"] = {
                "success": False,
                "count": 0,
                "message": "未启用"
            }

        # 16. Anthropic Blog
        if self.sources["anthropic_blog"]:
            anthropic_trends = self._get_anthropic_blog_trends()
            all_trends.extend(anthropic_trends)
            self.source_status["Anthropic"] = {
                "success": len(anthropic_trends) > 0,
                "count": len(anthropic_trends),
                "message": "正常" if len(anthropic_trends) > 0 else "无数据"
            }
        else:
            self.source_status["Anthropic"] = {
                "success": False,
                "count": 0,
                "message": "未启用"
            }

        # 17. MarkTechPost
        if self.sources["marktechpost"]:
            mtp_trends = self._get_marktechpost_trends()
            all_trends.extend(mtp_trends)
            self.source_status["MarkTechPost"] = {
                "success": len(mtp_trends) > 0,
                "count": len(mtp_trends),
                "message": "正常" if len(mtp_trends) > 0 else "无数据"
            }
        else:
            self.source_status["MarkTechPost"] = {
                "success": False,
                "count": 0,
                "message": "未启用"
            }

        # 18. KDnuggets
        if self.sources["kdnuggets"]:
            kd_trends = self._get_kdnuggets_trends()
            all_trends.extend(kd_trends)
            self.source_status["KDnuggets"] = {
                "success": len(kd_trends) > 0,
                "count": len(kd_trends),
                "message": "正常" if len(kd_trends) > 0 else "无数据"
            }
        else:
            self.source_status["KDnuggets"] = {
                "success": False,
                "count": 0,
                "message": "未启用"
            }

        # 19. AI Business
        if self.sources["ai_business"]:
            aib_trends = self._get_ai_business_trends()
            all_trends.extend(aib_trends)
            self.source_status["AI Business"] = {
                "success": len(aib_trends) > 0,
                "count": len(aib_trends),
                "message": "正常" if len(aib_trends) > 0 else "无数据"
            }
        else:
            self.source_status["AI Business"] = {
                "success": False,
                "count": 0,
                "message": "未启用"
            }

        # 20. Unite.AI
        if self.sources["unite_ai"]:
            unite_trends = self._get_unite_ai_trends()
            all_trends.extend(unite_trends)
            self.source_status["Unite.AI"] = {
                "success": len(unite_trends) > 0,
                "count": len(unite_trends),
                "message": "正常" if len(unite_trends) > 0 else "无数据"
            }
        else:
            self.source_status["Unite.AI"] = {
                "success": False,
                "count": 0,
                "message": "未启用"
            }

        # 21. The Gradient
        if self.sources["the_gradient"]:
            gradient_trends = self._get_gradient_trends()
            all_trends.extend(gradient_trends)
            self.source_status["The Gradient"] = {
                "success": len(gradient_trends) > 0,
                "count": len(gradient_trends),
                "message": "正常" if len(gradient_trends) > 0 else "无数据"
            }
        else:
            self.source_status["The Gradient"] = {
                "success": False,
                "count": 0,
                "message": "未启用"
            }

        # 22. Fast Company AI
        if self.sources["fastcompany_ai"]:
            fastcompany_trends = self._get_fastcompany_ai_trends()
            all_trends.extend(fastcompany_trends)
            self.source_status["Fast Company"] = {
                "success": len(fastcompany_trends) > 0,
                "count": len(fastcompany_trends),
                "message": "正常" if len(fastcompany_trends) > 0 else "无数据"
            }
        else:
            self.source_status["Fast Company"] = {
                "success": False,
                "count": 0,
                "message": "未启用"
            }

        # 23. InfoQ AI
        if self.sources["infoq_ai"]:
            infoq_trends = self._get_infoq_ai_trends()
            all_trends.extend(infoq_trends)
            self.source_status["InfoQ"] = {
                "success": len(infoq_trends) > 0,
                "count": len(infoq_trends),
                "message": "正常" if len(infoq_trends) > 0 else "无数据"
            }
        else:
            self.source_status["InfoQ"] = {
                "success": False,
                "count": 0,
                "message": "未启用"
            }

        # 24. Hugging Face Blog
        if self.sources["hugging_face_blog"]:
            hf_trends = self._get_hugging_face_blog_trends()
            all_trends.extend(hf_trends)
            self.source_status["Hugging Face"] = {
                "success": len(hf_trends) > 0,
                "count": len(hf_trends),
                "message": "正常" if len(hf_trends) > 0 else "无数据"
            }
        else:
            self.source_status["Hugging Face"] = {
                "success": False,
                "count": 0,
                "message": "未启用"
            }

        # ===== 科技新闻（过滤产品类） =====

        # 25. HackerNews（产品类过滤）
        if self.sources["hackernews"]:
            hn_trends = self._get_hacker_news_products()
            all_trends.extend(hn_trends)
            self.source_status["Hacker News"] = {
                "success": len(hn_trends) > 0,
                "count": len(hn_trends),
                "message": "正常" if len(hn_trends) > 0 else "无数据"
            }
        else:
            self.source_status["Hacker News"] = {
                "success": False,
                "count": 0,
                "message": "未启用"
            }

        # ========== 新增实时数据源 (v10.0) ==========

        # 26. NewsData.io（实时新闻API，秒级更新）
        if self.sources["newsdata_io"]:
            newsdata_trends = self._get_newsdata_io_trends()
            all_trends.extend(newsdata_trends)
            self.source_status["NewsData.io"] = {
                "success": len(newsdata_trends) > 0,
                "count": len(newsdata_trends),
                "message": "正常" if len(newsdata_trends) > 0 else "无数据"
            }
        else:
            self.source_status["NewsData.io"] = {
                "success": False,
                "count": 0,
                "message": "未启用"
            }

        # 27. Reddit Stream（实时社区讨论）
        if self.sources["reddit_stream"]:
            reddit_trends = self._get_reddit_ai_stream()
            all_trends.extend(reddit_trends)
            self.source_status["Reddit"] = {
                "success": len(reddit_trends) > 0,
                "count": len(reddit_trends),
                "message": "正常" if len(reddit_trends) > 0 else "无数据"
            }
        else:
            self.source_status["Reddit"] = {
                "success": False,
                "count": 0,
                "message": "未启用"
            }

        # 28. GitHub Trending（开发者关注热点）
        if self.sources["github_trending"]:
            github_trends = self._get_github_ai_trending()
            all_trends.extend(github_trends)
            self.source_status["GitHub Trending"] = {
                "success": len(github_trends) > 0,
                "count": len(github_trends),
                "message": "正常" if len(github_trends) > 0 else "无数据"
            }
        else:
            self.source_status["GitHub Trending"] = {
                "success": False,
                "count": 0,
                "message": "未启用"
            }

        # ========== 新增免费RSS数据源 (v10.1) ==========

        # 29. AI News（顶级AI新闻媒体，免费RSS）
        if self.sources["ai_news"]:
            ai_news_trends = self._get_ai_news_trends()
            all_trends.extend(ai_news_trends)
            self.source_status["AI News"] = {
                "success": len(ai_news_trends) > 0,
                "count": len(ai_news_trends),
                "message": "正常" if len(ai_news_trends) > 0 else "无数据"
            }
        else:
            self.source_status["AI News"] = {
                "success": False,
                "count": 0,
                "message": "未启用"
            }

        # 30. The Decoder（AI专业新闻，免费RSS）
        if self.sources["the_decoder"]:
            decoder_trends = self._get_the_decoder_trends()
            all_trends.extend(decoder_trends)
            self.source_status["The Decoder"] = {
                "success": len(decoder_trends) > 0,
                "count": len(decoder_trends),
                "message": "正常" if len(decoder_trends) > 0 else "无数据"
            }
        else:
            self.source_status["The Decoder"] = {
                "success": False,
                "count": 0,
                "message": "未启用"
            }

        # 31. 量子位（中文AI第一媒体，免费RSS）
        if self.sources["qbitai"]:
            qbitai_trends = self._get_qbitai_trends()
            all_trends.extend(qbitai_trends)
            self.source_status["量子位"] = {
                "success": len(qbitai_trends) > 0,
                "count": len(qbitai_trends),
                "message": "正常" if len(qbitai_trends) > 0 else "无数据"
            }
        else:
            self.source_status["量子位"] = {
                "success": False,
                "count": 0,
                "message": "未启用"
            }

        # 32. 机器之心（深度AI报道，免费RSS）
        if self.sources["jiqizhixin"]:
            jiqizhixin_trends = self._get_jiqizhixin_trends()
            all_trends.extend(jiqizhixin_trends)
            self.source_status["机器之心"] = {
                "success": len(jiqizhixin_trends) > 0,
                "count": len(jiqizhixin_trends),
                "message": "正常" if len(jiqizhixin_trends) > 0 else "无数据"
            }
        else:
            self.source_status["机器之心"] = {
                "success": False,
                "count": 0,
                "message": "未启用"
            }

        # 33. Wired AI v2（AI专题新闻，免费RSS）
        if self.sources["wired_ai_v2"]:
            wired_ai_v2_trends = self._get_wired_ai_trends_v2()
            all_trends.extend(wired_ai_v2_trends)
            self.source_status["Wired AI"] = {
                "success": len(wired_ai_v2_trends) > 0,
                "count": len(wired_ai_v2_trends),
                "message": "正常" if len(wired_ai_v2_trends) > 0 else "无数据"
            }
        else:
            self.source_status["Wired AI"] = {
                "success": False,
                "count": 0,
                "message": "未启用"
            }

        # 34. VentureBeat AI v2（AI商业新闻，免费RSS）
        if self.sources["venturebeat_ai_v2"]:
            venturebeat_v2_trends = self._get_venturebeat_ai_trends_v2()
            all_trends.extend(venturebeat_v2_trends)
            self.source_status["VentureBeat AI (RSS)"] = {
                "success": len(venturebeat_v2_trends) > 0,
                "count": len(venturebeat_v2_trends),
                "message": "正常" if len(venturebeat_v2_trends) > 0 else "无数据"
            }
        else:
            self.source_status["VentureBeat AI (RSS)"] = {
                "success": False,
                "count": 0,
                "message": "未启用"
            }

        # 35. Google AI Blog v2（官方AI动态，免费RSS）
        if self.sources["google_ai_blog_v2"]:
            google_ai_v2_trends = self._get_google_ai_blog_v2()
            all_trends.extend(google_ai_v2_trends)
            self.source_status["Google AI Blog (RSS)"] = {
                "success": len(google_ai_v2_trends) > 0,
                "count": len(google_ai_v2_trends),
                "message": "正常" if len(google_ai_v2_trends) > 0 else "无数据"
            }
        else:
            self.source_status["Google AI Blog (RSS)"] = {
                "success": False,
                "count": 0,
                "message": "未启用"
            }

        # 36. Google DeepMind（顶级研究，免费RSS）
        if self.sources["deepmind_blog_v2"]:
            deepmind_v2_trends = self._get_deepmind_blog_v2()
            all_trends.extend(deepmind_v2_trends)
            self.source_status["Google DeepMind"] = {
                "success": len(deepmind_v2_trends) > 0,
                "count": len(deepmind_v2_trends),
                "message": "正常" if len(deepmind_v2_trends) > 0 else "无数据"
            }
        else:
            self.source_status["Google DeepMind"] = {
                "success": False,
                "count": 0,
                "message": "未启用"
            }

        # 37. arXiv NLP（自然语言处理论文，免费RSS）
        if self.sources["arxiv_cl"]:
            arxiv_cl_trends = self._get_arxiv_cl_trends()
            all_trends.extend(arxiv_cl_trends)
            self.source_status["arXiv NLP"] = {
                "success": len(arxiv_cl_trends) > 0,
                "count": len(arxiv_cl_trends),
                "message": "正常" if len(arxiv_cl_trends) > 0 else "无数据"
            }
        else:
            self.source_status["arXiv NLP"] = {
                "success": False,
                "count": 0,
                "message": "未启用"
            }

        # 38. arXiv CV（计算机视觉论文，免费RSS）
        if self.sources["arxiv_cv"]:
            arxiv_cv_trends = self._get_arxiv_cv_trends()
            all_trends.extend(arxiv_cv_trends)
            self.source_status["arXiv CV"] = {
                "success": len(arxiv_cv_trends) > 0,
                "count": len(arxiv_cv_trends),
                "message": "正常" if len(arxiv_cv_trends) > 0 else "无数据"
            }
        else:
            self.source_status["arXiv CV"] = {
                "success": False,
                "count": 0,
                "message": "未启用"
            }

        # 39. arXiv ML（机器学习论文，免费RSS）
        if self.sources["arxiv_lg"]:
            arxiv_lg_trends = self._get_arxiv_lg_trends()
            all_trends.extend(arxiv_lg_trends)
            self.source_status["arXiv ML"] = {
                "success": len(arxiv_lg_trends) > 0,
                "count": len(arxiv_lg_trends),
                "message": "正常" if len(arxiv_lg_trends) > 0 else "无数据"
            }
        else:
            self.source_status["arXiv ML"] = {
                "success": False,
                "count": 0,
                "message": "未启用"
            }

        # 40. Reddit ML RSS（机器学习社区，免费RSS）
        if self.sources["reddit_ml_rss"]:
            reddit_ml_trends = self._get_reddit_ml_trends()
            all_trends.extend(reddit_ml_trends)
            self.source_status["Reddit ML"] = {
                "success": len(reddit_ml_trends) > 0,
                "count": len(reddit_ml_trends),
                "message": "正常" if len(reddit_ml_trends) > 0 else "无数据"
            }
        else:
            self.source_status["Reddit ML"] = {
                "success": False,
                "count": 0,
                "message": "未启用"
            }

        # 41. Reddit AI RSS（AI讨论社区，免费RSS）
        if self.sources["reddit_ai_rss"]:
            reddit_ai_trends = self._get_reddit_ai_trends_v2()
            all_trends.extend(reddit_ai_trends)
            self.source_status["Reddit AI"] = {
                "success": len(reddit_ai_trends) > 0,
                "count": len(reddit_ai_trends),
                "message": "正常" if len(reddit_ai_trends) > 0 else "无数据"
            }
        else:
            self.source_status["Reddit AI"] = {
                "success": False,
                "count": 0,
                "message": "未启用"
            }

        # 42. Towards Data Science（数据科学文章，免费RSS）
        if self.sources["towards_data_science"]:
            tds_trends = self._get_towards_data_science_trends()
            all_trends.extend(tds_trends)
            self.source_status["Towards Data Science"] = {
                "success": len(tds_trends) > 0,
                "count": len(tds_trends),
                "message": "正常" if len(tds_trends) > 0 else "无数据"
            }
        else:
            self.source_status["Towards Data Science"] = {
                "success": False,
                "count": 0,
                "message": "未启用"
            }

        # 不再排序、去重、过滤，保留所有数据源的完整内容
        # 按数据源组织返回
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
            # 新增实时数据源 (v10.0)
            "NewsData.io": [],
            "Reddit": [],
            "GitHub Trending": [],
            # 新增免费RSS数据源 (v10.1)
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

        # 将热点按数据源分类
        for trend in all_trends:
            source = trend.get("source", "")
            # 确定数据源分类
            if "Product Hunt" in source:
                trends_by_source["Product Hunt"].append(trend)
            elif "TechCrunch" in source:
                trends_by_source["TechCrunch AI"].append(trend)
            elif "Verge" in source:
                trends_by_source["The Verge AI"].append(trend)
            elif "VentureBeat" in source:
                trends_by_source["VentureBeat AI"].append(trend)
            elif "NewsAPI" in source:
                trends_by_source["NewsAPI"].append(trend)
            elif "MIT" in source:
                trends_by_source["MIT Technology Review"].append(trend)
            elif "OpenAI" in source:
                trends_by_source["OpenAI Blog"].append(trend)
            elif "Google AI" in source:
                trends_by_source["Google AI Blog"].append(trend)
            elif "DeepMind" in source:
                trends_by_source["DeepMind Blog"].append(trend)
            elif "Wired" in source:
                trends_by_source["Wired"].append(trend)
            elif "BAIR" in source:
                trends_by_source["BAIR Blog"].append(trend)
            elif "Microsoft Research" in source:
                trends_by_source["Microsoft Research"].append(trend)
            elif "Meta AI" in source:
                trends_by_source["Meta AI"].append(trend)
            elif "Anthropic" in source:
                trends_by_source["Anthropic"].append(trend)
            elif "MarkTechPost" in source:
                trends_by_source["MarkTechPost"].append(trend)
            elif "KDnuggets" in source:
                trends_by_source["KDnuggets"].append(trend)
            elif "AI Business" in source:
                trends_by_source["AI Business"].append(trend)
            elif "Unite.AI" in source:
                trends_by_source["Unite.AI"].append(trend)
            elif "The Gradient" in source:
                trends_by_source["The Gradient"].append(trend)
            elif "Fast Company" in source:
                trends_by_source["Fast Company"].append(trend)
            elif "InfoQ" in source:
                trends_by_source["InfoQ"].append(trend)
            elif "Hugging Face" in source:
                trends_by_source["Hugging Face"].append(trend)
            elif "arXiv" in source:
                trends_by_source["arXiv"].append(trend)
            elif "Hacker" in source:
                trends_by_source["Hacker News"].append(trend)
            # 新增实时数据源 (v10.0)
            elif "NewsData.io" in source:
                trends_by_source["NewsData.io"].append(trend)
            elif "Reddit" in source:
                trends_by_source["Reddit"].append(trend)
            elif "GitHub" in source:
                trends_by_source["GitHub Trending"].append(trend)
            # 新增免费RSS数据源 (v10.1)
            elif "AI News" in source:
                trends_by_source["AI News"].append(trend)
            elif "The Decoder" in source:
                trends_by_source["The Decoder"].append(trend)
            elif "量子位" in source:
                trends_by_source["量子位"].append(trend)
            elif "机器之心" in source:
                trends_by_source["机器之心"].append(trend)
            elif "Wired AI" in source:
                trends_by_source["Wired AI"].append(trend)
            elif "VentureBeat AI (RSS)" in source:
                trends_by_source["VentureBeat AI (RSS)"].append(trend)
            elif "Google AI Blog (RSS)" in source:
                trends_by_source["Google AI Blog (RSS)"].append(trend)
            elif "Google DeepMind" in source:
                trends_by_source["Google DeepMind"].append(trend)
            elif "arXiv NLP" in source:
                trends_by_source["arXiv NLP"].append(trend)
            elif "arXiv CV" in source:
                trends_by_source["arXiv CV"].append(trend)
            elif "arXiv ML" in source:
                trends_by_source["arXiv ML"].append(trend)
            elif "Reddit ML" in source:
                trends_by_source["Reddit ML"].append(trend)
            elif "Reddit AI" in source:
                trends_by_source["Reddit AI"].append(trend)
            elif "Towards Data Science" in source:
                trends_by_source["Towards Data Science"].append(trend)

        total_count = sum(len(trends) for trends in trends_by_source.values())

        self.log(f"数据源汇总完成: 共{total_count}条热点")
        for source, trends in trends_by_source.items():
            if trends:
                self.log(f"  {source}: {len(trends)}条")

        return trends_by_source

    # ==================== 产品类数据源 ====================

    def _get_product_hunt_trends(self) -> List[Dict[str, Any]]:
        """获取Product Hunt热门AI产品（RSS）"""
        try:
            return self._get_rss_trends(
                rss_url="https://www.producthunt.com/posts/feed",
                source_name="Product Hunt",
                item_type="product",
                max_items=20
            )
        except Exception as e:
            self.log(f"Product Hunt RSS解析失败: {e}", "ERROR")
            return []

    # ==================== 新增权威媒体数据源 ====================

    def _get_mit_tech_review_trends(self) -> List[Dict[str, Any]]:
        """获取MIT Technology Review AI相关文章（RSS）"""
        try:
            return self._get_rss_trends(
                rss_url="https://www.technologyreview.com/feed/",
                source_name="MIT Technology Review",
                item_type="news",
                max_items=15
            )
        except Exception as e:
            self.log(f"MIT Technology Review RSS解析失败: {e}", "ERROR")
            return []

    def _get_openai_blog_trends(self) -> List[Dict[str, Any]]:
        """获取OpenAI官方博客文章（RSS）"""
        try:
            return self._get_rss_trends(
                rss_url="https://openai.com/news/rss.xml",
                source_name="OpenAI Blog",
                item_type="news",
                max_items=15
            )
        except Exception as e:
            self.log(f"OpenAI Blog RSS解析失败: {e}", "ERROR")
            return []

    def _get_google_ai_blog_trends(self) -> List[Dict[str, Any]]:
        """获取Google AI博客文章（RSS）"""
        try:
            return self._get_rss_trends(
                rss_url="https://ai.googleblog.com/feeds/posts/default",
                source_name="Google AI Blog",
                item_type="news",
                max_items=15
            )
        except Exception as e:
            self.log(f"Google AI Blog RSS解析失败: {e}", "ERROR")
            return []

    def _get_deepmind_blog_trends(self) -> List[Dict[str, Any]]:
        """获取DeepMind博客文章（RSS）"""
        try:
            return self._get_rss_trends(
                rss_url="https://deepmind.com/feed",
                source_name="DeepMind Blog",
                item_type="news",
                max_items=10
            )
        except Exception as e:
            self.log(f"DeepMind Blog RSS解析失败: {e}", "ERROR")
            return []

    def _get_wired_ai_trends(self) -> List[Dict[str, Any]]:
        """获取Wired AI相关文章（RSS）"""
        try:
            return self._get_rss_trends(
                rss_url="https://www.wired.com/feed/tag/artificial-intelligence/",
                source_name="Wired",
                item_type="news",
                max_items=15
            )
        except Exception as e:
            self.log(f"Wired AI RSS解析失败: {e}", "ERROR")
            return []

    # ==================== 原有新闻类数据源 ====================

    def _get_github_ai_apps(self) -> List[Dict[str, Any]]:
        """获取GitHub Trending AI应用项目（非框架库）"""
        try:
            api_url = "https://github-trending-api.now.sh/repositories"

            # 搜索AI应用相关的语言和关键词
            search_terms = [
                ("", "weekly"),  # 全局热门
                ("python", "weekly"),
                ("javascript", "weekly"),
                ("typescript", "weekly"),
            ]

            all_repos = []

            for lang, period in search_terms:
                try:
                    params = {
                        "language": lang if lang else None,
                        "since": period
                    }
                    params = {k: v for k, v in params.items() if v is not None}

                    response = requests.get(api_url, params=params, timeout=10)
                    repos = response.json()

                    for repo in repos[:10]:
                        repo["fetched_language"] = lang or "multi"
                        all_repos.append(repo)

                except Exception as e:
                    self.log(f"获取GitHub {lang}趋势失败: {e}", "WARNING")
                    continue

            trends = []

            for repo in all_repos[:50]:  # 取前50个候选
                # 过滤：保留AI应用类项目
                name = repo.get("name", "").lower()
                description = repo.get("description", "").lower()
                combined = f"{name} {description}"

                # 过滤掉纯技术框架/库
                skip_keywords = [
                    "framework", "library", "sdk", "api", "toolkit",
                    "boilerplate", "template", "wrapper", "binding"
                ]

                if any(kw in combined for kw in skip_keywords):
                    continue

                # 优先保留AI应用类项目
                ai_keywords = [
                    "ai", "gpt", "chatbot", "agent", "assistant", "automation",
                    "copilot", "llm", "openai", "claude", "gemini", "stable diffusion",
                    "image", "video", "audio", "text", "code", "generation"
                ]

                if not any(kw in combined for kw in ai_keywords):
                    # 非AI项目降低优先级
                    continue

                stars_str = repo.get("stars", "0")
                stars = self._parse_stars(stars_str)
                forks = self._parse_stars(repo.get("forks", "0"))

                # 计算热度评分
                heat_score = stars * 0.5 + forks * 0.3 + 50  # 基础分50

                description = repo.get("description", "") or "AI应用项目"
                lang = repo.get("fetched_language", repo.get("language", "Unknown"))

                trends.append({
                    "title": f"{repo['author']}/{repo['name']}",
                    "description": description[:200],
                    "url": repo["url"],
                    "source": f"GitHub ({lang})",
                    "timestamp": datetime.now().strftime("%Y-%m-%d"),
                    "metrics": {
                        "stars": stars_str,
                        "forks": repo.get("forks", "0"),
                        "language": lang
                    },
                    "heat_score": int(heat_score),
                    "tags": ["AI应用", "开源", lang]
                })

            return trends[:30]  # 返回前30个
        except Exception as e:
            self.log(f"GitHub AI应用获取失败: {e}", "ERROR")
            return []

    # ==================== 新闻类数据源 ====================

    def _get_techcrunch_ai_trends(self) -> List[Dict[str, Any]]:
        """获取TechCrunch AI新闻（RSS）"""
        try:
            return self._get_rss_trends(
                rss_url="https://techcrunch.com/category/artificial-intelligence/feed/",
                source_name="TechCrunch AI",
                item_type="news",
                max_items=15
            )
        except Exception as e:
            self.log(f"TechCrunch AI RSS解析失败: {e}", "ERROR")
            return []

    def _get_verge_ai_trends(self) -> List[Dict[str, Any]]:
        """获取The Verge AI新闻（RSS）"""
        try:
            return self._get_rss_trends(
                rss_url="https://www.theverge.com/ai-artificial-intelligence/rss/index.xml",
                source_name="The Verge AI",
                item_type="news",
                max_items=15
            )
        except Exception as e:
            self.log(f"The Verge AI RSS解析失败: {e}", "ERROR")
            return []

    def _get_venturebeat_ai_trends(self) -> List[Dict[str, Any]]:
        """获取VentureBeat AI新闻（RSS）"""
        try:
            return self._get_rss_trends(
                rss_url="https://venturebeat.com/ai/feed/",
                source_name="VentureBeat AI",
                item_type="news",
                max_items=10
            )
        except Exception as e:
            self.log(f"VentureBeat AI RSS解析失败: {e}", "ERROR")
            return []

    def _get_newsapi_trends(self) -> List[Dict[str, Any]]:
        """获取NewsAPI.org AI新闻"""
        try:
            from src.utils.api_config import APIConfigManager

            api_config = APIConfigManager()
            api_key = api_config.get_api_key("newsapi")

            # NewsAPI endpoint for everything
            url = "https://newsapi.org/v2/everything"

            # AI相关关键词搜索
            ai_keywords = [
                "artificial intelligence",
                "machine learning",
                "deep learning",
                "AI model",
                "ChatGPT",
                "GPT-4",
                "LLM",
                "large language model",
                "OpenAI",
                "Anthropic",
                "Google Gemini",
                "Claude",
                "neural network",
                "computer vision",
                "NLP"
            ]

            # 将关键词用 OR 连接
            query = " OR ".join(ai_keywords[:10])  # 限制关键词数量

            params = {
                "q": query,
                "apiKey": api_key,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 20,
                "searchIn": "title,description"
            }

            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()

            data = response.json()

            if data.get("status") != "ok":
                self.log(f"NewsAPI返回错误: {data.get('message', 'Unknown error')}", "ERROR")
                return []

            articles = data.get("articles", [])
            trends = []

            for article in articles:
                try:
                    title = article.get("title", "")
                    description = article.get("description", "")
                    url = article.get("url", "")
                    published = article.get("publishedAt", "")
                    source_name = article.get("source", {}).get("name", "NewsAPI")

                    # 过滤掉无标题或无URL的文章
                    if not title or not url or title == "[Removed]":
                        continue

                    # 清理描述
                    if description:
                        import re
                        description = re.sub(r'<[^>]+>', '', description)
                        description = description.strip()[:300]

                    # 计算热度评分
                    heat_score = 65  # NewsAPI基础分

                    # 根据关键词加分
                    title_lower = title.lower()
                    high_value_keywords = [
                        "breakthrough", "launch", "release", "announce", "unveil",
                        "openai", "gpt-4", "claude", "gemini", "llama", "mistral",
                        "billion", "funding", "investment", "acquisition"
                    ]

                    for keyword in high_value_keywords:
                        if keyword.lower() in title_lower:
                            heat_score += 5
                            break

                    # 解析时间戳（使用TimeFilter工具）
                    timestamp_iso = self._parse_published_date(published)

                    trends.append({
                        "title": title,
                        "description": description or title[:200],
                        "url": url,
                        "source": f"NewsAPI ({source_name})",
                        "timestamp": timestamp_iso,
                        "metrics": {
                            "published": published,
                            "source": source_name,
                            "type": "news"
                        },
                        "heat_score": heat_score,
                        "tags": ["AI新闻", "行业资讯"]
                    })

                except Exception as e:
                    self.log(f"处理NewsAPI文章失败: {e}", "WARNING")
                    continue

            return trends[:20]  # 返回前20条

        except ValueError as e:
            # API密钥未配置
            self.log(f"NewsAPI密钥未配置: {e}", "WARNING")
            return []
        except requests.exceptions.RequestException as e:
            self.log(f"NewsAPI请求失败: {e}", "ERROR")
            return []
        except Exception as e:
            self.log(f"NewsAI获取失败: {e}", "ERROR")
            return []

    def _get_rss_trends(self, rss_url: str, source_name: str, item_type: str, max_items: int = 15) -> List[Dict[str, Any]]:
        """通用RSS获取方法"""
        try:
            feed = feedparser.parse(rss_url)

            if feed.bozo:
                self.log(f"{source_name} RSS解析警告: {feed.bozo}", "WARNING")

            trends = []

            for entry in feed.entries[:max_items]:
                try:
                    title = entry.get("title", "")
                    description = entry.get("description", "")

                    # 清理HTML标签
                    if description:
                        import re
                        description = re.sub(r'<[^>]+>', '', description)
                        description = description.strip()[:300]

                    url = entry.get("link", "")
                    published = entry.get("published", "")

                    # 计算热度评分
                    heat_score = 60  # RSS源基础分

                    # 根据类型调整
                    if item_type == "product":
                        heat_score += 20
                    elif item_type == "news":
                        # 新闻类：关键词加分
                        news_keywords = ["发布", "推出", "融资", "收购", "突破", "发布", "launch", "raises", "acquisition"]
                        if any(kw.lower() in title.lower() for kw in news_keywords):
                            heat_score += 15

                        # 知名公司加分
                        companies = ["OpenAI", "Google", "Meta", "Microsoft", "Anthropic", "Apple", "Amazon"]
                        if any(company.lower() in title.lower() for company in companies):
                            heat_score += 10

                    # 解析时间戳（使用TimeFilter工具）
                    timestamp_iso = self._parse_published_date(published)

                    trends.append({
                        "title": title,
                        "description": description or title[:200],
                        "url": url,
                        "source": source_name,
                        "timestamp": timestamp_iso,
                        "metrics": {
                            "published": published,
                            "type": item_type
                        },
                        "heat_score": heat_score,
                        "tags": ["新闻", "AI资讯"] if item_type == "news" else ["产品", "AI工具"]
                    })

                except Exception as e:
                    self.log(f"处理{source_name}条目失败: {e}", "WARNING")
                    continue

            return trends
        except Exception as e:
            self.log(f"{source_name} RSS获取失败: {e}", "ERROR")
            return []

    # ==================== 学术类数据源（重大新闻） ====================

    def _get_arxiv_major_news(self) -> List[Dict[str, Any]]:
        """获取arXiv重大论文新闻（仅重大突破）"""
        try:
            import arxiv

            # 搜索AI相关分类
            query = "cat:cs.AI OR cat:cs.CL OR cat:cs.LG OR cat:cs.CV"

            search = arxiv.Search(
                query=query,
                max_results=50,  # 获取更多候选
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )

            trends = []
            cutoff_date = datetime.now() - timedelta(days=30)  # 扩展到30天

            # 知名机构列表
            major_institutions = [
                "openai", "google", "deepmind", "meta", "anthropic",
                "microsoft", "stanford", "mit", "berkeley", "carnegie",
                "nvidia", "amazon", "apple"
            ]

            # 重大突破关键词
            breakthrough_keywords = [
                "gpt", "claude", "gemini", "llama", "diffusion",
                "breakthrough", "sota", "record", "human-level",
                "reasoning", "agent", "multimodal", "vision"
            ]

            for result in search.results():
                pub_date = result.published.replace(tzinfo=None)
                if pub_date < cutoff_date:
                    continue

                title = result.title.lower()
                authors = [a.name.lower() for a in result.authors]

                # 过滤：必须是知名机构或重大突破
                is_major = False

                # 检查作者是否来自知名机构
                for author in authors[:5]:
                    if any(inst in author for inst in major_institutions):
                        is_major = True
                        break

                # 检查标题是否包含重大突破关键词
                if not is_major:
                    if any(kw in title for kw in breakthrough_keywords):
                        is_major = True

                if not is_major:
                    continue  # 跳过普通论文

                # 计算热度评分
                days_ago = (datetime.now() - pub_date).days
                heat_score = 80 - days_ago * 2  # 基础分更高

                # 重大关键词加分
                if any(kw in title for kw in breakthrough_keywords):
                    heat_score += 10

                trends.append({
                    "title": result.title,
                    "description": result.summary[:300],
                    "url": result.entry_id,
                    "source": "arXiv",
                    "timestamp": pub_date.strftime("%Y-%m-%d"),
                    "metrics": {
                        "authors": [a.name for a in result.authors[:3]],
                        "days_ago": days_ago
                    },
                    "heat_score": heat_score,
                    "tags": ["论文", "学术", "重大突破"]
                })

                if len(trends) >= 20:
                    break

            return trends
        except ImportError:
            self.log("arXiv库未安装，跳过。运行: pip install arxiv", "WARNING")
            return []
        except Exception as e:
            self.log(f"arXiv重大新闻获取失败: {e}", "ERROR")
            return []

    # ==================== 科技新闻（过滤产品类） ====================

    def _get_hacker_news_products(self) -> List[Dict[str, Any]]:
        """获取HackerNews产品类话题（过滤技术细节）"""
        try:
            stories_url = "https://hacker-news.firebaseio.com/v0/topstories.json"
            response = requests.get(stories_url, timeout=10)
            story_ids = response.json()[:50]

            trends = []

            # 保留的关键词
            keep_keywords = [
                "show hn:", "launch", "release", "ai", "gpt", "openai",
                "product", "startup", "company", "raises", "funding",
                "acquired", "microsoft", "google", "apple", "meta"
            ]

            # 过滤的关键词（技术细节）
            skip_keywords = [
                "tutorial", "how to", "guide", "tips", "best practices",
                "programming", "coding", "debug", "framework", "library"
            ]

            for story_id in story_ids:
                try:
                    item_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
                    item = requests.get(item_url, timeout=5).json()

                    if not item or "url" not in item:
                        continue

                    title = item.get("title", "").lower()

                    # 过滤：跳过技术细节类
                    if any(kw in title for kw in skip_keywords):
                        continue

                    # 优先保留产品类
                    if not any(kw in title for kw in keep_keywords):
                        # 非产品类降低优先级
                        continue

                    score = item.get("score", 0)
                    comments = item.get("descendants", 0)
                    heat_score = score * 2 + comments + 40  # 基础分40

                    trends.append({
                        "title": item.get("title", ""),
                        "description": item.get("text", item.get("title", ""))[:200],
                        "url": item.get("url", ""),
                        "source": "Hacker News",
                        "timestamp": datetime.fromtimestamp(item["time"]).strftime("%Y-%m-%d %H:%M"),
                        "metrics": {
                            "upvotes": score,
                            "comments": comments
                        },
                        "heat_score": heat_score,
                        "tags": ["科技新闻", "产品"]
                    })

                except Exception as e:
                    self.log(f"获取HN故事 {story_id} 失败: {e}", "WARNING")
                    continue

            return trends[:30]
        except Exception as e:
            self.log(f"HackerNews产品类获取失败: {e}", "ERROR")
            return []

    # ==================== 新增权威AI媒体数据源 ====================

    def _get_bair_blog_trends(self) -> List[Dict[str, Any]]:
        """获取BAIR Blog文章（RSS）- 顶级学术研究"""
        try:
            return self._get_rss_trends(
                rss_url="https://bair.berkeley.edu/blog/feed.xml",
                source_name="BAIR Blog",
                item_type="news",
                max_items=12
            )
        except Exception as e:
            self.log(f"BAIR Blog RSS解析失败: {e}", "ERROR")
            return []

    def _get_microsoft_research_trends(self) -> List[Dict[str, Any]]:
        """获取Microsoft Research Blog文章（RSS）"""
        try:
            return self._get_rss_trends(
                rss_url="https://www.microsoft.com/en-us/research/feed/",
                source_name="Microsoft Research",
                item_type="news",
                max_items=12
            )
        except Exception as e:
            self.log(f"Microsoft Research RSS解析失败: {e}", "ERROR")
            return []

    def _get_meta_ai_blog_trends(self) -> List[Dict[str, Any]]:
        """获取Meta AI Blog文章（RSS）"""
        try:
            return self._get_rss_trends(
                rss_url="https://ai.meta.com/blog/feed/",
                source_name="Meta AI",
                item_type="news",
                max_items=12
            )
        except Exception as e:
            self.log(f"Meta AI Blog RSS解析失败: {e}", "ERROR")
            return []

    def _get_anthropic_blog_trends(self) -> List[Dict[str, Any]]:
        """获取Anthropic Blog文章（RSS）"""
        try:
            return self._get_rss_trends(
                rss_url="https://www.anthropic.com/blog/rss.xml",
                source_name="Anthropic",
                item_type="news",
                max_items=10
            )
        except Exception as e:
            self.log(f"Anthropic Blog RSS解析失败: {e}", "ERROR")
            return []

    def _get_marktechpost_trends(self) -> List[Dict[str, Any]]:
        """获取MarkTechPost文章（RSS）"""
        try:
            return self._get_rss_trends(
                rss_url="https://www.marktechpost.com/feed",
                source_name="MarkTechPost",
                item_type="news",
                max_items=15
            )
        except Exception as e:
            self.log(f"MarkTechPost RSS解析失败: {e}", "ERROR")
            return []

    def _get_kdnuggets_trends(self) -> List[Dict[str, Any]]:
        """获取KDnuggets文章（RSS）"""
        try:
            return self._get_rss_trends(
                rss_url="https://www.kdnuggets.com/feed",
                source_name="KDnuggets",
                item_type="news",
                max_items=12
            )
        except Exception as e:
            self.log(f"KDnuggets RSS解析失败: {e}", "ERROR")
            return []

    def _get_ai_business_trends(self) -> List[Dict[str, Any]]:
        """获取AI Business文章（RSS）"""
        try:
            return self._get_rss_trends(
                rss_url="https://aibusiness.com/rss.xml",
                source_name="AI Business",
                item_type="news",
                max_items=12
            )
        except Exception as e:
            self.log(f"AI Business RSS解析失败: {e}", "ERROR")
            return []

    def _get_unite_ai_trends(self) -> List[Dict[str, Any]]:
        """获取Unite.AI文章（RSS）"""
        try:
            return self._get_rss_trends(
                rss_url="https://unite.ai/feed",
                source_name="Unite.AI",
                item_type="news",
                max_items=12
            )
        except Exception as e:
            self.log(f"Unite.AI RSS解析失败: {e}", "ERROR")
            return []

    def _get_gradient_trends(self) -> List[Dict[str, Any]]:
        """获取The Gradient文章（RSS）"""
        try:
            return self._get_rss_trends(
                rss_url="https://thegradient.pub/rss",
                source_name="The Gradient",
                item_type="news",
                max_items=8
            )
        except Exception as e:
            self.log(f"The Gradient RSS解析失败: {e}", "ERROR")
            return []

    def _get_fastcompany_ai_trends(self) -> List[Dict[str, Any]]:
        """获取Fast Company AI文章（RSS）"""
        try:
            return self._get_rss_trends(
                rss_url="https://www.fastcompany.com/section/artificial-intelligence/feed",
                source_name="Fast Company",
                item_type="news",
                max_items=12
            )
        except Exception as e:
            self.log(f"Fast Company AI RSS解析失败: {e}", "ERROR")
            return []

    def _get_infoq_ai_trends(self) -> List[Dict[str, Any]]:
        """获取InfoQ AI文章（RSS）"""
        try:
            return self._get_rss_trends(
                rss_url="https://feed.infoq.com/ai-ml-data-eng",
                source_name="InfoQ",
                item_type="news",
                max_items=12
            )
        except Exception as e:
            self.log(f"InfoQ AI RSS解析失败: {e}", "ERROR")
            return []

    def _get_hugging_face_blog_trends(self) -> List[Dict[str, Any]]:
        """获取Hugging Face Blog文章（RSS）"""
        try:
            return self._get_rss_trends(
                rss_url="https://huggingface.co/blog/feed.xml",
                source_name="Hugging Face",
                item_type="news",
                max_items=10
            )
        except Exception as e:
            self.log(f"Hugging Face Blog RSS解析失败: {e}", "ERROR")
            return []

    # ========== 新增实时数据源 (v10.0) ==========

    def _get_newsdata_io_trends(self, max_results: int = 20) -> List[Dict[str, Any]]:
        """从NewsData.io获取实时AI新闻（秒级更新，比NewsAPI更快）"""
        api_key = os.getenv("NEWSDATA_IO_API_KEY")
        if not api_key:
            self.log("未配置NEWSDATA_IO_API_KEY，跳过NewsData.io", "WARNING")
            return []

        try:
            url = "https://newsdata.io/api/1/news"
            params = {
                "apikey": api_key,
                "q": "artificial intelligence OR AI OR machine learning OR deep learning OR LLM OR ChatGPT OR GPT OR Claude OR Gemini",
                "language": "en,zh",
                "category": "technology,science,business",
                "size": max_results
            }

            response = self._safe_request(url, timeout=20, params=params)
            if not response:
                return []

            data = response.json()

            if data.get("status") != "success":
                self.log(f"NewsData.io返回失败: {data}", "WARNING")
                return []

            trends = []
            for item in data.get("results", []):
                # 解析发布时间
                pub_date_str = item.get("pubDate", "")
                try:
                    if pub_date_str:
                        pub_time = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
                        # 计算时效性分数（越新越高）
                        hours_ago = (datetime.now(pub_time.tzinfo) - pub_time).total_seconds() / 3600
                        freshness_score = max(0, 100 - hours_ago * 2)
                    else:
                        freshness_score = 50
                except:
                    freshness_score = 50

                trends.append({
                    "title": item["title"],
                    "description": item.get("description", "")[:500],
                    "url": item["link"],
                    "source": f"NewsData.io - {item.get('source', 'Unknown')}",
                    "published_date": pub_date_str,
                    "heat_score": freshness_score,
                    "timestamp": pub_date_str
                })

            self.log(f"NewsData.io: 获取 {len(trends)} 条实时新闻")
            return trends

        except Exception as e:
            self.log(f"NewsData.io请求失败: {e}", "ERROR")
            return []

    def _get_reddit_ai_stream(self, max_results: int = 15) -> List[Dict[str, Any]]:
        """从Reddit AI相关subreddit获取实时热点（社区讨论，更新极快）"""
        try:
            import praw

            reddit_client_id = os.getenv("REDDIT_CLIENT_ID")
            reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET")

            if not reddit_client_id or not reddit_client_secret:
                self.log("未配置REDDIT_CLIENT_ID或REDDIT_CLIENT_SECRET，跳过Reddit", "WARNING")
                return []

            reddit = praw.Reddit(
                client_id=reddit_client_id,
                client_secret=reddit_client_secret,
                user_agent="ContentForgeAI/1.0"
            )

            trends = []
            subreddits = ["MachineLearning", "artificial", "ChatGPT", "LocalLLaMA"]

            for sub_name in subreddits:
                try:
                    subreddit = reddit.subreddit(sub_name)

                    # 获取热门帖子（实时更新）
                    for post in subreddit.hot(limit=max_results // len(subreddits)):
                        # 计算时效性分数
                        created_time = datetime.fromtimestamp(post.created_utc)
                        hours_ago = (datetime.now() - created_time).total_seconds() / 3600
                        freshness_score = max(0, 100 - hours_ago * 3)

                        # 综合分数：upvotes + 时效性
                        engagement_score = min(100, post.score / 10)
                        heat_score = (freshness_score * 0.6 + engagement_score * 0.4)

                        # 描述：使用selftext或URL
                        description = post.selftext[:500] if post.selftext else post.url

                        trends.append({
                            "title": post.title,
                            "description": description,
                            "url": f"https://reddit.com{post.permalink}",
                            "source": f"Reddit r/{sub_name}",
                            "published_date": created_time.isoformat(),
                            "heat_score": heat_score,
                            "timestamp": created_time.isoformat(),
                            "comments_count": post.num_comments,
                            "upvotes": post.score
                        })
                except Exception as e:
                    self.log(f"Reddit r/{sub_name}获取失败: {e}", "WARNING")
                    continue

            self.log(f"Reddit: 获取 {len(trends)} 条实时热点")
            return trends

        except ImportError:
            self.log("未安装praw库，跳过Reddit数据源 (pip install praw)", "WARNING")
            return []
        except Exception as e:
            self.log(f"Reddit请求失败: {e}", "ERROR")
            return []

    def _get_github_ai_trending(self, max_results: int = 20) -> List[Dict[str, Any]]:
        """从GitHub获取AI相关热门项目（开发者关注热点）"""
        try:
            # GitHub Search API
            url = "https://api.github.com/search/repositories"
            params = {
                "q": "artificial intelligence OR machine learning OR LLM OR chatgpt OR transformer language:python",
                "sort": "updated",
                "order": "desc",
                "per_page": max_results
            }

            response = self._safe_request(url, timeout=15, params=params)
            if not response:
                return []

            data = response.json()
            trends = []

            for repo in data.get("items", []):
                # 检查是否为AI相关
                desc = repo.get("description", "").lower()
                topics = [t.lower() for t in repo.get("topics", [])]
                combined_text = f"{desc} {' '.join(topics)}"

                ai_keywords = ["ai", "machine learning", "deep learning", "llm", "gpt", "chatgpt",
                             "transformer", "neural network", "nlp", "computer vision", "reinforcement"]

                # 至少包含一个AI关键词
                if not any(keyword in combined_text for keyword in ai_keywords):
                    continue

                # 计算热度分数
                stars_score = min(100, repo.get("stargazers_count", 0) / 100)
                forks_score = min(100, repo.get("forks_count", 0) / 50)
                heat_score = (stars_score * 0.7 + forks_score * 0.3)

                # 获取最新更新时间
                updated_at = repo.get("updated_at", "")

                trends.append({
                    "title": repo["name"],
                    "description": repo.get("description", ""),
                    "url": repo["html_url"],
                    "source": "GitHub Trending",
                    "published_date": updated_at,
                    "heat_score": heat_score,
                    "timestamp": updated_at,
                    "stars": repo.get("stargazers_count", 0),
                    "forks": repo.get("forks_count", 0)
                })

            self.log(f"GitHub Trending: 获取 {len(trends)} 个热门AI项目")
            return trends

        except Exception as e:
            self.log(f"GitHub Trending请求失败: {e}", "ERROR")
            return []

    # ========== 新增免费RSS数据源 (v10.1) ==========

    def _get_ai_news_trends(self) -> List[Dict[str, Any]]:
        """从AI News获取实时AI新闻（顶级AI新闻媒体，免费RSS）"""
        try:
            return self._get_rss_trends(
                rss_url="https://www.artificialintelligence-news.com/feed/rss/",
                source_name="AI News",
                item_type="news",
                max_items=15
            )
        except Exception as e:
            self.log(f"AI News RSS解析失败: {e}", "ERROR")
            return []

    def _get_the_decoder_trends(self) -> List[Dict[str, Any]]:
        """从The Decoder获取AI新闻（AI专业新闻媒体，免费RSS）"""
        try:
            return self._get_rss_trends(
                rss_url="https://the-decoder.com/feed/",
                source_name="The Decoder",
                item_type="news",
                max_items=15
            )
        except Exception as e:
            self.log(f"The Decoder RSS解析失败: {e}", "ERROR")
            return []

    def _get_qbitai_trends(self) -> List[Dict[str, Any]]:
        """从量子位获取中文AI新闻（中文AI第一媒体，免费RSS）"""
        try:
            return self._get_rss_trends(
                rss_url="https://www.qbitai.com/feed",
                source_name="量子位",
                item_type="news",
                max_items=15
            )
        except Exception as e:
            self.log(f"量子位RSS解析失败: {e}", "ERROR")
            return []

    def _get_jiqizhixin_trends(self) -> List[Dict[str, Any]]:
        """从机器之心获取AI新闻（深度AI报道，免费RSS）"""
        try:
            return self._get_rss_trends(
                rss_url="https://www.jiqizhixin.com/rss",
                source_name="机器之心",
                item_type="news",
                max_items=15
            )
        except Exception as e:
            self.log(f"机器之心RSS解析失败: {e}", "ERROR")
            return []

    def _get_wired_ai_trends_v2(self) -> List[Dict[str, Any]]:
        """从Wired AI获取AI专题新闻（免费RSS）"""
        try:
            return self._get_rss_trends(
                rss_url="https://www.wired.com/feed/tag/ai/latest/rss",
                source_name="Wired AI",
                item_type="news",
                max_items=10
            )
        except Exception as e:
            self.log(f"Wired AI RSS解析失败: {e}", "ERROR")
            return []

    def _get_venturebeat_ai_trends_v2(self) -> List[Dict[str, Any]]:
        """从VentureBeat AI获取AI商业新闻（免费RSS）"""
        try:
            return self._get_rss_trends(
                rss_url="https://venturebeat.com/category/ai/feed/",
                source_name="VentureBeat AI",
                item_type="news",
                max_items=15
            )
        except Exception as e:
            self.log(f"VentureBeat AI RSS解析失败: {e}", "ERROR")
            return []

    def _get_google_ai_blog_v2(self) -> List[Dict[str, Any]]:
        """从Google AI Blog获取官方AI动态（免费RSS）"""
        try:
            return self._get_rss_trends(
                rss_url="https://blog.google/technology/ai/rss/",
                source_name="Google AI Blog",
                item_type="news",
                max_items=10
            )
        except Exception as e:
            self.log(f"Google AI Blog RSS解析失败: {e}", "ERROR")
            return []

    def _get_deepmind_blog_v2(self) -> List[Dict[str, Any]]:
        """从Google DeepMind获取顶级研究动态（免费RSS）"""
        try:
            return self._get_rss_trends(
                rss_url="https://deepmind.google/blog/feed/",
                source_name="Google DeepMind",
                item_type="news",
                max_items=10
            )
        except Exception as e:
            self.log(f"Google DeepMind RSS解析失败: {e}", "ERROR")
            return []

    def _get_arxiv_cl_trends(self) -> List[Dict[str, Any]]:
        """从arXiv获取NLP论文（cs.CL分类，免费RSS）"""
        try:
            return self._get_rss_trends(
                rss_url="https://arxiv.org/rss/cs.CL",
                source_name="arXiv NLP",
                item_type="academic",
                max_items=15
            )
        except Exception as e:
            self.log(f"arXiv NLP RSS解析失败: {e}", "ERROR")
            return []

    def _get_arxiv_cv_trends(self) -> List[Dict[str, Any]]:
        """从arXiv获取计算机视觉论文（cs.CV分类，免费RSS）"""
        try:
            return self._get_rss_trends(
                rss_url="https://arxiv.org/rss/cs.CV",
                source_name="arXiv CV",
                item_type="academic",
                max_items=15
            )
        except Exception as e:
            self.log(f"arXiv CV RSS解析失败: {e}", "ERROR")
            return []

    def _get_arxiv_lg_trends(self) -> List[Dict[str, Any]]:
        """从arXiv获取机器学习论文（cs.LG分类，免费RSS）"""
        try:
            return self._get_rss_trends(
                rss_url="https://arxiv.org/rss/cs.LG",
                source_name="arXiv ML",
                item_type="academic",
                max_items=15
            )
        except Exception as e:
            self.log(f"arXiv ML RSS解析失败: {e}", "ERROR")
            return []

    def _get_reddit_ml_trends(self) -> List[Dict[str, Any]]:
        """从Reddit r/MachineLearning获取社区热点（免费RSS）"""
        try:
            return self._get_rss_trends(
                rss_url="https://www.reddit.com/r/MachineLearning/.rss",
                source_name="Reddit ML",
                item_type="community",
                max_items=15
            )
        except Exception as e:
            self.log(f"Reddit ML RSS解析失败: {e}", "ERROR")
            return []

    def _get_reddit_ai_trends_v2(self) -> List[Dict[str, Any]]:
        """从Reddit r/artificial获取AI讨论（免费RSS）"""
        try:
            return self._get_rss_trends(
                rss_url="https://www.reddit.com/r/artificial/.rss",
                source_name="Reddit AI",
                item_type="community",
                max_items=15
            )
        except Exception as e:
            self.log(f"Reddit AI RSS解析失败: {e}", "ERROR")
            return []

    def _get_towards_data_science_trends(self) -> List[Dict[str, Any]]:
        """从Towards Data Science获取数据科学文章（Medium出版，免费RSS）"""
        try:
            return self._get_rss_trends(
                rss_url="https://towardsdatascience.com/feed",
                source_name="Towards Data Science",
                item_type="blog",
                max_items=15
            )
        except Exception as e:
            self.log(f"Towards Data Science RSS解析失败: {e}", "ERROR")
            return []

    # ==================== 辅助方法 ====================

    def _init_category_keywords(self):
        """初始化分类关键词"""
        # 按优先级排序的5大分类
        self.category_keywords = {
            "📈 行业动态": {
                "keywords": [
                    "raises", "funding", "investment", "acquisition", "acquired", "merger",
                    "ipo", "valuation", "revenue", "strategy", "partnership", "collaboration",
                    "ceo", "founder", "startup", "company", "corporation", "launch", "release",
                    "business", "commercial", "enterprise", "deal"
                ],
                "icon": "📈",
                "priority": 1
            },
            "🎓 学术突破": {
                "keywords": [
                    "paper", "research", "study", "arxiv", "publication", "publish",
                    "university", "institute", "lab", "professor", "scientist", "researcher",
                    "conference", "journal", "peer-reviewed", "dataset", "breakthrough",
                    "novel", "state-of-the-art", "sota"
                ],
                "icon": "🎓",
                "priority": 2
            },
            "🔬 技术创新": {
                "keywords": [
                    "model", "algorithm", "architecture", "gpt", "claude", "gemini", "llama",
                    "diffusion", "transformer", "neural", "network", "training", "inference",
                    "framework", "engine", "system", "upgrade", "advance", "breakthrough",
                    "sota", "record", "human-level", "reasoning", "multimodal"
                ],
                "icon": "🔬",
                "priority": 3
            },
            "🛠️ AI工具/产品": {
                "keywords": [
                    "tool", "platform", "service", "app", "software", "application",
                    "product", "saas", "solution", "assistant", "copilot", "chatbot",
                    "generator", "creator", "editor", "plugin", "extension", "integration",
                    "api", "sdk", "library", "package", "release", "launch", "update"
                ],
                "icon": "🛠️",
                "priority": 4
            },
            "💼 AI应用": {
                "keywords": [
                    "use case", "industry", "business", "workflow", "automation",
                    "implementation", "deployment", "integration", "solution", "case study",
                    "application", "enterprise", "organization", "company", "sector"
                ],
                "icon": "💼",
                "priority": 5
            }
        }

        # 数据源到分类的映射（用于初步分类）
        self.source_category_map = {
            "Product Hunt": "🛠️ AI工具/产品",
            "GitHub": "💼 AI应用",
            "TechCrunch AI": "📈 行业动态",
            "The Verge AI": "🔬 技术创新",
            "VentureBeat AI": "📈 行业动态",
            "arXiv": "🎓 学术突破",
            "Hacker News": None  # HN需要根据内容判断
        }

    def _classify_trend(self, trend: Dict[str, Any]) -> Dict[str, Any]:
        """
        智能分类热点话题

        Args:
            trend: 热点数据

        Returns:
            分类信息字典
        """
        title = trend.get("title", "").lower()
        description = trend.get("description", "").lower()
        text = f"{title} {description}"

        # 步骤1：基于数据源的初步分类
        source = trend.get("source", "")
        base_category = self.source_category_map.get(source)

        # 步骤2：基于关键词计算每个类别的匹配度
        category_scores = {}

        for category, config in self.category_keywords.items():
            keywords = config["keywords"]

            # 计算关键词匹配分数
            score = 0
            matched_keywords = []

            for keyword in keywords:
                if keyword.lower() in text:
                    score += 1
                    matched_keywords.append(keyword)

            # 如果有基础分类且匹配，加分
            if base_category == category:
                score += 2

            category_scores[category] = {
                "score": score,
                "matched_keywords": matched_keywords
            }

        # 步骤3：选择最高分类
        best_category = max(category_scores.items(), key=lambda x: x[1]["score"])
        category_name = best_category[0]
        category_info = self.category_keywords[category_name]

        # 步骤4：判断是否是有效分类
        if best_category[1]["score"] == 0:
            # 没有匹配到任何关键词，根据数据源分配默认分类
            if base_category:
                category_name = base_category
                category_info = self.category_keywords[base_category]
            else:
                # 兜底分类
                category_name = "🔬 技术创新"
                category_info = self.category_keywords[category_name]

        return {
            "category": category_name,
            "icon": category_info["icon"],
            "confidence": best_category[1]["score"],
            "matched_keywords": best_category[1]["matched_keywords"]
        }

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

    def _deduplicate_trends_v2(self, trends: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """增强的去重逻辑 - 基于URL和标题相似度（v10.1新增）"""
        try:
            import difflib
            from urllib.parse import urlparse
        except ImportError:
            # 如果difflib不可用，回退到简单版本
            return self._deduplicate_trends(trends)

        seen_urls = set()
        seen_titles_normalized = {}
        unique_trends = []

        for trend in trends:
            url = trend.get("url", "")
            title = trend.get("title", "").strip().lower()

            # 标准化标题：移除多余空格、标点
            title_normalized = title.replace("  ", " ").replace("   ", " ")
            title_normalized = ''.join(c for c in title_normalized if c.isalnum() or c.isspace())

            # 1. URL去重（最准确，同一URL必然是重复）
            if url and url in seen_urls:
                continue

            # 2. 标题相似度检查（处理同源新闻）
            is_similar = False
            for seen_id, seen_title in seen_titles_normalized.items():
                # 计算相似度（使用SequenceMatcher）
                similarity = difflib.SequenceMatcher(None, title_normalized, seen_title).ratio()

                # 85%相似度视为重复
                if similarity > 0.85:
                    is_similar = True
                    # 如果新标题更长且更详细，可以替换旧的（可选）
                    if len(title) > len(seen_id) and len(title) - len(seen_id) > 10:
                        # 找到并替换旧的
                        for i, ut in enumerate(unique_trends):
                            if ut.get("title", "").strip().lower() == seen_id:
                                unique_trends[i] = trend
                                seen_titles_normalized[title_normalized] = title_normalized
                                seen_urls.add(url)
                                break
                    break

            if not is_similar:
                unique_trends.append(trend)
                if url:
                    seen_urls.add(url)
                seen_titles_normalized[title_normalized] = title_normalized

        # 记录去重统计
        original_count = len(trends)
        unique_count = len(unique_trends)
        if original_count > unique_count:
            self.log(f"去重：{original_count}条 → {unique_count}条 (移除{original_count - unique_count}条重复)")

        return unique_trends

    def _save_trends(self, trends_by_source: Dict[str, List[Dict[str, Any]]]):
        """保存热点分析结果到raw目录"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trends_ai_{timestamp}.json"

            total_count = sum(len(trends) for trends in trends_by_source.values())

            output = {
                "timestamp": datetime.now().isoformat(),
                "total_trends": total_count,
                "data_sources": list(trends_by_source.keys()),
                "trends_by_source": trends_by_source,
                # 保留旧格式兼容
                "trends": []
            }

            # 展平所有趋势到旧格式
            for source, trends in trends_by_source.items():
                output["trends"].extend(trends)

            # 使用新的存储管理器，保存到raw目录
            filepath = self.storage.save_json("raw", filename, output)

            self.log(f"热点分析已保存: {filepath}")
        except Exception as e:
            self.log(f"保存热点分析失败: {str(e)}", "WARNING")

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
            time_filter = TimeFilter(hours=24)  # 创建实例，时间窗口不重要
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
