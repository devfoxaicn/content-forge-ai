"""
真实AI热点分析Agent - 产品+新闻+学术导向
聚焦热门AI产品、应用、行业新闻和重大突破
受众：广泛，非技术细节
"""

from typing import Dict, Any, List, Optional
import json
import os
import requests
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
            "github_apps": "github" in sources_config,
            # 新闻类
            "techcrunch_ai": "techcrunch_ai" in sources_config,
            "verge_ai": "verge_ai" in sources_config,
            "venturebeat_ai": "venturebeat_ai" in sources_config,
            # 学术类（重大新闻）
            "arxiv_news": "arxiv_news" in sources_config,
            # 科技新闻（过滤产品类）
            "hackernews": "hackernews" in sources_config,
        }

        # 获取配置
        agent_config = config.get("agents", {}).get("ai_trend_analyzer", {})
        self.max_trends = agent_config.get("max_trends", 20)
        self.min_score = agent_config.get("min_heat_score", 60)

        # 初始化分类关键词
        self._init_category_keywords()

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

            # 选择热度最高的话题
            if hot_topics:
                selected_topic = hot_topics[0]
                self.log(f"选择热点话题: {selected_topic['title']}")
            else:
                selected_topic = {
                    "title": "AI技术发展",
                    "description": "人工智能前沿动态",
                    "url": "",
                    "source": "默认"
                }

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

    def _get_real_ai_trends(self) -> List[Dict[str, Any]]:
        """
        从多个数据源获取真实AI热点（产品+新闻+学术）

        Returns:
            List[Dict[str, Any]]: 热点话题列表
        """
        all_trends = []

        # ===== 产品类数据源 =====

        # 1. Product Hunt - 热门AI产品
        if self.sources["producthunt"]:
            try:
                ph_trends = self._get_product_hunt_trends()
                all_trends.extend(ph_trends)
                self.log(f"Product Hunt: 获取 {len(ph_trends)} 条热点")
            except Exception as e:
                self.log(f"Product Hunt获取失败: {e}", "WARNING")

        # 2. GitHub Trending - AI应用项目
        if self.sources["github_apps"]:
            try:
                gh_trends = self._get_github_ai_apps()
                all_trends.extend(gh_trends)
                self.log(f"GitHub AI应用: 获取 {len(gh_trends)} 条热点")
            except Exception as e:
                self.log(f"GitHub AI应用获取失败: {e}", "WARNING")

        # ===== 新闻类数据源 =====

        # 3. TechCrunch AI
        if self.sources["techcrunch_ai"]:
            try:
                tc_trends = self._get_techcrunch_ai_trends()
                all_trends.extend(tc_trends)
                self.log(f"TechCrunch AI: 获取 {len(tc_trends)} 条热点")
            except Exception as e:
                self.log(f"TechCrunch AI获取失败: {e}", "WARNING")

        # 4. The Verge AI
        if self.sources["verge_ai"]:
            try:
                verge_trends = self._get_verge_ai_trends()
                all_trends.extend(verge_trends)
                self.log(f"The Verge AI: 获取 {len(verge_trends)} 条热点")
            except Exception as e:
                self.log(f"The Verge AI获取失败: {e}", "WARNING")

        # 5. VentureBeat AI
        if self.sources["venturebeat_ai"]:
            try:
                vb_trends = self._get_venturebeat_ai_trends()
                all_trends.extend(vb_trends)
                self.log(f"VentureBeat AI: 获取 {len(vb_trends)} 条热点")
            except Exception as e:
                self.log(f"VentureBeat AI获取失败: {e}", "WARNING")

        # ===== 学术类数据源（重大新闻） =====

        # 6. arXiv重大论文新闻
        if self.sources["arxiv_news"]:
            try:
                arxiv_trends = self._get_arxiv_major_news()
                all_trends.extend(arxiv_trends)
                self.log(f"arXiv重大新闻: 获取 {len(arxiv_trends)} 条热点")
            except Exception as e:
                self.log(f"arXiv重大新闻获取失败: {e}", "WARNING")

        # ===== 科技新闻（过滤产品类） =====

        # 7. HackerNews（产品类过滤）
        if self.sources["hackernews"]:
            try:
                hn_trends = self._get_hacker_news_products()
                all_trends.extend(hn_trends)
                self.log(f"HackerNews产品类: 获取 {len(hn_trends)} 条热点")
            except Exception as e:
                self.log(f"HackerNews产品类获取失败: {e}", "WARNING")

        # 按综合热度评分排序
        all_trends.sort(key=lambda x: x.get("heat_score", 0), reverse=True)

        # 去重（基于标题相似度）
        all_trends = self._deduplicate_trends(all_trends)

        # 过滤低分内容
        all_trends = [t for t in all_trends if t.get("heat_score", 0) >= self.min_score]

        # 对每个热点进行分类
        for trend in all_trends:
            classification = self._classify_trend(trend)
            trend["category"] = classification["category"]
            trend["category_icon"] = classification["icon"]
            trend["category_confidence"] = classification["confidence"]
            # 更新tags以包含分类信息
            if "tags" not in trend:
                trend["tags"] = []
            trend["tags"].append(classification["category"].replace("📈 ", "").replace("🎓 ", "").replace("🔬 ", "").replace("🛠️ ", "").replace("💼 ", ""))

        # 返回Top N
        return all_trends[:self.max_trends]

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

                    trends.append({
                        "title": title,
                        "description": description or title[:200],
                        "url": url,
                        "source": source_name,
                        "timestamp": published[:10] if published else datetime.now().strftime("%Y-%m-%d"),
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

    def _save_trends(self, trends: List[Dict[str, Any]]):
        """保存热点分析结果到raw目录"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trends_ai_{timestamp}.json"

            output = {
                "timestamp": datetime.now().isoformat(),
                "total_trends": len(trends),
                "data_sources": list(self.sources.keys()),
                "trends": trends
            }

            # 使用新的存储管理器，保存到raw目录
            filepath = self.storage.save_json("raw", filename, output)

            self.log(f"热点分析已保存: {filepath}")
        except Exception as e:
            self.log(f"保存热点分析失败: {str(e)}", "WARNING")
