"""
热点分类Agent v9.2 - 将按数据源组织的热点按6大分类重新组织

v9.0 更新:
- 5分类 → 6分类重构
- 新增: 🦾 AI Agent 分类
- 实现Top5截取逻辑（宁缺毋滥策略）
- 24小时严格过滤
- 30个数据源分类映射

v9.1 更新:
- 严格24小时时间过滤（时间解析失败或超过24h直接排除）
- 增强时间格式支持（RSS/Atom/HTTP Date等）

v9.2 更新:
- 去除24小时时间限制
- 优先最新数据（按时间戳排序，最新的在前）
- 确保每个分类Top5填满（6×5=30条）
- 只过滤掉没有时间戳的内容
"""

from typing import Dict, Any, List
from src.agents.base import BaseAgent
from src.utils.time_filter import TimeFilter


class TrendCategorizerAgent(BaseAgent):
    """热点分类Agent v9.2 - 按6大分类组织热点，优先最新数据，Top5截取"""

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        super().__init__(config, prompts)
        # 获取配置
        agent_config = config.get("agents", {}).get("trend_categorizer", {})
        self.max_per_category = agent_config.get("max_per_category", 5)  # Top5截取

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行热点分类 (v9.2: 6分类 + 优先最新 + Top5截取)

        Args:
            state: 包含 trends_by_source 的状态

        Returns:
            Dict[str, Any]: 更新后的状态，包含 categorized_trends
        """
        self.log("开始按6大分类组织热点 (v9.2: 优先最新，确保30条满)...")

        try:
            trends_by_source = state.get("trends_by_source", {})
            if not trends_by_source:
                self.log("未找到 trends_by_source，跳过分类")
                return state

            # ========== v9.0: 6大分类定义 ==========
            categories = {
                "📚 学术前沿": {
                    "icon": "📚",
                    "keywords": [
                        "paper", "research", "study", "arxiv", "publication", "publish",
                        "university", "institute", "lab", "professor", "scientist", "researcher",
                        "conference", "journal", "peer-reviewed", "dataset", "breakthrough",
                        "novel", "state-of-the-art", "sota", "semantic scholar", "openalex",
                        "papers with code", "openreview", "dblp", "citation", "theorem",
                        "algorithm", "machine learning", "deep learning", "neural network"
                    ],
                    "items": []
                },
                "🛠️ 开发工具": {
                    "icon": "🛠️",
                    "keywords": [
                        "library", "framework", "package", "sdk", "api", "tool",
                        "hugging face", "model", "dataset", "pypi", "npm", "github release",
                        "python", "javascript", "typescript", "langchain", "pytorch",
                        "tensorflow", "keras", "scikit-learn", "pandas", "numpy"
                    ],
                    "items": []
                },
                "🦾 AI Agent": {
                    "icon": "🦾",
                    "keywords": [
                        "agent", "autonomous", "multi-agent", "autogpt", "babyagi", "agentgpt",
                        "copilot", "assistant", "chatbot", "langchain agent", "ai agent",
                        "autonomous agent", "workflow", "task", "planning", "reasoning",
                        "tool use", "function calling", "openai function", "claude agent"
                    ],
                    "items": []
                },
                "💼 企业应用": {
                    "icon": "💼",
                    "keywords": [
                        "enterprise", "b2b", "business", "solution", "deployment",
                        "implementation", "integration", "workflow", "automation",
                        "industry", "sector", "startup", "funding", "investment",
                        "acquisition", "merger", "partnership", "collaboration"
                    ],
                    "items": []
                },
                "🌐 消费产品": {
                    "icon": "🌐",
                    "keywords": [
                        "product", "app", "service", "launch", "release", "update",
                        "consumer", "user", "mobile", "web", "desktop", "extension",
                        "plugin", "saas", "platform", "tool", "application",
                        "product hunt", "show hn", "startup", "app store", "google play"
                    ],
                    "items": []
                },
                "📰 行业资讯": {
                    "icon": "📰",
                    "keywords": [
                        "news", "report", "analysis", "trend", "forecast", "prediction",
                        "industry", "market", "regulation", "policy", "law", "ethics",
                        "safety", "alignment", "interpretability", "governance",
                        "mit technology review", "stanford hai", "accenture"
                    ],
                    "items": []
                }
            }

            # ========== v9.0: 数据源到分类的映射（30个数据源） ==========
            source_category_map = {
                # 学术前沿
                "arXiv": "📚 学术前沿",
                "Semantic Scholar": "📚 学术前沿",
                "OpenAlex": "📚 学术前沿",
                "Papers with Code": "📚 学术前沿",
                "OpenReview": "📚 学术前沿",
                "DBLP": "📚 学术前沿",

                # 开发工具
                "Hugging Face": "🛠️ 开发工具",
                "PyPI": "🛠️ 开发工具",
                "npm": "🛠️ 开发工具",
                "GitHub Releases": "🛠️ 开发工具",
                "PyTorch": "🛠️ 开发工具",
                "TensorFlow": "🛠️ 开发工具",

                # AI Agent
                "GitHub Trending": "🦾 AI Agent",
                "Product Hunt": "🦾 AI Agent",
                "Reddit": "🦾 AI Agent",
                "Hacker News": "🦾 AI Agent",

                # 企业应用
                "TechCrunch AI": "💼 企业应用",
                "VentureBeat AI": "💼 企业应用",
                "AI Business": "💼 企业应用",
                "InfoQ AI": "💼 企业应用",

                # 消费产品
                "Product Hunt": "🌐 消费产品",
                "Hacker News": "🌐 消费产品",
                "a16z": "🌐 消费产品",
                "App Store": "🌐 消费产品",
                "Google Play": "🌐 消费产品",

                # 行业资讯
                "NewsAPI": "📰 行业资讯",
                "MIT Tech Review": "📰 行业资讯",
                "The Gradient": "📰 行业资讯",
                "MarkTechPost": "📰 行业资讯",
                "Stanford HAI": "📰 行业资讯",
                "Accenture": "📰 行业资讯",
            }

            total_items = 0

            # 遍历所有数据源
            for source_name, trends in trends_by_source.items():
                if not trends:
                    continue

                # 获取该数据源的默认分类
                default_category = source_category_map.get(source_name)

                for trend in trends:
                    # 格式化热点条目
                    formatted_item = self._format_trend_item(trend, source_name)

                    # 确定分类
                    category = self._determine_category(
                        formatted_item,
                        default_category,
                        categories
                    )

                    # 添加到对应分类
                    categories[category]["items"].append(formatted_item)
                    total_items += 1

            # ========== v9.2: 优先最新数据 + Top5截取 + 确保30条满 ==========
            categorized_trends = {}
            total_after_top5 = 0
            total_no_timestamp = 0

            for cat_name, cat_data in categories.items():
                items = cat_data["items"]

                # ========== 第一步: 只过滤掉没有时间戳的内容 ==========
                valid_items = []
                no_ts_count = 0

                for item in items:
                    timestamp = item.get("timestamp", "")
                    if not timestamp:
                        # v9.2: 没有时间戳的直接过滤掉（无法排序）
                        no_ts_count += 1
                        continue
                    # v9.2: 所有的有时间的都保留，不限制24小时
                    valid_items.append(item)

                if no_ts_count > 0:
                    self.log(f"  {cat_name}: 过滤掉{no_ts_count}条无时间戳内容")

                # ========== 第二步: 按时间戳排序（最新的在前）+ 热度作为次要排序 ==========
                sorted_items = sorted(
                    valid_items,
                    key=lambda x: (x.get("timestamp", ""), x.get("heat_score", 0)),
                    reverse=True
                )

                # ========== 第三步: 截取Top5（确保有数据） ==========
                top_items = sorted_items[:self.max_per_category]

                categorized_trends[cat_name] = {
                    "icon": cat_data["icon"],
                    "items": top_items,
                    "count": len(top_items)
                }
                total_after_top5 += len(top_items)
                total_no_timestamp += no_ts_count

            self.log(f"分类完成(优先最新): 原始{total_items}条 -> 无时间戳{total_no_timestamp}条 -> 保留{total_after_top5}条")

            # 统计每个分类的数量
            for cat_name, cat_data in categorized_trends.items():
                if cat_data["count"] > 0:
                    self.log(f"  {cat_name}: {cat_data['count']}条")

            return {
                **state,
                "categorized_trends": categorized_trends,
                "total_trends_count": total_after_top5,
                "current_step": "trend_categorized"
            }

        except Exception as e:
            self.log(f"分类失败: {e}", "ERROR")
            return {
                **state,
                "error_message": f"分类失败: {e}",
                "current_step": "trend_categorizer_failed"
            }

    def _format_trend_item(self, trend: Dict[str, Any], source_name: str) -> Dict[str, Any]:
        """格式化热点条目，添加来源链接等信息"""
        title = trend.get("title", "")
        description = trend.get("description", "")
        url = trend.get("url", "")
        source = trend.get("source", source_name)
        heat_score = trend.get("heat_score", 0)
        tags = trend.get("tags", [])
        timestamp = trend.get("timestamp", "")

        # 提取数据源名称（去掉括号内容）
        if "NewsAPI" in source:
            # 格式: "NewsAPI (TechCrunch)" -> "NewsAPI"
            clean_source = "NewsAPI"
        elif "GitHub" in source:
            clean_source = "GitHub"
        else:
            clean_source = source

        return {
            "title": title,
            "description": description,
            "url": url,
            "source": clean_source,
            "full_source": source,  # 保留完整来源信息
            "heat_score": heat_score,
            "tags": tags,
            "timestamp": timestamp
        }

    def _determine_category(
        self,
        item: Dict[str, Any],
        default_category: str,
        categories: Dict[str, Dict]
    ) -> str:
        """
        确定热点条目的分类 (v9.0: 6分类系统)

        优先级:
        1. 基于数据源的默认分类
        2. 基于关键词匹配
        3. 兜底分类 (行业资讯)
        """
        title = item.get("title", "").lower()
        description = item.get("description", "").lower()
        text = f"{title} {description}"

        # 如果有默认分类且该分类不是None，优先使用
        if default_category and default_category in categories:
            return default_category

        # 基于关键词计算每个分类的匹配度
        category_scores = {}
        for cat_name, cat_data in categories.items():
            keywords = cat_data["keywords"]
            score = sum(1 for kw in keywords if kw.lower() in text)
            category_scores[cat_name] = score

        # 选择得分最高的分类
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            if category_scores[best_category] > 0:
                return best_category

        # v9.0: 兜底分类 - 行业资讯（最通用）
        return "📰 行业资讯"
