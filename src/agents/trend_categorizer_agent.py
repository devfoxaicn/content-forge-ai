"""
热点分类Agent - 将按数据源组织的热点按分类重新组织
"""

from typing import Dict, Any, List
from src.agents.base import BaseAgent


class TrendCategorizerAgent(BaseAgent):
    """热点分类Agent - 按分类组织热点"""

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        super().__init__(config, prompts)

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行热点分类

        Args:
            state: 包含 trends_by_source 的状态

        Returns:
            Dict[str, Any]: 更新后的状态，包含 categorized_trends
        """
        self.log("开始按分类组织热点...")

        try:
            trends_by_source = state.get("trends_by_source", {})
            if not trends_by_source:
                self.log("未找到 trends_by_source，跳过分类")
                return state

            # 5大分类定义
            categories = {
                "📈 行业动态": {
                    "icon": "📈",
                    "keywords": [
                        "raises", "funding", "investment", "acquisition", "acquired", "merger",
                        "ipo", "valuation", "revenue", "strategy", "partnership", "collaboration",
                        "ceo", "founder", "startup", "company", "corporation", "launch", "release",
                        "business", "commercial", "enterprise", "deal"
                    ],
                    "items": []
                },
                "🎓 学术突破": {
                    "icon": "🎓",
                    "keywords": [
                        "paper", "research", "study", "arxiv", "publication", "publish",
                        "university", "institute", "lab", "professor", "scientist", "researcher",
                        "conference", "journal", "peer-reviewed", "dataset", "breakthrough",
                        "novel", "state-of-the-art", "sota"
                    ],
                    "items": []
                },
                "🔬 技术创新": {
                    "icon": "🔬",
                    "keywords": [
                        "model", "algorithm", "architecture", "gpt", "claude", "gemini", "llama",
                        "diffusion", "transformer", "neural", "network", "training", "inference",
                        "framework", "engine", "system", "upgrade", "advance", "breakthrough",
                        "sota", "record", "human-level", "reasoning", "multimodal"
                    ],
                    "items": []
                },
                "🛠️ AI工具/产品": {
                    "icon": "🛠️",
                    "keywords": [
                        "tool", "platform", "service", "app", "software", "application",
                        "product", "saas", "solution", "assistant", "copilot", "chatbot",
                        "generator", "creator", "editor", "plugin", "extension", "integration",
                        "api", "sdk", "library", "package", "release", "launch", "update"
                    ],
                    "items": []
                },
                "💼 AI应用": {
                    "icon": "💼",
                    "keywords": [
                        "use case", "industry", "business", "workflow", "automation",
                        "implementation", "deployment", "integration", "solution", "case study",
                        "application", "enterprise", "organization", "company", "sector"
                    ],
                    "items": []
                }
            }

            # 数据源到分类的映射（用于初步分类）
            source_category_map = {
                "Product Hunt": "🛠️ AI工具/产品",
                "GitHub": "💼 AI应用",
                "TechCrunch AI": "📈 行业动态",
                "The Verge AI": "🔬 技术创新",
                "VentureBeat AI": "📈 行业动态",
                "arXiv": "🎓 学术突破",
                "Hacker News": None,  # HN需要根据内容判断
                "NewsAPI": "📈 行业动态"
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

            # 计算每个分类的数量
            categorized_trends = {}
            for cat_name, cat_data in categories.items():
                categorized_trends[cat_name] = {
                    "icon": cat_data["icon"],
                    "items": cat_data["items"],
                    "count": len(cat_data["items"])
                }

            self.log(f"分类完成: 共{total_items}条热点，分为5大类")

            # 统计每个分类的数量
            for cat_name, cat_data in categorized_trends.items():
                if cat_data["count"] > 0:
                    self.log(f"  {cat_name}: {cat_data['count']}条")

            return {
                **state,
                "categorized_trends": categorized_trends,
                "total_trends_count": total_items,
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
        确定热点条目的分类

        优先级:
        1. 基于数据源的默认分类
        2. 基于关键词匹配
        3. 兜底分类
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

        # 兜底分类
        return "🔬 技术创新"
