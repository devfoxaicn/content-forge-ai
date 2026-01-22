"""
AI热点汇总Agent v3.0 - 世界级科技简报生成
使用categorized_trends生成包含所有数据源完整内容的简报
"""

from typing import Dict, Any, List
import yaml
from pathlib import Path
from datetime import datetime
from src.agents.base import BaseAgent


class NewsDigestAgent(BaseAgent):
    """AI热点汇总Agent v3.0 - 按分类组织的世界级技术简报"""

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        super().__init__(config, prompts)

        digest_config = config.get("agents", {}).get("trends_digest", {})
        self.digest_style = digest_config.get("style", "magazine")
        self.include_analysis = digest_config.get("include_analysis", True)
        self.llm.temperature = 0.7

        # 加载简报 Prompt 模板
        self.digest_prompts = self._load_digest_prompts()

        # 5大分类顺序
        self.CATEGORY_ORDER = [
            "📈 行业动态",
            "🎓 学术突破",
            "🔬 技术创新",
            "🛠️ AI工具/产品",
            "💼 AI应用"
        ]

        self.log("v3.0简报标准已加载 - 按分类组织，包含所有数据源内容")

    def _load_digest_prompts(self) -> Dict[str, Any]:
        """加载简报 Prompt 配置"""
        try:
            if self.prompts and "trends_digest" in self.prompts:
                return self.prompts["trends_digest"]

            config_path = Path(__file__).parent.parent.parent / "config" / "prompts.yaml"
            with open(config_path, 'r', encoding='utf-8') as f:
                prompts = yaml.safe_load(f)
                return prompts.get("trends_digest", {})
        except Exception as e:
            self.log(f"加载 trends_digest 配置失败: {e}，使用默认模板", "WARNING")
            return self._get_default_prompts()

    def _get_default_prompts(self) -> Dict[str, Any]:
        """获取默认 Prompt 模板"""
        return {
            "category_descriptions": {
                "📈 行业动态": "聚焦AI产业投资、并购、战略合作等商业动态",
                "🎓 学术突破": "精选顶级期刊论文与前沿研究成果",
                "🔬 技术创新": "追踪模型架构、算法突破与工程创新",
                "🛠️ AI工具/产品": "发现提升开发效率的实用工具与产品",
                "💼 AI应用": "展示AI在各行业的创新应用案例"
            }
        }

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """生成热点汇总简报"""
        self.log("开始生成世界级AI热点简报（v3.0 - 分类组织版）")

        try:
            categorized_trends = state.get("categorized_trends")
            if not categorized_trends:
                # 如果没有categorized_trends，使用旧逻辑
                self.log("未找到 categorized_trends，使用旧版逻辑")
                return self._execute_legacy(state)

            total_count = state.get("total_trends_count", 0)
            self.log(f"汇总 {total_count} 个热点话题，按5大分类组织")

            digest = self._generate_digest_v3(state, categorized_trends, total_count)

            self.log(f"成功生成世界级热点简报，包含 {total_count} 个话题")

            return {
                **state,
                "trends_digest": digest,
                "current_step": "trends_digest_completed"
            }
        except Exception as e:
            self.log(f"热点简报生成失败: {str(e)}", "ERROR")
            return {
                **state,
                "error_message": f"热点简报生成失败: {str(e)}",
                "current_step": "trends_digest_failed"
            }

    def _execute_legacy(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """兼容旧版逻辑"""
        # 导入旧版Agent
        from src.agents.trends_digest_agent import TrendsDigestAgent
        legacy_agent = TrendsDigestAgent(self.config, self.prompts)
        return legacy_agent.execute(state)

    def _generate_digest_v3(
        self,
        state: Dict[str, Any],
        categorized_trends: Dict[str, Dict],
        total_count: int
    ) -> Dict[str, Any]:
        """生成v3.0格式简报"""
        # 1. 生成元数据
        digest_metadata = self._generate_metadata_v3(total_count)

        # 2. 生成每个分类的导语
        category_intros = self._generate_category_intros_v3(categorized_trends)

        # 3. 生成趋势分析
        trend_analysis = None
        if self.include_analysis:
            trend_analysis = self._generate_trend_analysis_v3(categorized_trends)

        # 4. 提取关键洞察
        key_insights = self._extract_key_insights_v3(categorized_trends)

        # 5. 组装完整简报
        full_content = self._assemble_digest_v3(
            digest_metadata,
            categorized_trends,
            category_intros,
            trend_analysis,
            key_insights
        )

        word_count = len(full_content)

        return {
            "title": digest_metadata["title"],
            "subtitle": digest_metadata["subtitle"],
            "issue_number": digest_metadata["issue_number"],
            "publish_date": digest_metadata["publish_date"],
            "full_content": full_content,
            "category_intros": category_intros,
            "trend_analysis": trend_analysis,
            "key_insights": key_insights,
            "word_count": word_count,
            "reading_time": f"{word_count // 500}-{word_count // 300}分钟",
            "total_topics": total_count,
            "category_stats": digest_metadata["category_stats"],
            "style": self.digest_style,
            "version": "v3.0"
        }

    def _generate_metadata_v3(self, total_count: int) -> Dict[str, Any]:
        """生成简报元数据"""
        today = datetime.now()
        issue_number = today.strftime("%Y%m%d")

        return {
            "title": f"AI每日热点 · {today.strftime('%Y年%m月%d日')}",
            "subtitle": f"汇聚8个数据源，共获取{total_count}条AI资讯",
            "issue_number": issue_number,
            "publish_date": today.strftime("%Y-%m-%d"),
            "category_stats": {}
        }

    def _generate_category_intros_v3(self, categorized_trends: Dict[str, Dict]) -> Dict[str, str]:
        """为每个分类生成导语"""
        intros = {}
        for category in self.CATEGORY_ORDER:
            if category not in categorized_trends:
                continue

            category_data = categorized_trends[category]
            items = category_data.get("items", [])
            count = category_data.get("count", 0)

            if count == 0:
                continue

            intros[category] = self._generate_category_intro_v3(category, items)

        return intros

    def _generate_category_intro_v3(self, category: str, items: List[Dict]) -> str:
        """为单个分类生成导语"""
        try:
            # 获取前5个标题
            titles = [item.get("title", "") for item in items[:5]]
            titles_text = "\n".join([f"- {title}" for title in titles])

            category_desc = self.digest_prompts.get("category_descriptions", {}).get(category, "")

            template = """你是世界级科技媒体编辑。请为"{category}"分类撰写简短导语（30-50字）

分类描述：{category_description}

本分类部分热点：
{topics_list}

导语："""

            prompt = template.format(
                category=category,
                category_description=category_desc,
                topics_list=titles_text
            )

            response = self._call_llm(prompt)
            return response.strip()
        except Exception as e:
            self.log(f"分类导语生成失败: {e}", "WARNING")
            return category_desc or f"{category}精选"

    def _generate_trend_analysis_v3(self, categorized_trends: Dict[str, Dict]) -> str:
        """生成深度趋势分析"""
        try:
            # 提取所有热点标题（每个分类取前5个）
            all_titles = []
            for category in self.CATEGORY_ORDER:
                if category in categorized_trends:
                    items = categorized_trends[category].get("items", [])[:5]
                    all_titles.extend([item.get("title", "") for item in items])

            top_titles = all_titles[:15]

            # 统计数据源
            all_sources = []
            for category_data in categorized_trends.values():
                items = category_data.get("items", [])
                all_sources.extend([item.get("source", "") for item in items])

            from collections import Counter
            source_counts = Counter(all_sources)

            # 统计分类
            category_counts = {
                cat: data.get("count", 0)
                for cat, data in categorized_trends.items()
                if data.get("count", 0) > 0
            }

            template = """你是世界级科技媒体资深分析师。请基于以下AI技术热点，撰写深度趋势分析（250-350字）。

本期热点分类：
{category_distribution}

数据来源：
{sources}

部分热点标题：
{top_titles}

分析要求：
1. 深度洞察：透过现象看本质
2. 逻辑清晰：现象-原因-影响-展望
3. 数据支撑：引用具体分类和来源
4. 前瞻观点：提供行业展望

分析文章："""

            prompt = template.format(
                category_distribution=', '.join([f'{cat}: {count}条' for cat, count in category_counts.items()]),
                sources=', '.join([f'{src}: {cnt}条' for src, cnt in source_counts.most_common(5)]),
                top_titles='\n'.join([f'{i+1}. {title}' for i, title in enumerate(top_titles[:10])])
            )

            response = self._call_llm(prompt)
            return response.strip()
        except Exception as e:
            self.log(f"趋势分析生成失败: {e}", "WARNING")
            return None

    def _extract_key_insights_v3(self, categorized_trends: Dict[str, Dict]) -> List[str]:
        """提取关键洞察"""
        try:
            # 提取所有高热度热点
            all_items = []
            for category_data in categorized_trends.values():
                items = category_data.get("items", [])
                all_items.extend(items)

            # 按热度排序
            all_items.sort(key=lambda x: x.get("heat_score", 0), reverse=True)
            top_items = all_items[:10]

            titles = [item.get("title", "") for item in top_items]
            titles_text = "\n".join([f"{i+1}. {title}" for i, title in enumerate(titles)])

            template = """你是世界级科技媒体编辑。请从以下AI热点中提取3-5个关键洞察（每条20-30字）。

热点列表：
{topics_list}

洞察要求：
1. 洞察深刻，揭示行业本质
2. 观点鲜明，避免泛泛而谈
3. 语言精炼，每条20-30字
4. 使用"揭示了"、"标志着"等动词

关键洞察（每条一行）："""

            prompt = template.format(topics_list=titles_text)

            response = self._call_llm(prompt)
            insights = [line.strip() for line in response.strip().split('\n') if line.strip()]

            return insights[:5]
        except Exception as e:
            self.log(f"关键洞察提取失败: {e}", "WARNING")
            return []

    def _assemble_digest_v3(
        self,
        metadata: Dict[str, Any],
        categorized_trends: Dict[str, Dict],
        category_intros: Dict[str, str],
        trend_analysis: str = None,
        key_insights: List[str] = None
    ) -> str:
        """组装v3.0简报"""
        content_parts = []

        # ========== 头部 ==========
        content_parts.append(f"# {metadata['title']}\n\n")
        content_parts.append("> 📡 汇聚8个数据源的AI技术资讯，每天为你精选行业前沿\n\n")
        content_parts.append(f"**{metadata['subtitle']}**\n\n")
        content_parts.append(f"📅 {metadata['publish_date']}  ·  🆔 Issue #{metadata['issue_number']}\n\n")
        content_parts.append("---\n\n")

        # ========== 关键洞察 ==========
        if key_insights:
            content_parts.append("## 💡 核心洞察\n\n")
            for insight in key_insights:
                content_parts.append(f"- {insight}\n")
            content_parts.append("\n---\n\n")

        # ========== 趋势分析 ==========
        if trend_analysis:
            content_parts.append("## 📰 深度观察\n\n")
            content_parts.append(f"{trend_analysis}\n")
            content_parts.append("\n---\n\n")

        # ========== 分类热点 ==========
        content_parts.append("## 🔍 本期热点\n\n")

        # 按分类顺序组织
        for category in self.CATEGORY_ORDER:
            if category not in categorized_trends:
                continue

            category_data = categorized_trends[category]
            items = category_data.get("items", [])
            count = category_data.get("count", 0)

            if count == 0:
                continue

            content_parts.append(f"### {category} ({count}条)\n\n")

            # 分类导语
            if category in category_intros:
                content_parts.append(f"*{category_intros[category]}*\n\n")

            # 该分类下的所有热点
            for item in items:
                title = item.get("title", "")
                description = item.get("description", "")
                url = item.get("url", "")
                source = item.get("source", "")
                heat_score = item.get("heat_score", 0)

                # 热点详情
                content_parts.append(f"#### [{title}]({url})\n\n")

                # 来源和热度
                content_parts.append(f"**来源**: {source}  ·  **热度**: {heat_score}\n\n")

                # 描述（如果有）
                if description and len(description) > 20:
                    content_parts.append(f"{description}\n\n")

                content_parts.append("---\n\n")

        # ========== 页脚 ==========
        content_parts.append("\n## 📊 数据来源统计\n\n")

        # 统计每个数据源的数量
        source_stats = {}
        for category_data in categorized_trends.values():
            items = category_data.get("items", [])
            for item in items:
                source = item.get("source", "未知")
                source_stats[source] = source_stats.get(source, 0) + 1

        for source, count in sorted(source_stats.items(), key=lambda x: x[1], reverse=True):
            content_parts.append(f"- **{source}**: {count} 条\n")

        content_parts.append("\n---\n\n")
        content_parts.append("<div align='center'>\n\n")
        content_parts.append("**AI Daily** · 由 ContentForge AI 自动生成\n\n")
        content_parts.append(f"{metadata['publish_date']}\n\n")
        content_parts.append("</div>\n")

        return "".join(content_parts)
