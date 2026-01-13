"""
AI热点汇总Agent - 世界级科技简报生成（标准固化版）
使用配置化的 Prompt 模板，确保每期简报都达到世界级水准
"""

from typing import Dict, Any, List
import yaml
from pathlib import Path
from datetime import datetime
from src.agents.base import BaseAgent


class TrendsDigestAgent(BaseAgent):
    """AI热点汇总Agent - 世界级技术简报（标准固化）"""

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        super().__init__(config, prompts)

        digest_config = config.get("agents", {}).get("trends_digest", {})
        self.digest_style = digest_config.get("style", "magazine")
        self.include_analysis = digest_config.get("include_analysis", True)
        self.max_topics = digest_config.get("max_topics", 20)
        self.llm.temperature = 0.7  # 稍高温度，增加创造力

        # 加载简报 Prompt 模板
        self.digest_prompts = self._load_digest_prompts()

        self.log("世界级简报标准已加载，使用配置化 Prompt 模板")

    def _load_digest_prompts(self) -> Dict[str, Any]:
        """加载简报 Prompt 配置（从统一的 prompts.yaml）"""
        try:
            # 使用现有的 prompts 参数，它已经加载了 prompts.yaml
            # 如果没有传入，则直接读取文件
            if self.prompts and "trends_digest" in self.prompts:
                return self.prompts["trends_digest"]

            # fallback：直接读取文件
            config_path = Path(__file__).parent.parent.parent / "config" / "prompts.yaml"
            with open(config_path, 'r', encoding='utf-8') as f:
                prompts = yaml.safe_load(f)
                return prompts.get("trends_digest", {})
        except Exception as e:
            self.log(f"加载 trends_digest 配置失败: {e}，使用默认模板", "WARNING")
            return self._get_default_prompts()

    def _get_default_prompts(self) -> Dict[str, Any]:
        """获取默认 Prompt 模板（fallback）"""
        return {
            "category_descriptions": {
                "📈 行业动态": "聚焦AI产业投资、并购、战略合作等商业动态",
                "🎓 学术突破": "精选顶级期刊论文与前沿研究成果",
                "🔬 技术创新": "追踪模型架构、算法突破与工程创新",
                "🛠️ AI工具/产品": "发现提升开发效率的实用工具与产品",
                "💼 AI应用": "展示AI在各行业的创新应用案例"
            },
            "formatting": {
                "category_order": [
                    "📈 行业动态",
                    "🎓 学术突破",
                    "🔬 技术创新",
                    "🛠️ AI工具/产品",
                    "💼 AI应用"
                ]
            }
        }

    @property
    def CATEGORY_ORDER(self) -> List[str]:
        """获取分类顺序"""
        return self.digest_prompts.get("formatting", {}).get("category_order", [
            "📈 行业动态",
            "🎓 学术突破",
            "🔬 技术创新",
            "🛠️ AI工具/产品",
            "💼 AI应用"
        ])

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """生成热点汇总简报"""
        if state.get("selected_ai_topic", {}).get("source") == "user_provided":
            self.log("检测到用户指定话题模式，跳过热点汇总")
            return state

        self.log("开始生成世界级AI热点简报（标准固化版）")

        try:
            hot_topics = state.get("ai_hot_topics", [])
            if not hot_topics:
                raise ValueError("没有找到AI热点话题列表")

            hot_topics = hot_topics[:self.max_topics]
            self.log(f"汇总 {len(hot_topics)} 个热点话题")

            digest = self._generate_digest(state, hot_topics)

            self.log(f"成功生成世界级热点简报，包含 {len(hot_topics)} 个话题")

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

    def _generate_digest(self, state: Dict[str, Any], hot_topics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成热点简报内容（使用固化标准）"""
        # 1. 生成元数据
        digest_metadata = self._generate_metadata(hot_topics)

        # 2. 按分类分组
        grouped_topics = self._group_topics_by_category(hot_topics)

        # 3. 为每个话题生成高质量摘要
        enriched_summaries = self._generate_topic_summaries(hot_topics)

        # 4. 生成每个分类的导语
        category_intros = self._generate_category_intros(grouped_topics)

        # 5. 生成趋势分析
        trend_analysis = None
        if self.include_analysis:
            trend_analysis = self._generate_trend_analysis(hot_topics)

        # 6. 提取关键洞察
        key_insights = self._extract_key_insights(hot_topics)

        # 7. 组装完整简报
        full_content = self._assemble_world_class_digest(
            digest_metadata,
            grouped_topics,
            enriched_summaries,
            category_intros,
            trend_analysis,
            key_insights
        )

        word_count = len(full_content)
        reading_time = f"{word_count // 500}-{word_count // 300}分钟"

        return {
            "title": digest_metadata["title"],
            "subtitle": digest_metadata["subtitle"],
            "issue_number": digest_metadata["issue_number"],
            "publish_date": digest_metadata["publish_date"],
            "full_content": full_content,
            "topics": enriched_summaries,
            "category_intros": category_intros,
            "trend_analysis": trend_analysis,
            "key_insights": key_insights,
            "word_count": word_count,
            "reading_time": reading_time,
            "total_topics": len(hot_topics),
            "sources": self._get_sources(hot_topics),
            "category_stats": digest_metadata["category_stats"],
            "style": self.digest_style
        }

    def _generate_metadata(self, hot_topics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成简报元数据"""
        today = datetime.now()
        issue_number = today.strftime("%Y%m%d")

        sources = self._get_sources(hot_topics)
        category_stats = self._get_category_statistics(hot_topics)
        category_summary = self._get_category_summary_text(category_stats)

        # 使用配置文件中的格式模板
        formatting = self.digest_prompts.get("formatting", {})
        title_template = formatting.get("digest_title", "AI Daily · {date}")
        subtitle_template = formatting.get("digest_subtitle", "本期精选 {count} 个前沿动态 | {category_summary}")

        title = title_template.format(date=today.strftime('%Y年%m月%d日'))
        subtitle = subtitle_template.format(
            count=len(hot_topics),
            category_summary=category_summary
        )

        return {
            "title": title,
            "subtitle": subtitle,
            "issue_number": issue_number,
            "publish_date": today.strftime("%Y-%m-%d"),
            "sources": sources,
            "category_stats": category_stats
        }

    def _generate_topic_summaries(self, hot_topics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """为每个话题生成高质量摘要（使用配置化 Prompt）"""
        summaries = []
        for idx, topic in enumerate(hot_topics, 1):
            summary = self._generate_enriched_summary(topic, idx)
            summaries.append(summary)
            self.log(f"生成摘要 {idx}/{len(hot_topics)}: {topic['title'][:30]}...")
        return summaries

    def _generate_enriched_summary(self, topic: Dict[str, Any], index: int) -> Dict[str, Any]:
        """为单个话题生成丰富摘要"""
        title = topic.get("title", "未知标题")
        description = topic.get("description", "")
        url = topic.get("url", "")
        source = topic.get("source", "未知来源")
        category = topic.get("category", "未分类")
        heat_score = topic.get("heat_score", 0)
        metrics = topic.get("metrics", {})

        # 使用LLM生成高质量摘要
        enriched_summary = self._generate_llm_summary(title, description, category)

        return {
            "index": index,
            "title": title,
            "summary": enriched_summary,
            "source": source,
            "url": url,
            "category": category,
            "heat_score": heat_score,
            "metrics": metrics
        }

    def _generate_llm_summary(self, title: str, description: str, category: str) -> str:
        """使用LLM生成高质量摘要（使用配置化 Prompt）"""
        try:
            # 使用配置文件中的模板
            template = self.digest_prompts.get("summary_template",
                "你是一位世界级科技媒体编辑。请为以下AI技术新闻撰写摘要（80-120字）：\n\n标题：{title}\n描述：{description}\n分类：{category}\n\n摘要：")

            prompt = template.format(
                title=title,
                description=description,
                category=category
            )

            response = self._call_llm(prompt)
            return response.strip()
        except Exception as e:
            self.log(f"LLM摘要生成失败: {e}", "WARNING")
            return description[:150] + "..."

    def _generate_category_intros(self, grouped_topics: Dict[str, List[Dict[str, Any]]]) -> Dict[str, str]:
        """为每个分类生成导语（使用配置化 Prompt）"""
        intros = {}
        for category in self.CATEGORY_ORDER:
            if category not in grouped_topics or not grouped_topics[category]:
                continue

            topics = grouped_topics[category]
            intro = self._generate_category_intro(category, topics)
            intros[category] = intro

        return intros

    def _generate_category_intro(self, category: str, topics: List[Dict[str, Any]]) -> str:
        """为单个分类生成导语（使用配置化 Prompt）"""
        try:
            titles = [t["title"] for t in topics[:5]]
            titles_text = "\n".join([f"- {title}" for title in titles])

            # 使用配置文件中的模板
            template = self.digest_prompts.get("category_intro_template",
                '你是世界级科技媒体编辑。请为"{category}"分类撰写导语（50-80字）\n\n本分类热点：\n{topics_list}\n\n导语：')

            # 获取增强描述
            enhanced_desc = self.digest_prompts.get("category_enhanced_descriptions", {}).get(category,
                self.digest_prompts.get("category_descriptions", {}).get(category, ""))

            prompt = template.format(
                category=category,
                category_description=enhanced_desc,
                topics_list=titles_text
            )

            response = self._call_llm(prompt)
            return response.strip()
        except Exception as e:
            self.log(f"分类导语生成失败: {e}", "WARNING")
            # 使用增强描述作为 fallback
            return self.digest_prompts.get("category_enhanced_descriptions", {}).get(category,
                self.digest_prompts.get("category_descriptions", {}).get(category, f"{category}"))

    def _generate_trend_analysis(self, hot_topics: List[Dict[str, Any]]) -> str:
        """生成深度趋势分析（使用配置化 Prompt）"""
        try:
            # 提取关键信息
            top_titles = [t["title"] for t in hot_topics[:10]]
            all_tags = []
            for topic in hot_topics:
                all_tags.extend(topic.get("tags", []))

            from collections import Counter
            top_tags = Counter(all_tags).most_common(10)

            # 按分类统计
            category_counts = {}
            for topic in hot_topics:
                cat = topic.get("category", "未分类")
                category_counts[cat] = category_counts.get(cat, 0) + 1

            # 使用配置文件中的模板
            template = self.digest_prompts.get("trend_analysis_template",
                """你是世界级科技媒体资深分析师。请基于以下AI技术热点，撰写深度趋势分析（300-400字）。

本期TOP热点：
{top_titles}

热门技术领域：
{top_tags}

分类分布：
{category_distribution}

分析要求：
1. 深度洞察：透过现象看本质
2. 逻辑清晰：现象-原因-影响-展望
3. 数据支撑：引用具体热点
4. 前瞻观点：提供行业展望

分析文章：""")

            prompt = template.format(
                top_titles="\n".join([f'{i+1}. {title}' for i, title in enumerate(top_titles)]),
                top_tags=', '.join([tag for tag, _ in top_tags[:8]]),
                category_distribution=', '.join([f'{cat}: {count}个' for cat, count in category_counts.items()])
            )

            response = self._call_llm(prompt)
            return response.strip()
        except Exception as e:
            self.log(f"趋势分析生成失败: {e}", "WARNING")
            return None

    def _extract_key_insights(self, hot_topics: List[Dict[str, Any]]) -> List[str]:
        """提取关键洞察（使用配置化 Prompt）"""
        try:
            # 选择高热度话题
            top_topics = sorted(hot_topics, key=lambda x: x.get("heat_score", 0), reverse=True)[:8]

            titles = [t["title"] for t in top_topics]
            titles_text = "\n".join([f"{i+1}. {title}" for i, title in enumerate(titles)])

            # 使用配置文件中的模板
            template = self.digest_prompts.get("key_insights_template",
                """你是世界级科技媒体编辑。请从以下AI热点中提取3-5个关键洞察（每条20-30字）。

热点列表：
{topics_list}

洞察要求：
1. 洞察深刻，揭示行业本质
2. 观点鲜明，避免泛泛而谈
3. 语言精炼，每条20-30字
4. 使用"揭示了"、"标志着"等动词

关键洞察（每条一行）：""")

            prompt = template.format(topics_list=titles_text)

            response = self._call_llm(prompt)
            insights = [line.strip() for line in response.strip().split('\n') if line.strip()]

            # 应用质量标准
            quality = self.digest_prompts.get("quality_standards", {}).get("key_insights", {})
            max_count = quality.get("max_count", 5)

            return insights[:max_count]
        except Exception as e:
            self.log(f"关键洞察提取失败: {e}", "WARNING")
            return []

    def _assemble_world_class_digest(
        self,
        metadata: Dict[str, Any],
        grouped_topics: Dict[str, List[Dict[str, Any]]],
        summaries: List[Dict[str, Any]],
        category_intros: Dict[str, str],
        trend_analysis: str = None,
        key_insights: List[str] = None
    ) -> str:
        """组装世界级简报"""
        content_parts = []

        # ========== 头部 ==========
        content_parts.append(f"# {metadata['title']}\n\n")
        content_parts.append("<div align='center'>\n\n")
        content_parts.append(f"**{metadata['subtitle']}**\n\n")
        content_parts.append(f"📅 {metadata['publish_date']}  ·  🆔 Issue #{metadata['issue_number']}\n\n")
        content_parts.append("</div>\n\n")
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
            if category not in grouped_topics:
                continue

            topics = grouped_topics[category]
            content_parts.append(f"### {category}\n\n")

            # 分类导语
            if category in category_intros:
                content_parts.append(f"*{category_intros[category]}*\n\n")

            # 该分类下的热点
            for topic in topics:
                # 找到对应的summary
                summary = next((s for s in summaries if s["title"] == topic["title"]), None)
                if not summary:
                    continue

                # 热点详情
                content_parts.append(f"#### {summary['title']}\n\n")

                # 来源和热度
                source_badge = self._format_source_badge(summary)
                content_parts.append(f"{source_badge}\n\n")

                # 摘要
                content_parts.append(f"{summary['summary']}\n\n")

                # 链接
                if summary.get('url'):
                    content_parts.append(f"🔗 **[阅读原文]({summary['url']})**\n\n")

                content_parts.append("---\n\n")

        # ========== 页脚 ==========
        content_parts.append("\n## 📊 数据来源\n\n")
        sources = metadata['sources']
        for source, count in sources.items():
            content_parts.append(f"- **{source}**: {count} 条\n")

        content_parts.append("\n---\n\n")
        content_parts.append("<div align='center'>\n\n")
        content_parts.append("**AI Daily** · 由 ContentForge AI 自动生成\n\n")
        content_parts.append(f"{metadata['publish_date']}\n\n")
        content_parts.append("</div>\n")

        return "".join(content_parts)

    def _format_source_badge(self, summary: Dict[str, Any]) -> str:
        """格式化来源徽章（使用配置化样式）"""
        source = summary.get('source', '')
        metrics = summary.get('metrics', {})

        # 使用配置文件中的样式
        source_badges = self.digest_prompts.get("formatting", {}).get("source_badges", {})
        badge_config = source_badges.get(source, {})

        color = badge_config.get("color", "#6c757d")
        icon = badge_config.get("icon", "📄")

        if source == "Hacker News":
            upvotes = metrics.get('upvotes', 0)
            comments = metrics.get('comments', 0)
            return f"<span style='background: {color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px;'>{source}</span> {icon} {upvotes} upvotes · 💬 {comments} comments"
        elif source == "arXiv":
            return f"<span style='background: {color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px;'>{source}</span> {icon} 学术论文"
        else:
            return f"<span style='background: {color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px;'>{source}</span>"

    def _group_topics_by_category(self, hot_topics: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """按分类分组话题"""
        grouped = {category: [] for category in self.CATEGORY_ORDER}
        grouped["未分类"] = []

        for topic in hot_topics:
            category = topic.get("category", "未分类")
            if category in grouped:
                grouped[category].append(topic)
            else:
                grouped["未分类"].append(topic)

        if not grouped["未分类"]:
            del grouped["未分类"]

        return grouped

    def _get_category_statistics(self, hot_topics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """获取分类统计信息"""
        from collections import Counter

        categories = [topic.get("category", "未分类") for topic in hot_topics]
        category_counts = dict(Counter(categories))

        total = len(hot_topics)
        category_stats = {}
        for category, count in category_counts.items():
            percentage = (count / total * 100) if total > 0 else 0
            category_stats[category] = {
                "count": count,
                "percentage": f"{percentage:.1f}%"
            }

        return category_stats

    def _get_category_summary_text(self, category_stats: Dict[str, Any]) -> str:
        """生成分类统计摘要文本"""
        summary_parts = []
        for category in self.CATEGORY_ORDER:
            if category in category_stats:
                stats = category_stats[category]
                summary_parts.append(f"{category} {stats['count']}个")

        return " · ".join(summary_parts)

    def _get_sources(self, hot_topics: List[Dict[str, Any]]) -> Dict[str, int]:
        """统计数据源"""
        from collections import Counter
        sources = [topic.get("source", "未知") for topic in hot_topics]
        return dict(Counter(sources))
