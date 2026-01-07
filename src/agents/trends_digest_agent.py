"""
AI热点汇总Agent - 生成类似杂志/简报的汇总文章
将多个热点话题整合成一篇结构化的简报
"""

from typing import Dict, Any, List
import re
from datetime import datetime
from src.agents.base import BaseAgent


class TrendsDigestAgent(BaseAgent):
    """AI热点汇总Agent - 生成技术简报"""

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        super().__init__(config, prompts)
        digest_config = config.get("agents", {}).get("trends_digest", {})
        self.digest_style = digest_config.get("style", "professional")  # professional, casual, magazine
        self.include_analysis = digest_config.get("include_analysis", True)
        self.max_topics = digest_config.get("max_topics", 10)
        self.llm.temperature = 0.6  # 稍低温度，保证客观准确

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成热点汇总简报

        Args:
            state: 当前工作流状态

        Returns:
            Dict[str, Any]: 更新后的状态
        """
        self.log("开始生成AI热点汇总简报")

        try:
            # 获取所有热点话题
            hot_topics = state.get("ai_hot_topics", [])
            if not hot_topics:
                raise ValueError("没有找到AI热点话题列表")

            # 限制热点数量
            hot_topics = hot_topics[:self.max_topics]
            self.log(f"汇总 {len(hot_topics)} 个热点话题")

            # 生成简报
            digest = self._generate_digest(state, hot_topics)

            self.log(f"成功生成热点简报，包含 {len(hot_topics)} 个话题")

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
        """
        生成热点简报内容

        Args:
            state: 当前状态
            hot_topics: 热点话题列表

        Returns:
            Dict[str, Any]: 简报内容
        """
        # 1. 生成简报标题和概述
        digest_metadata = self._generate_metadata(hot_topics)

        # 2. 为每个热点生成简短摘要
        topic_summaries = []
        for idx, topic in enumerate(hot_topics, 1):
            summary = self._generate_topic_summary(topic, idx)
            topic_summaries.append(summary)

        # 3. 生成汇总分析（可选）
        summary_analysis = None
        if self.include_analysis:
            summary_analysis = self._generate_summary_analysis(hot_topics, state)

        # 4. 组装完整简报
        full_content = self._assemble_digest_content(
            digest_metadata,
            topic_summaries,
            summary_analysis
        )

        # 5. 统计信息
        word_count = len(full_content)
        reading_time = f"{word_count // 500}-{word_count // 300}分钟"

        return {
            "title": digest_metadata["title"],
            "subtitle": digest_metadata["subtitle"],
            "issue_number": digest_metadata["issue_number"],
            "publish_date": digest_metadata["publish_date"],
            "full_content": full_content,
            "topics": topic_summaries,
            "summary_analysis": summary_analysis,
            "word_count": word_count,
            "reading_time": reading_time,
            "total_topics": len(hot_topics),
            "sources": self._get_sources(hot_topics),
            "style": self.digest_style
        }

    def _generate_metadata(self, hot_topics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成简报元数据"""
        # 根据当前日期生成期号
        today = datetime.now()
        issue_number = today.strftime("%Y%m%d")

        # 分析主要数据源
        sources = self._get_sources(hot_topics)
        main_sources = ", ".join(list(sources.keys())[:5])

        return {
            "title": f"AI技术热点简报 - {today.strftime('%Y年%m月%d日')}",
            "subtitle": f"本期精选 {len(hot_topics)} 个AI技术热点，来源：{main_sources}",
            "issue_number": issue_number,
            "publish_date": today.strftime("%Y-%m-%d"),
            "sources": sources
        }

    def _generate_topic_summary(self, topic: Dict[str, Any], index: int) -> Dict[str, Any]:
        """
        为单个热点生成摘要

        Args:
            topic: 热点数据
            index: 索引编号

        Returns:
            Dict[str, Any]: 热点摘要
        """
        # 提取热点信息
        title = topic.get("title", "未知标题")
        description = topic.get("description", "")[:300]
        source = topic.get("source", "未知来源")
        url = topic.get("url", "")
        heat_score = topic.get("heat_score", 0)
        tags = topic.get("tags", [])
        timestamp = topic.get("timestamp", "")
        metrics = topic.get("metrics", {})

        # 生成热度描述
        heat_description = self._describe_heat_score(heat_score, metrics, source)

        # 生成简短摘要（如果描述太长，用LLM压缩）
        if len(description) > 200:
            description = self._summarize_description(title, description)

        return {
            "index": index,
            "title": title,
            "summary": description,
            "source": source,
            "url": url,
            "heat_score": heat_score,
            "heat_description": heat_description,
            "tags": tags,
            "timestamp": timestamp,
            "metrics": metrics
        }

    def _describe_heat_score(self, heat_score: int, metrics: Dict[str, Any], source: str) -> str:
        """生成热度描述"""
        if source == "Hacker News":
            upvotes = metrics.get("upvotes", 0)
            comments = metrics.get("comments", 0)
            return f"🔥 热度 {heat_score} (👍{upvotes} 💬{comments})"
        elif source == "arXiv":
            days_ago = metrics.get("days_ago", 0)
            return f"📚 学术论文 (📅{days_ago}天前发布)"
        elif source == "Hugging Face":
            likes = metrics.get("likes", 0)
            return f"🤗 模型热度 (👍{likes} likes)"
        elif source == "GitHub Trending":
            stars = metrics.get("stars", "0")
            return f"⭐ GitHub热门 ({stars} stars)"
        elif source == "Stack Overflow":
            score = metrics.get("score", 0)
            answers = metrics.get("answers", 0)
            return f"❓ 技术问答 (📊{score}分 💡{answers}个回答)"
        else:
            return f"🔥 热度评分: {heat_score}"

    def _summarize_description(self, title: str, description: str) -> str:
        """使用LLM压缩描述"""
        try:
            prompt = f"""请将以下技术描述压缩为1-2句话的摘要（50-80字）：

标题：{title}

原始描述：
{description}

要求：
- 保留核心信息
- 语言简洁明了
- 适合快速阅读
- 不要丢失关键细节

摘要："""

            response = self._call_llm(prompt)
            return response.strip()
        except:
            # 如果LLM调用失败，返回截断的描述
            return description[:150] + "..."

    def _generate_summary_analysis(self, hot_topics: List[Dict[str, Any]], state: Dict[str, Any]) -> str:
        """
        生成汇总分析

        Args:
            hot_topics: 热点列表
            state: 当前状态

        Returns:
            str: 分析内容
        """
        try:
            # 提取所有标题和标签
            titles = [t["title"] for t in hot_topics[:5]]
            all_tags = []
            for topic in hot_topics:
                all_tags.extend(topic.get("tags", []))

            # 统计热门标签
            from collections import Counter
            top_tags = Counter(all_tags).most_common(10)

            # 构建提示词
            prompt = f"""基于以下AI技术热点，撰写一篇简短的汇总分析（200-300字）：

本期热点标题：
{chr(10).join([f'{i+1}. {title}' for i, title in enumerate(titles)])}

主要技术领域：
{', '.join([tag for tag, _ in top_tags[:5]])}

请从以下角度分析：
1. 本期热点的主要技术趋势
2. 值得关注的技术方向
3. 对行业的影响

要求：
- 客观准确
- 简洁有力
- 适合简报刊登
"""

            analysis = self._call_llm(prompt)
            return analysis.strip()
        except Exception as e:
            self.log(f"生成汇总分析失败: {e}", "WARNING")
            return None

    def _assemble_digest_content(
        self,
        metadata: Dict[str, Any],
        topic_summaries: List[Dict[str, Any]],
        summary_analysis: str = None
    ) -> str:
        """
        组装完整的简报内容

        Args:
            metadata: 简报元数据
            topic_summaries: 话题摘要列表
            summary_analysis: 汇总分析

        Returns:
            str: 完整的Markdown内容
        """
        # 根据风格选择模板
        if self.digest_style == "magazine":
            return self._assemble_magazine_style(metadata, topic_summaries, summary_analysis)
        elif self.digest_style == "casual":
            return self._assemble_casual_style(metadata, topic_summaries, summary_analysis)
        else:  # professional
            return self._assemble_professional_style(metadata, topic_summaries, summary_analysis)

    def _assemble_professional_style(
        self,
        metadata: Dict[str, Any],
        topic_summaries: List[Dict[str, Any]],
        summary_analysis: str = None
    ) -> str:
        """专业风格简报"""
        content_parts = []

        # 标题
        content_parts.append(f"# {metadata['title']}\n")
        content_parts.append(f"{metadata['subtitle']}\n")
        content_parts.append(f"**发布日期**: {metadata['publish_date']}  |  **期号**: #{metadata['issue_number']}\n")

        # 汇总分析（如果有）
        if summary_analysis:
            content_parts.append("\n## 📊 本期趋势分析\n")
            content_parts.append(f"{summary_analysis}\n")

        # 热点列表
        content_parts.append("\n## 🔥 本期热点详情\n")

        for topic in topic_summaries:
            content_parts.append(f"\n### {topic['index']}. {topic['title']}\n")
            content_parts.append(f"**来源**: {topic['source']}  |  {topic['heat_description']}\n")
            content_parts.append(f"**摘要**: {topic['summary']}\n")

            if topic['url']:
                content_parts.append(f"**原文链接**: [{topic['url']}]({topic['url']})\n")

            if topic['tags']:
                tags_str = " ".join([f"#{tag}" for tag in topic['tags'][:5]])
                content_parts.append(f"**标签**: {tags_str}\n")

        # 数据源统计
        sources = metadata['sources']
        content_parts.append("\n---\n")
        content_parts.append("\n## 📈 数据源统计\n")
        for source, count in sources.items():
            content_parts.append(f"- **{source}**: {count} 条热点\n")

        # 页脚
        content_parts.append("\n---\n")
        content_parts.append(f"\n*本简报由 ContentForge AI 自动生成 | 数据来源: {', '.join(sources.keys())}*\n")

        return "\n".join(content_parts)

    def _assemble_magazine_style(
        self,
        metadata: Dict[str, Any],
        topic_summaries: List[Dict[str, Any]],
        summary_analysis: str = None
    ) -> str:
        """杂志风格简报"""
        content_parts = []

        # 大标题
        content_parts.append(f"# {metadata['title']}\n")
        content_parts.append(f"```{metadata['subtitle']}```\n")
        content_parts.append(f"📅 {metadata['publish_date']}  |  🆔 {metadata['issue_number']}\n")

        # 本期导读
        content_parts.append("\n## ✨ 本期导读\n")
        for topic in topic_summaries[:5]:
            content_parts.append(f"- **{topic['title']}** ({topic['source']})\n")

        # 汇总分析
        if summary_analysis:
            content_parts.append("\n## 📰 趋势观察\n")
            content_parts.append(f"{summary_analysis}\n")

        # 热点详情
        content_parts.append("\n## 🔥 热点解读\n")

        for topic in topic_summaries:
            content_parts.append(f"\n### 📌 {topic['index']}. {topic['title']}\n")

            # 元信息框
            content_parts.append(f"> 💡 **{topic['source']}**  |  {topic['heat_description']}\n")

            content_parts.append(f"\n{topic['summary']}\n")

            if topic['url']:
                content_parts.append(f"\n🔗 **[阅读原文]({topic['url']})**\n")

            if topic['tags']:
                tags_str = " ".join([f"`{tag}`" for tag in topic['tags'][:5]])
                content_parts.append(f"\n🏷️ {tags_str}\n")

        # 数据源
        sources = metadata['sources']
        content_parts.append("\n---\n")
        content_parts.append("\n## 📊 本期数据来源\n")
        for source, count in sources.items():
            content_parts.append(f"`{source}`: **{count}** 条  ")

        content_parts.append("\n\n---\n")
        content_parts.append(f"\n<div align='center'>\n\n**ContentForge AI** · 自动生成 · {metadata['publish_date']}\n\n</div>\n")

        return "\n".join(content_parts)

    def _assemble_casual_style(
        self,
        metadata: Dict[str, Any],
        topic_summaries: List[Dict[str, Any]],
        summary_analysis: str = None
    ) -> str:
        """轻松风格简报"""
        content_parts = []

        # 标题
        content_parts.append(f"# 🤖 {metadata['title']}\n")
        content_parts.append(f"{metadata['subtitle']}\n")
        content_parts.append(f"📅 {metadata['publish_date']}\n")

        # 汇总分析
        if summary_analysis:
            content_parts.append(f"\n## 🎯 一句话总结\n{summary_analysis}\n")

        # 热点列表
        content_parts.append("\n## 🔥 今天的热点\n")

        for topic in topic_summaries:
            content_parts.append(f"\n### {topic['index']}. {topic['title']}\n\n")

            content_parts.append(f"{topic['summary']}\n\n")

            content_parts.append(f"📍 {topic['source']} · {topic['heat_description']}\n")

            if topic['url']:
                content_parts.append(f"🔗 [原文]({topic['url']})\n")

        # 数据源
        sources = metadata['sources']
        content_parts.append("\n---\n")
        content_parts.append(f"\n📊 数据来源: {', '.join(sources.keys())}\n")
        content_parts.append(f"\n---\n\n✨ 由 ContentForge AI 自动生成\n")

        return "\n".join(content_parts)

    def _get_sources(self, hot_topics: List[Dict[str, Any]]) -> Dict[str, int]:
        """统计数据源"""
        from collections import Counter
        sources = [topic.get("source", "未知") for topic in hot_topics]
        return dict(Counter(sources))
