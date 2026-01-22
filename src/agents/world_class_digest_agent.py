"""
World Class AI News Digest Generator
顶级AI新闻简报生成器 - 参考The Verge/TechCrunch/Wired设计风格
"""

from datetime import datetime
from typing import Dict, Any, List, Tuple
from loguru import logger
import json
from langchain_openai import ChatOpenAI


class WorldClassDigestAgent:
    """世界顶级AI新闻简报生成器 v6.0 - 全中文LLM生成"""

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        self.config = config
        self.prompts = prompts
        self.name = "world_class_digest"
        self.llm = self._init_llm()

        # 翻译配置（默认启用）
        agent_config = config.get("agents", {}).get("world_class_digest", {})
        self.translate_enabled = agent_config.get("translate_enabled", True)
        self.batch_size = agent_config.get("batch_size", 5)  # 批量处理大小
        self.max_items_per_category = agent_config.get("max_items_per_category", 15)  # 每个分类最多显示数量

        self.log(f"v6.0初始化完成，翻译功能: {'启用' if self.translate_enabled else '禁用'}")

    def _init_llm(self):
        """初始化LLM用于翻译"""
        try:
            import os
            from pathlib import Path
            from dotenv import load_dotenv

            # 显式加载.env文件（从项目根目录）
            project_root = Path(__file__).parent.parent.parent
            env_file = project_root / ".env"
            if env_file.exists():
                load_dotenv(env_file)
                self.log(f"已加载环境变量文件: {env_file}")

            llm_config = self.config.get("llm", {})
            provider = llm_config.get("provider", "zhipuai")

            if provider == "zhipuai":
                # 优先从环境变量获取API key
                api_key = os.getenv("ZHIPUAI_API_KEY")
                if not api_key:
                    # 尝试从config的api_keys获取
                    api_key = self.config.get("api_keys", {}).get("zhipuai")

                if not api_key:
                    self.log("未配置ZHIPUAI_API_KEY，翻译功能将不可用", "WARNING")
                    return None

                zhipu_config = llm_config.get("zhipuai", {})
                return ChatOpenAI(
                    model=zhipu_config.get("model", "glm-4-flash"),
                    openai_api_key=api_key,
                    base_url=zhipu_config.get("base_url", "https://open.bigmodel.cn/api/coding/paas/v4/"),
                    temperature=zhipu_config.get("temperature", 0.7),
                    max_tokens=zhipu_config.get("max_tokens", 8000),
                    timeout=zhipu_config.get("timeout", 600)
                )
            else:
                return None
        except Exception as e:
            self.log(f"LLM初始化失败: {e}", "WARNING")
            return None

    def log(self, message: str, level: str = "INFO"):
        logger.log(level, f"[WorldClassDigestAgent] {message}")

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """执行简报生成"""
        try:
            self.log("开始生成世界顶级AI新闻简报...")

            categorized_trends = state.get("categorized_trends", {})
            source_status = state.get("source_status", {})

            # 生成简报
            digest = self._generate_world_class_digest(
                categorized_trends,
                source_status
            )

            return {
                **state,
                "news_digest": digest
            }

        except Exception as e:
            self.log(f"简报生成失败: {e}", "ERROR")
            return {
                **state,
                "error_message": str(e),
                "current_step": "world_class_digest_failed"
            }

    def _generate_world_class_digest(
        self,
        categorized_trends: Dict[str, Dict],
        source_status: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成世界顶级AI新闻简报"""

        today = datetime.now()
        issue_number = today.strftime("%Y%m%d")

        # 计算总数（修复：从items中获取）
        total_count = 0
        for category_data in categorized_trends.values():
            if isinstance(category_data, dict) and "items" in category_data:
                total_count += len(category_data["items"])
            elif isinstance(category_data, list):
                total_count += len(category_data)

        self.log(f"生成简报: {total_count}条热点")

        # 生成中文简报
        chinese_content = self._generate_chinese_digest_v2(
            categorized_trends,
            source_status,
            today,
            issue_number,
            total_count
        )

        word_count = len(chinese_content)

        return {
            "title": f"AI Daily · {today.strftime('%Y年%m月%d日')}",
            "subtitle": f"今日{total_count}条前沿动态",
            "issue_number": issue_number,
            "publish_date": today.strftime("%Y-%m-%d"),
            "full_content": chinese_content,
            "word_count": word_count,
            "reading_time": f"{word_count // 500}分钟",
            "total_topics": total_count,
            "source_status": source_status,
            "version": "v5.0"
        }

    def _generate_chinese_digest_v2(
        self,
        categorized_trends: Dict[str, Dict],
        source_status: Dict[str, Any],
        today: datetime,
        issue_number: str,
        total_count: int
    ) -> str:
        """生成中文简报 v6.0 - 顶级科技媒体风格，全中文LLM生成"""

        parts = []

        # ========== Header ==========
        parts.append("# " + "━" * 50 + "\n")
        parts.append(f"# 🔥 AI Daily · {today.strftime('%Y年%m月%d日')}\n")
        parts.append("# " + "━" * 50 + "\n\n")
        parts.append(f"### 🌐 汇聚全球顶尖AI资讯  |  📊 今日 **{total_count}** 条更新\n\n")
        parts.append(f"**📅 {today.strftime('%Y年%m月%d日')}**  ·  **Issue #{issue_number}**\n\n")
        parts.append("---\n\n")

        # ========== 核心头条 ==========
        parts.append("## ⭐ 核心头条\n\n")

        # 获取最高热度的新闻（跨所有分类）
        top_stories = self._get_top_stories(categorized_trends, limit=5)

        # 批量生成核心头条的中文摘要
        if self.translate_enabled and self.llm:
            top_stories = self._batch_generate_summaries(top_stories)

        for i, story in enumerate(top_stories, 1):
            title_cn = story.get("title_cn", story.get("title", ""))
            summary_cn = story.get("summary_cn", story.get("description", ""))[:150]
            source = story.get("source", "").split("(")[0].strip()
            url = story.get("url", "")

            parts.append(f"### {i}. {title_cn}\n\n")
            parts.append(f"> 📰 {source}  |  🔗 [原文链接]({url})\n\n")
            parts.append(f"{summary_cn}\n\n")
            parts.append("---\n\n")

        # ========== 分类资讯 ==========
        parts.append("## 📂 分类资讯\n\n")

        category_icons = {
            "📈 产业动态": "💼",
            "🎓 学术前沿": "🔬",
            "🔧 技术创新": "⚡",
            "🛠️ AI工具/产品": "🚀",
            "💼 AI应用": "🌐"
        }

        for category_key, category_data in categorized_trends.items():
            # 获取实际的条目列表
            if isinstance(category_data, dict) and "items" in category_data:
                trends = category_data["items"]
            else:
                trends = category_data

            if not trends or not isinstance(trends, list):
                continue

            icon = category_icons.get(category_key, "📌")
            category_name = self._get_category_name(category_key)

            # 限制每个分类显示的数量
            display_trends = trends[:self.max_items_per_category]
            parts.append(f"### {icon} {category_name} ({len(display_trends)}条，共{len(trends)}条)\n\n")

            # 批量生成当前分类的中文摘要
            if self.translate_enabled and self.llm:
                display_trends = self._batch_generate_summaries(display_trends)

            for i, trend in enumerate(display_trends, 1):
                title_cn = trend.get("title_cn", trend.get("title", ""))
                summary_cn = trend.get("summary_cn", trend.get("description", ""))[:150]
                source = trend.get("source", "").split("(")[0].strip()
                url = trend.get("url", "")

                parts.append(f"**{i}. {title_cn}**\n\n")
                parts.append(f"{summary_cn}\n\n")
                parts.append(f"📎 {source} | [阅读更多]({url})\n\n")

        # ========== 数据来源 ==========
        parts.append("---\n\n")
        parts.append("## 📊 数据来源\n\n")

        success_sources = [s for s, status in source_status.items() if status.get("success", False)]
        total_sources = len(source_status)

        parts.append(f"**数据获取成功率**: {len(success_sources)}/{total_sources} ({len(success_sources)*100//total_sources}%)\n\n")

        if success_sources:
            parts.append("**✅ 成功获取的数据源:**\n\n")
            for source in success_sources:
                count = source_status[source].get("count", 0)
                parts.append(f"- **{source}**: {count}条\n")
            parts.append("\n")

        failed_sources = [s for s, status in source_status.items() if not status.get("success", False)]
        if failed_sources:
            parts.append("**❌ 暂时不可用的数据源:**\n\n")
            for source in failed_sources:
                parts.append(f"- **{source}**: {source_status[source].get('message', '未知错误')}\n")
            parts.append("\n")

        # ========== Footer ==========
        parts.append("---\n\n")
        parts.append("<div align='center'>\n\n")
        parts.append("### 🤖 由 ContentForge AI 自动生成\n\n")
        parts.append(f"{today.strftime('%Y年%m月%d日')}\n\n")
        parts.append("**数据来源**: TechCrunch AI · MIT Technology Review · OpenAI Blog · NewsAPI · arXiv · Hacker News\n\n")
        parts.append("</div>\n")

        return "".join(parts)

    def _get_top_stories(self, categorized_trends: Dict[str, Dict], limit: int = 5) -> List[Dict]:
        """获取最热门的新闻"""
        all_trends = []
        for category, category_data in categorized_trends.items():
            if isinstance(category_data, dict) and "items" in category_data:
                all_trends.extend(category_data["items"])
            else:
                # 兼容旧格式
                all_trends.extend(category_data)

        # 按热度分数排序
        all_trends.sort(key=lambda x: x.get("heat_score", 0), reverse=True)
        return all_trends[:limit]

    def _batch_generate_summaries(self, trends: List[Dict]) -> List[Dict]:
        """批量生成新闻的中文摘要

        Args:
            trends: 新闻列表，每条包含 title, description, url, source

        Returns:
            处理后的新闻列表，包含中文标题和摘要
        """
        if not trends or not self.translate_enabled:
            return trends

        # 如果已经是中文内容，直接返回
        first_item = trends[0] if trends else {}
        if first_item.get("title") and any('\u4e00' <= c <= '\u9fff' for c in first_item["title"]):
            return trends

        # 构建批量翻译提示
        news_items = []
        for i, item in enumerate(trends):
            title = item.get("title", "").replace('&amp;', '&').replace('&quot;', '"')
            desc = item.get("description", "").replace('&amp;', '&').replace('&quot;', '"')
            desc = desc.replace('<p>', '').replace('</p>', '').replace('<br>', ' ')[:200]  # 限制长度
            news_items.append(f"{i+1}. 标题: {title}\n   摘要: {desc}")

        prompt = f"""你是一位顶级科技媒体编辑（如TechCrunch、The Verge）。请将以下AI新闻翻译并精简成专业的中文简报。

要求：
1. 标题翻译要简洁有力，符合科技媒体风格
2. 摘要要精炼，控制在50字以内，突出核心信息
3. 保持专业术语准确性（如LLM、RAG、Transformer等）
4. 直接输出JSON格式，不要有任何额外说明

输出格式示例：
{{
  "1": {{"title": "中文标题", "summary": "中文摘要"}},
  "2": {{"title": "中文标题", "summary": "中文摘要"}}
}}

待处理的新闻：
{chr(10).join(news_items)}

请直接输出JSON："""

        try:
            from langchain_core.messages import HumanMessage
            response = self.llm.invoke([HumanMessage(content=prompt)])
            result = response.content.strip()

            # 清理可能的markdown代码块标记
            result = result.strip()
            if result.startswith('```'):
                result = result.split('```', 2)[1] if '```' in result[3:] else result
                result = result.strip()
                if result.startswith('json'):
                    result = result[4:].strip()
                if result.endswith('```'):
                    result = result[:-3].strip()

            # 解析JSON
            import json
            translated_data = json.loads(result)

            # 更新原始数据
            for i, item in enumerate(trends):
                key = str(i + 1)
                if key in translated_data:
                    item["title_cn"] = translated_data[key]["title"]
                    item["summary_cn"] = translated_data[key]["summary"]
                else:
                    # 解析失败，使用原数据
                    item["title_cn"] = item.get("title", "")
                    item["summary_cn"] = item.get("description", "")[:150] + "..."

            self.log(f"批量生成摘要完成: {len(trends)}条")
            return trends

        except Exception as e:
            self.log(f"批量生成摘要失败: {e}，使用原始内容", "WARNING")
            # 降级处理：直接使用原始内容
            for item in trends:
                item["title_cn"] = item.get("title", "")
                item["summary_cn"] = item.get("description", "")[:150] + "..." if len(item.get("description", "")) > 150 else item.get("description", "")
            return trends

    def _get_category_name(self, key: str) -> str:
        """获取分类中文名"""
        mapping = {
            "📈 产业动态": "产业动态",
            "🎓 学术前沿": "学术前沿",
            "🔧 技术创新": "技术创新",
            "🛠️ AI工具/产品": "产品工具",
            "💼 AI应用": "行业应用"
        }
        return mapping.get(key, key)
