"""
World Class AI News Digest Generator
顶级AI新闻简报生成器 v7.0 - 增强版

v7.0 新特性:
- 使用评分筛选后的新闻 (scored_trends)
- 生成完整JSON结构化数据
- 为每条新闻添加背景分析和行业影响
- 提取核心洞察 (Core Insights)
- 识别热门话题 (Trending Topics)
- 优化翻译质量 (Few-shot提示)
"""

from datetime import datetime
from typing import Dict, Any, List
from loguru import logger
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from collections import Counter


class WorldClassDigestAgent:
    """世界顶级AI新闻简报生成器 v7.0"""

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        self.config = config
        self.prompts = prompts
        self.name = "world_class_digest"
        self.llm = self._init_llm()

        # 翻译配置
        agent_config = config.get("agents", {}).get("world_class_digest", {})
        self.translate_enabled = agent_config.get("translate_enabled", True)
        self.batch_size = agent_config.get("batch_size", 5)

        self.log(f"v7.0初始化完成，翻译功能: {'启用' if self.translate_enabled else '禁用'}")

    def _init_llm(self) -> ChatOpenAI:
        """初始化LLM"""
        try:
            import os
            from pathlib import Path
            from dotenv import load_dotenv

            # 加载.env文件
            project_root = Path(__file__).parent.parent.parent
            env_file = project_root / ".env"
            if env_file.exists():
                load_dotenv(env_file)
                self.log(f"已加载环境变量文件: {env_file}")

            llm_config = self.config.get("llm", {})
            provider = llm_config.get("provider", "zhipuai")

            if provider == "zhipuai":
                api_key = os.getenv("ZHIPUAI_API_KEY")
                if not api_key:
                    api_key = self.config.get("api_keys", {}).get("zhipuai")

                if not api_key:
                    self.log("未配置ZHIPUAI_API_KEY", "WARNING")
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
            return None
        except Exception as e:
            self.log(f"LLM初始化失败: {e}", "WARNING")
            return None

    def log(self, message: str, level: str = "INFO"):
        logger.log(level, f"[WorldClassDigestAgent] {message}")

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """执行简报生成"""
        try:
            self.log("开始生成世界顶级AI新闻简报 v7.0...")

            # 使用 scored_trends 而不是 categorized_trends
            scored_trends = state.get("scored_trends", {})
            editors_pick = state.get("editors_pick", [])
            source_status = state.get("source_status", {})

            # 生成简报
            digest = self._generate_world_class_digest(
                scored_trends,
                editors_pick,
                source_status
            )

            return {
                **state,
                "news_digest": digest,
                "current_step": "digest_generated"
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
        scored_trends: Dict[str, Dict],
        editors_pick: List[Dict],
        source_status: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成世界顶级AI新闻简报 v7.0"""

        today = datetime.now()
        issue_number = today.strftime("%Y%m%d")

        # 计算总数
        total_count = sum(
            cat_data.get("count", 0)
            for cat_data in scored_trends.values()
        )

        self.log(f"生成简报: {total_count}条精选热点")

        # 为新闻增强信息（翻译、背景、影响、标签）
        enhanced_editors_pick = self._enhance_news_items(editors_pick)
        for cat_name, cat_data in scored_trends.items():
            items = cat_data.get("items", [])
            enhanced_items = self._enhance_news_items(items)
            cat_data["items"] = enhanced_items

        # 提取核心洞察
        all_items = []
        for cat_data in scored_trends.values():
            all_items.extend(cat_data.get("items", []))

        core_insights = self._extract_core_insights(all_items)

        # 识别热门话题
        trending_topics = self._identify_trending_topics(all_items)

        # 生成Markdown内容
        markdown_content = self._generate_markdown_v7(
            scored_trends,
            enhanced_editors_pick,
            core_insights,
            trending_topics,
            source_status,
            today,
            issue_number,
            total_count
        )

        # 生成JSON数据
        json_data = self._generate_json_v7(
            scored_trends,
            enhanced_editors_pick,
            core_insights,
            trending_topics,
            source_status,
            today,
            issue_number,
            total_count,
            markdown_content
        )

        return json_data

    def _enhance_news_items(self, items: List[Dict]) -> List[Dict]:
        """为新闻条目增强信息（翻译、背景、影响、标签）"""
        if not items:
            return items

        # 批量翻译
        if self.translate_enabled and self.llm:
            items = self._batch_translate_items(items)

        # 为每条新闻生成背景、影响、标签（仅对重要新闻）
        for item in items:
            importance = item.get("importance_score", 0)
            if importance >= 70:  # 只为重要新闻生成详细分析
                enhanced = self._generate_background_analysis(item)
                item.update(enhanced)

        return items

    def _batch_translate_items(self, items: List[Dict]) -> List[Dict]:
        """批量翻译新闻标题和摘要"""
        if not items or not self.translate_enabled:
            return items

        # 检查是否已有中文
        if items[0].get("title_cn"):
            return items

        # 构建翻译提示（使用Few-shot示例）
        news_items = []
        for i, item in enumerate(items):
            title = item.get("title", "").replace('&amp;', '&').replace('&quot;', '"')
            desc = item.get("description", "").replace('&amp;', '&').replace('&quot;', '"')
            desc = desc.replace('<p>', '').replace('</p>', '').replace('<br>', ' ')[:200]
            news_items.append(f"{i+1}. 标题: {title}\n   摘要: {desc}")

        # 构建顶级科技媒体级别的翻译prompt（v8.0）
        prompt = f"""你是TechCrunch、The Verge、36氪、虎嗅等世界顶级科技媒体的中文主编，拥有20年科技新闻翻译经验。

【重要】你必须将英文翻译成中文，输出格式必须是JSON对象，键名为序号（"1", "2", "3"...）。

【翻译原则 - 顶级科技媒体标准】
1. **标题要求**：
   - 简洁有力，直击要点（不超过30字）
   - ✅ 好的风格："OpenAI发布GPT-5，支持100万tokens上下文"
   - ❌ 差的风格："OpenAI今天发布了新的GPT-5模型"
   - 突出技术亮点或商业价值
   - 使用主动语态，避免拖沓

2. **术语处理**：
   - 保留专业术语不翻译：LLM、RAG、Transformer、Agent、GPU、API、SDK等
   - 机构名保留原文：OpenAI、Meta、Google、Microsoft等
   - 产品名保留原文：ChatGPT、GitHub、Hugging Face等

3. **摘要要求**：
   - 精炼有力，控制在60-100字
   - 突出核心信息，去除冗余
   - 使用科技媒体常用表达

4. **语言风格**：
   - 符合中文科技媒体习惯
   - 简洁流畅，易于快速阅读
   - 准确传达技术细节

【输出格式示例】
{{
  "1": {{"title": "OpenAI发布GPT-5，支持100万tokens上下文", "summary": "OpenAI推出GPT-5，上下文窗口扩大至100万tokens，性能提升40%"}},
  "2": {{"title": "Meta发布新开源大模型", "summary": "Meta推出全新开源LLM，性能媲美GPT-4，支持商用"}}
}}

【待翻译新闻】
{chr(10).join(news_items)}

请直接输出JSON格式（不要有任何额外说明，所有标题和摘要必须是中文）："""

        try:
            # 使用SystemMessage + HumanMessage增强指令遵循（v8.0 - 顶级科技媒体级别）
            system_msg = """你是TechCrunch、The Verge、36氪、虎嗅等世界顶级科技媒体的中文主编，拥有20年科技新闻翻译经验。

【核心要求】
1. 你必须将英文翻译成中文
2. 输出必须是JSON格式，键名为序号："1", "2", "3"...
3. 保留专业术语不翻译：LLM、RAG、Transformer、Agent、GPU、API、SDK等
4. 机构名保留原文：OpenAI、Meta、Google、Microsoft等

【翻译示例】
{{"1": {{"title": "OpenAI发布GPT-5，支持100万tokens上下文", "summary": "OpenAI推出GPT-5，上下文窗口扩大至100万tokens，性能提升40%"}}}}
{{"2": {{"title": "Meta发布新开源大模型", "summary": "Meta推出全新开源LLM，性能媲美GPT-4，支持商用"}}}}"""

            response = self.llm.invoke([
                SystemMessage(content=system_msg),
                HumanMessage(content=prompt)
            ])
            result = response.content.strip()

            # 清理markdown代码块
            if result.startswith('```'):
                result = result.split('```', 2)[1] if '```' in result[3:] else result
                result = result.strip()
                if result.startswith('json'):
                    result = result[4:].strip()
                if result.endswith('```'):
                    result = result[:-3].strip()

            # 解析JSON
            translated_data = json.loads(result)

            # 处理不同格式
            # 格式1: {"1": {"title": "...", "summary": "..."}, ...}
            # 格式2: [{"title": "...", "summary": "..."}, ...] (按顺序对应)
            if isinstance(translated_data, list):
                # 如果返回列表，按顺序映射
                for i, item in enumerate(items):
                    if i < len(translated_data):
                        item["title_cn"] = translated_data[i].get("title", item.get("title", ""))
                        item["summary_cn"] = translated_data[i].get("summary", item.get("description", ""))[:150]
                    else:
                        item["title_cn"] = item.get("title", "")
                        item["summary_cn"] = item.get("description", "")[:150]
            else:
                # 字典格式，按编号键查找
                for i, item in enumerate(items):
                    key = str(i + 1)
                    if key in translated_data:
                        item["title_cn"] = translated_data[key]["title"]
                        item["summary_cn"] = translated_data[key].get("summary", item.get("description", ""))[:150]
                    else:
                        item["title_cn"] = item.get("title", "")
                        item["summary_cn"] = item.get("description", "")[:150]

            self.log(f"批量翻译完成: {len(items)}条")
            return items

        except json.JSONDecodeError as e:
            self.log(f"JSON解析失败: {e}，LLM返回: {result[:200]}...", "WARNING")
            for item in items:
                item["title_cn"] = item.get("title", "")
                item["summary_cn"] = item.get("description", "")[:150]
            return items
        except Exception as e:
            self.log(f"批量翻译失败: {e}，使用原始内容", "WARNING")
            for item in items:
                item["title_cn"] = item.get("title", "")
                item["summary_cn"] = item.get("description", "")[:150]
            return items

    def _generate_background_analysis(self, item: Dict) -> Dict:
        """为单条新闻生成背景分析和行业影响"""
        if not self.llm:
            return {"background": "", "impact": "", "tags": []}

        title = item.get("title_cn", item.get("title", ""))
        summary = item.get("summary_cn", item.get("description", ""))

        prompt = f"""基于以下AI新闻，生成背景分析和行业影响：

【新闻】
标题: {title}
摘要: {summary}

请生成：
1. background (100-150字): 背景介绍，帮助读者理解上下文
2. impact (100-150字): 行业影响分析，说明为什么重要
3. tags (3-5个关键词): 用于分类和检索

直接输出JSON格式，不要有任何额外说明。"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            result = response.content.strip()

            if result.startswith('```'):
                result = result.split('```', 2)[1] if '```' in result[3:] else result
                result = result.strip()
                if result.startswith('json'):
                    result = result[4:].strip()
                if result.endswith('```'):
                    result = result[:-3].strip()

            analysis = json.loads(result)
            return {
                "background": analysis.get("background", ""),
                "impact": analysis.get("impact", ""),
                "tags": analysis.get("tags", [])
            }
        except Exception as e:
            self.log(f"背景分析生成失败: {e}", "DEBUG")
            return {"background": "", "impact": "", "tags": []}

    def _extract_core_insights(self, items: List[Dict]) -> List[str]:
        """从所有新闻中提取核心洞察（v8.0 - 顶级科技媒体级别）"""
        if not self.llm or not items:
            return []

        # 选择最重要的10条新闻
        top_items = sorted(items, key=lambda x: x.get("importance_score", 0), reverse=True)[:10]

        news_summary = "\n".join([
            f"- {item.get('title_cn', item.get('title', ''))}"
            for item in top_items
        ])

        # v8.0 - 顶级科技媒体级别的核心洞察提取
        prompt = f"""你是TechCrunch、36氪、虎嗅等世界顶级科技媒体的总编辑，拥有20年AI行业深度报道经验。

请基于今日AI新闻，提取3-5条**具有前瞻性的核心洞察**。

【今日重要新闻】
{news_summary}

【洞察要求 - 顶级科技媒体标准】
1. **洞察深度**: 透过表象看到行业本质变化，揭示深层趋势
2. **观点鲜明**: 避免泛泛而谈，每条都要有独特观点
3. **前瞻性**: 能够预见6-12个月的行业发展方向
4. **启发性**: 能引发读者思考和讨论
5. **语言精炼**: 每条30-50字，信息密度高
6. **使用有力动词**: "揭示了"、"标志着"、"预示着"、"重塑了"、"颠覆了"
7. **避免罗列**: 不列举具体公司/产品名称，聚焦行业趋势
8. **多角度**: 从技术、商业、生态等不同维度观察

【洞察示例】（参考风格）:
- ❌ 差: "今天有很多AI新闻"
- ✅ 好: "多智能体协作范式确立，标志着AI从单一对话迈向自主执行新阶段"
- ❌ 差: "大模型性能持续提升"
- ✅ 好: "开源模型逼近闭源水平，重塑AI产业竞争格局"

直接输出JSON数组格式，不要有任何额外说明。"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            result = response.content.strip()

            if result.startswith('```'):
                result = result.split('```', 2)[1] if '```' in result[3:] else result
                result = result.strip()
                if result.startswith('json'):
                    result = result[4:].strip()
                if result.endswith('```'):
                    result = result[:-3].strip()

            insights = json.loads(result)
            return insights if isinstance(insights, list) else []
        except Exception as e:
            self.log(f"核心洞察提取失败: {e}", "DEBUG")
            return []

    def _identify_trending_topics(self, items: List[Dict]) -> List[Dict]:
        """识别热门话题"""
        # 从所有标签中统计热门话题
        all_tags = []
        for item in items:
            tags = item.get("tags", [])
            all_tags.extend(tags)

        # 如果没有标签，从标题中提取关键词
        if not all_tags:
            for item in items:
                title = item.get("title_cn", item.get("title", ""))
                # 简单的关键词提取（实际应该用更复杂的NLP）
                keywords = ["GPT", "LLM", "RAG", "Agent", "Transformer", "AI", "大模型", "开源"]
                for kw in keywords:
                    if kw in title:
                        all_tags.append(kw)

        # 统计词频
        tag_counts = Counter(all_tags)

        # 转换为热门话题列表
        trending = []
        for tag, count in tag_counts.most_common(10):
            if count >= 2:  # 至少出现2次
                trending.append({
                    "name": tag,
                    "count": count,
                    "trend": "rising" if count >= 4 else "stable"
                })

        return trending[:5]  # 返回Top 5

    def _generate_markdown_v7(
        self,
        scored_trends: Dict[str, Dict],
        editors_pick: List[Dict],
        core_insights: List[str],
        trending_topics: List[Dict],
        source_status: Dict[str, Any],
        today: datetime,
        issue_number: str,
        total_count: int
    ) -> str:
        """生成Markdown格式简报 v8.0 - 顶级科技媒体风格"""

        parts = []

        # ========== Header - 增强版 ==========
        parts.append(f"# AI每日热点 · {today.strftime('%Y年%m月%d日')}\n\n")
        parts.append(f"> **期号**: #{issue_number}  |  **阅读时间**: ~{max(3, total_count * 15 // 60)}分钟  |  **本期精选**: {total_count}条前沿动态\n\n")
        parts.append("---\n\n")

        # ========== 核心洞察 - 优化展示 ==========
        if core_insights:
            parts.append("## 💡 核心洞察\n\n")
            parts.append("*基于今日AI热点，提炼行业本质变化与未来趋势*\n\n")
            for insight in core_insights:
                parts.append(f"- **{insight}**\n")
            parts.append("\n---\n\n")

        # ========== 热门话题 ==========
        if trending_topics:
            parts.append("## 📊 热门话题\n\n")
            parts.append("| 话题 | 相关新闻 | 趋势 |\n")
            parts.append("|------|---------|------|\n")
            for topic in trending_topics:
                trend_icon = "📈 上升" if topic.get("trend") == "rising" else "➡️ 稳定"
                parts.append(f"| {topic['name']} | {topic['count']}条 | {trend_icon} |\n")
            parts.append("\n---\n\n")

        # ========== 分类热点 - 优化展示格式 ==========
        parts.append("## 🔍 分类热点\n\n")

        for cat_name, cat_data in scored_trends.items():
            items = cat_data.get("items", [])
            if not items:
                continue

            icon = cat_data.get("icon", "📌")
            name = self._get_category_name(cat_name)
            parts.append(f"### {icon} {name} ({len(items)}条)\n\n")

            for i, item in enumerate(items, 1):
                title = item.get("title_cn", item.get("title", ""))
                summary = item.get("summary_cn", item.get("description", ""))
                source = item.get("source", "")
                url = item.get("url", "")
                score = item.get("importance_score", 0)
                background = item.get("background", "")
                impact = item.get("impact", "")

                # 优化展示格式
                parts.append(f"#### {i}. {title}\n\n")
                parts.append(f"> 📰 **{source}**  |  ⭐ **评分**: {int(score)}/100  |  🔗 [原文链接]({url})\n\n")

                if summary:
                    parts.append(f"{summary}\n\n")

                if background:
                    parts.append(f"📖 **背景**: {background}\n\n")

                if impact:
                    parts.append(f"🎯 **影响**: {impact}\n\n")

                parts.append("---\n\n")

        # ========== 数据来源 ==========
        parts.append("## 📚 数据来源\n\n")
        success_sources = [s for s, status in source_status.items() if status.get("success", False)]
        for source in success_sources:
            count = source_status[source].get("count", 0)
            parts.append(f"- **{source}**: {count}条\n")
        parts.append("\n---\n\n")

        # ========== Footer ==========
        parts.append("*🤖 Generated by [ContentForge AI](https://github.com/devfoxaicn/content-forge-ai)*\n")

        return "".join(parts)

    def _generate_json_v7(
        self,
        scored_trends: Dict[str, Dict],
        editors_pick: List[Dict],
        core_insights: List[str],
        trending_topics: List[Dict],
        source_status: Dict[str, Any],
        today: datetime,
        issue_number: str,
        total_count: int,
        markdown_content: str
    ) -> Dict[str, Any]:
        """生成JSON格式数据 v7.0"""

        # 构建分类数据
        categories = []
        category_id_map = {
            "📈 行业动态": ("industry", "产业动态", "📈"),
            "🎓 学术前沿": ("academic", "学术前沿", "🎓"),
            "🔬 技术创新": ("tech", "技术创新", "🔬"),
            "🛠️ AI工具/产品": ("product", "产品工具", "🛠️"),
            "💼 AI应用": ("application", "行业应用", "💼")
        }

        for cat_name, cat_data in scored_trends.items():
            cat_id, name, icon = category_id_map.get(cat_name, (cat_name, cat_name, "📌"))
            items = []
            for item in cat_data.get("items", []):
                # 使用hash()函数生成唯一ID
                url_hash = hash(item.get("url", "")) & 0xffffff
                items.append({
                    "id": f"{cat_id}_{url_hash:06x}",
                    "title": item.get("title", ""),
                    "title_cn": item.get("title_cn", ""),
                    "summary": item.get("description", "")[:200],
                    "summary_cn": item.get("summary_cn", "")[:200],
                    "url": item.get("url", ""),
                    "source": item.get("source", ""),
                    "category": name,
                    "importance_score": item.get("importance_score", 0),
                    "published_at": item.get("timestamp", ""),
                    "tags": item.get("tags", []),
                    "background": item.get("background", ""),
                    "impact": item.get("impact", "")
                })

            categories.append({
                "id": cat_id,
                "name": name,
                "icon": icon,
                "count": len(items),
                "items": items
            })

        # 构建编辑精选
        editors_pick_data = []
        for item in editors_pick:
            editors_pick_data.append({
                "id": item.get("id", ""),
                "title": item.get("title", ""),
                "title_cn": item.get("title_cn", ""),
                "summary": item.get("description", "")[:200],
                "summary_cn": item.get("summary_cn", "")[:200],
                "url": item.get("url", ""),
                "source": item.get("source", ""),
                "category": self._get_category_name(item.get("category", "")),
                "importance_score": item.get("importance_score", 0),
                "published_at": item.get("timestamp", ""),
                "tags": item.get("tags", []),
                "background": item.get("background", ""),
                "impact": item.get("impact", ""),
                "pick_rank": item.get("pick_rank", 0)
            })

        # 构建数据来源
        sources = []
        for source, status in source_status.items():
            if status.get("success", False):
                sources.append({
                    "name": source,
                    "count": status.get("count", 0)
                })

        return {
            "metadata": {
                "title": f"AI每日热点 · {today.strftime('%Y年%m月%d日')}",
                "issue_number": issue_number,
                "publish_date": today.strftime("%Y-%m-%d"),
                "generated_at": today.isoformat(),
                "word_count": len(markdown_content),
                "reading_time": f"{total_count * 15 // 60}分钟",
                "total_items": total_count,
                "version": "v7.0"
            },
            "editors_pick": editors_pick_data,
            "categories": categories,
            "core_insights": core_insights,
            "trending_topics": trending_topics,
            "sources": sources,
            "markdown_content": markdown_content
        }

    def _get_category_name(self, key: str) -> str:
        """获取分类中文名"""
        mapping = {
            "📈 行业动态": "产业动态",
            "🎓 学术前沿": "学术前沿",
            "🔬 技术创新": "技术创新",
            "🛠️ AI工具/产品": "产品工具",
            "💼 AI应用": "行业应用"
        }
        return mapping.get(key, key)
