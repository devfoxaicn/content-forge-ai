"""
World Class AI News Digest Generator v8.0
顶级AI新闻简报生成器 v8.0 - 增强版

v8.0 新特性:
- 整合 copywriting 原则（清晰度、利益导向、具体性）
- 整合 copy-editing 原则（7次扫描优化）
- 强化核心洞察提取（行业深度分析）
- 优化翻译质量（科技媒体标准）
- 增强可读性和吸引力
"""

from datetime import datetime
from typing import Dict, Any, List
from loguru import logger
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from collections import Counter


class WorldClassDigestAgentV8:
    """世界顶级AI新闻简报生成器 v8.0"""

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        self.config = config
        self.prompts = prompts
        self.name = "world_class_digest_v8"
        self.llm = self._init_llm()

        # 翻译配置
        agent_config = config.get("agents", {}).get("world_class_digest", {})
        self.translate_enabled = agent_config.get("translate_enabled", True)
        self.batch_size = agent_config.get("batch_size", 5)

        self.log(f"v8.0初始化完成，整合 copywriting + copy-editing 原则")

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
        logger.log(level, f"[WorldClassDigestAgentV8] {message}")

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """执行简报生成"""
        try:
            self.log("开始生成世界顶级AI新闻简报 v8.0...")

            # 使用 scored_trends
            scored_trends = state.get("scored_trends", {})
            editors_pick = state.get("editors_pick", [])
            source_status = state.get("source_status", {})

            # 生成简报
            digest = self._generate_world_class_digest_v8(
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

    def _generate_world_class_digest_v8(
        self,
        scored_trends: Dict[str, Dict],
        editors_pick: List[Dict],
        source_status: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成世界顶级AI新闻简报 v8.0"""

        today = datetime.now()
        issue_number = today.strftime("%Y%m%d")

        # 计算总数
        total_count = sum(
            cat_data.get("count", 0)
            for cat_data in scored_trends.values()
        )

        self.log(f"生成简报: {total_count}条精选热点")

        # 第1步：翻译和增强新闻（应用 copywriting 原则）
        enhanced_editors_pick = self._enhance_news_items_v8(editors_pick)
        for cat_name, cat_data in scored_trends.items():
            items = cat_data.get("items", [])
            enhanced_items = self._enhance_news_items_v8(items)
            cat_data["items"] = enhanced_items

        # 第2步：提取核心洞察（应用 content-research-writer 原则）
        all_items = []
        for cat_data in scored_trends.values():
            all_items.extend(cat_data.get("items", []))

        core_insights = self._extract_core_insights_v8(all_items)

        # 第3步：识别热门话题
        trending_topics = self._identify_trending_topics(all_items)

        # 第4步：生成深度观察（新增）
        deep_observation = self._generate_deep_observation(all_items, core_insights)

        # 第5步：生成Markdown内容（应用 copy-editing 7次扫描）
        markdown_content = self._generate_markdown_v8(
            scored_trends,
            enhanced_editors_pick,
            core_insights,
            trending_topics,
            deep_observation,
            source_status,
            today,
            issue_number,
            total_count
        )

        # 第6步：生成JSON数据
        json_data = self._generate_json_v8(
            scored_trends,
            enhanced_editors_pick,
            core_insights,
            trending_topics,
            deep_observation,
            source_status,
            today,
            issue_number,
            total_count,
            markdown_content
        )

        return json_data

    def _enhance_news_items_v8(self, items: List[Dict]) -> List[Dict]:
        """为新闻条目增强信息 v8.0（应用 copywriting 原则）"""
        if not items:
            return items

        # 批量翻译（使用改进的 prompt）
        if self.translate_enabled and self.llm:
            items = self._batch_translate_items_v8(items)

        # 为重要新闻生成增强分析
        for item in items:
            importance = item.get("importance_score", 0)
            if importance >= 75:  # 只为最重要新闻生成详细分析
                enhanced = self._generate_enhanced_analysis_v8(item)
                item.update(enhanced)

        return items

    def _batch_translate_items_v8(self, items: List[Dict]) -> List[Dict]:
        """批量翻译新闻标题和摘要 v8.0（应用 copywriting 原则）"""
        if not items or not self.translate_enabled:
            return items

        # 检查是否已有中文
        if items[0].get("title_cn"):
            return items

        # 构建翻译提示（整合 copywriting 原则）
        news_items = []
        for i, item in enumerate(items):
            title = item.get("title", "").replace('&amp;', '&').replace('&quot;', '"')
            desc = item.get("description", "").replace('&amp;', '&').replace('&quot;', '"')
            desc = desc.replace('<p>', '').replace('</p>', '').replace('<br>', ' ')[:200]
            news_items.append(f"{i+1}. 标题: {title}\n   摘要: {desc}")

        prompt = f"""你是TechCrunch、The Verge、Wired等顶级科技媒体的中文主编。

【核心翻译原则 - Copywriting Standards】

1. **清晰度优于聪明** (Clarity Over Cleverness)
   - 直接传达核心信息，不要故弄玄虚
   - 示例："GPT-5发布" 优于 "GPT-5震撼登场"

2. **利益优于功能** (Benefits Over Features)
   - 标题要突出"这对读者意味着什么"
   - 示例："支持100万tokens，可处理整本书" 优于 "具有100万tokens上下文窗口"

3. **具体性优于模糊** (Specificity Over Vagueness)
   - 使用具体数字，避免"强大、优秀、先进"等词
   - 示例："性能提升300%" 优于 "性能大幅提升"

4. **标题公式**（从顶级科技媒体学习）
   - "Company发布Product，核心Benefit"
   - "突破数字：具体成就"
   - "Industry实现Milestone，意味着Impact"

5. **专业术语处理**
   - 保留不翻译：LLM、RAG、Transformer、Agent、GPU、API、SDK
   - 翻译但保留英文：人工智能(AI)、机器学习(ML)、深度学习(DL)

6. **摘要精炼原则**
   - 控制在60-80字
   - 突出核心价值和影响
   - 使用主动语态

【优秀翻译示例】

输入: "OpenAI Announces GPT-5 With 1M Context Window, 300% Better Reasoning"
输出: {{"title": "OpenAI发布GPT-5：支持100万tokens，推理提升300%", "summary": "OpenAI推出GPT-5，上下文窗口扩展至100万tokens（可处理整本书），推理能力提升300%"}

输入: "Meta Releases New Open Source LLM to Compete with GPT-4"
输出: {{"title": "Meta开源新大模型，性能媲美GPT-4可免费商用", "summary": "Meta发布全新开源LLM，性能达到GPT-4水平，企业可免费用于商业产品"}}

输入: "Google DeepMind's AlphaFold 3 Can Now Predict Protein Interactions"
输出: {{"title": "DeepMind推出AlphaFold 3：可预测蛋白质相互作用", "summary": "DeepMind发布AlphaFold 3，突破性升级可预测蛋白质间相互作用，加速新药研发"}}

【输出格式】
必须是JSON对象，键名为序号（"1", "2", "3"...）：
{{
  "1": {{"title": "中文标题", "summary": "中文摘要（60-80字）"}},
  "2": {{"title": "中文标题", "summary": "中文摘要（60-80字）"}}
}}

【待翻译新闻】
{chr(10).join(news_items)}

请严格按照以上原则，直接输出JSON格式："""

        try:
            system_msg = """你是TechCrunch、The Verge、Wired等顶级科技媒体的中文主编。

【必须遵守】
1. 所有标题和摘要必须是中文
2. 输出必须是JSON格式，键名为序号："1", "2", "3"...
3. 保留专业术语不翻译：LLM、RAG、Transformer、Agent、GPU等
4. 突出利益和价值，而不仅仅是功能"""

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
            if isinstance(translated_data, list):
                for i, item in enumerate(items):
                    if i < len(translated_data):
                        item["title_cn"] = translated_data[i].get("title", item.get("title", ""))
                        item["summary_cn"] = translated_data[i].get("summary", item.get("description", ""))[:150]
                    else:
                        item["title_cn"] = item.get("title", "")
                        item["summary_cn"] = item.get("description", "")[:150]
            else:
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

    def _generate_enhanced_analysis_v8(self, item: Dict) -> Dict:
        """为单条新闻生成增强分析 v8.0（应用 copy-editing 原则）"""
        if not self.llm:
            return {"background": "", "impact": "", "tags": []}

        title = item.get("title_cn", item.get("title", ""))
        summary = item.get("summary_cn", item.get("description", ""))

        prompt = f"""你是资深科技行业分析师，基于以下新闻生成深度分析。

【新闻】
标题: {title}
摘要: {summary}

请生成：

1. **background** (120-150字): 背景介绍
   - 使用"所以..."、"在此之前..."等连接词增强可读性
   - 突出技术发展脉络
   - 帮助读者理解上下文

2. **impact** (120-150字): 行业影响分析
   - 使用"这意味着..."、"具体来说..."等具体化表达
   - 避免"重大"、"深远"等空洞词汇
   - 说明对不同群体的具体影响

3. **tags** (3-5个关键词): 用于分类和检索
   - 选择用户会搜索的术语
   - 避免过于宽泛的词

【写作原则】
- 具体性：使用具体数字和例子
- 利益导向：说明"这对读者意味着什么"
- 可读性：段落控制在3-4句话

直接输出JSON格式："""

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
            self.log(f"增强分析生成失败: {e}", "DEBUG")
            return {"background": "", "impact": "", "tags": []}

    def _extract_core_insights_v8(self, items: List[Dict]) -> List[str]:
        """提取核心洞察 v8.0（应用 content-research-writer 原则）"""
        if not self.llm or not items:
            return []

        # 选择最重要的10条新闻
        top_items = sorted(items, key=lambda x: x.get("importance_score", 0), reverse=True)[:10]

        news_summary = "\n".join([
            f"- {item.get('title_cn', item.get('title', ''))}"
            for item in top_items
        ])

        prompt = f"""你是资深科技行业观察家，基于今日AI新闻提取核心洞察。

【今日重要新闻】
{news_summary}

请生成3-5条核心洞察，每条40-60字，要求：

1. **捕捉趋势**：识别不同新闻背后的共同趋势
2. **关联分析**：发现看似无关事件之间的联系
3. **预见未来**：基于当前动态推断发展方向
4. **具体表达**：避免"重要"、"突破"等空洞词汇
5. **利益导向**：说明"这对行业/读者意味着什么"

【洞察示例】
❌ "AI技术持续发展，对行业产生重大影响"
✅ "从GPT-5到开源模型竞争，大模型进入'性能+成本'双轮驱动阶段，企业选型更务实"

【输出格式】
直接输出JSON数组格式，不要有任何额外说明："""

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
                keywords = ["GPT", "LLM", "RAG", "Agent", "Transformer", "AI", "大模型", "开源", "多模态"]
                for kw in keywords:
                    if kw in title:
                        all_tags.append(kw)

        # 统计词频
        tag_counts = Counter(all_tags)

        # 转换为热门话题列表
        trending = []
        for tag, count in tag_counts.most_common(10):
            if count >= 2:
                trending.append({
                    "name": tag,
                    "count": count,
                    "trend": "rising" if count >= 4 else "stable"
                })

        return trending[:5]

    def _generate_deep_observation(self, items: List[Dict], core_insights: List[str]) -> str:
        """生成深度观察（新增部分）"""
        if not self.llm or not items:
            return ""

        top_items = sorted(items, key=lambda x: x.get("importance_score", 0), reverse=True)[:5]
        news_list = "\n".join([
            f"- {item.get('title_cn', item.get('title', ''))}"
            for item in top_items
        ])

        prompt = f"""你是TechCrunch、Wired等顶级科技媒体的专栏作家。

基于今日AI新闻和核心洞察，写一篇350-450字的深度观察文章。

【参考的核心洞察】
{chr(10).join(f"- {insight}" for insight in core_insights)}

【今日重要新闻】
{news_list}

【写作要求】
1. **强Hook开头**：用数据、反常识观点或具体场景开头
2. **具体化表达**：避免"重大"、"突破"等空洞词汇
3. **关联分析**：发现不同新闻之间的内在联系
4. **行业视角**：从产业、技术、应用多角度分析
5. **可读性**：段落控制在3-5句话，使用连接词
6. **长度**：350-450字

【优秀开头示例】
❌ "今天AI行业有很多重要新闻..."
✅ "OpenAI、Meta、Google三家在同一天发布新模型，这不是巧合，而是大模型竞争进入新阶段的信号"

直接输出文章内容（不要标题，不要额外说明）："""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            result = response.content.strip()
            return result
        except Exception as e:
            self.log(f"深度观察生成失败: {e}", "DEBUG")
            return ""

    def _generate_markdown_v8(
        self,
        scored_trends: Dict[str, Dict],
        editors_pick: List[Dict],
        core_insights: List[str],
        trending_topics: List[Dict],
        deep_observation: str,
        source_status: Dict[str, Any],
        today: datetime,
        issue_number: str,
        total_count: int
    ) -> str:
        """生成Markdown格式简报 v8.0"""

        parts = []

        # ========== Header ==========
        parts.append(f"# AI每日热点 · {today.strftime('%Y年%m月%d日')}\n\n")
        parts.append(f"> **期号**: #{issue_number}  |  **阅读时间**: ~{max(5, total_count * 12 // 60)}分钟  |  **精选**: {total_count}条\n\n")
        parts.append("---\n\n")

        # ========== 核心洞察 ==========
        if core_insights:
            parts.append("## 💡 核心洞察\n\n")
            for insight in core_insights:
                parts.append(f"- {insight}\n")
            parts.append("\n---\n\n")

        # ========== 深度观察（新增） ==========
        if deep_observation:
            parts.append("## 📰 深度观察\n\n")
            parts.append(f"{deep_observation}\n\n")
            parts.append("---\n\n")

        # ========== 编辑精选 ==========
        if editors_pick:
            parts.append("## ⭐ 编辑精选 (Editor's Picks)\n\n")

            for i, item in enumerate(editors_pick, 1):
                title = item.get("title_cn", item.get("title", ""))
                summary = item.get("summary_cn", item.get("description", ""))
                source = item.get("source", "")
                url = item.get("url", "")
                score = item.get("importance_score", 0)
                background = item.get("background", "")
                impact = item.get("impact", "")

                parts.append(f"### {i}. {title}\n\n")
                parts.append(f"> 📰 **{source}**  |  ⭐ **重要性**: {int(score)}/100  |  🔗 [原文链接]({url})\n\n")

                if summary:
                    parts.append(f"**核心内容**: {summary}\n\n")

                if background:
                    parts.append(f"**背景**: {background}\n\n")

                if impact:
                    parts.append(f"**行业影响**: {impact}\n\n")

                parts.append("---\n\n")

        # ========== 热门话题 ==========
        if trending_topics:
            parts.append("## 📊 热门话题\n\n")
            parts.append("| 话题 | 相关新闻 | 趋势 |\n")
            parts.append("|------|---------|------|\n")
            for topic in trending_topics:
                trend_icon = "📈 上升" if topic.get("trend") == "rising" else "➡️ 稳定"
                parts.append(f"| {topic['name']} | {topic['count']}条 | {trend_icon} |\n")
            parts.append("\n---\n\n")

        # ========== 分类热点 ==========
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

                parts.append(f"#### {i}. {title}\n\n")
                parts.append(f"> 📰 **{source}**  |  ⭐ **重要性**: {int(score)}/100  |  🔗 [原文]({url})\n\n")

                if summary:
                    parts.append(f"**摘要**: {summary}\n\n")

                if background:
                    parts.append(f"**背景**: {background}\n\n")

                if impact:
                    parts.append(f"**影响**: {impact}\n\n")

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

    def _generate_json_v8(
        self,
        scored_trends: Dict[str, Dict],
        editors_pick: List[Dict],
        core_insights: List[str],
        trending_topics: List[Dict],
        deep_observation: str,
        source_status: Dict[str, Any],
        today: datetime,
        issue_number: str,
        total_count: int,
        markdown_content: str
    ) -> Dict[str, Any]:
        """生成JSON格式数据 v8.0"""

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
                "reading_time": f"{max(5, total_count * 12 // 60)}分钟",
                "total_items": total_count,
                "version": "v8.0"
            },
            "editors_pick": editors_pick_data,
            "categories": categories,
            "core_insights": core_insights,
            "deep_observation": deep_observation,  # 新增
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
