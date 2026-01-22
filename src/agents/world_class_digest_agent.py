"""
世界顶级AI新闻简报Agent - 全中文版
按照世界一流科技媒体标准，生成专业的AI新闻简报
"""

from typing import Dict, Any, List
import yaml
from pathlib import Path
from datetime import datetime
from src.agents.base import BaseAgent


class WorldClassDigestAgent(BaseAgent):
    """
    世界顶级AI新闻简报Agent
    参照36氪、虎嗅、品玩等专业科技媒体的中文写作标准
    """

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        super().__init__(config, prompts)

        digest_config = config.get("agents", {}).get("trends_digest", {})
        self.include_analysis = digest_config.get("include_analysis", True)
        self.llm.temperature = 0.8  # 更高温度，增强中文表达的丰富性

        # 5大分类定义
        self.CATEGORIES = {
            "industry": {
                "name": "产业动态",
                "icon": "📈",
                "description": "追踪AI产业资本流向、巨头战略布局、初创企业融资，第一时间掌握全球商业核心动态"
            },
            "academic": {
                "name": "学术前沿",
                "icon": "🎓",
                "description": "精选顶级期刊论文、前沿研究成果，解读学术界最新突破与理论创新"
            },
            "tech": {
                "name": "技术创新",
                "icon": "🔬",
                "description": "深度解析模型架构、算法突破、工程创新，追踪AI技术底层演进"
            },
            "product": {
                "name": "产品工具",
                "icon": "🛠️",
                "description": "发现提升开发效率的实用工具、颠覆性的AI产品，评测最新应用体验"
            },
            "application": {
                "name": "行业应用",
                "icon": "💼",
                "description": "展示AI在各行业的创新应用案例，分析落地实践与商业价值"
            }
        }

        self.log("世界顶级中文简报标准已加载 - 专业科技媒体风格")

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """生成世界顶级AI新闻简报"""
        self.log("开始生成世界顶级AI新闻简报（全中文版）")

        try:
            categorized_trends = state.get("categorized_trends")
            if not categorized_trends:
                self.log("未找到 categorized_trends，无法生成简报")
                return {
                    **state,
                    "error_message": "未找到分类后的热点数据",
                    "current_step": "digest_failed"
                }

            total_count = state.get("total_trends_count", 0)
            self.log(f"开始处理 {total_count} 条AI资讯")

            # 生成完整简报
            digest = self._generate_world_class_digest(categorized_trends, total_count)

            self.log(f"成功生成世界顶级AI新闻简报，共 {total_count} 条资讯")

            return {
                **state,
                "news_digest": digest,
                "current_step": "digest_completed"
            }
        except Exception as e:
            self.log(f"简报生成失败: {str(e)}", "ERROR")
            return {
                **state,
                "error_message": f"简报生成失败: {str(e)}",
                "current_step": "digest_failed"
            }

    def _generate_world_class_digest(
        self,
        categorized_trends: Dict[str, Dict],
        total_count: int
    ) -> Dict[str, Any]:
        """生成世界顶级简报内容"""

        today = datetime.now()
        issue_number = today.strftime("%Y%m%d")

        # 1. 生成核心洞察
        key_insights = self._generate_key_insights(categorized_trends)

        # 2. 生成深度观察
        deep_analysis = self._generate_deep_analysis(categorized_trends)

        # 3. 生成每个分类的导语
        category_intros = self._generate_category_intros(categorized_trends)

        # 4. 翻译并整理所有热点内容
        translated_items = self._translate_and_format_items(categorized_trends)

        # 5. 组装完整简报
        full_content = self._assemble_full_content(
            today, issue_number, total_count,
            key_insights, deep_analysis,
            category_intros, translated_items
        )

        word_count = len(full_content)

        return {
            "title": f"AI每日热点 · {today.strftime('%Y年%m月%d日')}",
            "subtitle": f"汇聚全球8大AI资讯源，精选{total_count}条前沿动态",
            "issue_number": issue_number,
            "publish_date": today.strftime("%Y-%m-%d"),
            "full_content": full_content,
            "key_insights": key_insights,
            "deep_analysis": deep_analysis,
            "category_intros": category_intros,
            "word_count": word_count,
            "reading_time": f"{word_count // 400}-{word_count // 250}分钟",
            "total_topics": total_count,
            "version": "v4.0"
        }

    def _generate_key_insights(self, categorized_trends: Dict[str, Dict]) -> List[str]:
        """生成核心洞察（中文）"""
        try:
            # 提取所有高热度热点
            all_items = []
            for category_data in categorized_trends.values():
                items = category_data.get("items", [])
                all_items.extend(items)

            # 按热度排序
            all_items.sort(key=lambda x: x.get("heat_score", 0), reverse=True)
            top_items = all_items[:15]

            # 构建标题列表
            titles_text = "\n".join([
                f"{i+1}. {item.get('title', '')}"
                for i, item in enumerate(top_items)
            ])

            prompt = f"""你是一位世界顶级科技媒体的总编辑。请从以下AI热点中提取3-5个核心洞察（每条25-35字）。

热点列表：
{titles_text}

要求：
1. 洞察深刻，透过现象看本质
2. 观点鲜明，一针见血
3. 语言精炼，专业表达
4. 使用"标志着"、"揭示了"、"反映出"等判断性动词
5. 全中文表述

核心洞察（每条一行）："""

            response = self._call_llm(prompt)
            insights = [line.strip() for line in response.strip().split('\n') if line.strip()]
            return insights[:5]
        except Exception as e:
            self.log(f"核心洞察生成失败: {e}", "WARNING")
            return []

    def _generate_deep_analysis(self, categorized_trends: Dict[str, Dict]) -> str:
        """生成深度观察（中文）"""
        try:
            # 统计各分类数量
            category_counts = {}
            all_titles = {}
            for key, cat_data in categorized_trends.items():
                count = cat_data.get("count", 0)
                if count > 0:
                    cat_name = self._get_category_name(key)
                    category_counts[cat_name] = count
                    items = cat_data.get("items", [])[:8]
                    all_titles[cat_name] = [item.get("title", "") for item in items]

            # 构建输入文本
            input_text = "本期热点分类统计：\n"
            for cat, count in category_counts.items():
                input_text += f"- {cat}: {count}条\n"

            input_text += "\n各分类代表性热点：\n"
            for cat, titles in all_titles.items():
                input_text += f"\n【{cat}】\n"
                for title in titles[:5]:
                    input_text += f"  • {title}\n"

            prompt = f"""你是一位世界顶级科技媒体的资深分析师。请基于以下AI热点，撰写一篇深度观察文章（350-450字）。

{input_text}

写作要求：
1. 立意高远：从产业格局、技术演进、商业价值等宏观视角切入
2. 逻辑清晰：现象描述 → 原因分析 → 影响判断 → 趋势展望
3. 数据支撑：引用具体分类数量和代表性热点
4. 观点鲜明：提出有深度的判断和预测
5. 语言专业：使用"底层逻辑"、"范式转移"、"生态重构"等专业表达
6. 全中文写作

深度观察文章："""

            response = self._call_llm(prompt)
            return response.strip()
        except Exception as e:
            self.log(f"深度观察生成失败: {e}", "WARNING")
            return ""

    def _generate_category_intros(self, categorized_trends: Dict[str, Dict]) -> Dict[str, str]:
        """生成各分类导语（中文）"""
        intros = {}

        for key, cat_data in categorized_trends.items():
            items = cat_data.get("items", [])
            count = cat_data.get("count", 0)

            if count == 0:
                continue

            cat_name = self._get_category_name(key)
            cat_desc = self.CATEGORIES.get(self._get_key_by_name(cat_name), {}).get("description", "")

            # 获取前5个标题
            titles = [item.get("title", "") for item in items[:5]]
            titles_text = "\n".join([f"  • {title}" for title in titles])

            prompt = f"""你是一位专业科技媒体的版块主编。请为"{cat_name}"分类撰写一段精彩导语（50-70字）。

分类定位：{cat_desc}

本分类精选热点：
{titles_text}

要求：
1. 突出分类价值
2. 语言生动有力
3. 吸引读者继续阅读
4. 全中文表达

导语："""

            try:
                response = self._call_llm(prompt)
                intros[cat_name] = response.strip()
            except Exception as e:
                self.log(f"{cat_name}导语生成失败: {e}", "WARNING")
                intros[cat_name] = cat_desc

        return intros

    def _translate_and_format_items(self, categorized_trends: Dict[str, Dict]) -> Dict[str, List[Dict]]:
        """翻译并格式化所有热点条目（全中文）"""
        result = {}

        for key, cat_data in categorized_trends.items():
            items = cat_data.get("items", [])
            cat_name = self._get_category_name(key)

            formatted_items = []
            for item in items:
                # 翻译标题和描述
                translated = self._translate_item(item)
                formatted_items.append(translated)

            result[cat_name] = formatted_items

        return result

    def _translate_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """翻译单条热点（全中文）"""
        title = item.get("title", "")
        description = item.get("description", "")
        url = item.get("url", "")
        source = item.get("source", "")
        heat_score = item.get("heat_score", 0)

        # 使用LLM翻译标题和描述
        prompt = f"""你是一位专业科技媒体的翻译编辑。请将以下AI新闻翻译成流畅的中文。

原文标题：
{title}

原文描述：
{description}

翻译要求：
1. 标题：简洁有力，保留关键信息，符合中文新闻标题习惯
2. 描述：完整准确，语言流畅，不超过150字
3. 专业术语：如"Generative AI"译为"生成式AI"，"Large Language Model"译为"大语言模型"
4. 保留英文专有名词（如产品名、公司名）

请按以下格式返回：
标题：[中文标题]
描述：[中文描述]"""

        try:
            response = self._call_llm(prompt)

            # 解析翻译结果
            lines = response.strip().split('\n')
            zh_title = title
            zh_description = description

            for line in lines:
                if line.startswith("标题："):
                    zh_title = line.replace("标题：", "").strip()
                elif line.startswith("描述："):
                    zh_description = line.replace("描述：", "").strip()

            return {
                "title": zh_title,
                "description": zh_description,
                "url": url,
                "source": source,
                "heat_score": heat_score
            }
        except Exception as e:
            self.log(f"翻译失败: {e}", "WARNING")
            return {
                "title": title,
                "description": description,
                "url": url,
                "source": source,
                "heat_score": heat_score
            }

    def _assemble_full_content(
        self,
        today: datetime,
        issue_number: str,
        total_count: int,
        key_insights: List[str],
        deep_analysis: str,
        category_intros: Dict[str, str],
        translated_items: Dict[str, List[Dict]]
    ) -> str:
        """组装完整简报内容（全中文）"""
        parts = []

        # ========== 头部 ==========
        parts.append(f"# AI每日热点 · {today.strftime('%Y年%m月%d日')}\n\n")
        parts.append("> 📡 汇聚全球8大AI资讯源，每天为你精选最前沿的技术动态\n\n")
        parts.append(f"**本期共收录 {total_count} 条AI资讯**\n\n")
        parts.append(f"📅 {today.strftime('%Y年%m月%d日')}  ·  🆔 第 {issue_number} 期\n\n")
        parts.append("---\n\n")

        # ========== 核心洞察 ==========
        if key_insights:
            parts.append("## 💡 核心洞察\n\n")
            for insight in key_insights:
                parts.append(f"- {insight}\n")
            parts.append("\n---\n\n")

        # ========== 深度观察 ==========
        if deep_analysis:
            parts.append("## 📰 深度观察\n\n")
            parts.append(f"{deep_analysis}\n")
            parts.append("\n---\n\n")

        # ========== 分类热点 ==========
        parts.append("## 🔍 本期热点\n\n")

        # 按顺序输出各分类
        category_order = ["产业动态", "学术前沿", "技术创新", "产品工具", "行业应用"]

        for category in category_order:
            if category not in translated_items or not translated_items[category]:
                continue

            items = translated_items[category]

            # 获取icon
            icon = self._get_category_icon(category)

            parts.append(f"### {icon} {category}（{len(items)}条）\n\n")

            # 分类导语
            if category in category_intros:
                parts.append(f"*{category_intros[category]}*\n\n")

            # 该分类的所有热点
            for item in items:
                title = item.get("title", "")
                description = item.get("description", "")
                url = item.get("url", "")
                source = item.get("source", "")
                heat_score = item.get("heat_score", 0)

                parts.append(f"#### [{title}]({url})\n\n")
                parts.append(f"**来源**：{source}  ·  **热度**：{heat_score}\n\n")

                if description and len(description) > 20:
                    parts.append(f"{description}\n\n")

                parts.append("---\n\n")

        # ========== 数据来源 ==========
        parts.append("## 📊 数据来源\n\n")
        parts.append("本期数据来自以下全球AI资讯源：\n\n")
        parts.append("- **TechCrunch AI** - 硅谷科技媒体的AI风向标\n")
        parts.append("- **NewsAPI** - 全球AI新闻聚合平台\n")
        parts.append("- **arXiv** - 预印本论文库，学术前沿首发\n")
        parts.append("- **Hacker News** - 硅谷技术社区热议\n")
        parts.append("- **Product Hunt** - 全球AI产品发现平台\n")
        parts.append("- **GitHub Trending** - 开源AI项目趋势\n")
        parts.append("- **The Verge AI** - 深度技术报道\n")
        parts.append("- **VentureBeat AI** - AI商业资讯\n")

        parts.append("\n---\n\n")
        parts.append("<div align='center'>\n\n")
        parts.append("**AI每日热点** · 由 ContentForge AI 自动生成\n\n")
        parts.append(f"{today.strftime('%Y年%m月%d日')}\n\n")
        parts.append("</div>\n")

        return "".join(parts)

    def _get_category_name(self, key: str) -> str:
        """获取分类中文名"""
        mapping = {
            "📈 行业动态": "产业动态",
            "🎓 学术突破": "学术前沿",
            "🔬 技术创新": "技术创新",
            "🛠️ AI工具/产品": "产品工具",
            "💼 AI应用": "行业应用"
        }
        return mapping.get(key, key)

    def _get_key_by_name(self, name: str) -> str:
        """根据中文名获取key"""
        reverse_mapping = {
            "产业动态": "industry",
            "学术前沿": "academic",
            "技术创新": "tech",
            "产品工具": "product",
            "行业应用": "application"
        }
        return reverse_mapping.get(name, name)

    def _get_category_icon(self, name: str) -> str:
        """获取分类图标"""
        mapping = {
            "产业动态": "📈",
            "学术前沿": "🎓",
            "技术创新": "🔬",
            "产品工具": "🛠️",
            "行业应用": "💼"
        }
        return mapping.get(name, "📌")
