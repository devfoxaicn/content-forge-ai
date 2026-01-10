"""
长文本技术文章生成Agent
根据AI热点话题生成深度技术文章
"""

from typing import Dict, Any
import re
import json
from datetime import datetime
from src.agents.base import BaseAgent


class LongFormGeneratorAgent(BaseAgent):
    """长文本技术文章生成Agent"""

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        super().__init__(config, prompts)
        generator_config = config.get("agents", {}).get("longform_generator", {})
        self.article_length = generator_config.get("article_length", "long")  # 默认生成长文章
        self.technical_depth = generator_config.get("technical_depth", "advanced")  # 默认高级技术深度
        self.max_tokens = generator_config.get("max_tokens", 8000)  # 增加token限制以支持更长文章
        self.llm.max_tokens = self.max_tokens
        self.llm.temperature = 0.7  # 平衡创意和准确性
        self.mock_mode = config.get("agents", {}).get("ai_trend_analyzer", {}).get("mock_mode", False)

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成长文本技术文章（分阶段生成）

        Args:
            state: 当前工作流状态

        Returns:
            Dict[str, Any]: 更新后的状态
        """
        self.log("开始生成长文本技术文章（分阶段生成）")

        try:
            # 获取选中的热点话题
            selected_topic = state.get("selected_ai_topic")
            if not selected_topic:
                raise ValueError("没有找到选中的AI热点话题")

            self.log(f"基于热点生成文章: {selected_topic['title']}")

            # Mock模式或API调用失败时返回模拟数据
            if self.mock_mode:
                self.log("使用Mock模式生成模拟文章")
                article = self._generate_mock_article(selected_topic)
            else:
                # 分阶段生成
                article = self._generate_article_stages(state, selected_topic)

            self.log(f"成功生成技术文章，字数: {article['word_count']}")

            return {
                **state,
                "longform_article": article,
                "current_step": "longform_generator_completed"
            }
        except Exception as e:
            self.log(f"长文本生成失败: {str(e)}", "ERROR")
            # 失败时也返回模拟数据以便测试后续流程
            self.log("使用模拟数据继续测试", "WARNING")
            selected_topic = state.get("selected_ai_topic", {"title": "AI技术", "description": "技术热点"})
            article = self._generate_mock_article(selected_topic)
            return {
                **state,
                "longform_article": article,
                "current_step": "longform_generator_completed"
            }

    def _generate_article_stages(self, state: Dict[str, Any], topic_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        分阶段生成文章（避免超时）

        Args:
            state: 当前状态
            topic_data: 热点数据

        Returns:
            Dict[str, Any]: 文章数据
        """
        # 获取研究数据
        research_data = state.get("research_data", {})
        research_summary = state.get("research_summary", "")

        # 第一阶段：生成大纲
        self.log("第一阶段：生成文章大纲...")
        outline = self._generate_outline(state, topic_data, research_data)

        # 第二阶段：逐节展开
        self.log(f"第二阶段：展开 {len(outline.get('sections', []))} 个章节...")
        # 不在这里添加主标题，在最后统一添加
        full_content = ""
        sections_content = {}
        previous_sections = []  # 跟踪前面已生成的章节

        # 检查是否启用上下文窗口
        generator_config = self.config.get("agents", {}).get("longform_generator", {})
        enable_context = generator_config.get("enable_context_window", True)

        for idx, section in enumerate(outline.get('sections', []), 1):
            self.log(f"  正在生成第 {idx}/{len(outline.get('sections', []))} 节: {section.get('title', '')}")

            # 使用研究数据和上下文展开章节
            if enable_context:
                section_content = self._expand_section(section, research_data, topic_data, previous_sections)
            else:
                section_content = self._expand_section(section, research_data, topic_data)

            # 规范化章节标题，避免重复
            section_content = self._normalize_section_headers(section_content, section.get('title'))

            # 添加到完整内容
            full_content += f"{section_content}\n\n"
            sections_content[section.get('title')] = section_content

            # 记录当前章节标题，供下一节使用
            previous_sections.append(section.get('title'))

        # 第三阶段：生成总结
        self.log("第三阶段：生成总结...")
        summary = self._generate_summary(topic_data, outline, research_data)
        full_content += f"## {summary.get('title', '总结')}\n\n{summary.get('content', '')}\n\n"

        # 清理重复内容（在添加主标题之前）
        generator_config = self.config.get("agents", {}).get("longform_generator", {})
        if generator_config.get("enable_format_fix", True):
            full_content = self._clean_duplicate_content(full_content)

        # 在所有内容生成完成后，统一添加主标题
        if not full_content.startswith('#'):
            full_content = f"# {topic_data['title']}\n\n" + full_content

        # 添加元数据
        metadata = self._generate_metadata(topic_data, full_content, research_data)
        full_content += metadata

        # 计算字数
        word_count = len(full_content)

        return {
            "title": topic_data['title'],
            "full_content": full_content,
            "sections": sections_content,
            "word_count": word_count,
            "source_topic": topic_data['title'],
            "tags": topic_data.get('tags', []),
            "reading_time": f"{word_count // 400}-{word_count // 300}分钟",
            "outline": outline
        }

    def _generate_outline(self, state: Dict[str, Any], topic_data: Dict[str, Any], research_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成文章大纲

        Args:
            state: 当前状态
            topic_data: 热点数据
            research_data: 研究数据

        Returns:
            Dict[str, Any]: 大纲
        """
        # 构建研究背景
        research_info = ""
        if research_data and research_data.get("detailed_info"):
            details = research_data["detailed_info"]
            research_info = f"""
**技术背景**：{details.get('background', '')}

**核心特性**：{details.get('core_features', '')}

**应用场景**：{details.get('use_cases', '')}

**发展趋势**：{details.get('trends', '')}
"""

        prompt = f"""请为以下技术主题生成详细的文章大纲。

**主题**：{topic_data['title']}
**描述**：{topic_data.get('description', '')}

{research_info}

**要求**：
1. 大纲应包含 10-12 个主要章节（确保总字数达到1.5万字）
2. 每个章节要有明确的主题和要点
3. 注明每节的预计字数
4. 确保逻辑连贯、层次清晰
5. 重点章节（如核心技术解析、实践应用）应细分更多子节

**字数分配建议**：
- 引言: 600字
- 技术背景: 1000字
- 核心原理: 2000字
- 架构设计: 1800字
- 关键特性: 1800字
- 实践应用: 2000字
- 技术对比: 1200字
- 性能优化: 1000字
- 最佳实践: 1500字
- 未来展望: 1000字
- 总结: 500字
────────────────
总计: 约1.5万字

**输出格式（JSON）**：
{{
  "title": "主标题",
  "sections": [
    {{"title": "引言", "words": 600, "points": ["要点1", "要点2"]}},
    {{"title": "技术背景", "words": 1000, "points": ["要点1", "要点2"]}},
    {{"title": "核心原理", "words": 2000, "points": ["要点1", "要点2", "要点3"]}},
    ...
  ]
}}

请生成大纲：
"""

        try:
            response = self._call_llm(prompt)

            # 解析JSON响应
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                outline = json.loads(json_match.group())
                return outline

        except Exception as e:
            self.log(f"大纲生成失败，使用默认大纲: {str(e)}", "WARNING")

        # 返回默认大纲（1.5万字版本）
        return {
            "title": topic_data['title'],
            "sections": [
                {"title": "引言", "words": 600, "points": "引入话题，说明重要性，概述文章结构"},
                {"title": "技术背景", "words": 1000, "points": "发展历程、现状、挑战、为什么需要这项技术"},
                {"title": "核心原理", "words": 2000, "points": "底层原理、关键算法、技术细节"},
                {"title": "架构设计", "words": 1800, "points": "系统架构、模块设计、数据流向"},
                {"title": "关键特性", "words": 1800, "points": "核心功能、技术亮点、创新点"},
                {"title": "实践应用", "words": 2000, "points": "实际案例、部署方法、应用场景"},
                {"title": "技术对比", "words": 1200, "points": "与同类技术对比、优缺点分析"},
                {"title": "性能优化", "words": 1000, "points": "性能瓶颈、优化策略、最佳实践"},
                {"title": "最佳实践", "words": 1500, "points": "开发指南、避坑指南、推荐工具"},
                {"title": "未来展望", "words": 1000, "points": "发展趋势、机遇挑战、生态建设"},
                {"title": "总结", "words": 500, "points": "核心观点、行动建议、学习路径"}
            ]
        }

    def _expand_section(self, section: Dict[str, Any], research_data: Dict[str, Any],
                       topic_data: Dict[str, Any], previous_sections: list = None) -> str:
        """
        展开单个章节内容（增强版，包含上下文）

        Args:
            section: 章节信息
            research_data: 研究数据
            topic_data: 热点数据
            previous_sections: 前面章节的标题列表（用于保持连贯性）

        Returns:
            str: 章节内容
        """
        if previous_sections is None:
            previous_sections = []

        section_title = section.get('title', '')
        section_words = section.get('words', 500)
        section_points = section.get('points', '')

        # 构建上下文信息
        context = self._build_section_context(section_title, previous_sections, topic_data)

        # 根据章节类型选择不同的展开策略（注意：更具体的条件要放在前面）
        if '引言' in section_title:
            return self._expand_introduction(section_title, section_words, topic_data, context)
        elif '背景' in section_title:
            return self._expand_background(section_title, section_words, research_data, context)
        elif '对比' in section_title:
            return self._expand_comparison(section_title, section_words, research_data, topic_data, context)
        elif '核心' in section_title or '技术' in section_title:
            return self._expand_core_tech(section_title, section_words, research_data, topic_data, context)
        elif '实践' in section_title or '应用' in section_title:
            return self._expand_practice(section_title, section_words, research_data, topic_data, context)
        elif '未来' in section_title or '展望' in section_title:
            return self._expand_future(section_title, section_words, research_data, context)
        else:
            return self._expand_generic(section_title, section_words, section_points, research_data, context)

    def _expand_introduction(self, title: str, words: int, topic_data: Dict[str, Any],
                          context: str = "") -> str:
        """展开引言章节（增强版）"""
        # 如果words小于600，使用600作为默认值
        target_words = max(words, 600)
        prompt = f"""请撰写文章引言部分。

**文章主题**：{topic_data['title']}

{context}

**要求字数**：{target_words}字

**内容要求**：
1. 用引人入胜的开场白，吸引读者注意
2. 介绍技术背景和重要性
3. 点明文章将讨论的核心问题
4. 简要概述文章结构（将在哪些方面展开）

请撰写引言（{target_words}字）：
"""

        response = self._call_llm(prompt)
        return response.strip()

    def _expand_background(self, title: str, words: int, research_data: Dict[str, Any],
                          context: str = "") -> str:
        """展开技术背景章节（增强版）"""
        background = research_data.get("detailed_info", {}).get("background", "")
        # 如果words小于1000，使用1000作为默认值
        target_words = max(words, 1000)

        prompt = f"""请撰写技术背景部分。

{context}

**背景资料**：{background}

**要求字数**：{target_words}字

**内容要求**：
1. 相关技术的发展历程
2. 当前技术现状和竞争格局
3. 面临的挑战或问题
4. 为什么需要这项技术

请撰写技术背景（{target_words}字）：
"""

        response = self._call_llm(prompt)
        return response.strip()

    def _expand_core_tech(self, title: str, words: int, research_data: Dict[str, Any],
                          topic_data: Dict[str, Any], context: str = "") -> str:
        """展开核心技术解析章节（最重要）- 进一步细分为4-5个子节以避免超时"""
        details = research_data.get("detailed_info", {})
        self.log(f"    [{title}] 分为4个子节生成，避免超时...")

        # 定义4个子节，每个500-700字，总计约2000-2800字
        subsections = [
            {
                "subtitle": "技术架构与原理",
                "focus": details.get('core_features', ''),
                "words": 700,
                "content": """
1. 整体架构设计
2. 核心组件和模块
3. 工作流程和数据流
4. 关键技术原理
"""
            },
            {
                "subtitle": "关键特性详解",
                "focus": details.get('specs', ''),
                "words": 700,
                "content": """
1. 主要功能特性
2. 性能指标和规格
3. 技术优势和创新点
4. 适用场景分析
"""
            },
            {
                "subtitle": "核心算法与实现",
                "focus": details.get('core_features', ''),
                "words": 700,
                "content": """
1. 核心算法原理
2. 关键数据结构
3. 实现细节分析
4. 代码示例与解析
"""
            },
            {
                "subtitle": "技术对比与选型",
                "focus": details.get('pros_cons', ''),
                "words": 600,
                "content": """
1. 与同类技术对比
2. 优缺点分析
3. 使用场景选型建议
4. 迁移注意事项
"""
            }
        ]

        # 不在这里添加章节标题，由上层的 _normalize_section_headers 统一处理
        full_content = ""

        for idx, subsection in enumerate(subsections, 1):
            self.log(f"      正在生成子节 {idx}/3: {subsection['subtitle']}")

            prompt = f"""请撰写核心技术解析的子章节：{subsection['subtitle']}

**主题**：{topic_data['title']}

{context}

**参考资料**：{subsection['focus']}

**要求字数**：{subsection['words']}字

**内容要求**：{subsection['content']}

请使用Markdown格式，包含代码块、表格等（{subsection['words']}字，要求详细专业）：
"""

            response = self._call_llm(prompt)
            # 子节标题使用 ####，但检查内容是否已包含标题
            subsection_content = response.strip()
            if not subsection_content.startswith('#'):
                subsection_content = f"#### {idx}. {subsection['subtitle']}\n\n{subsection_content}"
            full_content += f"\n{subsection_content}\n\n"

        return full_content

    def _expand_practice(self, title: str, words: int, research_data: Dict[str, Any],
                         topic_data: Dict[str, Any], context: str = "") -> str:
        """展开实践应用章节 - 分为3个子节以避免超时"""
        use_cases = research_data.get("detailed_info", {}).get("use_cases", "")
        self.log(f"    [{title}] 分为3个子节生成，避免超时...")

        # 定义3个子节，每个600-700字，总计约2000字
        subsections = [
            {
                "subtitle": "应用场景与案例",
                "words": 700,
                "content": f"""
1. 主要应用场景分析
2. 真实案例详细解析（至少2个）
3. 应用效果和成果展示
4. ROI分析

**参考资料**：{use_cases}
"""
            },
            {
                "subtitle": "实施指南与部署方法",
                "words": 700,
                "content": """
1. 环境准备和前置条件
2. 详细实施步骤
3. 部署方法和配置说明
4. 验证和测试方法
"""
            },
            {
                "subtitle": "最佳实践与避坑指南",
                "words": 600,
                "content": """
1. 生产环境最佳实践
2. 常见问题和解决方案
3. 性能优化建议
4. 推荐工具和资源
"""
            }
        ]

        # 不在这里添加章节标题，由上层的 _normalize_section_headers 统一处理
        full_content = ""

        for idx, subsection in enumerate(subsections, 1):
            self.log(f"      正在生成子节 {idx}/2: {subsection['subtitle']}")

            prompt = f"""请撰写实践应用的子章节：{subsection['subtitle']}

**主题**：{topic_data['title']}

{context}

{subsection['content']}

**要求字数**：{subsection['words']}字

请撰写实践应用子章节（{subsection['words']}字，要求实用专业）：
"""

            response = self._call_llm(prompt)
            # 子节标题使用 ####，但检查内容是否已包含标题
            subsection_content = response.strip()
            if not subsection_content.startswith('#'):
                subsection_content = f"#### {idx}. {subsection['subtitle']}\n\n{subsection_content}"
            full_content += f"\n{subsection_content}\n\n"

        return full_content

    def _expand_comparison(self, title: str, words: int, research_data: Dict[str, Any],
                          topic_data: Dict[str, Any], context: str = "") -> str:
        """展开技术对比章节（增强版）"""
        # 如果words小于1200，使用1200作为默认值
        target_words = max(words, 1200)
        prompt = f"""请撰写技术对比部分。

**主题**：{topic_data['title']}

{context}

**要求字数**：{target_words}字

**内容要求**：
1. 与同类技术的详细对比
2. 不同场景下的选型建议
3. 迁移路径和注意事项
4. 使用对比表格展示

请撰写技术对比（{target_words}字）：
"""

        response = self._call_llm(prompt)
        return response.strip()

    def _expand_future(self, title: str, words: int, research_data: Dict[str, Any],
                       context: str = "") -> str:
        """展开未来展望章节（增强版）"""
        trends = research_data.get("detailed_info", {}).get("trends", "")
        # 如果words小于1000，使用1000作为默认值
        target_words = max(words, 1000)

        prompt = f"""请撰写未来展望部分。

{context}

**发展趋势**：{trends}

**要求字数**：{target_words}字

**内容要求**：
1. 分析技术发展趋势
2. 讨论潜在的改进方向
3. 预测对行业的影响
4. 面临的挑战与机遇
5. 生态建设展望

请撰写未来展望（{target_words}字）：
"""

        response = self._call_llm(prompt)
        return response.strip()

    def _expand_generic(self, title: str, words: int, points: str, research_data: Dict[str, Any],
                        context: str = "") -> str:
        """展开通用章节（增强版）"""
        # 确保最小字数为800字
        target_words = max(words, 800)
        prompt = f"""请撰写"{title}"章节。

{context}

**要点**：{points}

**要求字数**：{target_words}字

请撰写该章节内容（{target_words}字，要求详细专业）：
"""

        response = self._call_llm(prompt)
        return response.strip()

    def _generate_summary(self, topic_data: Dict[str, Any], outline: Dict[str, Any], research_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成总结章节

        Args:
            topic_data: 热点数据
            outline: 文章大纲
            research_data: 研究数据

        Returns:
            Dict[str, Any]: 总结内容
        """
        trends = research_data.get("detailed_info", {}).get("trends", "")

        prompt = f"""请撰写文章总结部分。

**主题**：{topic_data['title']}

**发展趋势**：{trends}

**要求**：
1. 总结核心观点和关键洞察
2. 给不同角色读者（开发者、企业决策者、投资者）的具体建议
3. 500字左右
4. 提供学习路径和行动指南

请撰写总结（500字）：
"""

        response = self._call_llm(prompt)

        return {
            "title": "总结",
            "content": response.strip()
        }

    def _generate_metadata(self, topic_data: Dict[str, Any], content: str, research_data: Dict[str, Any]) -> str:
        """生成元数据"""
        word_count = len(content)
        tags = ', '.join(topic_data.get('tags', ['AI', '技术']))
        date_str = datetime.now().strftime("%Y-%m-%d")

        return f"""
---

**关于作者**：本文由ContentForge AI自动生成，基于最新的AI技术热点分析。

**延伸阅读**：
- 官方文档和GitHub仓库
- 社区最佳实践案例
- 相关技术论文和研究报告

**互动交流**：欢迎在评论区分享你的观点和经验，让我们一起探讨技术的未来！

---

📌 **关键词**：{tags}

📅 **发布日期**：{date_str}

🔖 **字数统计**：约{word_count}字

⏱️ **阅读时间**：{word_count // 400}-{word_count // 300}分钟


---
**元数据**:
- 字数: {word_count}
- 阅读时间: {word_count // 400}-{word_count // 300}分钟
- 来源热点: {topic_data['title']}
- 标签: {tags}
- 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

    def _normalize_section_headers(self, content: str, section_title: str) -> str:
        """
        规范化章节标题，避免重复

        Args:
            content: 原始章节内容
            section_title: 章节标题

        Returns:
            str: 规范化后的内容
        """
        lines = content.split('\n')
        normalized = []
        title_added = False

        for line in lines:
            # 检查是否是标题行
            if line.strip().startswith('#'):
                # 如果内容已经以标题开头，不再添加章节标题
                title_added = True
                # 规范化标题层级（移除多余的#）
                normalized.append(self._fix_heading_level(line))
            elif not title_added and line.strip():
                # 第一行非标题内容前添加章节标题
                normalized.append(f"## {section_title}")
                normalized.append("")
                normalized.append(line)
                title_added = True
            else:
                normalized.append(line)

        # 如果整个section都是空的，添加标题
        if not title_added:
            normalized.append(f"## {section_title}")
            normalized.append("")

        return '\n'.join(normalized)

    def _fix_heading_level(self, line: str) -> str:
        """
        修复标题层级

        Args:
            line: 标题行

        Returns:
            str: 修复后的标题行
        """
        stripped = line.strip()
        if not stripped.startswith('#'):
            return stripped

        # 计算当前层级
        level = 0
        for char in stripped:
            if char == '#':
                level += 1
            else:
                break

        # 规范化规则
        # - 如果层级大于4（#####），降级到####
        # - 移除emoji后的多余空格
        content = stripped[level:].strip()

        if level > 4:
            level = 4

        return '#' * level + ' ' + content

    def _clean_duplicate_content(self, content: str) -> str:
        """
        清理重复内容（如重复的标题、段落）

        Args:
            content: 原始内容

        Returns:
            str: 清理后的内容
        """
        lines = content.split('\n')
        cleaned = []
        seen_titles = set()

        for line in lines:
            # 检测标题行
            if line.strip().startswith('#'):
                # 提取标题文本（去除#和emoji）
                title_text = line.strip()
                for prefix in ['#', '##', '###', '####', '#####']:
                    if title_text.startswith(prefix):
                        title_text = title_text[len(prefix):].strip()
                        break

                # 移除emoji（用于比较）
                title_text_clean = re.sub(r'[^\w\s\u4e00-\u9fff]', '', title_text)

                # 检查是否重复
                if title_text_clean and title_text_clean not in seen_titles:
                    seen_titles.add(title_text_clean)
                    cleaned.append(line)
                # 跳过重复的标题
            else:
                cleaned.append(line)

        return '\n'.join(cleaned)

    def _build_section_context(self, current_title: str, previous_sections: list,
                              topic_data: Dict[str, Any]) -> str:
        """
        构建章节上下文信息，用于保持内容连贯性

        Args:
            current_title: 当前章节标题
            previous_sections: 前面章节的标题列表
            topic_data: 主题数据

        Returns:
            str: 上下文信息
        """
        context_parts = []

        # 添加文章整体主题
        context_parts.append(f"**文章主题**：{topic_data['title']}")
        context_parts.append(f"**主题描述**：{topic_data.get('description', '')}")

        # 添加前面章节概要
        if previous_sections:
            context_parts.append(f"\n**已讨论的章节**：")
            for i, prev_title in enumerate(previous_sections, 1):
                context_parts.append(f"{i}. {prev_title}")

        # 添加当前章节在文章中的位置
        if len(previous_sections) == 0:
            position = "第一个章节"
        elif len(previous_sections) == 1:
            position = "第二个章节"
        elif len(previous_sections) == 2:
            position = "第三个章节"
        else:
            position = f"第{len(previous_sections) + 1}个章节"

        context_parts.append(f"\n**当前章节位置**：{position}")

        # 添加连贯性要求
        if previous_sections:
            last_section = previous_sections[-1]
            context_parts.append(f"\n**连贯性要求**：")
            context_parts.append(f"- 上一节讨论了：{last_section}")
            context_parts.append(f"- 本节要自然承接上一节的内容")
            context_parts.append(f"- 避免重复前面已经详细讨论的内容")
            context_parts.append(f"- 可以引用前面提到的概念（用'如前所述'、'前面提到'等连接词）")

        return "\n".join(context_parts)

    def _build_prompt(self, state: Dict[str, Any], topic_data: Dict[str, Any]) -> str:
        """
        构建生成提示词

        Args:
            state: 当前状态
            topic_data: 热点话题数据

        Returns:
            str: 提示词
        """
        # 调试：检查研究数据是否存在
        research_data = state.get("research_data", {})
        research_summary = state.get("research_summary", "")
        if research_data:
            self.log(f"研究数据存在: {len(research_data.get('search_results', []))} 条搜索结果, {len(research_data.get('official_docs', []))} 个官方文档")
            if research_data.get("detailed_info"):
                self.log(f"详细分析存在: {list(research_data.get('detailed_info', {}).keys())}")
        else:
            self.log("警告: 研究数据为空，将使用通用内容", "WARNING")

        prompts = self.prompts.get("prompts", {})
        prompt_template = prompts.get("longform_generator", {}).get("user", "")

        # 构建热点话题描述
        topic_desc = f"""
标题：{topic_data['title']}
描述：{topic_data['description']}
来源：{topic_data.get('source', '未知')}
关键信息：
{chr(10).join([f"- {point}" for point in topic_data.get('key_points', [])])}
标签：{', '.join(topic_data.get('tags', []))}
"""

        # 获取研究数据（ResearchAgent提供）
        research_data = state.get("research_data", {})
        research_summary = state.get("research_summary", "")

        # 构建研究背景部分
        research_context = ""
        if research_data:
            detailed_info = research_data.get("detailed_info", {})

            research_context = f"""
**深度研究资料**：

### 技术背景
{detailed_info.get('background', '暂无技术背景信息')}

### 核心特性
{detailed_info.get('core_features', '暂无核心特性信息')}

### 技术规格
{detailed_info.get('specs', '暂无技术规格信息')}

### 应用场景
{detailed_info.get('use_cases', '暂无应用场景信息')}

### 优缺点分析
{detailed_info.get('pros_cons', '暂无优缺点分析')}

### 发展趋势
{detailed_info.get('trends', '暂无发展趋势信息')}
"""

            # 添加收集到的资料统计
            if research_data.get("search_results"):
                research_context += f"""
### 参考资料来源
- 搜索结果：{len(research_data.get('search_results', []))} 条
- 官方文档：{len(research_data.get('official_docs', []))} 个
- GitHub项目：{len(research_data.get('github_repos', []))} 个
- 技术文章：{len(research_data.get('technical_articles', []))} 篇
"""

        # 获取目标受众
        target_audience = state.get("target_audience", "技术从业者")

        if prompt_template:
            return prompt_template.format(
                topic_desc=topic_desc.strip(),
                research_context=research_context.strip(),
                research_summary=research_summary.strip(),
                target_audience=target_audience,
                article_length=self.article_length,
                technical_depth=self.technical_depth
            )
        else:
            # 使用默认提示词（简化版，更直接）
            research_info = ""
            if research_data and research_data.get("detailed_info"):
                details = research_data["detailed_info"]
                research_info = f"""
**技术背景**：{details.get('background', '')}

**核心特性**：{details.get('core_features', '')}

**技术规格**：{details.get('specs', '')}

**应用场景**：{details.get('use_cases', '')}

**优缺点**：{details.get('pros_cons', '')}

**发展趋势**：{details.get('trends', '')}
"""

            return f"""请撰写一篇深度技术文章，字数必须达到1.5万字左右。

**主题**：{topic_data['title']}

**研究资料**：
{research_info}

**文章结构**（按顺序展开，每部分都要详细）：

## 引言（600字）
引入话题，说明重要性，概述文章结构

## 技术背景（1000字）
发展历程、现状、挑战、为什么需要这项技术

## 核心原理（2000字）
底层原理、关键算法、技术细节

## 架构设计（1800字）
系统架构、模块设计、数据流向

## 关键特性（1800字）
核心功能、技术亮点、创新点

## 实践应用（2000字）
实际案例、部署方法、应用场景、最佳实践

## 技术对比（1200字）
与同类技术对比、优缺点分析、选型建议

## 性能优化（1000字）
性能瓶颈、优化策略、调优方法

## 最佳实践（1500字）
开发指南、避坑指南、推荐工具

## 未来展望（1000字）
发展趋势、机遇挑战、生态建设

## 总结（500字）
核心观点、行动建议、学习路径

**要求**：
1. 使用Markdown格式
2. 每个部分都要深入展开，不要简略
3. 使用研究资料中的具体信息
4. 包含代码示例和实际案例（如适用）
5. 总字数1.5万字左右

请开始撰写：
"""

    def _parse_article(self, response: str, topic_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析生成的文章

        Args:
            response: LLM响应
            topic_data: 热点话题数据

        Returns:
            Dict[str, Any]: 结构化文章
        """
        # 提取标题
        title_match = re.search(r'^#\s+(.+)$', response, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else topic_data['title']

        # 提取各部分内容
        sections = {}
        section_patterns = {
            'introduction': r'##\s*[引言|前言|简介|Introduction][^\n]*\n+(.+?)(?=##\s|$)',
            'background': r'##\s*[技术背景|背景|Background][^\n]*\n+(.+?)(?=##\s|$)',
            'core_analysis': r'##\s*[核心解析|技术解析|深度分析|Core Analysis][^\n]*\n+(.+?)(?=##\s|$)',
            'practice': r'##\s*[实践应用|应用案例|最佳实践|Practice][^\n]*\n+(.+?)(?=##\s|$)',
            'future': r'##\s*[未来展望|发展趋势|Future][^\n]*\n+(.+?)(?=##\s|$)',
            'summary': r'##\s*[总结|结语|Summary][^\n]*\n+(.+?)(?=##\s|$|$)'
        }

        for key, pattern in section_patterns.items():
            match = re.search(pattern, response, re.MULTILINE | re.DOTALL)
            if match:
                sections[key] = match.group(1).strip()

        # 计算字数
        word_count = len(response)

        return {
            "title": title,
            "full_content": response,
            "sections": sections,
            "word_count": word_count,
            "source_topic": topic_data['title'],
            "tags": topic_data.get('tags', []),
            "reading_time": f"{word_count // 400}-{word_count // 300}分钟"
        }

    def _generate_mock_article(self, topic_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成模拟文章（用于Mock模式或API失败时）

        Args:
            topic_data: 热点话题数据

        Returns:
            Dict[str, Any]: 结构化文章
        """
        title = topic_data.get('title', 'AI技术突破')
        description = topic_data.get('description', 'AI领域出现重大技术突破')

        # 生成模拟的微信格式文章
        mock_content = f"""# {title}：深度解析技术突破与未来趋势

## 引言

近期，{title}成为技术圈热议的焦点。这一技术突破不仅引发了开发者社区的广泛关注，更预示着整个行业即将迎来新的变革。本文将从技术背景、核心突破、实践应用等多个维度，深入解析这一热点话题。

## 技术背景

{description}

在过去几年中，该领域经历了快速的发展。从最初的理论探索，到如今的技术成熟，我们已经见证了多个关键节点的突破。这些进展为当前的成果奠定了坚实的基础。

### 技术演进历程

- **初期阶段**：基础理论研究和概念验证
- **发展阶段**：技术迭代和性能优化
- **成熟阶段**：大规模应用和生态建设

## 核心技术解析

### 关键创新点

这一突破的核心在于以下几个方面：

#### 1. 性能大幅提升

通过创新的算法设计和架构优化，新技术在多个关键指标上实现了显著提升：

- 处理速度提升3-5倍
- 资源消耗降低40%-60%
- 准确率提高15%-25%

#### 2. 易用性显著改善

技术门槛的降低使得更多开发者能够快速上手：

- 简洁的API设计
- 完善的文档和示例
- 活跃的社区支持

#### 3. 应用场景不断扩展

从传统的应用场景，到新兴的领域探索，技术边界正在被不断突破：

```
传统应用 → 性能优化 → 场景扩展 → 生态繁荣
```

## 实践应用

### 真实案例分享

让我们通过几个实际案例，了解这一技术在生产环境中的应用价值。

#### 案例1：互联网企业

某知名互联网企业在引入该技术后，实现了：

- 系统响应时间缩短50%
- 运维成本降低30%
- 用户满意度提升20%

#### 案例2：创业公司

一家AI创业公司基于该技术开发新产品，在3个月内：

- 完成产品MVP开发
- 获得种子轮融资
- 积累首批1000+付费用户

### 最佳实践建议

基于实际应用经验，我们总结出以下最佳实践：

1. **渐进式迁移**：先从非核心业务开始试点
2. **充分测试**：建立完善的测试体系
3. **团队培训**：确保团队掌握相关技能
4. **持续优化**：根据反馈不断改进

## 未来展望

### 发展趋势

基于当前的技术发展轨迹，我们可以预见以下几个趋势：

1. **技术融合**：与其他技术栈的深度整合
2. **标准化**：行业标准逐渐形成
3. **普及化**：从早期采用者走向大众市场
4. **智能化**：引入更多AI能力

### 挑战与机遇

虽然前景广阔，但仍面临一些挑战：

- **技术挑战**：性能优化、稳定性提升
- **生态挑战**：工具链完善、人才培养
- **商业挑战**：商业模式探索、市场教育

同时，这些挑战也孕育着巨大的机遇：

- 新的工具和服务需求
- 专业咨询服务市场
- 培训和认证体系建立

## 总结

{title}标志着技术发展进入了一个新的阶段。对于开发者和企业来说，现在正是关注和布局的最佳时机。

**关键要点**：

1. ✅ 技术已经相对成熟，可以投入生产使用
2. ✅ 社区活跃，文档和资源丰富
3. ✅ 实际案例证明了其商业价值
4. ⚠️ 需要根据具体场景进行评估和选择
5. 🚀 未来发展空间巨大，值得关注

**行动建议**：

- 对开发者：学习并掌握这项技术
- 对企业：评估并开展小规模试点
- 对投资者：关注相关领域的创业机会

---

**关于作者**：本文由ContentForge AI自动生成，基于最新的AI技术热点分析。

**延伸阅读**：
- 官方文档和GitHub仓库
- 社区最佳实践案例
- 相关技术论文和研究报告

**互动交流**：欢迎在评论区分享你的观点和经验，让我们一起探讨技术的未来！

---

📌 **关键词**：{', '.join(topic_data.get('tags', ['AI', '技术', '创新']))}

📅 **发布日期**：2026-01-07

🔖 **字数统计**：约2800字

⏱️ **阅读时间**：8-10分钟
"""

        word_count = len(mock_content)

        return {
            "title": title,
            "full_content": mock_content,
            "sections": {
                "introduction": "引言部分内容...",
                "background": "技术背景部分内容...",
                "core_analysis": "核心技术解析部分内容...",
                "practice": "实践应用部分内容...",
                "future": "未来展望部分内容...",
                "summary": "总结部分内容..."
            },
            "word_count": word_count,
            "source_topic": title,
            "tags": topic_data.get('tags', ['AI', '技术']),
            "reading_time": f"{word_count // 400}-{word_count // 300}分钟"
        }
