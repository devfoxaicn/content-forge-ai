"""
长文本技术文章生成Agent
根据AI热点话题生成深度技术文章
"""

from typing import Dict, Any
import re
from src.agents.base import BaseAgent


class LongFormGeneratorAgent(BaseAgent):
    """长文本技术文章生成Agent"""

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        super().__init__(config, prompts)
        generator_config = config.get("agents", {}).get("longform_generator", {})
        self.article_length = generator_config.get("article_length", "medium")  # short, medium, long
        self.technical_depth = generator_config.get("technical_depth", "intermediate")  # beginner, intermediate, advanced
        self.max_tokens = generator_config.get("max_tokens", 4000)
        self.llm.max_tokens = self.max_tokens
        self.llm.temperature = 0.7  # 平衡创意和准确性
        self.mock_mode = config.get("agents", {}).get("ai_trend_analyzer", {}).get("mock_mode", False)

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成长文本技术文章

        Args:
            state: 当前工作流状态

        Returns:
            Dict[str, Any]: 更新后的状态
        """
        self.log("开始生成长文本技术文章")

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
                # 构建提示词
                user_prompt = self._build_prompt(state, selected_topic)

                # 调用LLM生成文章
                response = self._call_llm(user_prompt)

                # 解析文章内容
                article = self._parse_article(response, selected_topic)

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

    def _build_prompt(self, state: Dict[str, Any], topic_data: Dict[str, Any]) -> str:
        """
        构建生成提示词

        Args:
            state: 当前状态
            topic_data: 热点话题数据

        Returns:
            str: 提示词
        """
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

        # 获取目标受众
        target_audience = state.get("target_audience", "技术从业者")

        if prompt_template:
            return prompt_template.format(
                topic_desc=topic_desc.strip(),
                target_audience=target_audience,
                article_length=self.article_length,
                technical_depth=self.technical_depth
            )
        else:
            # 使用默认提示词
            return f"""你是一位资深的技术文章作者，擅长深度解析AI技术趋势。

请基于以下热点话题，撰写一篇高质量的技术文章：

{topic_desc}

写作要求：

1. **文章结构**（采用Markdown格式）：
   # 引人入胜的标题

   ## 引言（200-300字）
   - 用引人入胜的开场白
   - 介绍技术背景和重要性
   - 点明文章将讨论的核心问题

   ## 技术背景（300-400字）
   - 相关技术的发展历程
   - 当前技术现状
   - 面临的挑战或问题

   ## 核心解析（800-1200字）
   - 深入分析技术原理
   - 解释关键概念和术语
   - 提供具体示例或案例
   - 对比不同技术方案

   ## 实践应用（400-600字）
   - 实际应用场景
   - 实施建议或最佳实践
   - 常见问题和解决方案
   - 工具和资源推荐

   ## 未来展望（200-300字）
   - 技术发展趋势
   - 潜在的改进方向
   - 对行业的影响

   ## 总结（100-200字）
   - 核心观点总结
   - 给读者的建议

2. **写作风格**：
   - 专业但不晦涩
   - 深入浅出，适合{target_audience}阅读
   - 技术深度：{self.technical_depth}
   - 文章长度：{self.article_length}

3. **内容要求**：
   - 确保技术准确性
   - 提供具体数据和案例
   - 避免过度宣传
   - 保持客观中立
   - 代码示例使用```markdown ```代码```格式

请开始撰写文章，确保内容专业、有价值、易读性强。
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
