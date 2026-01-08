"""
小红书笔记精炼Agent
将长文本技术文章精炼成小红书风格的笔记
"""

from typing import Dict, Any
import re
from src.agents.base import BaseAgent


class XiaohongshuRefinerAgent(BaseAgent):
    """小红书笔记精炼Agent"""

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        super().__init__(config, prompts)
        refiner_config = config.get("agents", {}).get("xiaohongshu_refiner", {})
        self.style = refiner_config.get("style", "professional")  # professional, casual, humorous
        self.content_density = refiner_config.get("content_density", "medium")  # light, medium, dense
        self.max_tokens = refiner_config.get("max_tokens", 2000)
        self.llm.max_tokens = self.max_tokens
        self.llm.temperature = 0.8  # 适应小红书风格，需要一些灵活性
        self.mock_mode = config.get("agents", {}).get("ai_trend_analyzer", {}).get("mock_mode", False)

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        精炼技术文章为小红书笔记

        Args:
            state: 当前工作流状态

        Returns:
            Dict[str, Any]: 更新后的状态
        """
        self.log("开始精炼技术文章为小红书笔记")

        try:
            # 获取长文本文章
            article = state.get("longform_article")
            if not article:
                raise ValueError("没有找到长文本文章")

            self.log(f"精炼文章: {article['title']}")

            # Mock模式或API失败时生成模拟数据
            if self.mock_mode:
                self.log("使用Mock模式生成小红书笔记")
                xhs_note = self._generate_mock_note(article)
            else:
                # 构建提示词
                user_prompt = self._build_prompt(state, article)

                # 调用LLM精炼内容
                response = self._call_llm(user_prompt)

                # 解析小红书笔记
                xhs_note = self._parse_xiaohongshu_note(response, article)

            self.log(f"成功生成小红书笔记，字数: {xhs_note['word_count']}")

            return {
                **state,
                "xiaohongshu_note": xhs_note,
                "refined_content": xhs_note,  # 兼容后续Agent
                "generated_content": xhs_note,  # 兼容后续Agent
                "current_step": "xiaohongshu_refiner_completed"
            }
        except Exception as e:
            self.log(f"笔记精炼失败: {str(e)}", "ERROR")
            # 失败时也返回模拟数据
            self.log("使用模拟数据继续测试", "WARNING")
            article = state.get("longform_article", {"title": "AI技术", "full_content": "内容"})
            xhs_note = self._generate_mock_note(article)
            return {
                **state,
                "xiaohongshu_note": xhs_note,
                "refined_content": xhs_note,
                "generated_content": xhs_note,
                "current_step": "xiaohongshu_refiner_completed"
            }

    def _build_prompt(self, state: Dict[str, Any], article: Dict[str, Any]) -> str:
        """
        构建精炼提示词

        Args:
            state: 当前状态
            article: 长文本文章

        Returns:
            str: 提示词
        """
        prompts = self.prompts.get("prompts", {})
        prompt_template = prompts.get("xiaohongshu_refiner", {}).get("user", "")

        # 提取文章核心内容（前2000字）
        content_preview = article['full_content'][:2000] + "..." if len(article['full_content']) > 2000 else article['full_content']

        # 获取目标受众和风格
        target_audience = state.get("target_audience", "技术从业者")
        topic = state.get("topic", "AI工具")

        if prompt_template:
            return prompt_template.format(
                article_title=article['title'],
                article_content=content_preview,
                target_audience=target_audience,
                style=self.style,
                content_density=self.content_density
            )
        else:
            # 使用默认提示词
            return f"""你是一位专业的小红书内容创作者，擅长将技术文章转化为受欢迎的小红书笔记。

请将以下技术文章精炼为小红书风格的笔记：

**原文章标题**：{article['title']}

**原文章内容**：
{content_preview}

**精炼要求**：

1. **内容提取与重组**：
   - 提取最核心、最有价值的3-5个要点
   - 每个要点用简练的语言总结（50-100字）
   - 保留关键数据、技术名词、工具名称

2. **小红书风格要求**：
   - **开头**：用痛点、反常识观点、或惊艳数据开场（3-5句话）
   - **正文**：
     * 用emoji分隔各个要点（每段开头加emoji）
     * 使用简洁的标题（如"✨ 核心特性"、"💡 实践建议"）
     * 每个要点控制在2-4句话
     * 适当使用"亲测"、"必看"、"干货"等小红书常用词
   - **结尾**：总结价值+行动号召（引导点赞、收藏、评论）

3. **内容密度**：{self.content_density}
   - light: 简洁易懂，适合快速阅读
   - medium: 适中平衡，保留核心细节
   - dense: 信息量大，适合深度阅读

4. **写作风格**：{self.style}
   - professional: 专业但不枯燥
   - casual: 轻松友好，像朋友分享
   - humorous: 幽默风趣，增加趣味性

5. **输出格式**：
```markdown
# 吸引人的小红书标题（20字内，可用emoji）

🔥 引人入胜的开场

## ✨ 核心要点1
简洁描述...

## 💡 核心要点2
简洁描述...

## 🎯 实践建议
可操作的建议...

## 💬 总结
简短总结+行动号召

---
标签：#标签1 #标签2 #标签3 #标签4 #标签5
```

6. **重要注意事项**：
   - 不要直接复制原文句子，必须用自己的话重新表达
   - 保留关键技术术语和数据
   - 使用emoji增强可读性，但不要过度（整个笔记5-10个emoji为宜）
   - 标题要有吸引力但不标题党
   - 标签要精准且热门（从原文章标签中选择3-5个）

**目标受众**：{target_audience}

**主题**：{topic}

请开始精炼，确保内容既专业又符合小红书调性！
"""

    def _parse_xiaohongshu_note(self, response: str, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析小红书笔记

        Args:
            response: LLM响应
            article: 原文章

        Returns:
            Dict[str, Any]: 结构化笔记
        """
        # 提取标题
        title_match = re.search(r'^#\s+(.+)$', response, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else article['title']

        # 提取开篇
        intro_match = re.search(r'^#\s+.+?\n+(.+?)(?=^##\s|\Z)', response, re.MULTILINE | re.DOTALL)
        intro = intro_match.group(1).strip() if intro_match else ""

        # 提取正文
        body_parts = []
        for match in re.finditer(r'^##\s+(.+?)\n+(.+?)(?=^##\s|^###\s|---|\Z)', response, re.MULTILINE | re.DOTALL):
            section_title = match.group(1).strip()
            section_content = match.group(2).strip()
            body_parts.append(f"## {section_title}\n{section_content}")
        body = "\n\n".join(body_parts)

        # 提取结尾
        ending_match = re.search(r'---\n+(.+?)(?=\n\n#|$)', response, re.DOTALL)
        ending = ending_match.group(1).strip() if ending_match else ""

        # 提取标签
        hashtags = []
        hashtag_match = re.search(r'标签[:：](.+)', response)
        if hashtag_match:
            hashtag_text = hashtag_match.group(1).strip()
            hashtags = [tag.strip() for tag in re.findall(r'#[\w\u4e00-\u9fff]+', hashtag_text)]

        if not hashtags:
            hashtags = re.findall(r'#[\w\u4e00-\u9fff]+', response)

        # 继承原文章的标签
        original_tags = article.get('tags', [])
        all_tags = list(set(hashtags + [f"#{tag}" for tag in original_tags]))[:5]

        # 计算字数
        word_count = len(response)

        return {
            "title": title,
            "intro": intro,
            "body": body,
            "ending": ending,
            "full_content": response,
            "hashtags": all_tags,
            "word_count": word_count,
            "original_article_word_count": article.get('word_count', 0),
            "compression_ratio": f"{(1 - word_count / article.get('word_count', 1)) * 100:.1f}%"
        }

    def _generate_mock_note(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成模拟小红书笔记（用于Mock模式或API失败时）

        Args:
            article: 长文本文章

        Returns:
            Dict[str, Any]: 小红书笔记
        """
        title = article.get('title', 'AI技术突破')
        # 提取文章标题的主标题部分（去掉冒号后的内容）
        main_title = title.split('：')[0].split(':')[0]

        # 生成完整的mock内容（以标题开头）
        mock_content = f"""{main_title}✨最新技术突破太惊艳了！

姐妹们👋，今天给大家分享一个超级震撼的AI技术突破！

🔥 **核心亮点**

1️⃣ 性能大幅提升
- 处理速度提升3-5倍⚡
- 资源消耗降低40%-60%💪
- 准确率提高15%-25%📈

2️⃣ 超级好用
- API设计简洁明了✨
- 文档完善，小白也能上手📚
- 社区活跃，问题快速解决🤝

3️⃣ 应用场景广
从互联网大厂到创业公司都在用！✅

💡 **实战案例**

某互联网大厂实测：
✅ 响应时间缩短50%
✅ 运维成本降低30%
✅ 用户满意度提升20%

创业公司3个月战绩：
✅ 完成MVP开发
✅ 获得种子轮融资
✅ 1000+付费用户

📌 **使用建议**

1. 渐进式迁移，先从非核心业务开始
2. 充分测试，建立完善的测试体系
3. 团队培训，确保掌握相关技能
4. 持续优化，根据反馈不断改进

🚀 **未来趋势**

技术融合 + 标准化 + 普及化 + 智能化

现在是关注和布局的最佳时机！💰

✨ **总结**

✅ 技术成熟，可投入生产
✅ 社区活跃，资源丰富
✅ 商业价值已验证
⚠️ 需根据场景评估选择
🚀 发展空间巨大

---

💬 评论区聊聊你的看法～
❤️ 觉得有用记得点赞收藏哦！
👆 关注我，获取更多AI干货！

#AI技术 #技术分享 #干货 #职场技能
"""

        # 提取body部分（去掉第一行标题，避免重复）
        body_lines = mock_content.split('\n')[1:]  # 跳过第一行标题
        body_content = '\n'.join(body_lines)

        word_count = len(mock_content)

        return {
            "title": main_title,
            "intro": f"{main_title}✨最新技术突破太惊艳了！",
            "body": body_content,  # 使用去掉标题的body
            "ending": "觉得有用记得点赞收藏哦！关注我，获取更多AI干货！",
            "full_content": mock_content,
            "hashtags": ["#AI技术", "#技术分享", "#干货", "#职场技能"],
            "word_count": word_count,
            "original_article_word_count": article.get('word_count', 2800),
            "compression_ratio": f"{(1 - word_count / article.get('word_count', 2800)) * 100:.1f}%"
        }
