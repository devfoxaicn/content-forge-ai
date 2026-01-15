"""
小红书短笔记精炼Agent（快速阅读版本）
生成500-900字的精简小红书笔记，适合快速浏览和传播
"""

from typing import Dict, Any
import re
from src.agents.base import BaseAgent


class XiaohongshuShortRefinerAgent(BaseAgent):
    """小红书短笔记精炼Agent - 快速阅读版本（500-900字）"""

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        super().__init__(config, prompts)
        refiner_config = config.get("agents", {}).get("xiaohongshu_short_refiner", {})
        self.style = refiner_config.get("style", "viral")
        self.max_tokens = refiner_config.get("max_tokens", 4000)  # 短笔记需要较少token
        self.llm.max_tokens = self.max_tokens
        self.llm.temperature = 0.98  # 更高的创造性，短内容需要更吸引人
        self.include_test_case = refiner_config.get("include_test_case", True)
        self.target_word_count = refiner_config.get("target_word_count", 700)  # 目标700字
        self.mock_mode = config.get("agents", {}).get("ai_trend_analyzer", {}).get("mock_mode", False)

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """精炼技术文章为小红书短笔记（快速阅读版本）"""
        self.log("开始精炼小红书短笔记（快速阅读版本）")

        try:
            article = state.get("longform_article")
            if not article:
                raise ValueError("没有找到长文本文章")

            self.log(f"精炼文章: {article['title']}")

            if self.mock_mode:
                self.log("使用Mock模式生成短笔记")
                xhs_note = self._generate_mock_note(article)
            else:
                user_prompt = self._build_prompt(state, article)
                response = self._call_llm(user_prompt)
                xhs_note = self._parse_xiaohongshu_note(response, article)

            self.log(f"成功生成短笔记，字数: {xhs_note['word_count']}")
            return {
                **state,
                "xiaohongshu_short_note": xhs_note,
                "current_step": "xiaohongshu_short_refiner_completed"
            }
        except Exception as e:
            self.log(f"短笔记精炼失败: {str(e)}", "ERROR")
            self.log("使用模拟数据继续测试", "WARNING")
            article = state.get("longform_article", {"title": "AI技术", "full_content": "内容"})
            xhs_note = self._generate_mock_note(article)
            return {
                **state,
                "xiaohongshu_short_note": xhs_note,
                "current_step": "xiaohongshu_short_refiner_completed"
            }

    def _build_prompt(self, state: Dict[str, Any], article: Dict[str, Any]) -> str:
        """构建短笔记提示词"""
        prompts = self.prompts.get("prompts", {})
        prompt_template = prompts.get("xiaohongshu_short_refiner", {}).get("user", "")

        # 提取文章内容（2000字符足够）
        content_preview = article['full_content'][:2000] + "..." if len(article['full_content']) > 2000 else article['full_content']
        target_audience = state.get("target_audience", "技术从业者")
        topic = state.get("topic", "AI工具")

        if prompt_template:
            return prompt_template.format(
                article_title=article['title'],
                article_content=content_preview,
                target_audience=target_audience,
                target_word_count=self.target_word_count,
                style=self.style
            )
        else:
            # 短笔记默认提示词
            return f"""你是一位小红书爆款短笔记创作者，擅长创作500-900字的快速阅读笔记。

**原文章标题**：{article['title']}

**原文章内容**：
{content_preview}

---

## 📝 短笔记创作要求

**格式限制**：
- 标题：10字以内，含emoji
- 正文：500-900字，绝不超过1000字
- 使用`---`分隔不同部分
- 4-6个章节，精简有力

**必须包含**：
1. 黄金3秒开篇（数字/痛点/震惊）
2. 3个核心亮点（用|分隔）
3. 真实效果数据（效率↑70%）
4. 2-3个快速技巧
5. 简短总结 + CTA

**写作风格**：{self.style}（爆款风格）
- 数字驱动，对比强烈
- emoji适中（8-12个）
- 情感共鸣，紧迫感强
- **简洁有力，每段1-2句话**

**内容特点**：
- 重点突出，不说废话
- 数据驱动，真实可信
- 快速阅读，3分钟看完
- 易于分享，适合传播

**严格限制**：
- 字数必须在500-900之间
- 不要超过1000字
- 不要少于500字

现在开始创作小红书爆款短笔记！🚀
"""

    def _parse_xiaohongshu_note(self, response: str, article: Dict[str, Any]) -> Dict[str, Any]:
        """解析小红书短笔记"""
        # 提取markdown代码块内容（如果被```markdown包裹）
        markdown_match = re.search(r'```markdown\n(.*?)```', response, re.DOTALL)
        if markdown_match:
            content = markdown_match.group(1).strip()
        else:
            content = response.strip()

        # 提取标题
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else article['title']

        # 提取标签
        hashtags = []
        hashtag_match = re.search(r'标签[:：](.+)', content)
        if hashtag_match:
            hashtag_text = hashtag_match.group(1).strip()
            hashtags = [tag.strip() for tag in re.findall(r'#[\w\u4e00-\u9fff]+', hashtag_text)]

        if not hashtags:
            hashtags = re.findall(r'#[\w\u4e00-\u9fff]+', content)

        # 继承原文章的标签
        original_tags = article.get('tags', [])
        all_tags = list(set(hashtags + [f"#{tag}" for tag in original_tags]))[:5]

        # 计算字数
        word_count = len(content)

        # 计算emoji数量
        emoji_count = len(re.findall(r'[🚀🔥💡⚡✅📊📈💰⏱️🎯📌❌⚠️🚨🎁✨🏆💪👇💬🔄❤️😭😱]', content))

        return {
            "title": title,
            "full_content": content,
            "hashtags": all_tags,
            "word_count": word_count,
            "original_article_word_count": article.get('word_count', 0),
            "compression_ratio": f"{(1 - word_count / article.get('word_count', 1)) * 100:.1f}%",
            "emoji_count": emoji_count,
            "note_type": "short",
            "target_word_count": self.target_word_count
        }

    def _generate_mock_note(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """生成模拟小红书短笔记"""
        title = article.get('title', 'AI技术突破')
        main_title = title.split('：')[0].split(':')[0]

        mock_content = f"""# 亲测7天！{main_title}真香💥

宝子们！被问了100次，今天必须分享！

以前改代码要在IDE和网页间来回切换，复制粘贴到手酸！

用了一周后，我直接下班早了2小时！😭

---

## ✨ 核心发现

**AI主动干活**：不用反复解释｜**深度理解**：像真同事一样｜**效率翻倍**：编码时间↓50%

---

## 🚀 真实效果

**Day 1**：配置2小时，有点懵
**Day 3**：效率提升30%
**Day 7**：完全上手，翻倍！

**数据对比**：
• 编码：4h→2h（↓50%）
• Bug修复：1.5h→0.5h（↓67%）
• 下班时间：提前2小时🎉

---

## 📌 3个核心技巧

1️⃣ **像同事一样沟通**
别说"帮我X"，而是"我们来解决X"

2️⃣ **给完整上下文**
第一次就告诉项目结构，不用反复说

3️⃣ **建立标准流程**
需求→代码→测试→文档，一套模板

---

## 💬 总结

不是AI取代你，而是会用AI的人取代你！

早用早享受！现在开始还不晚🔥

🔗 搜索关键词 `{main_title}`

---

#AI编程 #程序员 #效率神器 #干货分享
"""

        word_count = len(mock_content)
        emoji_count = len(re.findall(r'[🚀🔥💡⚡✅📊📈💰⏱️🎯📌❌⚠️🚨🎁✨🏆💪👇💬🔄❤️😭😱]', mock_content))

        return {
            "title": f"亲测7天！{main_title}真香💥",
            "full_content": mock_content,
            "hashtags": ["#AI编程", "#程序员", "#效率神器", "#干货分享"],
            "word_count": word_count,
            "original_article_word_count": article.get('word_count', 40000),
            "compression_ratio": f"{(1 - word_count / article.get('word_count', 40000)) * 100:.1f}%",
            "emoji_count": emoji_count,
            "note_type": "short",
            "target_word_count": 700
        }
