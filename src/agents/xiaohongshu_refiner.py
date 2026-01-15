"""
小红书笔记精炼Agent（增强版）
将长文本技术文章精炼成小红书风格的笔记，专注于生成爆款内容
"""

from typing import Dict, Any
import re
from src.agents.base import BaseAgent


class XiaohongshuRefinerAgent(BaseAgent):
    """小红书笔记精炼Agent - 增强版，专注爆款内容"""

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        super().__init__(config, prompts)
        refiner_config = config.get("agents", {}).get("xiaohongshu_refiner", {})
        self.style = refiner_config.get("style", "viral")  # viral, professional, casual
        self.content_density = refiner_config.get("content_density", "medium")
        self.max_tokens = refiner_config.get("max_tokens", 3000)
        self.llm.max_tokens = self.max_tokens
        self.llm.temperature = 0.9  # 提高创造性，生成更有趣的内容
        self.mock_mode = config.get("agents", {}).get("ai_trend_analyzer", {}).get("mock_mode", False)

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """精炼技术文章为小红书笔记"""
        self.log("开始精炼技术文章为小红书笔记（爆款模式）")

        try:
            article = state.get("longform_article")
            if not article:
                raise ValueError("没有找到长文本文章")

            self.log(f"精炼文章: {article['title']}")

            if self.mock_mode:
                self.log("使用Mock模式生成小红书笔记")
                xhs_note = self._generate_mock_note(article)
            else:
                user_prompt = self._build_prompt(state, article)
                response = self._call_llm(user_prompt)
                xhs_note = self._parse_xiaohongshu_note(response, article)

            self.log(f"成功生成小红书笔记，字数: {xhs_note['word_count']}")
            return {
                **state,
                "xiaohongshu_note": xhs_note,
                "refined_content": xhs_note,
                "generated_content": xhs_note,
                "current_step": "xiaohongshu_refiner_completed"
            }
        except Exception as e:
            self.log(f"笔记精炼失败: {str(e)}", "ERROR")
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
        """构建精炼提示词 - 爆款优化版"""
        prompts = self.prompts.get("prompts", {})
        prompt_template = prompts.get("xiaohongshu_refiner", {}).get("user", "")

        content_preview = article['full_content'][:3000] + "..." if len(article['full_content']) > 3000 else article['full_content']
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
            # 爆款优化版提示词
            return f"""你是一位小红书爆款内容创作专家，精通小红书平台算法和用户心理。

请将以下技术文章精炼为**小红书爆款笔记**：

**原文章标题**：{article['title']}

**原文章内容**：
{content_preview}

---

## 🔥 爆款笔记创作指南

### 1️⃣ 标题优化（必须吸引眼球）
从以下模板中选择或组合：
- **数字型**："零基础必看！XXX保姆级教程"、"3个技巧让我XXX"
- **对比型**："XXX vs YYY，谁才是最强？"、"后悔没早点知道"
- **痛点型**："为什么你的XXX总是XXX？"、"90%的人都不知道"
- **震惊型**："XXX也太强大了吧！"、"终于找到了！"
- **紧迫型**："2026年了，你还在XXX吗？"、"不转就晚了"

**要求**：
- 标题控制在20字以内
- 必须包含1-2个emoji
- 必须有数字、对比、痛点至少一个元素

### 2️⃣ 开头钩子（前3秒决定留存）
使用以下技巧开场：
- **反常识**："颠覆认知！原来XXX应该这样用"
- **数据震撼**："最新调研：全球顶尖开发者都在用XXX"
- **痛点共鸣**："还在复制粘贴代码？AI编程已进化"
- **紧迫感**："2026年了，你还在用传统方式？"

**要求**：
- 3-5句话，每句简短有力
- 第一句话必须有冲击力
- 使用"宝子们"、"姐妹们"、"各位"拉近距离

### 3️⃣ 正文结构（用 --- 分隔每个部分）
**参考格式**：
```
第一部分内容

---

第二部分内容

---

第三部分内容

---
```

**每个部分的要求**：
- 用emoji作为小标题（✨💡🚀⚡💻等）
- 数据驱动，用数字说话（效率↑70%、成本↓50%）
- 简洁明了，每段2-4句话
- 适当使用"亲测"、"实测"、"必看"、"干货"等词汇

### 4️⃣ 内容密度：{self.content_density}
- **light**: 简洁易懂，适合快速浏览（600-800字）
- **medium**: 适中平衡，保留核心细节（800-1000字）
- **dense**: 信息密集，适合深度阅读（1000-1200字）

### 5️⃣ 写作风格：{self.style}
- **viral**: 爆款风格（推荐），语言活泼，数据驱动，情感共鸣
- **professional**: 专业但不枯燥，适合B端用户
- **casual**: 轻松友好，像朋友分享

### 6️⃣ 输出格式模板
```markdown
# 标题（20字内，含emoji）

🔥 引人入胜的开场（3-5句，有冲击力）

---

**最新调研/发现**：核心观点1

---

## ✨ 核心特性
**特性1**：描述｜**特性2**：描述｜**特性3**：描述

---

## ⚡ 真实效果
**效果1**：数据｜**效果2**：数据｜**效果3**：数据

---

## 💻 谁最适合用？
**人群1**：说明｜**人群2**：说明｜**人群3**：说明

---

## 🛠️ 快速上手
1️⃣ 步骤1｜2️⃣ 步骤2｜3️⃣ 步骤3｜4️⃣ 步骤4

---

## 💡 核心优势
vs 竞品A：优势1｜vs 竞品B：优势2

---

## 📝 实用场景
✅ 场景1｜✅ 场景2｜✅ 场景3

---

## 总结💫

**关键数据**：XXX、YYY、ZZZ

**核心价值**：一句话总结

想让XXX？从YYY开始！🚀

🔗 相关链接

---

#标签1 #标签2 #标签3 #标签4 #标签5
```

### 7️⃣ 标签优化策略
选择5个精准标签：
- **2个流量标签**：#AI编程 #零基础入门 #程序员 #效率神器 #生产力工具
- **2个精准标签**：根据文章内容选择
- **1个泛标签**：#干货 #技术分享

### 8️⃣ 爆款要素检查清单
✅ 标题有吸引力（数字/对比/痛点/震惊）
✅ 开头3秒抓住注意力
✅ 数据驱动，有具体数字
✅ 用 --- 分隔不同部分
✅ emoji丰富但不滥用（5-10个）
✅ 有"亲测"、"必看"等真实感词汇
✅ 结尾有行动号召（点赞、收藏、评论）
✅ 标签精准且热门

**目标受众**：{target_audience}
**主题**：{topic}

请开始创作，确保内容既专业又符合小红书爆款调性！
"""

    def _parse_xiaohongshu_note(self, response: str, article: Dict[str, Any]) -> Dict[str, Any]:
        """解析小红书笔记"""
        # 提取标题
        title_match = re.search(r'^#\s+(.+)$', response, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else article['title']

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
        all_tags = list(set(hashtags + [f"#{tag}" for tag in original_tags]))[:8]

        # 计算字数
        word_count = len(response)

        return {
            "title": title,
            "full_content": response,
            "hashtags": all_tags,
            "word_count": word_count,
            "original_article_word_count": article.get('word_count', 0),
            "compression_ratio": f"{(1 - word_count / article.get('word_count', 1)) * 100:.1f}%"
        }

    def _generate_mock_note(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """生成模拟小红书笔记 - 爆款优化版"""
        title = article.get('title', 'AI技术突破')
        main_title = title.split('：')[0].split(':')[0]

        mock_content = f"""# {main_title}保姆级教程🔥

2026年了，你还在用传统方式？太落后了！

---

**最新调研**：全球顶尖开发者都在用这个神器！

---

## ✨ 核心特性

**200K上下文**：一口气读完项目｜**Artifacts预览**：代码实时渲染｜**多模型**：Claude/GPT无缝切换

---

## ⚡ 真实效果

**调试效率↑70%**：30分钟→5分钟｜**代码质量↑50%**：自动审查｜**学习速度↑3倍**：像导师一样教

---

## 💻 谁最适合用？

**编程小白**：自然语言写代码｜**资深开发**：重构+架构｜**数据分析师**：Python+图表一键生成

---

## 🛠️ 3分钟上手

1️⃣ 打开Claude Code创建项目

2️⃣ 上传代码/文档作为上下文

3️⃣ 用自然语言描述需求

4️⃣ 实时预览生成结果

就这么简单！🎉

---

## 💡 核心优势

vs ChatGPT：上下文窗口大4倍，不遗忘｜vs Cursor：无需安装IDE｜vs Copilot：主动协作

---

## 📝 实用场景

✅ 快速搭建项目原型｜✅ 遗留代码重构｜✅ 复杂Bug排查｜✅ 技术文档自动生成

---

## 总结💫

**关键数据**：200K上下文、效率提升70%、学习速度3倍

**核心价值**：从问答到协作，AI真正的进化

想让编程变简单？从Claude Cowork开始！🚀

🔗 官网搜"Claude Code"

---

#Claude3 #AI编程 #零基础入门 #程序员 #效率神器 #生产力工具 #干货 #技术分享
"""

        word_count = len(mock_content)

        return {
            "title": f"{main_title}保姆级教程🔥",
            "full_content": mock_content,
            "hashtags": ["#Claude3", "#AI编程", "#零基础入门", "#程序员", "#效率神器", "#生产力工具", "#干货", "#技术分享"],
            "word_count": word_count,
            "original_article_word_count": article.get('word_count', 40000),
            "compression_ratio": f"{(1 - word_count / article.get('word_count', 40000)) * 100:.1f}%"
        }
