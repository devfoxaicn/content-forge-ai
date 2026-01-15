"""
小红书笔记精炼Agent（世界级运营专家）
基于小红书顶级博主的内容创作方法论，打造100万+阅读的爆款笔记
"""

from typing import Dict, Any
import re
from src.agents.base import BaseAgent


class XiaohongshuRefinerAgent(BaseAgent):
    """小红书笔记精炼Agent - 世界级运营专家，专注爆款内容"""

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        super().__init__(config, prompts)
        refiner_config = config.get("agents", {}).get("xiaohongshu_refiner", {})
        self.style = refiner_config.get("style", "viral")  # viral, professional, casual
        self.content_density = refiner_config.get("content_density", "rich")  # light, medium, rich
        self.max_tokens = refiner_config.get("max_tokens", 8000)  # 增加到8000避免截断
        self.llm.max_tokens = self.max_tokens
        self.llm.temperature = 0.95  # 提高创造性
        self.include_test_case = refiner_config.get("include_test_case", True)  # 包含实测案例
        self.use_emotional_language = refiner_config.get("use_emotional_language", True)  # 情感化语言
        self.mock_mode = config.get("agents", {}).get("ai_trend_analyzer", {}).get("mock_mode", False)

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """精炼技术文章为小红书笔记（世界级爆款标准）"""
        self.log("开始精炼技术文章为小红书笔记（世界级运营专家）")

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
                # 记录原始响应用于调试
                self.log(f"LLM原始响应长度: {len(response)} 字符")
                self.log(f"LLM原始响应预览: {response[:500]}...")
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
        """构建世界级提示词"""
        prompts = self.prompts.get("prompts", {})
        prompt_template = prompts.get("xiaohongshu_refiner", {}).get("user", "")

        # 提取更多文章内容（3500字符）
        content_preview = article['full_content'][:3500] + "..." if len(article['full_content']) > 3500 else article['full_content']
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
            # 世界级提示词 - 基于小红书顶级博主的创作方法论
            return f"""你是一位拥有100万+粉丝的小红书顶级博主，你的笔记经常获得100万+阅读和10万+点赞。你深谙小红书算法和用户心理。

**原文章标题**：{article['title']}

**原文章内容**：
{content_preview}

---

## 🎯 小红书爆款笔记核心公式（必须遵守）

### 1️⃣ 爆款标题公式（15字内，决定80%点击率）

**数字型**（最有效）：
```
"零基础必看！XXX保姆级教程"
"3个技巧让我效率翻倍"
"5个误区90%的人都踩过"
"10天从入门到精通"
```

**对比型**：
```
"后悔没早点知道XXX"
"XXX vs YYY，谁才是王者？"
"用对XXX，效率提升10倍"
```

**痛点型**：
```
"为什么你的XXX总是XXX？"
"终于找到困扰我3年的方法"
"不要再浪费时间在XXX上"
```

**震惊型**：
```
"XXX也太强大了吧！"
"这个功能我居然今天才知道"
"发现一个被埋没的神器"
```

**紧迫型**：
```
"2026年了，你还不懂XXX？"
"现在布局XXX正是时候"
"早知道就好了系列"
```

### 2️⃣ 开篇钩子（前3秒决定留存）

**反常识型**：
```
"颠覆认知！原来XXX应该这样用..."

"大多数人觉得X，但其实Y才是真相..."

"我用了10年才发现，XXX最大的误区是..."
```

**数据震撼型**：
```
"最新调研：90%的人都做错了..."

"某大厂内部数据：使用X后效率提升70%..."

"我实测100次，发现这个隐藏功能..."
```

**痛点共鸣型**：
```
"宝子们！你是不是也遇到过XXX..."

"终于找到了困扰我3年的解决方案..."

"别再踩坑了！今天分享XXX正确用法..."
```

**亲身实测型**（最推荐！86%爆款笔记采用）：
```
"亲测1个月，我发现XXX真的太香了！"

"用了一周后，我必须安利给你们！"

"被问了100次，今天终于决定分享..."
```

### 3️⃣ 正文结构（用 --- 分隔每个部分）

**标准爆款结构**：
```
[爆款标题]

[引人入胜的开篇3-5句]

---

[第一部分：痛点/问题]

---

[第二部分：解决方案/核心方法]

---

[第三部分：实操步骤/案例]

---

[第四部分：效果数据/对比]

---

[第五部分：避坑指南/注意事项]

---

[总结 + CTA]

[话题标签]
```

**每个部分的要求**：
- 用emoji作为小标题（✨💡🚀⚡💻📌等）
- 数据驱动，用数字说话（效率↑70%、成本↓50%）
- 适当使用"亲测"、"实测"、"必看"、"干货"等词汇
- 简洁明了，每段2-4句话
- 使用|分隔多个要点

### 4️⃣ 内容密度：{self.content_density}

**light**: 简洁易懂，适合快速浏览（800-1000字）
- 重点突出，不说废话
- 适合工具类、技巧类内容

**medium**: 适中平衡，保留核心细节（1000-1500字）
- 有数据有案例
- 适合科普类、教程类内容

**rich**: 信息密集，适合深度阅读（1500-2000字）
- 详细步骤 + 多个案例
- 适合经验分享、深度测评
- **强烈推荐使用rich密度**

### 5️⃣ 写作风格：{self.style}

**viral（爆款风格）** - 推荐：
- 数字驱动，对比强烈
- emoji丰富（10-15个）
- 情感共鸣，紧迫感强
- 互动设计，引导评论

**professional（专业风格）**：
- 数据严谨，逻辑清晰
- 权威引用，案例详实
- 适合B端用户、深度内容

**casual（轻松风格）**：
- 语言口语化，像对话
- 使用"宝子们"、"姐妹们"拉近距离
- 适合年轻受众、生活方式内容

### 6️⃣ 亲身实测话术（{self.include_test_case}）

如果include_test_case=True，必须包含：

**时间线**：
```
"使用前：XXX"
"使用3天后：XXX"
"使用1个月后：XXX"
```

**真实数据**：
```
"效率提升：70%"
"成本降低：50%"
"时间节省：从3小时→30分钟"
```

**对比图描述**：
```
"Before vs After对比图请看↓"

"第1张图：传统方法"
"第2张图：使用XXX后"
```

**情感变化**：
```
"一开始我也怀疑..."
"但用了一周后..."
"现在我离不开它了！"
```

### 7️⃣ Emoji视觉策略

**表情包逻辑**：
- 开篇：🚀 🔥 💡 ⚡ （抓眼球）
- 要点：✅ 📌 🎯 💪 （重点）
- 数据：📊 📈 💰 ⏱️ （可信度）
- 警告：⚠️ 🚨 ❌ （注意）
- 推荐：✨ 🎁 ⭐ （价值）

**密度控制**：
- 标题：1-2个emoji
- 正文：10-15个emoji
- 每部分：2-3个emoji

### 8️⃣ 话题标签策略（公式）

**标签公式**：
```
2个流量标签 + 2个精准标签 + 1个泛标签
```

**流量标签池**（必须包含）：
#零基础入门 #程序员 #效率神器 #生产力工具 #AI编程 #干货分享 #技术博主

**精准标签**（根据内容）：
#Python入门 #JavaScript #Web开发 #数据分析 #机器学习

**泛标签**：
#学习笔记 #职场必备 #干货

**标签位置**：
最后5个字符

### 9️⃣ 爆款要素检查清单（发布前必查）

✅ 标题15字内，包含数字/对比/痛点
✅ 开篇前3秒抓住注意力（反常识/数据震撼/亲身实测）
✅ 数据驱动，有具体数字
✅ 用 --- 分隔不同部分（5-7个部分）
✅ emoji丰富但不滥用（10-15个）
✅ 有"亲测"、"实测"、"必看"等真实感词汇
✅ 包含Before/After对比
✅ 有具体的步骤或案例
✅ 结尾有行动号召（点赞/收藏/评论/关注）
✅ 标签精准且热门（5个）

### 🔟 情感化语言技巧

**拉近距离**：
```
"宝子们"
"姐妹们"
"各位"
"家人们"
```

**真实感**：
```
"亲测"
"实测"
"真心推荐"
"用了一周后..."
"被问了100次..."
```

**紧迫感**：
```
"别再踩坑了"
"现在知道还不晚"
"早用早享受"
```

**价值感**：
```
"干货满满"
"建议收藏"
"值得反复观看"
"太香了"
```

### 📤 输出格式模板

```markdown
# 爆款标题（15字内，含emoji）

🔥 引人入胜的开篇（3-5句，有冲击力，亲身实测最佳）

---

## ✨ 核心发现

**发现1**：描述｜**发现2**：描述｜**发现3**：描述

---

## 💡 为什么这么好用？

[核心原因]
[具体数据]

---

## 🚀 我的使用体验

**第1天**：XXX
**第3天**：XXX
**第7天**：XXX

真实数据：效率↑70%｜成本↓50%｜时间节省60%

---

## 📌 3个核心技巧

1️⃣ [技巧1]：详细说明

2️⃣ [技巧2]：详细说明

3️⃣ [技巧3]：详细说明

---

## ⚠️ 避坑指南

❌ [错误1]
❌ [错误2]

✅ 正确做法：XXX

---

## 💬 总结

[核心价值]

[行动号召]

🔗 相关链接

---

#标签1 #标签2 #标签3 #标签4 #标签5
```

---

## 🎨 世界级爆款笔记示例（亲身实测型）

```markdown
# 亲测1个月！这个AI工具让我效率翻倍💥

宝子们！被问了100次，今天终于决定分享这个被埋没的神器！

使用前我每天加班到10点，现在6点就能下班了！😭

---

## ✨ 核心发现

**200K上下文**：一口气读完整个项目｜**智能理解**：不用反复解释｜**主动协作**：像真正的同事

---

## 💡 为什么这么好用？

传统AI：你问它才答，被动响应，像个孤岛

这个工具：主动理解上下文，预判你的需求，直接操作文件！

这是质的飞跃！🚀

---

## 🚀 我的使用体验

**第1天**：花了2小时配置，有点懵

**第3天**：开始熟悉，效率提升30%

**第7天**：完全上手，效率翻倍！

**第30天**：已经离不开它了！

真实数据：
• 编码时间：4小时→2小时（↓50%）
• Bug修复：1.5小时→0.5小时（↓67%）
• 加班时间：减少60%

---

## 📌 3个核心技巧

1️⃣ **像同事一样沟通**
不要"请帮我X"，而是"我们来解决X"

2️⃣ **给予完整上下文**
把项目结构告诉AI，让它理解全局

3️⃣ **建立工作流**
设计标准流程，让AI无缝融入

---

## ⚠️ 避坑指南

❌ 期望AI一次性完美
❌ 不理解直接复制代码
❌ 过度依赖放弃思考

✅ 正确姿势：AI是助手，你是决策者！

---

## 💬 总结

不是AI取代你，而是会用AI的人取代你！

早用早享受！现在开始还来得及！🔥

🔗 官网搜"Claude Code"

---

#AI编程 #零基础入门 #程序员 #效率神器 #生产力工具
```

---

**目标受众**：{target_audience}
**主题**：{topic}
**内容密度**：{self.content_density}
**包含实测案例**：{self.include_test_case}

现在开始创作小红书爆款笔记！🚀
"""

    def _parse_xiaohongshu_note(self, response: str, article: Dict[str, Any]) -> Dict[str, Any]:
        """解析小红书笔记"""
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
        all_tags = list(set(hashtags + [f"#{tag}" for tag in original_tags]))[:8]

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
            "content_density": self.content_density
        }

    def _generate_mock_note(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """生成模拟小红书笔记（世界级爆款标准）"""
        title = article.get('title', 'AI技术突破')
        main_title = title.split('：')[0].split(':')[0]

        mock_content = f"""# 亲测1个月！{main_title}让我效率翻倍💥

宝子们！被问了100次，今天终于决定分享这个被埋没的神器！

使用前我每天加班到10点，现在6点就能下班了！😭

---

## ✨ 核心发现炸裂

**200K上下文**：一口气读完整个项目｜**Artifacts预览**：代码实时渲染｜**多模型切换**：Claude/GPT无缝切换

---

## 💡 为什么这么好用？

传统AI：你问它才答，被动响应，像个孤岛

{main_title}：主动理解上下文，预判你的需求，直接操作文件！

这是质的飞跃！🚀

---

## 🚀 我的使用体验

**第1天**：花了2小时配置，有点懵

**第3天**：开始熟悉，效率提升30%

**第7天**：完全上手，效率翻倍！

**第30天**：已经离不开它了！

真实数据：
• 编码时间：4小时→2小时（↓50%）
• Bug修复：1.5小时→0.5小时（↓67%）
• 加班时间：减少60%

---

## 📌 3个核心技巧

1️⃣ **像同事一样沟通**
不要"请帮我X"，而是"我们来解决X"

2️⃣ **给予完整上下文**
把项目结构告诉AI，让它理解全局

3️⃣ **建立工作流**
设计标准流程，让AI无缝融入

---

## ⚠️ 避坑指南

❌ 期望AI一次性完美
❌ 不理解直接复制代码
❌ 过度依赖放弃思考

✅ 正确姿势：AI是助手，你是决策者！

---

## 💬 总结

不是AI取代你，而是会用AI的人取代你！

早用早享受！现在开始还来得及！🔥

🔗 官网搜相关工具

---

#AI编程 #零基础入门 #程序员 #效率神器 #生产力工具
"""

        word_count = len(mock_content)
        emoji_count = len(re.findall(r'[🚀🔥💡⚡✅📊📈💰⏱️🎯📌❌⚠️🚨🎁✨🏆💪👇💬🔄❤️😭😱]', mock_content))

        return {
            "title": f"亲测1个月！{main_title}让我效率翻倍💥",
            "full_content": mock_content,
            "hashtags": ["#AI编程", "#零基础入门", "#程序员", "#效率神器", "#生产力工具"],
            "word_count": word_count,
            "original_article_word_count": article.get('word_count', 40000),
            "compression_ratio": f"{(1 - word_count / article.get('word_count', 40000)) * 100:.1f}%",
            "emoji_count": emoji_count,
            "content_density": "rich"
        }
