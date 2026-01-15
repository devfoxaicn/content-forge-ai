"""
Twitter帖子生成Agent（专家级）
将专业文章精炼为高互动Twitter Thread，专注于生成爆款内容
"""

from typing import Dict, Any, List
import re
from src.agents.base import BaseAgent


class TwitterGeneratorAgent(BaseAgent):
    """Twitter帖子生成Agent - 专家级，专注爆款内容"""

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        super().__init__(config, prompts)
        twitter_config = config.get("agents", {}).get("twitter_generator", {})
        self.style = twitter_config.get("style", "viral")  # viral, professional, casual, storytelling
        self.thread_mode = twitter_config.get("thread_mode", True)
        self.max_tweets = twitter_config.get("max_tweets", 8)  # 增加到8条以支持更丰富的内容
        self.max_tokens = twitter_config.get("max_tokens", 4000)  # 增加token以支持更详细的prompt
        self.llm.max_tokens = self.max_tokens
        self.llm.temperature = 0.95  # 提高创造性，生成更有趣的内容
        self.cta_type = twitter_config.get("cta_type", "engagement")  # engagement, follow, retweet, reply
        self.use_emojis = twitter_config.get("use_emojis", True)
        self.hook_strategy = twitter_config.get("hook_strategy", "auto")  # auto, curiosity, controversy, data
        self.mock_mode = config.get("agents", {}).get("ai_trend_analyzer", {}).get("mock_mode", False)

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """生成Twitter帖子（专家级）"""
        self.log("开始生成Twitter帖子（专家级爆款模式）")

        try:
            article = state.get("longform_article")
            if not article:
                raise ValueError("没有找到长文本文章")

            self.log(f"基于文章生成Twitter: {article['title']}")

            if self.mock_mode:
                self.log("使用Mock模式生成Twitter帖子")
                twitter_post = self._generate_mock_thread(article)
            else:
                user_prompt = self._build_prompt(state, article)
                response = self._call_llm(user_prompt)
                twitter_post = self._parse_twitter_post(response, article)

            self.log(f"成功生成Twitter帖子: {twitter_post['tweet_count']}条推文")
            return {
                **state,
                "twitter_post": twitter_post,
                "generated_content": twitter_post,
                "current_step": "twitter_generator_completed"
            }
        except Exception as e:
            self.log(f"Twitter帖子生成失败: {str(e)}", "ERROR")
            self.log("使用模拟数据继续测试", "WARNING")
            article = state.get("longform_article", {"title": "AI技术", "full_content": "内容"})
            twitter_post = self._generate_mock_thread(article)
            return {
                **state,
                "twitter_post": twitter_post,
                "generated_content": twitter_post,
                "current_step": "twitter_generator_completed"
            }

    def _build_prompt(self, state: Dict[str, Any], article: Dict[str, Any]) -> str:
        """构建专家级提示词"""
        prompts = self.prompts.get("prompts", {})
        prompt_template = prompts.get("twitter_generator", {}).get("user", "")

        # 提取文章核心内容（增加到2500字符以获取更多上下文）
        content_preview = article['full_content'][:2500] + "..." if len(article['full_content']) > 2500 else article['full_content']

        target_audience = state.get("target_audience", "技术从业者")
        topic = state.get("topic", "AI技术")

        thread_instruction = f"生成一个Twitter viral thread（{self.max_tweets}条推文）" if self.thread_mode else "生成一条推文"

        if prompt_template:
            return prompt_template.format(
                article_title=article['title'],
                article_content=content_preview,
                target_audience=target_audience,
                style=self.style,
                thread_mode=self.thread_mode,
                max_tweets=self.max_tweets,
                cta_type=self.cta_type,
                hook_strategy=self.hook_strategy
            )
        else:
            # 专家级提示词
            return f"""你是一位Twitter顶级内容创作者，拥有100万+粉丝，精通Twitter算法和用户心理。你的Thread经常获得10万+互动。

**原文章标题**：{article['title']}

**原文章内容**：
{content_preview}

---

## 🎯 任务：{thread_instruction}

## 1️⃣ 开头钩子策略（Hook Strategy）

选择以下策略之一或组合使用：

**A. 反常识型**（最适合技术话题）：
```
"大多数人都认为X，但实际上Y才是真相..."

"我在[领域]工作了10年，发现一个反直觉的真相..."

"如果你还在做X，那你可能已经落后了..."
```

**B. 数据震撼型**（最具说服力）：
```
"最新调研震惊：90%的开发者都做错了..."

"某大厂内部数据：使用X后效率提升70%..."

"我分析了1000个项目，发现..."
```

**C. 痛点共鸣型**（最强互动）：
```
"你是不是也遇到过这个问题..."

"终于找到了困扰我3年的解决方案..."

"不要再浪费时间在X上了..."
```

**D. 紧迫感型**（最高转化）：
```
"2026年了，如果你还不懂X，就真的out了..."

"这个改变正在发生，越早布局越好..."

"下个月可能就晚了..."
```

**第1条推文黄金公式**：
```
[强钩子：1句话抓住注意力]
[核心价值：2句话说明价值]
[Thread提示：🧵]
```

示例：
```
🚀 我发现一个提升效率70%的AI工具，90%的人都不知道！

使用3个月，我的代码质量提升50%，调试时间缩短70%。

🧵👇
```

---

## 2️⃣ Thread叙事结构（Narrative Arc）

采用**英雄之旅**结构：

**推文1（召唤）**：用钩子抓住注意力，引发好奇
**推文2-3（挑战）**：描述问题/痛点，建立共鸣
**推文4-5（解决方案）**：揭示核心解决方案，展示价值
**推文6-7（证据）**：用数据/案例证明效果
**推文8（行动）**：总结价值，强行动号召

每条推文格式：
```
[Emoji标题/要点]
[1-2句话核心内容]
[1-2句话详细说明/案例]
```

---

## 3️⃣ 内容密度控制

**总字符数**：{self.max_tweets * 200}-{self.max_tweets * 250}字符
**每条推文**：200-250字符（留空间给emoji和图片）
**视觉节奏**：
- 推文1：简洁有力（150-180字符）
- 推文2-3：痛点详细（220-250字符）
- 推文4-5：解决方案（220-250字符）
- 推文6-7：证据数据（200-240字符）
- 推文8：总结+CTA（180-220字符）

---

## 4️⃣ Emoji使用策略（提升视觉吸引力）

**表情包组合公式**：
```
主题emoji + 动作emoji + 结果emoji = 高互动
```

**常用emoji组合**：
- 开场用：🚀 🔥 💡 ⚡ 🎯
- 列举用：✅ 1️⃣ 2️⃣ 3️⃣ 📌
- 数据用：📊 📈 💰 ⏱️
- 警告用：⚠️ 🚨 ❌
- 结论用：✨ 🎁 🏆 💪

**emoji密度**：
- 每条推文：2-4个emoji
- 开头emoji：必须使用（视觉锚点）
- 列表emoji：必须统一（✅或📌）
- 结尾emoji：1个强化情感

---

## 5️⃣ 爆款写作技巧

**A. 数字驱动**：
```
❌ "这个工具很快"
✅ "响应时间从30分钟降到5分钟（提速83%）"

❌ "很多人都在用"
✅ "已有10万+开发者，覆盖50+国家"
```

**B. 对比强烈**：
```
"传统方法：耗时30分钟，准确率70%"
"新方法：耗时5分钟，准确率95%"

"大多数人都X，但顶尖高手都Y"
```

**C. 权威背书**：
```
"某大厂内部测试数据..."
"Gartner报告显示..."
"斯坦福最新研究..."
```

**D. 社交证明**：
```
"我的团队使用后..."
"评论区有人反馈..."
"真实案例：某创业公司..."
```

---

## 6️⃣ 互动优化（Engagement Hacks）

**A. 提问互动**：
```
"你觉得呢？👇"

"你的选择是？"

"评论区告诉我你的经历..."
```

**B. 紧迫互动**：
```
"趁现在还没烂大街..."

"前1000个关注者送..."

"点赞超过1万就出第二期..."
```

**C. 资源引导**：
```
"想要完整教程？关注+回复'666'"

"关注我，每天分享一个AI技巧"

"转发给需要的朋友..."
```

---

## 7️⃣ 行动号召（CTA）优化

**类型：{self.cta_type}**

**A. Engagement型**（提升互动）：
```
"💬 你觉得这个工具怎么样？评论区聊聊！

🔄 转发给需要的朋友

❤️ 点赞让更多人看到"
```

**B. Follow型**（增长粉丝）：
```
"🎯 关注我，每天分享一个AI技巧

🚀 立即关注，不错过下一个干货"
```

**C. Retweet型**（扩大传播）：
```
"🔄 觉得有用？转发给需要的朋友！

让更多人知道这个神器..."
```

**D. Reply型**（获取线索）：
```
"🎁 想要完整教程？

👇 关注+回复'教程'，私信发你！"
```

---

## 8️⃣ 话题标签策略（Tag Strategy）

**标签公式**：
```
2个流量标签 + 2个精准标签 + 1个行动标签
```

**流量标签**（必须包含）：
- #AI #MachineLearning #Coding #Developer #Tech
- #JavaScript #Python #Productivity #Automation

**精准标签**（根据内容）：
- #AIAutomation #CodeAssistant #DeveloperTools
- #PromptEngineering #LLM #ChatGPT

**行动标签**（可选）：
- #FollowForMore #LearnAI #TechTips

**标签位置**：
- 在最后一条推文的最后
- 或者放在第1条推文的最后（提高曝光）

---

## 9️⃣ 写作风格：{self.style}

**viral（爆款风格）** - 推荐：
- 数字驱动，对比强烈
- emoji丰富，视觉吸引
- 情感共鸣，紧迫感强
- 互动设计，引导评论

**professional（专业风格）**：
- 数据严谨，逻辑清晰
- 权威引用，案例详实
- 适合B端用户
- 语气稳重

**casual（轻松风格）**：
- 语言口语化，像对话
- 幽默轻松，拉近距离
- 适合年轻受众
- 互动性强

**storytelling（故事风格）**：
- 叙事驱动，情节吸引
- 个人经历，真实感强
- 情感连接，记忆深刻
- 适合品牌故事

---

## 🔟 Thread检查清单（发布前必查）

✅ 第1条有强钩子（反常识/数据震撼/痛点/紧迫感）
✅ 钩子在前20字符内
✅ 每条推文都有独立价值
✅ 推文之间逻辑连贯
✅ 使用数字增强说服力
✅ emoji使用合理（2-4个/条）
✅ 最后1条有强CTA
✅ 话题标签精准（5个）
✅ 总字符数合适（{self.max_tweets * 200}-{self.max_tweets * 250}）
✅ 有提问或互动设计

---

## 📤 输出格式模板

```
Tweet 1:
[强钩子]
[核心价值]
🧵

Tweet 2:
[Emoji] [要点标题]
[详细说明]

Tweet 3:
[Emoji] [要点标题]
[详细说明]
[互动提问]

...

Tweet {self.max_tweets}:
[Emoji] 总结
✅ [要点1]  |  ✅ [要点2]  |  ✅ [要点3]

[强CTA]

#标签1 #标签2 #标签3 #标签4 #标签5
```

---

**目标受众**：{target_audience}
**主题**：{topic}
**钩子策略**：{self.hook_strategy}

---

## 🎨 爆款Thread示例参考

**示例1：反常识型**
```
🚀 我发现90%的开发者都在浪费时间写重复代码...

使用AI助手后，我的开发效率提升3倍！

🧵👇

1/8
❌ 传统方式：
- 手写CRUD操作（耗时2小时）
- 查阅文档（30分钟）
- 调试Bug（1小时）
✅ AI助手：
- 自动生成代码（5分钟）
- 智能解释（即时）
- 自动优化（零成本）

2/8
💡 关键洞察：

AI不是替代你，而是让你从重复劳动中解放出来

把时间花在真正有价值的事情上：
- 架构设计
- 业务逻辑
- 产品创新

你还在纠结CRUD吗？🤔

3/8
📊 真实数据（我的团队）：

• 开发时间：60% ↓
• Bug数量：40% ↓
• 代码质量：50% ↑
• 团队满意度：⭐⭐⭐⭐⭐

某大厂内部测试，效果更显著...

4/8
🔥 3个核心技巧：

1️⃣ 描述需求，让AI生成代码框架
2️⃣ 逐层优化，从基础到进阶
3️⃣ 持续学习，积累提示词库

掌握这些，你就是10x开发者！💪

5/8
⚠️ 常见误区：

❌ 期望AI一次性完美
❌ 不理解就直接复制
❌ 过度依赖，放弃思考

正确姿势：
✅ AI是助手，你是决策者
✅ 理解原理，灵活运用
✅ 持续验证，积累经验

6/8
🎯 实战案例：

某创业公司使用AI助手：
- MVP开发时间：2个月 → 2周
- 人力成本：节省60%
- 产品质量：显著提升

投资人评价："执行力超强！"

7/8
💬 评论区有朋友问：

"AI会取代程序员吗？"

我的答案：
不会取代，但会"AI增强型"取代"传统型"

未来不是AI vs 人
而是 会用AI的人 vs 不会用AI的人

8/8
✨ 总结：

✅ AI是工具，不是威胁
✅ 越早使用，优势越大
✅ 持续学习，保持领先

🚀 立即开始你的AI之旅！

💬 你的选择是？评论区聊聊！

🔄 觉得有用？转发给朋友

#AI #编程 #效率 #开发者 #工具
```

---

**重要提醒**：
- 第1条推文决定成败，必须反复打磨
- 每条推文都要有独立价值
- 推文之间要有"悬念"连接
- 最后1条必须有强CTA
- emoji是视觉语言，善用它们

请开始创作爆款Twitter Thread！🚀
"""

    def _parse_twitter_post(self, response: str, article: Dict[str, Any]) -> Dict[str, Any]:
        """解析Twitter帖子（专家级）"""
        # 分割推文（多种格式支持）
        tweets = self._extract_tweets(response)

        # 如果没有成功分割，尝试其他方法
        if not tweets:
            tweets = self._fallback_parse(response)

        # 确保不超过最大推文数
        tweets = tweets[:self.max_tweets]

        # 提取话题标签
        all_hashtags = self._extract_hashtags(tweets, article)

        # 计算统计信息
        total_chars = sum(len(tweet) for tweet in tweets)
        avg_chars = total_chars // len(tweets) if tweets else 0

        # 分析质量指标
        quality_metrics = self._analyze_quality(tweets)

        return {
            "tweets": tweets,
            "tweet_count": len(tweets),
            "total_characters": total_chars,
            "average_characters": avg_chars,
            "hashtags": all_hashtags,
            "style": self.style,
            "is_thread": len(tweets) > 1,
            "original_article_title": article.get('title'),
            "full_content": '\n\n'.join(tweets),
            "quality_metrics": quality_metrics,
            "hook_type": self._detect_hook_type(tweets[0] if tweets else ""),
            "cta_strength": self._evaluate_cta_strength(tweets[-1] if tweets else "")
        }

    def _extract_tweets(self, response: str) -> List[str]:
        """提取推文（支持多种格式）"""
        tweets = []
        lines = response.strip().split('\n')
        current_tweet = []

        for line in lines:
            line = line.strip()
            if not line:
                if current_tweet:
                    tweet_text = ' '.join(current_tweet).strip()
                    if tweet_text and len(tweet_text) > 10:  # 过滤过短内容
                        tweets.append(tweet_text)
                    current_tweet = []
            else:
                # 处理各种推文标记格式
                if re.match(r'^Tweet \d+:', line, re.IGNORECASE) or \
                   re.match(r'^\d+/?\d+', line) or \
                   line.startswith('Tweet '):
                    if current_tweet:
                        tweet_text = ' '.join(current_tweet).strip()
                        if tweet_text and len(tweet_text) > 10:
                            tweets.append(tweet_text)
                        current_tweet = []
                    # 移除标记前缀
                    line = re.sub(r'^Tweet \d+:\s*', '', line, flags=re.IGNORECASE)
                    line = re.sub(r'^\d+/?\d+\s*', '', line)
                    line = re.sub(r'^Tweet\s+\d+\s*', '', line, flags=re.IGNORECASE)

                # 跳过纯数字标记
                if not re.match(r'^\d+/$', line):
                    current_tweet.append(line)

        # 添加最后一条
        if current_tweet:
            tweet_text = ' '.join(current_tweet).strip()
            if tweet_text and len(tweet_text) > 10:
                tweets.append(tweet_text)

        return tweets

    def _fallback_parse(self, response: str) -> List[str]:
        """备用解析方法"""
        # 按空行分割
        tweets = [t.strip() for t in response.split('\n\n') if t.strip()]

        # 如果还是不行，按单行分割（适合短推文）
        if len(tweets) <= 1:
            tweets = [t.strip() for t in response.split('\n') if t.strip() and len(t.strip()) > 20]

        return tweets[:self.max_tweets]

    def _extract_hashtags(self, tweets: List[str], article: Dict[str, Any]) -> List[str]:
        """提取话题标签（专家级）"""
        # 从推文中提取标签
        all_hashtags = []
        for tweet in tweets:
            hashtags = re.findall(r'#\w+', tweet)
            all_hashtags.extend(hashtags)

        # 如果没有找到标签，智能生成
        if not all_hashtags:
            all_hashtags = self._generate_smart_hashtags(article)

        # 去重并限制数量
        unique_hashtags = list(set(all_hashtags))
        return unique_hashtags[:8]

    def _generate_smart_hashtags(self, article: Dict[str, Any]) -> List[str]:
        """智能生成话题标签"""
        title = article.get('title', '').lower()
        content = article.get('full_content', '').lower()

        hashtags = []

        # 流量标签（必选）
        traffic_tags = ['#AI', '#MachineLearning', '#Tech', '#Developer', '#Coding']
        for tag in traffic_tags:
            if tag.lower() in title or any(t in title for t in tag[1:].split('_')):
                hashtags.append(tag)
                break

        # 根据内容生成精准标签
        if 'python' in title or 'python' in content:
            hashtags.append('#Python')
        if 'javascript' in title or 'javascript' in content or 'js' in title:
            hashtags.append('#JavaScript')
        if 'tool' in title or '工具' in title:
            hashtags.append('#Tools')
        if '效率' in title or 'productivity' in content:
            hashtags.append('#Productivity')
        if '自动化' in title or 'automation' in content:
            hashtags.append('#Automation')

        # 确保至少有3个标签
        while len(hashtags) < 3:
            default_tags = ['#AI', '#Tech', '#Innovation']
            for tag in default_tags:
                if tag not in hashtags:
                    hashtags.append(tag)
                    break

        return hashtags[:5]

    def _analyze_quality(self, tweets: List[str]) -> Dict[str, Any]:
        """分析Thread质量"""
        metrics = {
            "has_hook": False,
            "has_cta": False,
            "has_question": False,
            "emoji_count": 0,
            "number_count": 0,
            "avg_char_count": 0
        }

        if not tweets:
            return metrics

        # 分析第1条推文（钩子）
        first_tweet = tweets[0]
        hook_indicators = ['🚀', '🔥', '⚡', '震惊', '发现', '秘密', '90%', '大多数']
        metrics["has_hook"] = any(indicator in first_tweet for indicator in hook_indicators)

        # 分析最后1条推文（CTA）
        if len(tweets) > 1:
            last_tweet = tweets[-1]
            cta_indicators = ['关注', '转发', '评论', 'Follow', 'Retweet', 'Reply', '点赞']
            metrics["has_cta"] = any(indicator in last_tweet for indicator in cta_indicators)

        # 统计emoji、数字、提问
        for tweet in tweets:
            emoji_count = len(re.findall(r'[🚀🔥💡⚡✅📊🎯📌❌⚠️🎁✨🏆💪👇💬🔄❤️]', tweet))
            metrics["emoji_count"] += emoji_count
            metrics["number_count"] += len(re.findall(r'\d+', tweet))
            if '？' in tweet or '?' in tweet or '吗' in tweet:
                metrics["has_question"] = True

        # 平均字符数
        metrics["avg_char_count"] = sum(len(t) for t in tweets) // len(tweets)

        return metrics

    def _detect_hook_type(self, first_tweet: str) -> str:
        """检测钩子类型"""
        if any(word in first_tweet for word in ['但', '但是', '实际上', '然而']):
            return "counter_intuitive"  # 反常识
        elif any(word in first_tweet for word in ['%', '倍', '数据', '调研']):
            return "data_driven"  # 数据驱动
        elif any(word in first_tweet for word in ['痛', '困扰', '问题', '烦恼']):
            return "pain_point"  # 痛点
        elif any(word in first_tweet for word in ['2026', '晚了', '错过', '最后']):
            return "urgency"  # 紧迫感
        else:
            return "general"

    def _evaluate_cta_strength(self, last_tweet: str) -> str:
        """评估CTA强度"""
        if any(word in last_tweet for word in ['立即', '马上', '现在', '今天']):
            return "strong"
        elif any(word in last_tweet for word in ['关注', '转发', '评论']):
            return "medium"
        else:
            return "weak"

    def _generate_mock_thread(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """生成模拟Twitter Thread（专家级）"""
        title = article.get('title', 'AI技术突破')
        main_title = title.split('：')[0].split(':')[0]

        # 生成8条推文的爆款Thread
        tweets = [
            f"""🚀 我发现一个让90%开发者都惊掉下巴的AI神器...

效率提升70%，代码质量提高50%，调试时间缩短80%！

🧵👇""",

            f"""1/8
❌ 传统方式：
• 写重复代码（2小时）
• 查文档（30分钟）
• 调试Bug（1小时）

✅ 新方式：
• 自动生成（5分钟）
• 智能解释（即时）
• 零Bug优化

差距太大了！""",

            f"""2/8
💡 核心洞察：

AI不是来替代你的
而是来让你成为10x开发者的

关键在于：
❌ 不是复制粘贴
✅ 是理解+运用

你还在纠结重复代码吗？🤔""",

            f"""3/8
📊 真实数据震撼：

某大厂内部测试：
⏱️ 开发时间：60% ↓
🐛 Bug数量：40% ↓
⭐ 代码质量：50% ↑
😊 团队满意度：100%

这不是魔法，是工具的力量！""",

            f"""4/8
🔥 3个核心技巧：

1️⃣ 描述需求，让AI生成框架
2️⃣ 逐层优化，从基础到进阶
3️⃣ 持续学习，积累提示词库

掌握这3个，你就是高手！💪""",

            f"""5/8
⚠️ 常见误区：

❌ 期望AI一次性完美
❌ 不理解直接复制
❌ 过度依赖放弃思考

正确姿势：
✅ AI=助手，你=决策者
✅ 理解原理灵活运用
✅ 持续验证积累经验""",

            f"""6/8
🎯 实战案例：

某创业公司使用AI助手：
• MVP：2个月→2周
• 成本：节省60%
• 质量：显著提升

投资人："执行力超强！"

这就是工具的价值！""",

            f"""7/8
💬 有人问：

"AI会取代程序员吗？"

我的答案：
不会取代
但"会用AI的"会取代"不会用AI的"

未来不是 AI vs 人
而是 会AI的人 vs 不会AI的人

8/8
✨ 总结：

✅ AI是工具不是威胁
✅ 越早使用优势越大
✅ 持续学习保持领先

🚀 立即开始你的AI之旅！

💬 你的选择？评论区聊聊！

🔄 转发给需要的朋友

#AI #编程 #效率 #开发者 #工具"""
        ]

        all_hashtags = ["#AI", "#编程", "#效率", "#开发者", "#工具"]
        total_chars = sum(len(tweet) for tweet in tweets)
        avg_chars = total_chars // len(tweets)

        return {
            "tweets": tweets,
            "tweet_count": len(tweets),
            "total_characters": total_chars,
            "average_characters": avg_chars,
            "hashtags": all_hashtags,
            "style": "viral",
            "is_thread": True,
            "original_article_title": title,
            "full_content": '\n\n'.join(tweets),
            "quality_metrics": {
                "has_hook": True,
                "has_cta": True,
                "has_question": True,
                "emoji_count": 45,
                "number_count": 30,
                "avg_char_count": avg_chars
            },
            "hook_type": "counter_intuitive",
            "cta_strength": "strong"
        }
