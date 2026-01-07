"""
推特帖子生成Agent
将专业文章精炼为爆款Twitter帖子
"""

from typing import Dict, Any
import re
from src.agents.base import BaseAgent


class TwitterGeneratorAgent(BaseAgent):
    """Twitter帖子生成Agent"""

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        super().__init__(config, prompts)
        twitter_config = config.get("agents", {}).get("twitter_generator", {})
        self.style = twitter_config.get("style", "engaging")  # engaging, professional, casual
        self.thread_mode = twitter_config.get("thread_mode", True)  # 是否生成thread形式
        self.max_tweets = twitter_config.get("max_tweets", 5)  # thread最多几条
        self.max_tokens = twitter_config.get("max_tokens", 1500)
        self.llm.max_tokens = self.max_tokens
        self.llm.temperature = 0.9  # Twitter需要更高的创意性
        self.mock_mode = config.get("agents", {}).get("ai_trend_analyzer", {}).get("mock_mode", False)

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成Twitter帖子

        Args:
            state: 当前工作流状态

        Returns:
            Dict[str, Any]: 更新后的状态
        """
        self.log("开始生成Twitter帖子")

        try:
            # 获取长文本文章
            article = state.get("longform_article")
            if not article:
                raise ValueError("没有找到长文本文章")

            self.log(f"基于文章生成Twitter: {article['title']}")

            # Mock模式或API失败时生成模拟数据
            if self.mock_mode:
                self.log("使用Mock模式生成Twitter帖子")
                twitter_post = self._generate_mock_thread(article)
            else:
                # 构建提示词
                user_prompt = self._build_prompt(state, article)

                # 调用LLM生成Twitter帖子
                response = self._call_llm(user_prompt)

                # 解析Twitter帖子
                twitter_post = self._parse_twitter_post(response, article)

            self.log(f"成功生成Twitter帖子: {twitter_post['tweet_count']}条推文")

            return {
                **state,
                "twitter_post": twitter_post,
                "current_step": "twitter_generator_completed"
            }
        except Exception as e:
            self.log(f"Twitter帖子生成失败: {str(e)}", "ERROR")
            # 失败时也返回模拟数据
            self.log("使用模拟数据继续测试", "WARNING")
            article = state.get("longform_article", {"title": "AI技术", "full_content": "内容"})
            twitter_post = self._generate_mock_thread(article)
            return {
                **state,
                "twitter_post": twitter_post,
                "current_step": "twitter_generator_completed"
            }

    def _build_prompt(self, state: Dict[str, Any], article: Dict[str, Any]) -> str:
        """
        构建生成提示词

        Args:
            state: 当前状态
            article: 长文本文章

        Returns:
            str: 提示词
        """
        prompts = self.prompts.get("prompts", {})
        prompt_template = prompts.get("twitter_generator", {}).get("user", "")

        # 提取文章核心内容（前1500字）
        content_preview = article['full_content'][:1500] + "..." if len(article['full_content']) > 1500 else article['full_content']

        # 获取目标受众
        target_audience = state.get("target_audience", "技术从业者")
        topic = state.get("topic", "AI技术")

        thread_instruction = f"生成一个Twitter thread（最多{self.max_tweets}条推文）" if self.thread_mode else "生成一条推文"

        if prompt_template:
            return prompt_template.format(
                article_title=article['title'],
                article_content=content_preview,
                target_audience=target_audience,
                style=self.style,
                thread_mode=self.thread_mode,
                max_tweets=self.max_tweets
            )
        else:
            # 使用默认提示词
            return f"""你是一位Twitter爆款内容创作者，擅长将技术文章转化为高互动的Twitter帖子。

**原文章标题**：{article['title']}

**原文章内容**：
{content_preview}

**任务**：{thread_instruction}

**Twitter帖子要求**：

1. **结构要求**：
   - **第1条推文**：必须有强钩子（Hook）
     * 用反常识观点、惊人数据、或痛点开场
     * 1-2句话点明主题
     * 结尾用"🧵"提示这是thread
   - **中间推文**：
     * 每条推文讲一个核心要点
     * 使用简单易懂的语言
     * 适当使用emoji增强表达
     * 每条控制在2-3句话
   - **最后1条**：
     * 总结价值
     * 行动号召（关注/转发/评论）
     * 相关话题标签

2. **写作风格**：{self.style}
   - engaging: 引人入胜，用故事和案例
   - professional: 专业权威，数据和事实
   - casual: 轻松友好，像对话一样

3. **爆款技巧**：
   - 开头用数字："5个技巧"、"3个误区"
   - 用对比："大多数人都...但其实..."
   - 用疑问："你知道...吗？"
   - 用权威："研究表明..."
   - 用紧迫感："现在开始..."

4. **字符限制**：
   - 每条推文控制在250字符以内（留空间给emoji）
   - 使用简洁的表达
   - 避免冗长句子

5. **emoji使用**：
   - 适当使用emoji（每条1-3个）
   - 相关emoji：🚀 🔥 💡 ✅ ⚡ 📊 🎯
   - 不要过度使用

6. **话题标签**：
   - 在最后一条推文添加3-5个相关话题标签
   - 使用英文标签：#AI #MachineLearning #Tech
   - 混合热门和精准标签

**输出格式**（每条推文之间用空行分隔）：

```
Tweet 1内容...

Tweet 2内容...

Tweet 3内容...

...

最后一条... 标签1 标签2 标签3
```

**目标受众**：{target_audience}
**主题**：{topic}

**重要**：
- 确保每条推文都有独立价值
- 推文之间要有逻辑连贯性
- 语言要口语化、易理解
- 必须有明确的行动号召

请开始生成Twitter帖子！
"""

    def _parse_twitter_post(self, response: str, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析Twitter帖子

        Args:
            response: LLM响应
            article: 原文章

        Returns:
            Dict[str, Any]: 结构化Twitter帖子
        """
        # 分割推文（按空行或"Tweet"标记）
        tweets = []
        lines = response.strip().split('\n')

        current_tweet = []
        for line in lines:
            line = line.strip()
            if not line:
                if current_tweet:
                    tweet_text = ' '.join(current_tweet).strip()
                    if tweet_text:
                        tweets.append(tweet_text)
                    current_tweet = []
            else:
                # 跳过"Tweet 1:"这样的标记
                if re.match(r'^Tweet \d+:', line, re.IGNORECASE):
                    if current_tweet:
                        tweet_text = ' '.join(current_tweet).strip()
                        if tweet_text:
                            tweets.append(tweet_text)
                        current_tweet = []
                    # 提取实际内容（去掉"Tweet 1:"前缀）
                    line = re.sub(r'^Tweet \d+:\s*', '', line, flags=re.IGNORECASE)
                current_tweet.append(line)

        # 添加最后一条
        if current_tweet:
            tweet_text = ' '.join(current_tweet).strip()
            if tweet_text:
                tweets.append(tweet_text)

        # 如果没有成功分割，尝试按"Tweet"标记分割
        if not tweets:
            tweet_matches = re.split(r'Tweet \d+:', response, flags=re.IGNORECASE)
            tweets = [t.strip() for t in tweet_matches if t.strip()]

        # 提取话题标签
        all_hashtags = []
        for tweet in tweets:
            hashtags = re.findall(r'#\w+', tweet)
            all_hashtags.extend(hashtags)

        # 如果没有找到标签，从原文章标签生成
        if not all_hashtags:
            original_tags = article.get('tags', [])
            all_hashtags = [f"#{tag.replace(' ', '').replace('-', '')}" for tag in original_tags[:5]]

        # 去重
        all_hashtags = list(set(all_hashtags))[:5]

        # 计算总字符数
        total_chars = sum(len(tweet) for tweet in tweets)

        return {
            "tweets": tweets,
            "tweet_count": len(tweets),
            "total_characters": total_chars,
            "average_characters": total_chars // len(tweets) if tweets else 0,
            "hashtags": all_hashtags,
            "style": self.style,
            "is_thread": len(tweets) > 1,
            "original_article_title": article.get('title'),
            "full_content": '\n\n'.join(tweets)  # 完整内容用于展示
        }

    def _generate_mock_thread(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成模拟Twitter Thread（用于Mock模式或API失败时）

        Args:
            article: 长文本文章

        Returns:
            Dict[str, Any]: Twitter帖子
        """
        title = article.get('title', 'AI技术突破')
        # 提取主标题
        main_title = title.split('：')[0].split(':')[0]

        # 生成5条推文的Thread
        tweets = [
            f"🚀 {main_title}：技术突破太惊艳了！\n\n性能提升3-5倍，资源消耗降低60%，准确率提高25%！",
            f"💡 关键创新点：\n\n✅ API设计简洁\n✅ 文档完善易上手\n✅ 社区活跃支持好\n\n从大厂到创业公司都在用！",
            f"📊 实测数据（某互联网大厂）：\n\n• 响应时间缩短50%\n• 运维成本降低30%\n• 用户满意度提升20%\n\n真实有效！",
            f"🔥 最佳实践：\n\n1. 渐进式迁移\n2. 充分测试\n3. 团队培训\n4. 持续优化\n\n现在就是布局的最佳时机！",
            f"✨ 总结：\n\n✅ 技术成熟\n✅ 资源丰富\n✅ 价值验证\n🚀 空间巨大\n\n#AI #技术 #创新"
        ]

        all_hashtags = ["#AI", "#技术", "#创新"]
        total_chars = sum(len(tweet) for tweet in tweets)

        return {
            "tweets": tweets,
            "tweet_count": len(tweets),
            "total_characters": total_chars,
            "average_characters": total_chars // len(tweets),
            "hashtags": all_hashtags,
            "style": "engaging",
            "is_thread": True,
            "original_article_title": title,
            "full_content": '\n\n'.join(tweets)
        }
