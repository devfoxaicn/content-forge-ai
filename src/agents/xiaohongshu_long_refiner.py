"""
小红书长笔记精炼Agent（深度内容版本）
生成约2000字的深度小红书笔记，适合技术教程、经验分享类内容
"""

from typing import Dict, Any
import re
from src.agents.base import BaseAgent


class XiaohongshuLongRefinerAgent(BaseAgent):
    """小红书长笔记精炼Agent - 深度内容版本（约2000字）"""

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        super().__init__(config, prompts)
        refiner_config = config.get("agents", {}).get("xiaohongshu_long_refiner", {})
        self.style = refiner_config.get("style", "viral")
        self.max_tokens = refiner_config.get("max_tokens", 12000)  # 长笔记需要更多token
        self.llm.max_tokens = self.max_tokens
        self.llm.temperature = 0.95
        self.include_test_case = refiner_config.get("include_test_case", True)
        self.use_emotional_language = refiner_config.get("use_emotional_language", True)
        self.target_word_count = refiner_config.get("target_word_count", 2000)
        self.mock_mode = config.get("agents", {}).get("ai_trend_analyzer", {}).get("mock_mode", False)

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """精炼技术文章为小红书长笔记（深度内容版本）"""
        self.log("开始精炼小红书长笔记（深度内容版本）")

        try:
            article = state.get("longform_article")
            if not article:
                raise ValueError("没有找到长文本文章")

            self.log(f"精炼文章: {article['title']}")

            if self.mock_mode:
                self.log("使用Mock模式生成长笔记")
                xhs_note = self._generate_mock_note(article)
            else:
                user_prompt = self._build_prompt(state, article)
                response = self._call_llm(user_prompt)
                xhs_note = self._parse_xiaohongshu_note(response, article)

            self.log(f"成功生成长笔记，字数: {xhs_note['word_count']}")
            return {
                **state,
                "xiaohongshu_long_note": xhs_note,
                "current_step": "xiaohongshu_long_refiner_completed"
            }
        except Exception as e:
            self.log(f"长笔记精炼失败: {str(e)}", "ERROR")
            self.log("使用模拟数据继续测试", "WARNING")
            article = state.get("longform_article", {"title": "AI技术", "full_content": "内容"})
            xhs_note = self._generate_mock_note(article)
            return {
                **state,
                "xiaohongshu_long_note": xhs_note,
                "current_step": "xiaohongshu_long_refiner_completed"
            }

    def _build_prompt(self, state: Dict[str, Any], article: Dict[str, Any]) -> str:
        """构建长笔记提示词"""
        prompts = self.prompts.get("prompts", {})
        prompt_template = prompts.get("xiaohongshu_long_refiner", {}).get("user", "")

        # 提取更多文章内容（5000字符，长笔记需要更多上下文）
        content_preview = article['full_content'][:5000] + "..." if len(article['full_content']) > 5000 else article['full_content']
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
            # 长笔记默认提示词（6章节结构）
            return f"""你是一位小红书深度内容创作者，擅长创作技术教程和经验分享笔记。

**原文章标题**：{article['title']}

**原文章内容**：
{content_preview}

---

## 📝 长笔记创作要求（6章节结构）

**格式限制**：
- 标题：20字以内，含emoji
- 正文：1500-2500字
- 使用`---`分隔6个章节
- 严格遵循6章节结构

**6个章节结构**：
1. **引人入胜的开篇**（黄金3秒钩子）
   - 亲身实测/数据震撼/痛点共鸣
   - 制造紧迫感和好奇心

2. **核心发现/亮点**
   - 3-5个核心发现，用|分隔
   - 每个发现一句话描述

3. **为什么这么好用/技术原理**
   - 详细的技术背景或原理
   - 核心优势解析
   - 与传统方法对比

4. **使用体验/效果数据**
   - 时间线体验（第1天/第3天/第7天/第30天）
   - 真实数据对比（Before vs After）
   - 具体案例展示

5. **核心技巧/避坑指南**
   - 3-5个核心技巧（详细说明）
   - 3个常见错误及解决方案

6. **总结 + 行动号召**
   - 核心价值总结
   - 强有力的CTA
   - 话题标签（5个）

**写作风格**：{self.style}（爆款风格）
- 数字驱动，对比强烈
- emoji丰富（15-20个）
- 情感共鸣，紧迫感强
- 包含真实数据和案例

现在开始创作小红书深度长笔记！🚀
"""

    def _parse_xiaohongshu_note(self, response: str, article: Dict[str, Any]) -> Dict[str, Any]:
        """解析小红书长笔记"""
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
            "note_type": "long",
            "target_word_count": self.target_word_count
        }

    def _generate_mock_note(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """生成模拟小红书长笔记"""
        title = article.get('title', 'AI技术突破')
        main_title = title.split('：')[0].split(':')[0]

        mock_content = f"""# 亲测30天！{main_title}完整实战攻略💥

宝子们！被问了100次，今天终于决定分享这个完整的实战攻略！

使用前我每天加班到10点，现在6点就能下班了！用了一个月深度体验，整理了这份超详细教程！😭

---

## ✨ 核心发现

**200K上下文**：一口气读完整个项目｜**智能理解**：不用反复解释｜**主动协作**：像真正的同事

---

## 💡 技术背景

传统AI工具的痛点：
- 只能被动问答，需要反复解释上下文
- 无法直接操作文件，需要复制粘贴
- 不理解项目结构，每次都要重新说明

{main_title}通过以下技术解决：
1. 深度上下文理解：支持200K token上下文窗口
2. 主动文件操作：可以直接读取、编辑、创建文件
3. 智能项目分析：自动理解项目结构和依赖关系

这是质的飞跃！🚀

---

## 🚀 完整部署步骤

### 第1步：环境准备

确保你的系统满足以下要求：
- Node.js 18+ 或 Python 3.8+
- 8GB以上内存
- 稳定的网络连接

安装命令：
```bash
npm install -g @claude-code/cli
```

### 第2步：API配置

1. 注册账号获取API密钥
2. 配置环境变量：
```bash
export ANTHROPIC_API_KEY="your-key-here"
```

### 第3步：项目初始化

在你的项目目录运行：
```bash
claude-code init
```

系统会自动：
- 分析项目结构
- 识别技术栈
- 创建配置文件

### 第4步：开始使用

启动交互式界面：
```bash
claude-code chat
```

现在你可以像和同事聊天一样和AI协作了！

### 第5步：高级配置

创建自定义配置文件`.claude-code.yaml`：
```yaml
preferences:
  code_style: "your-style"
  framework: "your-framework"
  test_framework: "jest"
```

---

## 📚 实战案例1：重构老旧代码

**问题**：一个1000行的遗留文件，逻辑混乱

**传统方法**：
- 手动阅读：2小时
- 理解逻辑：1小时
- 重构代码：3小时
- 测试验证：1小时
- **总计：7小时**

**使用{main_title}**：
1. 提交文件给AI
2. 说明重构目标
3. AI分析并生成新代码
4. 人工审查调整
5. **总计：2小时**

**效率提升：71%**

---

## 📚 实战案例2：快速原型开发

**需求**：开发一个REST API接口

**使用流程**：
1. 描述需求给AI（5分钟）
2. AI生成数据库模型（10分钟）
3. AI生成API路由（15分钟）
4. AI生成测试用例（10分钟）
5. 人工审查优化（20分钟）

**总计：1小时完成完整API**

传统方式需要：3-4小时

---

## 📈 30天使用体验

**第1天**：
花了2小时配置环境，有点懵。不知道怎么高效提问，经常重复解释。

**第3天**：
开始熟悉最佳实践，学会给予完整上下文。效率提升约30%。

**第7天**：
完全上手！建立了自己的工作流模板。效率提升约50%。

**第14天**：
发现高级功能（多文件编辑、命令执行）。效率提升约70%。

**第30天**：
已经离不开它了！工作效率翻倍，每天准时下班。

**真实数据对比**：
- 编码时间：4小时→2小时（↓50%）
- Bug修复：1.5小时→0.5小时（↓67%）
- 文档编写：1小时→0.3小时（↓70%）
- 加班时间：减少60%
- 代码质量：提升40%（通过测试覆盖率衡量）

---

## 📌 5个核心技巧

1️⃣ **像同事一样沟通，不是命令**
不要说"帮我写个函数"，试着说"我们来重构这个模块，提升性能"。给予AI决策参与权，它能预判你的需求。

2️⃣ **给予完整上下文**
第一次交互时，把项目结构、技术栈、编码规范都告诉AI。使用`/context`命令上传关键文件，让它理解全局。

3️⃣ **建立专属工作流**
设计标准流程：
- 需求分析 → 代码生成 → 测试编写 → 文档更新
保存为模板，下次一键调用

4️⃣ **善用迭代优化**
第一版代码通常不够完美。通过3-4轮迭代：
- "优化性能"
- "增加错误处理"
- "改进可读性"
最终得到高质量代码。

5️⃣ **建立知识库**
把常见解决方案保存为snippets。AI会学习你的模式，写出来的代码越来越像你的风格。

---

## ⚠️ 避坑指南

❌ **错误1：期望一次性完美**
AI生成的第一版代码通常需要优化。要通过多轮迭代逐步完善。

✅ **正确做法**：分步骤验证，逐步优化

❌ **错误2：不理解就复制**
不要直接复制粘贴AI生成的代码！要理解每行代码的作用。

✅ **正确做法**：先让AI解释代码逻辑，再审查使用

❌ **错误3：过度依赖放弃思考**
AI是助手，不是替代。你才是决策者和架构师。

✅ **正确做法**：AI负责实现，你负责设计和把关

---

## 💬 深度总结

30天深度使用后，我的结论是：

**不是AI取代你，而是会用AI的人取代你！**

{main_title}不仅仅是一个工具，它是：
- 🚀 **效率放大器**：让优秀的人更优秀
- 🧠 **知识外脑**：补充你的技能短板
- 🤝 **协作伙伴**：像真正的同事一样工作

**早用早享受！现在开始还来得及！**

🔗 官方文档：搜索关键词 `{main_title}`

💬 **有问题评论区留言，我会一一回复！**

---

#AI编程 #零基础入门 #程序员 #效率神器 #生产力工具 #技术教程 #深度好文 #职场必备
"""

        word_count = len(mock_content)
        emoji_count = len(re.findall(r'[🚀🔥💡⚡✅📊📈💰⏱️🎯📌❌⚠️🚨🎁✨🏆💪👇💬🔄❤️😭😱]', mock_content))

        return {
            "title": f"亲测30天！{main_title}完整实战攻略💥",
            "full_content": mock_content,
            "hashtags": ["#AI编程", "#零基础入门", "#程序员", "#效率神器", "#生产力工具", "#技术教程", "#深度好文", "#职场必备"],
            "word_count": word_count,
            "original_article_word_count": article.get('word_count', 40000),
            "compression_ratio": f"{(1 - word_count / article.get('word_count', 40000)) * 100:.1f}%",
            "emoji_count": emoji_count,
            "note_type": "long",
            "target_word_count": 2000
        }
