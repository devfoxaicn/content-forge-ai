"""
小红书短笔记精炼Agent（快速阅读版本）
生成800-1000字的精简小红书笔记，适合快速浏览和传播
"""

from typing import Dict, Any
import re
from src.agents.base import BaseAgent


class XiaohongshuShortRefinerAgent(BaseAgent):
    """小红书短笔记精炼Agent - 快速阅读版本（800-1000字）"""

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        super().__init__(config, prompts)
        refiner_config = config.get("agents", {}).get("xiaohongshu_short_refiner", {})
        self.style = refiner_config.get("style", "viral")
        self.max_tokens = refiner_config.get("max_tokens", 4000)  # 短笔记需要较少token
        self.llm.max_tokens = self.max_tokens
        self.llm.temperature = 0.98  # 更高的创造性，短内容需要更吸引人
        self.include_test_case = refiner_config.get("include_test_case", True)
        self.target_word_count = refiner_config.get("target_word_count", 900)  # 目标900字（800-1000中间值）
        self.mock_mode = config.get("agents", {}).get("ai_trend_analyzer", {}).get("mock_mode", False)

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """基于长版本精炼小红书短笔记"""
        self.log("开始基于长版本精炼小红书短笔记")

        try:
            # 优先使用长版本，如果没有则使用原始文章
            long_note = state.get("xiaohongshu_long_note")
            article = state.get("longform_article")

            if not long_note and not article:
                raise ValueError("没有找到长笔记或原始文章")

            # 如果有长版本，基于长版本精炼；否则基于原始文章
            if long_note:
                self.log(f"基于长版本精炼短笔记: {long_note['title']}")
                source_content = long_note['full_content']
                source_title = long_note['title']
            else:
                self.log(f"基于原始文章精炼短笔记: {article['title']}")
                source_content = article['full_content']
                source_title = article['title']

            # 预处理：移除长笔记中的---分隔符，避免LLM模仿原文格式
            source_content = source_content.replace('---', '\n')
            self.log("已预处理：移除原文分隔符以避免格式模仿")

            if self.mock_mode:
                self.log("使用Mock模式生成短笔记")
                xhs_note = self._generate_mock_note(article or {"title": source_title, "full_content": source_content})
            else:
                user_prompt = self._build_prompt(state, source_title, source_content)
                response = self._call_llm(user_prompt)
                xhs_note = self._parse_xiaohongshu_note(response, article or {"title": source_title, "full_content": source_content})

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

    def _build_prompt(self, state: Dict[str, Any], source_title: str, source_content: str) -> str:
        """构建短笔记提示词（基于长版本，世界级小红书专家）"""
        prompts = self.prompts.get("prompts", {})
        prompt_template = prompts.get("xiaohongshu_short_refiner", {}).get("user", "")

        target_audience = state.get("target_audience", "技术从业者")

        if prompt_template:
            return prompt_template.format(
                source_title=source_title,
                source_content=source_content,
                target_audience=target_audience,
                target_word_count=self.target_word_count,
                style=self.style
            )
        else:
            # 世界级小红书专家的短笔记提示词（精炼版，保留核心信息）
            return f"""你是世界顶尖的小红书内容专家，精通将深度长文精炼为高传播短笔记。

**长版本标题**：{source_title}

**长版本内容**：
{source_content[:3000]}

---

## 🎯 短笔记精炼策略（世界级标准）

你的任务：将长笔记精炼为800-1000字的**精华版**，要求：

### 📐 结构要求（必须保持6章节）
使用 `---` 分隔以下6个章节，每个章节都要对应长版本：

**第1章：黄金3秒钩子**（2-3句话，制造紧迫感）
- 从长版本开篇提取最震撼的痛点/数据/发现
- 必须包含：数字/对比/情感共鸣

**第2章：核心发现**（3-5个，用 | 分隔）
- 从长版本提取最关键的发现
- 每个发现8字以内，数字驱动

**第3章：为什么有效**（1段核心逻辑）
- 保留长版本的技术原理/核心优势
- 精简到3-4句话，删除背景描述

**第4章：真实效果**（保留关键数据）
- Before/After对比（必保留）
- 1个具体案例（从长版本选最震撼的）
- 用数据说话，时间线压缩

**第5章：核心技巧**（3个技巧+2个避坑）
- 技巧：从长版本选最重要的3个
- 避坑：选最常见的2个错误
- 每个技巧2句话，直击要点

**第6章：行动号召**（2句话CTA）
- 总结核心价值（1句）
- 强有力的行动号召（1句）

### ✍️ 写作标准
- **标题**：15字以内，含emoji，制造好奇
- **字数**：800-1000字，精准控制
- **emoji**：8-12个，点缀关键点
- **短句为主**：每段1-2句话，节奏快
- **数字驱动**：保留所有数据对比
- **情感共鸣**：紧迫感+获得感

### 🚫 删减原则
❌ 删除：背景铺垫、冗余案例、过渡句、重复表达
✅ 保留：核心数据、关键结论、实用技巧、对比效果

### 🔥 世界级标准
- 开篇3秒内抓住注意力
- 每个章节都有信息增量
- 数据真实，对比强烈
- 可操作性强，拿来即用
- 结尾CTA有力，引发行动

**现在开始精炼，记住：这是长笔记的精华浓缩，不是简单删减！**
"""

    def _parse_xiaohongshu_note(self, response: str, article: Dict[str, Any]) -> Dict[str, Any]:
        """解析小红书短笔记"""
        # 提取markdown代码块内容（如果被```markdown包裹）
        markdown_match = re.search(r'```markdown\n(.*?)```', response, re.DOTALL)
        if markdown_match:
            content = markdown_match.group(1).strip()
        else:
            content = response.strip()

        # 后处理：将---分隔符转换为## 1️⃣ 2️⃣ 3️⃣编号格式
        content = self._convert_to_numbered_format(content)
        self.log("已后处理：转换分隔符为编号格式")

    def _convert_to_numbered_format(self, content: str) -> str:
        """将---分隔符转换为## 1️⃣ 2️⃣ 3️⃣编号格式"""
        # 定义章节编号emoji
        number_emojis = ['1️⃣', '2️⃣', '3️⃣', '4️⃣', '5️⃣', '6️⃣']
        # 定义章节类型映射（根据内容特征）
        section_types = [
            '核心发现',      # 第1节
            '架构设计',      # 第2节
            '实测效果',      # 第3节
            '实战技巧',      # 第4节
            '避坑指南',      # 第5节
            '总结'          # 第6节
        ]

        # 按分隔符分割内容
        parts = content.split('---')

        # 过滤空部分
        parts = [p.strip() for p in parts if p.strip()]

        # 如果没有分隔符，返回原内容
        if len(parts) <= 1:
            return content

        # 第一部分是标题和开头，保持不变
        result = parts[0].strip() + '\n\n'

        # 检查是否有"## 开头先说痛点"，如果没有则添加
        if '## 开头先说痛点' not in result and '有没有发现' in result:
            # 找到开头的位置，在其前插入标题
            lines = result.split('\n')
            for i, line in enumerate(lines):
                if '有没有发现' in line or '凌晨' in line or '今天分享' in line:
                    lines.insert(i, '## 开头先说痛点')
                    lines.insert(i + 1, '')
                    break
            result = '\n'.join(lines)

        # 处理后续部分，添加编号
        for i, part in enumerate(parts[1:], start=1):
            if i > len(number_emojis):
                break

            # 跳过空行
            if not part.strip():
                continue

            # 提取第一行作为小标题，如果没有则使用默认类型
            part_lines = part.strip().split('\n')
            first_line = part_lines[0].strip() if part_lines else ''

            # 判断第一行是否像标题（短且没有句号）
            if len(first_line) < 30 and '。' not in first_line and '！' not in first_line and first_line and not first_line.startswith('#'):
                # 第一行作为小标题
                section_title = first_line
                remaining_content = '\n'.join(part_lines[1:]) if len(part_lines) > 1 else ''
            else:
                # 使用默认类型
                section_title = section_types[min(i-1, len(section_types)-1)]
                remaining_content = part.strip()

            # 添加编号章节
            result += f'## {number_emojis[i-1]} {section_title}\n\n'
            if remaining_content.strip():
                result += remaining_content.strip() + '\n\n'

        # 确保结尾有"## 总结一下"
        if '## 总结一下' not in result:
            # 检查最后一段是否像总结
            last_part = parts[-1].strip() if parts else ''
            if last_part and len(last_part) < 200 and ('效率' in last_part or '工具' in last_part or '立刻' in last_part):
                # 最后一段作为总结
                result = result.rstrip() + '\n\n## 总结一下\n\n' + last_part + '\n\n'

        return result.rstrip()

    def _parse_xiaohongshu_note(self, response: str, article: Dict[str, Any]) -> Dict[str, Any]:
        """解析小红书短笔记"""
        # 提取markdown代码块内容（如果被```markdown包裹）
        markdown_match = re.search(r'```markdown\n(.*?)```', response, re.DOTALL)
        if markdown_match:
            content = markdown_match.group(1).strip()
        else:
            content = response.strip()

        # 后处理：将---分隔符转换为## 1️⃣ 2️⃣ 3️⃣编号格式
        content = self._convert_to_numbered_format(content)
        self.log("已后处理：转换分隔符为编号格式")

        # 提取标题（使用转换后的content）
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

    def _convert_to_numbered_format(self, content: str) -> str:
        """将---分隔符转换为## 1️⃣ 2️⃣ 3️⃣编号格式"""
        # 定义章节编号emoji
        number_emojis = ['1️⃣', '2️⃣', '3️⃣', '4️⃣', '5️⃣', '6️⃣']
        # 定义章节类型映射（根据内容特征）
        section_types = [
            '核心发现',      # 第1节
            '架构设计',      # 第2节
            '实测效果',      # 第3节
            '实战技巧',      # 第4节
            '避坑指南',      # 第5节
            '总结'          # 第6节
        ]

        # 按分隔符分割内容
        parts = content.split('---')

        # 过滤空部分
        parts = [p.strip() for p in parts if p.strip()]

        # 如果没有分隔符，返回原内容
        if len(parts) <= 1:
            return content

        # 第一部分是标题和开头，保持不变
        result = parts[0].strip() + '\n\n'

        # 检查是否有"## 开头先说痛点"，如果没有则添加
        if '## 开头先说痛点' not in result and '有没有发现' in result:
            # 找到开头的位置，在其前插入标题
            lines = result.split('\n')
            for i, line in enumerate(lines):
                if '有没有发现' in line or '凌晨' in line or '今天分享' in line:
                    lines.insert(i, '## 开头先说痛点')
                    lines.insert(i + 1, '')
                    break
            result = '\n'.join(lines)

        # 处理后续部分，添加编号
        for i, part in enumerate(parts[1:], start=1):
            if i > len(number_emojis):
                break

            # 跳过空行
            if not part.strip():
                continue

            # 提取第一行作为小标题，如果没有则使用默认类型
            part_lines = part.strip().split('\n')
            first_line = part_lines[0].strip() if part_lines else ''

            # 判断第一行是否像标题（短且没有句号）
            if len(first_line) < 30 and '。' not in first_line and '！' not in first_line and first_line and not first_line.startswith('#'):
                # 第一行作为小标题
                section_title = first_line
                remaining_content = '\n'.join(part_lines[1:]) if len(part_lines) > 1 else ''
            else:
                # 使用默认类型
                section_title = section_types[min(i-1, len(section_types)-1)]
                remaining_content = part.strip()

            # 添加编号章节
            result += f'## {number_emojis[i-1]} {section_title}\n\n'
            if remaining_content.strip():
                result += remaining_content.strip() + '\n\n'

        # 确保结尾有"## 总结一下"
        if '## 总结一下' not in result:
            # 检查最后一段是否像总结
            last_part = parts[-1].strip() if parts else ''
            if last_part and len(last_part) < 200 and ('效率' in last_part or '工具' in last_part or '立刻' in last_part):
                # 最后一段作为总结
                result = result.rstrip() + '\n\n## 总结一下\n\n' + last_part + '\n\n'

        return result.rstrip()

    def _generate_mock_note(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """生成模拟小红书短笔记（世界级标准）"""
        title = article.get('title', 'AI技术突破')
        main_title = title.split('：')[0].split(':')[0]

        mock_content = f"""# 7天上手！{main_title}真香💥

每天加班到10点？用了一周直接早下班2小时😭

---

## ✨ 核心发现

200K上下文一口气读完项目｜像真同事一样主动干活｜编码效率直接翻倍

---

## 💡 为什么好用

传统AI要反复解释上下文，每次复制粘贴到手酸

这个直接读项目文件，自动理解依赖关系，不用你说它就懂

---

## 📊 真实数据

编码：4h→2h（↓50%）
Bug修复：1.5h→0.5h（↓67%）

**真实案例**：重构1000行老代码
传统方法7小时｜用它2小时搞定

---

## 📌 3个技巧

1️⃣ 像同事一样沟通
别说"帮我X"，说"我们一起解决X"

2️⃣ 第一次就给完整上下文
上传项目结构，不用反复说

3️⃣ 建立标准流程
需求→代码→测试→文档，一套模板

❌ 别指望一次完美，要迭代优化

---

## 💬 总结

不是AI取代你，是会用AI的人取代你！早用早享受🔥

🔗 搜索 `{main_title}`

#AI编程 #程序员 #效率神器
"""

        word_count = len(mock_content)
        emoji_count = len(re.findall(r'[🚀🔥💡⚡✅📊📈💰⏱️🎯📌❌⚠️🚨🎁✨🏆💪👇💬🔄❤️😭😱]', mock_content))

        return {
            "title": f"7天上手！{main_title}真香💥",
            "full_content": mock_content,
            "hashtags": ["#AI编程", "#程序员", "#效率神器"],
            "word_count": word_count,
            "original_article_word_count": article.get('word_count', 40000),
            "compression_ratio": f"{(1 - word_count / article.get('word_count', 40000)) * 100:.1f}%",
            "emoji_count": emoji_count,
            "note_type": "short",
            "target_word_count": 900
        }
