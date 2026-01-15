"""
微信公众号文章生成 Agent（专家版）
将长文本技术文章转换为适合微信公众号发布的专业级HTML格式

特点：
- 专业级排版设计
- 移动端优化
- SEO友好
- 交互元素增强
"""

from typing import Dict, Any
from src.agents.base import BaseAgent


class WechatGeneratorAgent(BaseAgent):
    """微信公众号文章生成 Agent - 专家版"""

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        super().__init__(config, prompts)
        wechat_config = config.get("agents", {}).get("wechat_generator", {})
        self.enable_toc = wechat_config.get("enable_toc", True)  # 目录导航
        self.enable_reading_progress = wechat_config.get("enable_reading_progress", True)  # 阅读进度
        self.enable_share_buttons = wechat_config.get("enable_share_buttons", True)  # 分享按钮
        self.theme = wechat_config.get("theme", "professional")  # professional, minimal, colorful
        self.max_tokens = wechat_config.get("max_tokens", 16000)
        self.llm.max_tokens = self.max_tokens
        self.llm.temperature = 0.7
        self.mock_mode = config.get("agents", {}).get("wechat_generator", {}).get("mock_mode", False)

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """生成微信公众号文章"""
        self.log("开始生成微信公众号文章（专家级排版）")

        try:
            article = state.get("longform_article")
            if not article:
                raise ValueError("没有找到长文本文章")

            self.log(f"转换文章: {article['title']}")

            if self.mock_mode:
                self.log("使用Mock模式生成微信文章")
                wechat_content = self._generate_mock_wechat(article)
            else:
                user_prompt = self._build_prompt(state, article)
                response = self._call_llm(user_prompt)
                wechat_content = self._parse_wechat_content(response, article)

            self.log(f"成功生成微信文章，字数: {wechat_content['word_count']}")
            return {
                **state,
                "wechat_article": wechat_content,
                "current_step": "wechat_generator_completed"
            }
        except Exception as e:
            self.log(f"微信文章生成失败: {str(e)}", "ERROR")
            article = state.get("longform_article", {"title": "AI技术", "full_content": "内容"})
            wechat_content = self._generate_mock_wechat(article)
            return {
                **state,
                "wechat_article": wechat_content,
                "current_step": "wechat_generator_completed"
            }

    def _build_prompt(self, state: Dict[str, Any], article: Dict[str, Any]) -> str:
        """构建专家级提示词"""
        prompts = self.prompts.get("prompts", {})
        prompt_template = prompts.get("wechat_generator", {}).get("user", "")

        content = article['full_content']
        if len(content) > 12000:
            content = content[:12000] + "\n\n... (内容已截断，完整版请查看原文)"

        if not prompt_template:
            prompt_template = """
你是一位微信公众号内容创作专家，擅长将技术文章转化为专业、易读、高互动的微信文章。

**原标题**：{title}

**文章内容**：
{content}

---

## 📱 微信文章专家级创作指南

### 1️⃣ 标题优化
- 使用吸引人但不夸张的标题
- 长度建议：15-25字
- 可以包含emoji，但不要过度
- 避免标题党，确保内容与标题匹配

### 2️⃣ 开头设计（前300字决定留存率）
**黄金3法则**：
- **第1句**：痛点共鸣或数据震撼
- **第2句**：文章价值预告
- **第3句**：建立信任感（经验、数据、权威）

**示例结构**：
```
【痛点】你是否遇到过...？
【价值】今天分享的XXX能帮你解决...
【信任】亲测/实测数据，XXX效率提升70%
```

### 3️⃣ 正文结构优化
**使用以下元素增强可读性**：

**a) 目录导航**（长文必备）
```
📖 本期目录
01 核心概念解析
02 实战应用场景
03 最佳实践指南
04 常见问题解答
```

**b) 章节标题层次**
- H1: 主标题（文章顶部）
- H2: 大章节标题（左右绿边框）
- H3: 小节标题（加粗，黑色）
- H4: 小知识点（用引用框）

**c) 重点内容突出**
- 使用绿色强调关键数据（`<span style="color: #07c160">`)
- 使用引用框展示金句
- 使用提示框展示注意事项
- 使用代码块展示技术细节

**d) 移动端优化**
- 段落控制在3-5句话
- 每段开头用emoji或数字标记
- 使用分隔线区分不同主题
- 图片/代码块前后留白

### 4️⃣ 交互元素
**添加以下互动元素**：
- 💬 思考题：在关键章节后添加"💭 思考：..."
- 📊 数据卡片：用表格或图表展示数据
- ⚠️ 注意事项：用警告框提醒重要信息
- ✅ 检查清单：用勾选框列出步骤
- 🔗 相关阅读：推荐相关文章

### 5️⃣ 结尾设计（黄金3步走）
**第1步**：总结核心价值（3-5个要点）
**第2步**：行动号召（关注、点赞、收藏、分享）
**第3步**：互动引导（提问、话题讨论）

**示例**：
```
📌 总结一下今天分享的要点：
✅ 要点1：...
✅ 要点2：...
✅ 要点3：...

💡 觉得有用？
👍 点赞让更多人看到
⭐ 收藏方便以后查阅
🔄 转发给需要的朋友

💬 评论区聊聊：
你在XXX方面遇到过什么问题？
或者有什么好的经验分享？
```

### 6️⃣ SEO优化
- 在文章开头添加150字摘要
- 在全文中自然穿插3-5个关键词
- 在结尾添加相关话题标签

### 7️⃣ 输出格式
```
【标题】（优化后的标题）

【摘要】（150字摘要，包含核心关键词）

【正文】
（使用Markdown格式，会自动转换为HTML）
```

请开始创作，确保内容既专业又符合微信公众号调性！
"""

        return prompt_template.format(
            title=article['title'],
            content=content
        )

    def _parse_wechat_content(self, response: str, article: Dict[str, Any]) -> Dict[str, Any]:
        """解析微信内容"""
        lines = response.split('\n')
        title = article['title']
        summary = ""
        full_content = response

        # 提取标题
        for i, line in enumerate(lines):
            if line.startswith("【标题】"):
                if i + 1 < len(lines):
                    title = lines[i + 1].strip()
                    break
            elif line.startswith("# ") and not line.startswith("## "):
                title = line[2:].strip()
                break

        # 提取摘要
        for i, line in enumerate(lines):
            if line.startswith("【摘要】"):
                if i + 1 < len(lines):
                    summary_lines = []
                    j = i + 1
                    while j < len(lines) and not lines[j].startswith("【"):
                        summary_lines.append(lines[j])
                        j += 1
                    summary = '\n'.join(summary_lines).strip()
                    break

        # 生成专业级HTML
        html_content = self._generate_professional_html(full_content, title)

        # 生成配图提示词
        image_prompts = self._generate_image_prompts(full_content, title)

        return {
            "title": title,
            "html_content": html_content,
            "markdown_content": full_content,
            "word_count": len(full_content),
            "reading_time": f"{len(full_content) // 500 + 1}",
            "cover_image_prompt": image_prompts[0] if image_prompts else f"生成一个关于{title}的微信封面图，技术风格，蓝绿渐变，简洁大气，高质量",
            "image_prompts": image_prompts,
            "summary": summary or full_content[:150] + "...",
            "seo_keywords": self._extract_keywords(full_content)
        }

    def _extract_keywords(self, content: str) -> list:
        """提取SEO关键词"""
        # 简单的关键词提取（可以后续用NLP优化）
        tech_keywords = ["AI", "人工智能", "机器学习", "深度学习", "Python", "JavaScript", "代码", "开发", "算法", "数据", "架构", "性能", "优化", "实战"]
        found = [kw for kw in tech_keywords if kw in content]
        return found[:5] if found else ["AI技术", "开发"]

    def _generate_image_prompts(self, content: str, title: str) -> list:
        """生成文章配图提示词（用于AI绘画）"""
        prompts = []

        # 1. 封面图（最重要）
        cover_prompt = f"""【封面图】
位置：文章开头
尺寸：900x500px（16:9）
描述：生成一个关于"{title}"的微信公众号封面图
风格要求：
- 现代科技风格，蓝绿渐变配色（#07c160到#1890ff）
- 扁平化设计，简洁大气
- 包含主题相关的图标或元素
- 高质量，适合作为首图吸引点击
- 字体清晰易读
- 背景干净，突出主题
"""
        prompts.append(cover_prompt.strip())

        # 2. 概念图（技术原理）
        concept_prompt = f"""【概念图-技术原理】
位置：文章中段（介绍技术背景时）
尺寸：900x500px
描述：生成"{title}"的核心技术概念图
风格要求：
- 信息图表风格，清晰易懂
- 使用流程图或架构图形式
- 蓝白色调，专业感强
- 展示技术流程或架构关系
- 包含适当的图标和箭头指示
- 适合技术类公众号配图
"""
        prompts.append(concept_prompt.strip())

        # 3. 对比图（前后对比）
        comparison_prompt = f"""【对比图-效果展示】
位置：文章中后段（展示效果对比时）
尺寸：900x500px
描述：生成"{title}"使用前后的效果对比图
风格要求：
- 左右对比布局（Before vs After）
- 使用暗色背景突出对比
- 绿色箭头表示改进提升
- 数据可视化风格
- 包含具体的数字或指标
- 视觉冲击力强
"""
        prompts.append(comparison_prompt.strip())

        # 4. 实景图（应用场景）
        scenario_prompt = f"""【实景图-应用场景】
位置：文章后段（介绍应用场景时）
尺寸：900x500px
描述：生成"{title}"在实际工作场景中的应用图
风格要求：
- 办公室或工作场景
- 电脑屏幕展示相关界面
- 温暖的灯光效果
- 现代简约风格
- 人物背影或侧面（聚焦工作）
- 专业感强
"""
        prompts.append(scenario_prompt.strip())

        # 5. 总结图（核心要点）
        summary_prompt = f"""【总结图-核心要点】
位置：文章结尾（总结部分）
尺寸：900x500px
描述：生成"{title}"的核心要点总结图
风格要求：
- 清单式布局（checklist风格）
- 3-5个关键点
- 使用大号emoji或图标
- 浅色背景，清晰易读
- 绿色勾选标记
- 适合截图分享
"""
        prompts.append(summary_prompt.strip())

        return prompts

    def _generate_professional_html(self, markdown_content: str, title: str) -> str:
        """生成专业级HTML（带所有优化）"""
        html_lines = []

        # 根据主题选择配色方案
        if self.theme == "professional":
            primary_color = "#07c160"  # 微信绿
            accent_color = "#f5a623"
            bg_color = "#ffffff"
        elif self.theme == "minimal":
            primary_color = "#333333"
            accent_color = "#666666"
            bg_color = "#fafafa"
        else:  # colorful
            primary_color = "#1890ff"
            accent_color = "#7cfc00"
            bg_color = "#ffffff"

        html_lines.append(f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        /* 全局样式 */
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "PingFang SC", "Microsoft YaHei", sans-serif;
            font-size: 16px;
            line-height: 1.8;
            color: #333;
            max-width: 677px;
            margin: 0 auto;
            padding: 20px;
            background-color: {bg_color};
        }}

        /* 标题样式 */
        h1 {{
            font-size: 26px;
            font-weight: 700;
            color: #000;
            margin: 30px 0 20px;
            text-align: center;
            line-height: 1.4;
            letter-spacing: 0.5px;
        }}

        h2 {{
            font-size: 22px;
            font-weight: 700;
            color: #000;
            margin: 40px 0 20px;
            padding-left: 16px;
            border-left: 5px solid {primary_color};
            position: relative;
            line-height: 1.5;
        }}

        h2::before {{
            content: "";
            position: absolute;
            left: -21px;
            top: 50%;
            transform: translateY(-50%);
            width: 8px;
            height: 8px;
            background: {primary_color};
            border-radius: 50%;
        }}

        h3 {{
            font-size: 19px;
            font-weight: 600;
            color: #222;
            margin: 30px 0 15px;
            display: flex;
            align-items: center;
        }}

        h3::before {{
            content: "";
            display: inline-block;
            width: 4px;
            height: 16px;
            background: {accent_color};
            margin-right: 8px;
            border-radius: 2px;
        }}

        h4 {{
            font-size: 17px;
            font-weight: 600;
            color: #444;
            margin: 20px 0 10px;
        }}

        /* 段落样式 */
        p {{
            margin: 16px 0;
            text-align: justify;
            text-indent: 2em;
        }}

        p:first-of-type {{
            text-indent: 0;
        }}

        /* 强调文本 */
        strong {{
            color: {primary_color};
            font-weight: 600;
        }}

        /* 代码样式 */
        code {{
            background: linear-gradient(135deg, #f5f7fa 0%, #e8eef5 100%);
            padding: 3px 8px;
            border-radius: 4px;
            font-family: "SF Mono", "Monaco", "Courier New", monospace;
            font-size: 14px;
            color: #d63384;
            border: 1px solid #e8eef5;
        }}

        pre {{
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 20px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 20px 0;
            border: 1px solid #dee2e6;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}

        pre code {{
            background: none;
            padding: 0;
            border: none;
            color: #333;
            font-size: 13px;
            line-height: 1.6;
        }}

        /* 引用块 */
        blockquote {{
            border-left: 4px solid {primary_color};
            padding: 15px 20px;
            margin: 20px 0;
            background: linear-gradient(135deg, rgba(7, 193, 96, 0.05) 0%, rgba(7, 193, 96, 0.08) 100%);
            color: #555;
            font-style: italic;
            position: relative;
        }}

        blockquote::before {{
            content: "";
            position: absolute;
            left: 4px;
            top: 15px;
            bottom: 15px;
            width: 2px;
            background: rgba(7, 193, 96, 0.2);
        }}

        /* 列表样式 */
        ul, ol {{
            padding-left: 25px;
            margin: 16px 0;
        }}

        li {{
            margin: 8px 0;
            line-height: 1.6;
        }}

        li::marker {{
            color: {primary_color};
            font-weight: bold;
        }}

        /* 分隔线 */
        hr {{
            border: none;
            height: 2px;
            background: linear-gradient(90deg, transparent 0%, {primary_color} 50%, transparent 100%);
            margin: 40px 0;
            opacity: 0.3;
        }}

        /* 表格样式 */
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}

        th, td {{
            border: 1px solid #e0e0e0;
            padding: 12px 16px;
            text-align: left;
        }}

        th {{
            background: linear-gradient(135deg, {primary_color} 0%, {primary_color}dd 100%);
            color: white;
            font-weight: 600;
        }}

        tr:nth-child(even) {{
            background: #f8f9fa;
        }}

        /* 高亮框样式 */
        .highlight-box {{
            background: linear-gradient(135deg, rgba(7, 193, 96, 0.08) 0%, rgba(7, 193, 96, 0.12) 100%);
            border-left: 4px solid {primary_color};
            padding: 15px 20px;
            margin: 20px 0;
            border-radius: 4px;
        }}

        .tip-box {{
            background: linear-gradient(135deg, #fff3cd 0%, #ffe69c 100%);
            border-left: 4px solid #ffc107;
            padding: 15px 20px;
            margin: 20px 0;
            border-radius: 4px;
        }}

        .warning-box {{
            background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
            border-left: 4px solid #dc3545;
            padding: 15px 20px;
            margin: 20px 0;
            border-radius: 4px;
        }}

        /* 数据卡片 */
        .data-card {{
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }}

        .data-card h4 {{
            color: {primary_color};
            margin-bottom: 10px;
        }}

        /* 响应式优化 */
        @media (max-width: 480px) {{
            body {{
                padding: 15px;
            }}

            h1 {{
                font-size: 22px;
            }}

            h2 {{
                font-size: 19px;
                padding-left: 12px;
            }}

            h3 {{
                font-size: 17px;
            }}

            pre {{
                padding: 15px;
                font-size: 12px;
            }}

            table {{
                font-size: 14px;
            }}
        }}

        /* 目录样式 */
        .toc {{
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }}

        .toc h3 {{
            text-align: center;
            margin: 0 0 15px 0;
            color: {primary_color};
            font-size: 18px;
        }}

        .toc ul {{
            list-style: none;
            padding: 0;
        }}

        .toc li {{
            padding: 8px 0;
            border-bottom: 1px dashed #e0e0e0;
        }}

        .toc li:last-child {{
            border-bottom: none;
        }}

        .toc a {{
            color: #333;
            text-decoration: none;
            display: flex;
            align-items: center;
        }}

        .toc a:hover {{
            color: {primary_color};
        }}
    </style>
</head>
<body>
""")

        # 转换 Markdown 内容
        in_code_block = False
        in_highlight_box = False
        in_toc = False
        code_lines = []
        toc_items = []

        for line in markdown_content.split('\n'):
            # 处理代码块
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                if in_code_block:
                    html_lines.append('<pre><code>')
                else:
                    html_lines.append('</code></pre>')
                continue

            if in_code_block:
                html_lines.append(line)
                continue

            # 处理目录
            if '目录' in line and ('📖' in line or '目录' in line):
                in_toc = True
                html_lines.append('<div class="toc">')
                html_lines.append('<h3>📖 本期目录</h3>')
                html_lines.append('<ul>')
                continue

            if in_toc and any(line.strip().startswith(f'{d}.') for d in '0123456789'):
                toc_items.append(line.strip())
                html_lines.append(f'<li><a href="#">{line.strip()[3:]}</a></li>')
                continue

            if in_toc and not line.strip():
                html_lines.append('</ul>')
                html_lines.append('</div>')
                in_toc = False
                continue

            # 处理标题
            if line.startswith('# '):
                level = len(line) - len(line.lstrip('#'))
                text = line.lstrip('#').strip()
                if level == 1:
                    html_lines.append(f'<h1>{text}</h1>')
                elif level == 2:
                    html_lines.append(f'<h2>{text}</h2>')
                elif level == 3:
                    html_lines.append(f'<h3>{text}</h3>')
                else:
                    html_lines.append(f'<h{level}>{text}</h{level}>')
            # 处理列表
            elif line.strip().startswith('- ') or line.strip().startswith('* '):
                text = line.strip()[2:]
                html_lines.append(f'<li>{text}</li>')
            # 处理分隔线
            elif line.strip() == '---':
                html_lines.append('<hr>')
            # 处理引用
            elif line.strip().startswith('> '):
                text = line.strip()[2:]
                html_lines.append(f'<blockquote>{text}</blockquote>')
            # 处理普通段落
            elif line.strip():
                # 处理行内代码
                line = line.replace('`', '<code>').replace('`', '</code>')
                # 处理加粗
                line = line.replace('**', '<strong>').replace('**', '</strong>')
                html_lines.append(f'<p>{line}</p>')

        html_lines.append("""
</body>
</html>
""")

        return '\n'.join(html_lines)

    def _generate_mock_wechat(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """生成模拟微信内容"""
        title = article.get('title', 'AI技术文章')
        content = article.get('full_content', '文章内容')[:3000]

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 16px; line-height: 1.8; color: #333; max-width: 677px; margin: 0 auto; padding: 20px; }}
        h1 {{ font-size: 26px; font-weight: 700; text-align: center; margin: 30px 0; }}
        h2 {{ font-size: 22px; font-weight: 700; border-left: 5px solid #07c160; padding-left: 16px; margin: 40px 0 20px; }}
        p {{ margin: 16px 0; text-align: justify; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="highlight-box">
        <strong>💡 核心价值</strong>：本篇文章将带你深入了解{title}的方方面面
    </div>
    <p>{content}</p>
</body>
</html>
"""

        return {
            "title": title,
            "html_content": html_content,
            "markdown_content": content,
            "word_count": len(content),
            "reading_time": f"{len(content) // 500 + 1}",
            "cover_image_prompt": f"{title}封面图，技术风格，蓝绿渐变",
            "summary": f"本文深入解析{title}，涵盖核心概念、实战案例和最佳实践。",
            "seo_keywords": ["AI技术", "开发", "实战"]
        }
