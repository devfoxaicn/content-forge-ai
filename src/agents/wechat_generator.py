"""
微信公众号文章生成 Agent

将长文本技术文章转换为适合微信公众号发布的 HTML 格式
"""

from typing import Dict, Any
from src.agents.base import BaseAgent


class WechatGeneratorAgent(BaseAgent):
    """微信公众号文章生成 Agent"""

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        super().__init__(config, prompts)
        wechat_config = config.get("agents", {}).get("wechat_generator", {})
        self.enable_html = wechat_config.get("enable_html", True)
        self.enable_code_highlight = wechat_config.get("enable_code_highlight", True)
        self.max_tokens = wechat_config.get("max_tokens", 16000)
        self.llm.max_tokens = self.max_tokens
        self.llm.temperature = 0.7
        self.mock_mode = config.get("agents", {}).get("wechat_generator", {}).get("mock_mode", False)

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        将长文本转换为微信公众号 HTML 格式

        输出格式：
        {
            "title": "文章标题",
            "html_content": "<html>...</html>",  # 带排版的完整HTML
            "word_count": 5000,
            "reading_time": "15分钟",
            "cover_image_prompt": "封面图提示词",
            "summary": "文章摘要"
        }
        """
        self.log("开始生成微信公众号文章")

        try:
            # 获取长文本文章
            article = state.get("longform_article")
            if not article:
                raise ValueError("没有找到长文本文章")

            self.log(f"转换文章: {article['title']}")

            # Mock模式
            if self.mock_mode:
                self.log("使用Mock模式生成微信文章")
                wechat_content = self._generate_mock_wechat(article)
            else:
                # 构建提示词
                user_prompt = self._build_prompt(state, article)

                # 调用LLM生成内容
                response = self._call_llm(user_prompt)

                # 解析微信内容
                wechat_content = self._parse_wechat_content(response, article)

            self.log(f"成功生成微信文章，字数: {wechat_content['word_count']}")

            return {
                **state,
                "wechat_article": wechat_content,
                "current_step": "wechat_generator_completed"
            }
        except Exception as e:
            self.log(f"微信文章生成失败: {str(e)}", "ERROR")
            # 失败时返回模拟数据
            article = state.get("longform_article", {"title": "AI技术", "full_content": "内容"})
            wechat_content = self._generate_mock_wechat(article)
            return {
                **state,
                "wechat_article": wechat_content,
                "current_step": "wechat_generator_completed"
            }

    def _build_prompt(self, state: Dict[str, Any], article: Dict[str, Any]) -> str:
        """构建生成提示词"""
        prompts = self.prompts.get("prompts", {})
        prompt_template = prompts.get("wechat_generator", {}).get("user", "")

        # 获取文章内容（截取前10000字，避免超出限制）
        content = article['full_content']
        if len(content) > 10000:
            content = content[:10000] + "\n\n... (内容已截断)"

        # 使用默认提示词
        if not prompt_template:
            prompt_template = """
请将以下技术文章转换为适合微信公众号发布的格式：

**原标题**：{title}

**文章内容**：
{content}

**要求**：
1. 保持技术准确性和深度
2. 优化段落结构，使其更适合移动端阅读
3. 添加适当的过渡段落和小标题
4. 使用正式但易懂的语言风格
5. 生成100字左右的文章摘要

请以以下格式输出：

【标题】
（优化后的标题）

【摘要】
100字摘要

【正文】
（优化后的正文内容，使用 Markdown 格式）
"""

        return prompt_template.format(
            title=article['title'],
            content=content
        )

    def _parse_wechat_content(self, response: str, article: Dict[str, Any]) -> Dict[str, Any]:
        """解析LLM生成的微信内容"""
        # 简单解析：按标题、摘要、正文分段
        lines = response.split('\n')

        title = article['title']
        summary = ""
        full_content = response

        # 尝试提取标题
        for i, line in enumerate(lines):
            if line.startswith("【标题】") or line.startswith("# ") or line.startswith("标题："):
                if i + 1 < len(lines):
                    title = lines[i + 1].strip()
                    break

        # 尝试提取摘要
        for i, line in enumerate(lines):
            if line.startswith("【摘要】") or line.startswith("摘要："):
                if i + 1 < len(lines):
                    summary_lines = []
                    j = i + 1
                    while j < len(lines) and not lines[j].startswith("【"):
                        summary_lines.append(lines[j])
                        j += 1
                    summary = '\n'.join(summary_lines).strip()
                    break

        # 生成HTML
        html_content = self._generate_html(full_content, title)

        return {
            "title": title,
            "html_content": html_content,
            "markdown_content": full_content,
            "word_count": len(full_content),
            "reading_time": f"{len(full_content) // 500 + 1}分钟",
            "cover_image_prompt": f"生成一个关于{title}的封面图，技术风格，简洁大气",
            "summary": summary or full_content[:100] + "..."
        }

    def _generate_html(self, markdown_content: str, title: str) -> str:
        """将 Markdown 转换为带样式的 HTML"""
        # 简单的 Markdown 转 HTML
        html_lines = []
        html_lines.append("""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>""" + title + """</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            font-size: 16px;
            line-height: 1.8;
            color: #333;
            max-width: 677px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            font-size: 24px;
            font-weight: bold;
            color: #000;
            margin: 30px 0 20px;
            text-align: center;
        }
        h2 {
            font-size: 20px;
            font-weight: bold;
            color: #000;
            margin: 30px 0 15px;
            padding-left: 10px;
            border-left: 4px solid #07c160;
        }
        h3 {
            font-size: 18px;
            font-weight: bold;
            color: #333;
            margin: 25px 0 10px;
        }
        p {
            margin: 15px 0;
            text-align: justify;
        }
        strong {
            color: #07c160;
            font-weight: bold;
        }
        code {
            background: #f5f5f5;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: "Courier New", monospace;
            font-size: 14px;
        }
        pre {
            background: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            margin: 15px 0;
        }
        pre code {
            background: none;
            padding: 0;
        }
        blockquote {
            border-left: 4px solid #ddd;
            padding-left: 15px;
            margin: 15px 0;
            color: #666;
            font-style: italic;
        }
        ul, ol {
            padding-left: 20px;
            margin: 15px 0;
        }
        li {
            margin: 5px 0;
        }
        hr {
            border: none;
            border-top: 1px solid #eee;
            margin: 30px 0;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px 12px;
            text-align: left;
        }
        th {
            background: #f5f5f5;
            font-weight: bold;
        }
    </style>
</head>
<body>
""")

        # 转换 Markdown 内容
        in_code_block = False
        code_lines = []

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

        # 生成简单的 HTML
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{ font-family: -apple-system, sans-serif; font-size: 16px; line-height: 1.8; color: #333; max-width: 677px; margin: 0 auto; padding: 20px; }}
        h1 {{ font-size: 24px; font-weight: bold; text-align: center; margin: 30px 0; }}
        h2 {{ font-size: 20px; font-weight: bold; border-left: 4px solid #07c160; padding-left: 10px; margin: 30px 0 15px; }}
        p {{ margin: 15px 0; text-align: justify; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p>{content}</p>
</body>
</html>
"""

        return {
            "title": title,
            "html_content": html_content,
            "markdown_content": content,
            "word_count": len(content),
            "reading_time": f"{len(content) // 500 + 1}分钟",
            "cover_image_prompt": f"{title}封面图",
            "summary": content[:100] + "..."
        }
