"""
图像建议Agent
提供配图建议和AI绘画提示词
"""

from typing import Dict, Any, List
import re
from src.agents.base import BaseAgent


class ImageAdvisorAgent(BaseAgent):
    """图像建议Agent"""

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        super().__init__(config, prompts)
        self.num_suggestions = config.get("agents", {}).get("image_advisor", {}).get("num_suggestions", 3)
        self.generate_sd_prompts = config.get("agents", {}).get("image_advisor", {}).get("generate_sd_prompts", True)

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行图像建议生成

        Args:
            state: 当前工作流状态

        Returns:
            Dict[str, Any]: 更新后的状态
        """
        self.log("开始生成图像建议")

        try:
            # 获取AI热点话题（适配新工作流）
            ai_topic = state.get("selected_ai_topic", {})
            xiaohongshu = state.get("xiaohongshu_note", {})
            longform = state.get("longform_article", {})

            # 优先使用小红书笔记，其次长文本文章
            if xiaohongshu and xiaohongshu.get("title"):
                content = xiaohongshu
                content_type = "小红书笔记"
            elif longform and longform.get("title"):
                content = longform
                content_type = "长文本文章"
            elif ai_topic and ai_topic.get("title"):
                # 如果都没有，使用AI热点话题
                content = ai_topic
                content_type = "AI热点话题"
            else:
                raise ValueError("没有找到可用的内容（需要xiaohongshu_note、longform_article或selected_ai_topic）")

            self.log(f"基于{content_type}生成图像建议: {content.get('title', '')}")

            # 构建提示词
            user_prompt = self._build_prompt(state, content, ai_topic)

            # 调用LLM生成建议
            response = self._call_llm(user_prompt)

            # 解析响应
            suggestions = self._parse_suggestions(response)

            self.log(f"成功生成 {len(suggestions)} 个图像建议")

            return {
                **state,
                "image_suggestions": suggestions,
                "current_step": "image_advisor_completed"
            }
        except Exception as e:
            self.log(f"图像建议生成失败: {str(e)}", "ERROR")
            # 失败时返回基于AI热点的默认建议
            ai_topic = state.get("selected_ai_topic", {})
            fallback_suggestions = self._generate_fallback_suggestions(ai_topic)
            return {
                **state,
                "image_suggestions": fallback_suggestions,
                "error_message": f"图像建议生成失败: {str(e)}",
                "current_step": "image_advisor_failed"
            }

    def _build_prompt(self, state: Dict[str, Any], content: Dict[str, Any], ai_topic: Dict[str, Any] = None) -> str:
        """
        构建提示词

        Args:
            state: 当前状态
            content: 生成的内容
            ai_topic: AI热点话题

        Returns:
            str: 提示词
        """
        prompts = self.prompts.get("prompts", {})
        prompt_template = prompts.get("image_advisor", {}).get("user", "")

        # 如果有AI热点，使用热点信息
        if ai_topic and ai_topic.get("title"):
            topic_info = f"热点话题：{ai_topic.get('title')}\n描述：{ai_topic.get('description', '')}"
        else:
            topic_info = f"主题：{state.get('topic', 'AI技术')}"

        return f"""请为以下内容生成配图建议（小红书风格）：

{topic_info}

内容标题：{content.get('title', '')}

请提供3个配图方案，每个方案包括：
1. 画面描述
2. 色彩风格
3. Stable Diffusion提示词（英文）

输出格式：
图片 1：
画面描述：[详细的场景描述]
色彩风格：[配色方案]
SD提示词：[英文提示词]
SD负面提示词：[需要避免的内容]

请直接输出，不要额外解释。"""

    def _parse_suggestions(self, response: str) -> List[Dict[str, Any]]:
        """
        解析LLM响应为建议列表

        Args:
            response: LLM响应

        Returns:
            List[Dict[str, Any]]: 建议列表
        """
        suggestions = []

        # 尝试解析结构化响应
        lines = response.split("\n")
        current_suggestion = {}
        suggestion_count = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 检测新的建议开始
            if line.startswith(("图片1", "图片2", "图片3", "建议1", "建议2", "建议3")):
                if current_suggestion and len(current_suggestion) > 1:
                    suggestions.append(current_suggestion)
                    suggestion_count += 1
                    if suggestion_count >= self.num_suggestions:
                        break
                current_suggestion = {}

            elif "画面描述" in line or "场景描述" in line:
                current_suggestion["scene_description"] = line.split(":", 1)[1].strip() if ":" in line else line.split("：", 1)[1].strip()
            elif "色彩风格" in line or "风格建议" in line:
                current_suggestion["color_scheme"] = line.split(":", 1)[1].strip() if ":" in line else line.split("：", 1)[1].strip()
            elif "SD提示词" in line or "Stable" in line:
                # 提取SD提示词
                prompt_match = re.search(r'Positive[:：](.+?)(?=Negative:|$)', response, re.DOTALL)
                if prompt_match:
                    current_suggestion["sd_prompt"] = prompt_match.group(1).strip()
                neg_match = re.search(r'Negative[:：](.+?)(?=$|Positive:)', response, re.DOTALL)
                if neg_match:
                    current_suggestion["sd_negative_prompt"] = neg_match.group(1).strip()

        # 添加最后一个建议
        if current_suggestion:
            suggestions.append(current_suggestion)

        # 如果解析失败，返回默认建议
        if not suggestions:
            suggestions = [{
                "style": "技术简约风",
                "scene_description": "包含电脑屏幕、代码编辑器界面、AI工具logo的桌面场景",
                "color_scheme": "深色模式，蓝紫色调",
                "sd_prompt": "computer screen with code editor, AI tools interface, minimalist desk setup, dark mode, blue and purple lighting, high quality, 4k, professional photography",
                "sd_negative_prompt": "text, watermark, blurry, low quality, distorted"
            }]

        return suggestions[:self.num_suggestions]

    def _generate_fallback_suggestions(self, ai_topic: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        生成默认的图像建议（当LLM调用失败时）

        Args:
            ai_topic: AI热点话题

        Returns:
            List[Dict[str, Any]]: 默认建议列表
        """
        topic_title = ai_topic.get("title", "") if ai_topic else ""
        source = ai_topic.get("source", "") if ai_topic else ""

        # 根据热点类型生成相关建议
        source_lower = source.lower() if source else ""

        if "huggingface" in source_lower or "模型" in topic_title:
            return [{
                "style": "科技模型风",
                "scene_description": f"展示{topic_title}的神经网络结构图，发光的节点和连接线，背景深蓝色科技感",
                "color_scheme": "科技蓝、紫色、深色背景",
                "sd_prompt": "AI neural network visualization, glowing nodes and connections, deep blue tech background, futuristic style, clean design, high quality, 4k",
                "sd_negative_prompt": "text, watermark, blurry, low quality, distorted, ugly"
            }]
        elif "github" in source_lower or "代码" in topic_title:
            return [{
                "style": "代码编程风",
                "scene_description": "代码编辑器界面，显示Python或JavaScript代码，语法高亮，深色模式，桌上放咖啡和植物",
                "color_scheme": "深色背景、明亮代码高亮、暖色点缀",
                "sd_prompt": "code editor screen with syntax highlighted code, dark mode, coffee cup and plant on desk, cozy workspace, warm lighting, high quality, professional",
                "sd_negative_prompt": "text, watermark, blurry, low quality"
            }]
        elif "arxiv" in source_lower or "论文" in topic_title:
            return [{
                "style": "学术研究风",
                "scene_description": "学术论文和数据可视化图表，白色背景，几何图形和数学符号",
                "color_scheme": "白色背景、蓝色图形、黑色文字",
                "sd_prompt": "academic paper with charts and data visualization, white background, geometric shapes and math symbols, clean professional style, high quality",
                "sd_negative_prompt": "text, watermark, blurry, low quality, messy"
            }]
        else:
            return [{
                "style": "通用科技风",
                "scene_description": f"{topic_title}的概念示意图，AI元素，现代科技感",
                "color_scheme": "科技蓝、紫色渐变、白色背景",
                "sd_prompt": f"AI technology concept art, {topic_title[:30]}, modern tech style, blue and purple gradient, clean background, high quality, 4k, futuristic",
                "sd_negative_prompt": "text, watermark, blurry, low quality, distorted"
            }]
