"""
图片生成Agent
使用Gemini或其他AI服务生成配图
"""

from typing import Dict, Any, Optional, List
import os
import base64
from datetime import datetime
from src.agents.base import BaseAgent
from loguru import logger


class ImageGeneratorAgent(BaseAgent):
    """图片生成Agent"""

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        super().__init__(config, prompts)
        generator_config = config.get("agents", {}).get("image_generator", {})
        self.provider = generator_config.get("provider", "gemini")  # gemini, stability, openai
        self.model = generator_config.get("model", "gemini-2.0-flash-exp")
        self.image_size = generator_config.get("image_size", "1024x1024")
        self.num_images = generator_config.get("num_images", 1)
        self.enable_generation = generator_config.get("enable_generation", False)  # 默认关闭，等待API key

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成配图

        Args:
            state: 当前工作流状态

        Returns:
            Dict[str, Any]: 更新后的状态
        """
        self.log("开始生成配图")

        try:
            # 如果未启用图片生成，只返回提示词
            if not self.enable_generation:
                self.log("图片生成未启用，仅生成提示词", "WARNING")
                return self._generate_prompts_only(state)

            # 获取SD提示词（来自ImageAdvisorAgent）
            image_suggestions = state.get("image_suggestions", [])
            if not image_suggestions:
                # 如果没有SD提示词，使用标题生成
                title = state.get("recommended_title") or state.get("xiaohongshu_note", {}).get("title", "")
                sd_prompt = self._generate_prompt_from_title(title, state.get("topic", ""))
            else:
                sd_prompt = image_suggestions[0].get("sd_prompt", "")

            if not sd_prompt:
                raise ValueError("没有可用的图片提示词")

            self.log(f"使用提示词生成图片: {sd_prompt[:100]}...")

            # 根据provider生成图片
            if self.provider == "gemini":
                images = self._generate_with_gemini(sd_prompt, state)
            elif self.provider == "openai":
                images = self._generate_with_openai(sd_prompt, state)
            elif self.provider == "stability":
                images = self._generate_with_stability(sd_prompt, state)
            else:
                raise ValueError(f"不支持的图片生成provider: {self.provider}")

            self.log(f"成功生成 {len(images)} 张图片")

            return {
                **state,
                "generated_images": images,
                "current_step": "image_generator_completed"
            }

        except Exception as e:
            self.log(f"图片生成失败: {str(e)}", "ERROR")
            # 失败时至少保存提示词
            return self._generate_prompts_only(state, error=str(e))

    def _generate_prompts_only(self, state: Dict[str, Any], error: str = None) -> Dict[str, Any]:
        """
        仅生成提示词（不实际生成图片）
        注意：提示词由 auto_orchestrator.py 统一保存到各个媒体目录

        Args:
            state: 当前状态
            error: 错误信息

        Returns:
            Dict[str, Any]: 更新后的状态
        """
        # 获取或生成提示词
        image_suggestions = state.get("image_suggestions", [])

        if image_suggestions:
            # 尝试从image_suggestions中提取sd_prompt
            prompts = [s.get("sd_prompt", "") for s in image_suggestions if s.get("sd_prompt")]

            if prompts:
                self.log(f"使用ImageAdvisorAgent提供的 {len(prompts)} 个提示词")
            else:
                # 如果image_suggestions存在但没有sd_prompt，使用其他信息生成
                self.log(f"ImageAdvisorAgent提供了建议但没有sd_prompt，基于其他信息生成")
                ai_topic = state.get("selected_ai_topic", {})
                title = state.get("recommended_title") or state.get("xiaohongshu_note", {}).get("title", "")

                if ai_topic and ai_topic.get("title"):
                    prompt = self._generate_prompt_from_topic(ai_topic, title)
                else:
                    prompt = self._generate_prompt_from_title(title, state.get("topic", ""))

                prompts = [prompt]
        else:
            # 基于热点话题生成相关提示词
            ai_topic = state.get("selected_ai_topic", {})
            title = state.get("recommended_title") or state.get("xiaohongshu_note", {}).get("title", "")

            # 如果有AI热点话题，基于话题生成
            if ai_topic and ai_topic.get("title"):
                self.log(f"基于AI热点生成配图提示词: {ai_topic.get('title')}")
                prompt = self._generate_prompt_from_topic(ai_topic, title)
            else:
                self.log(f"基于标题生成配图提示词: {title}")
                prompt = self._generate_prompt_from_title(title, state.get("topic", ""))

            prompts = [prompt]

        self.log(f"生成 {len(prompts)} 个图片生成提示词（由orchestrator保存到媒体目录）")

        if error:
            self.log(f"图片生成失败: {error}", "ERROR")
        else:
            self.log("提示词生成完成，将保存到各媒体目录")

        return {
            **state,
            "image_prompts": prompts,
            "generated_images": [],
            "current_step": "image_generator_prompts_only"
        }

    def _generate_with_gemini(self, prompt: str, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        使用Gemini优化提示词或尝试生成图片

        注意：
        1. Gemini API本身不支持直接生成图片
        2. Vertex AI Imagen API需要Google Cloud认证（不是简单的API Key）
        3. 当前实现：使用Gemini优化提示词，返回供其他工具使用

        Args:
            prompt: 图片生成提示词
            state: 当前状态

        Returns:
            List[Dict[str, Any]]: 优化的提示词或生成的图片信息
        """
        try:
            import google.generativeai as genai

            # 获取API key
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("请设置环境变量 GEMINI_API_KEY")

            genai.configure(api_key=api_key)

            self.log("使用Gemini优化图片生成提示词", "INFO")

            model = genai.GenerativeModel(self.model)

            # 让Gemini帮我们优化提示词
            optimization_prompt = f"""请将以下图片生成提示词优化为高质量的中文AI绘图提示词：

原始提示词：{prompt}

请提供：
1. 优化后的中文提示词（详细描述画面内容、风格、光线、构图、色彩等）
2. 负面提示词（需要避免的内容）
3. 推荐的艺术风格和画面细节

输出格式：
正向提示词: [详细的中文提示词]
负面提示词: [中文负面提示词]

要求：
- 提示词要具体、详细、有画面感
- 适合小红书配图风格：现代、简洁、科技感、吸引人
- 重点描述：主体内容、背景、色彩搭配、光线效果、构图方式、艺术风格
- 字数150-300字
- 使用描述性强的词汇，让AI能准确理解你想要的画面
"""

            response = model.generate_content(optimization_prompt)
            optimized = response.text

            # 解析优化后的提示词
            positive_match = optimized.find("正向提示词:")
            negative_match = optimized.find("负面提示词:")

            if positive_match != -1 and negative_match != -1:
                positive_prompt = optimized[positive_match+6:negative_match].strip()
                negative_prompt = optimized[negative_match+6:].strip()
            else:
                # 如果没有找到标记，尝试提取整个内容
                lines = optimized.split('\n')
                positive_prompt = prompt
                negative_prompt = "低质量、模糊、变形、丑陋、构图差、颜色失真"

                # 尝试提取优化后的内容
                for i, line in enumerate(lines):
                    if '提示词' in line and '优化' in line:
                        if i + 1 < len(lines):
                            positive_prompt = lines[i + 1].strip()
                            break

            self.log("中文提示词优化完成", "SUCCESS")

            # 提示词由 auto_orchestrator.py 保存到各个媒体目录
            self.log("优化后的提示词将由orchestrator保存到媒体目录", "INFO")

            # 尝试使用其他方式生成图片
            # 注意：Vertex AI Imagen需要OAuth认证，不是简单的API Key
            # 如果用户配置了Google Cloud，可以尝试使用（需要额外配置）
            images = []

            try:
                # 检查是否有可用的图片生成方式
                # 这里我们优先返回提示词，让用户手动生成或使用其他服务
                self.log("提示：当前配置生成优化提示词，可使用这些提示词在Stable Diffusion、DALL-E等工具中生成图片", "INFO")

            except Exception as e:
                self.log(f"图片生成跳过: {str(e)}", "INFO")

            # 返回优化的提示词（由orchestrator保存）
            return [{
                "provider": "gemini",
                "prompt": positive_prompt,
                "negative_prompt": negative_prompt,
                "original_prompt": prompt,
                "note": "已使用Gemini优化为中文提示词。可在支持中文的AI绘图工具中使用，或翻译后用于Stable Diffusion、DALL-E等工具",
                "optimized_prompt": optimized,
                "created_at": datetime.now().isoformat()
            }]

        except ImportError:
            self.log("未安装google-generativeai库，请运行: pip install google-generativeai", "ERROR")
            raise
        except Exception as e:
            self.log(f"Gemini调用失败: {str(e)}", "ERROR")
            raise

    def _generate_with_openai(self, prompt: str, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        使用OpenAI DALL-E生成图片

        Args:
            prompt: 图片生成提示词
            state: 当前状态

        Returns:
            List[Dict[str, Any]]: 生成的图片信息
        """
        try:
            from openai import OpenAI

            # 获取API key
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("请设置环境变量 OPENAI_API_KEY")

            client = OpenAI(api_key=api_key)

            self.log(f"使用DALL-E生成 {self.num_images} 张图片")

            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size=self.image_size,
                n=self.num_images,
                quality="standard"
            )

            # 图片由 orchestrator 保存到对应媒体目录
            self.log("图片生成成功，将由orchestrator保存到媒体目录", "INFO")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            images = []
            for i, image_data in enumerate(response.data):
                # 下载图片
                image_url = image_data.url
                revised_prompt = image_data.revised_prompt

                # 保存图片信息（不保存文件）
                image_info = {
                    "provider": "openai",
                    "model": "dall-e-3",
                    "url": image_url,
                    "prompt": prompt,
                    "revised_prompt": revised_prompt,
                    "size": self.image_size,
                    "created_at": datetime.now().isoformat()
                }

                images.append(image_info)

            return images

        except ImportError:
            self.log("未安装openai库，请运行: pip install openai", "ERROR")
            raise
        except Exception as e:
            self.log(f"OpenAI DALL-E调用失败: {str(e)}", "ERROR")
            raise

    def _generate_with_stability(self, prompt: str, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        使用Stability AI生成图片

        Args:
            prompt: 图片生成提示词
            state: 当前状态

        Returns:
            List[Dict[str, Any]]: 生成的图片信息
        """
        try:
            import stability_sdk
            from stability_sdk import interface

            # 获取API key
            api_key = os.getenv("STABILITY_API_KEY")
            if not api_key:
                raise ValueError("请设置环境变量 STABILITY_API_KEY")

            self.log(f"使用Stability AI生成 {self.num_images} 张图片")

            # 初始化Stability AI
            stability_api = interface.StabilityInference(
                key=api_key,
                verbose=True,
            )

            # 生成图片
            answers = stability_api.generate(
                prompt=prompt,
                seed=self._get_seed(),
                steps=30,
                cfg_scale=8.0,
                width=1024,
                height=1024,
                samples=self.num_images,
                sampler=interface.generation.SAMPLER_K_DPMPP_2M
            )

            # 图片由 orchestrator 保存到对应媒体目录
            self.log("图片生成成功，将由orchestrator保存到媒体目录", "INFO")

            images = []
            for i, resp in enumerate(answers):
                for artifact in resp.artifacts:
                    if artifact.type == interface.generation.ARTIFACT_IMAGE:
                        # 不保存文件，只返回图片数据
                        images.append({
                            "provider": "stability",
                            "model": "stable-diffusion",
                            "image_data": artifact.binary,  # base64编码的图片数据
                            "prompt": prompt,
                            "seed": artifact.seed,
                            "created_at": datetime.now().isoformat()
                        })

            return images

        except ImportError:
            self.log("未安装stability-sdk库，请运行: pip install stability-sdk", "ERROR")
            raise
        except Exception as e:
            self.log(f"Stability AI调用失败: {str(e)}", "ERROR")
            raise

    def _generate_prompt_from_title(self, title: str, topic: str) -> str:
        """
        从标题生成图片提示词

        Args:
            title: 笔记标题
            topic: 主题

        Returns:
            str: 图片生成提示词
        """
        # 使用LLM生成图片提示词
        prompt = f"""Based on the following Xiaohongshu note title, generate a detailed image generation prompt for AI image creation tools:

Title: {title}
Topic: {topic}

Generate an English prompt for image generation that:
1. Describes the main subject clearly
2. Includes style details (clean, modern, tech-focused)
3. Specifies the mood (professional, inspiring, approachable)
4. Mentions composition (centered, clean background, good lighting)
5. Suggests color palette (modern, vibrant but professional)

Output only the prompt text, no explanations."""

        try:
            response = self._call_llm(prompt)
            # 提取提示词（移除可能的引号和多余文字）
            prompt_lines = response.strip().split('\n')
            for line in prompt_lines:
                if line.strip() and not line.startswith('#') and not line.startswith('Prompt:'):
                    return line.strip()
            return response.strip()
        except Exception as e:
            self.log(f"生成图片提示词失败: {str(e)}", "WARNING")
            # 返回默认提示词
            return f"Modern AI technology concept, clean design, professional style, {topic}"

    def _generate_prompt_from_topic(self, ai_topic: Dict[str, Any], title: str = "") -> str:
        """
        基于AI热点话题生成图片提示词

        Args:
            ai_topic: AI热点话题数据
            title: 标题（可选）

        Returns:
            str: 图片生成提示词
        """
        topic_title = ai_topic.get("title", "")
        topic_desc = ai_topic.get("description", "")
        source = ai_topic.get("source", "")
        tags = ai_topic.get("tags", [])

        self.log(f"生成配图提示词 - 话题: {topic_title}, 来源: {source}")

        # 使用LLM生成与话题相关的配图提示词
        prompt = f"""请基于以下AI技术热点，生成一个高质量的小红书配图提示词（中文）：

热点话题：{topic_title}
描述：{topic_desc}
来源：{source}
标签：{', '.join(tags)}
笔记标题：{title}

要求：
1. 画面要与AI技术主题相关（如：神经网络、AI模型、代码、数据可视化等）
2. 小红书风格：现代、简洁、科技感、吸引眼球
3. 配色方案：科技蓝、紫色渐变、白色背景
4. 构图：居中或三分法，主体突出
5. 光线：明亮、柔和
6. 风格：扁平化设计或3D渲染，专业但易理解

输出格式：
描述一个完整的画面场景，包括：
- 主体内容（如：AI芯片、神经网络图、代码界面等）
- 背景（简洁现代）
- 色彩搭配
- 光线效果
- 整体风格

字数：100-200字

请直接输出画面描述，不要解释。"""

        try:
            response = self._call_llm(prompt)
            # 清理响应
            prompt_text = response.strip()

            # 如果响应太长，截取主要部分
            if len(prompt_text) > 500:
                prompt_text = prompt_text[:500]

            self.log(f"基于AI热点生成配图提示词成功: {prompt_text[:50]}...")
            return prompt_text

        except Exception as e:
            self.log(f"基于AI热点生成配图提示词失败: {str(e)}", "WARNING")

            # Fallback: 基于热点类型生成默认提示词
            source_lower = source.lower() if source else ""

            if "huggingface" in source_lower or "模型" in topic_title:
                return """现代AI模型概念图，中心是发光的神经网络节点，周围环绕着数据流。背景是深蓝色科技感渐变，点缀着紫色和青色的光点。扁平化设计风格，简洁专业，适合小红书科技类配图。整体色调：科技蓝、紫色、白色。构图：居中对称，主体突出。光线：柔和的中心光晕效果。"""
            elif "github" in source_lower or "代码" in topic_title:
                return """代码编辑器界面，屏幕上显示着Python或JavaScript代码，语法高亮色彩丰富。背景是深色模式，窗口带有现代感的阴影和圆角。桌上放着一杯咖啡，旁边是植物。温馨的程序员工作场景，适合技术分享类内容。配色：深色背景、明亮代码高亮、暖色点缀。"""
            elif "arxiv" in source_lower or "论文" in topic_title:
                return """学术论文概念图，展示图表、公式和数据可视化。背景是简洁的白色或浅灰色，上面漂浮着几何图形和数学符号。现代学术风格，清晰专业。配色：白色背景、蓝色几何图形、黑色文字。适合AI研究类内容。"""
            else:
                return f"""AI技术概念图，展示{topic_title[:20]}的核心元素。现代科技风格，简洁专业的配图，适合小红书技术分享。配色：科技蓝、紫色渐变、白色背景。构图：居中，主体突出。"""

    def _get_aspect_ratio(self) -> str:
        """
        根据image_size返回aspect ratio
        """
        size_to_ratio = {
            "1024x1024": "1:1",
            "1792x1024": "16:9",
            "1024x1792": "9:16",
            "512x512": "1:1"
        }
        return size_to_ratio.get(self.image_size, "1:1")

    def _get_seed(self) -> int:
        """生成随机seed"""
        import random
        return random.randint(0, 4294967295)
