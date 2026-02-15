#!/usr/bin/env python3
"""
小红书图片生成器
使用 Gemini 3 Pro Image Preview (Nano Banana Pro) 生成图片
支持累积传入之前生成的图片以保持风格一致性
"""
import asyncio
import base64
import json
import sys
from pathlib import Path
from typing import Optional, List

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("错误: 未安装 google-genai SDK")
    print("请运行: pip install google-genai")
    sys.exit(1)


# 配置 - 使用项目根目录的配置
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_FILE = PROJECT_ROOT / ".claude" / "settings.local.json"
OUTPUT_DIR = Path(__file__).parent / "generated_images"
OUTPUT_DIR.mkdir(exist_ok=True)

# IP形象目录 - 按用途分类存储
IP_IMAGES_DIR = PROJECT_ROOT / "pictures"
IP_COVER_DIR = IP_IMAGES_DIR / "cover"      # 仅用于小红书封面图
IP_CHAPTER_DIR = IP_IMAGES_DIR / "chapter"  # 仅用于小红书章节图
IP_COMMON_DIR = IP_IMAGES_DIR / "common"    # 通用IP（微信专用、小红书通用）
IP_TWEET_DIR = IP_IMAGES_DIR / "twitter" / "tweet"  # Twitter推文专用IP
IP_TWITTER_COMMON_DIR = IP_IMAGES_DIR / "twitter" / "common"  # Twitter通用IP

# ⚠️ 微信公众号IP说明：只使用 pictures/common/ 目录，不使用 wechat 子目录

# 支持的宽高比
ASPECT_RATIOS = ["1:1", "16:9", "9:16", "4:3", "3:4", "4:5", "5:4", "21:9", "9:21"]


class ImageGenerator:
    """图片生成器"""

    def __init__(self, api_key: Optional[str] = None):
        """初始化 Gemini API 客户端"""
        self.api_key = api_key or self._load_api_key()
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY 未设置，请在 .claude/settings.local.json 中配置")

        self.client = genai.Client(api_key=self.api_key)
        self.model_name = "gemini-3-pro-image-preview"  # Nano Banana Pro

    def _load_api_key(self) -> Optional[str]:
        """从配置文件加载 API Key"""
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                return config.get("env", {}).get("GEMINI_API_KEY")
        return None

    def generate_image(
        self,
        prompt: str,
        aspect_ratio: str = "3:4",
        filename: Optional[str] = None,
        use_ip: bool = True,
        image_type: str = "chapter",
        previous_images: Optional[List[str]] = None,
        chapter_title: Optional[str] = None
    ) -> dict:
        """
        生成单张图片

        Args:
            prompt: 图片描述提示词
            aspect_ratio: 宽高比，默认 "3:4"
            filename: 输出文件名（可选）
            use_ip: 是否使用 IP 形象作为参考
            image_type: 图片类型
                - "wechat": 微信公众号封面图（使用 pictures/common/）
                - "cover": 小红书封面图（使用 pictures/cover/ + pictures/common/）
                - "chapter": 小红书章节图（使用 pictures/chapter/ + pictures/common/）
                - "tweet": Twitter推文图（使用 pictures/twitter/tweet/ + pictures/twitter/common/）
            previous_images: 已废弃参数，保留用于兼容性，不再使用（只参考pictures目录的IP形象）
            chapter_title: 章节标题（可选），用于在提示词开头添加章节信息

        Returns:
            包含图片信息的字典
        """
        if aspect_ratio not in ASPECT_RATIOS:
            raise ValueError(f"不支持的宽高比: {aspect_ratio}")

        try:
            # 构建内容列表
            contents = []

            # 宽高比描述
            aspect_ratio_prompts = {
                "1:1": "正方形 (1:1)",
                "16:9": "横向宽屏 (16:9)",
                "9:16": "竖向 (9:16)",
                "4:3": "横向 (4:3)",
                "3:4": "竖向 (3:4)",
                "4:5": "竖向 (4:5)",
                "5:4": "横向 (5:4)",
                "21:9": "超宽屏 (21:9)",
                "9:21": "超长竖向 (9:21)"
            }

            # 添加参考图片说明
            reference_info = []

            # 1. 首先添加 IP 形象（根据图片类型加载不同目录）
            if use_ip:
                # 确定要加载的目录
                ip_dirs_to_load = []
                if image_type == "wechat":
                    # 微信公众号封面图：只加载 common/
                    ip_dirs_to_load = [IP_COMMON_DIR]
                elif image_type == "tweet":
                    # Twitter推文图：加载 twitter/tweet/ + twitter/common/
                    ip_dirs_to_load = [IP_TWEET_DIR, IP_TWITTER_COMMON_DIR]
                elif image_type == "cover":
                    # 小红书封面图：加载 cover/ + common/
                    ip_dirs_to_load = [IP_COVER_DIR, IP_COMMON_DIR]
                elif image_type == "chapter":
                    # 小红书章节图：加载 chapter/ + common/
                    ip_dirs_to_load = [IP_CHAPTER_DIR, IP_COMMON_DIR]
                else:
                    # 其他情况：加载所有目录
                    ip_dirs_to_load = [IP_COVER_DIR, IP_CHAPTER_DIR, IP_COMMON_DIR]

                # 从指定目录加载所有图片
                loaded_images = []
                for ip_dir in ip_dirs_to_load:
                    if ip_dir.exists() and ip_dir.is_dir():
                        ip_images = list(ip_dir.glob("*.png")) + list(ip_dir.glob("*.jpg")) + list(ip_dir.glob("*.jpeg"))
                        ip_images = [img for img in ip_images if img.is_file()]

                        if ip_images:
                            for ip_image_path in sorted(ip_images):
                                ip_image_bytes = ip_image_path.read_bytes()
                                contents.append(
                                    types.Part.from_bytes(
                                        data=ip_image_bytes,
                                        mime_type="image/png"
                                    )
                                )
                                loaded_images.append(str(ip_image_path))

                # 打印加载的IP形象图片信息
                if loaded_images:
                    print(f"✓ 已加载 {len(loaded_images)} 张IP形象参考图片:")
                    for img in loaded_images:
                        print(f"  - {img}")
                else:
                    print("⚠ 警告: 未找到IP形象图片")

                if contents:
                    if image_type == "wechat":
                        reference_info.append("前面几张是我的个人IP形象极客狐DevFox，来自pictures/common目录")
                    elif image_type == "tweet":
                        reference_info.append("前面几张是我的个人IP形象极客狐DevFox，来自pictures目录的twitter子目录")
                    elif image_type == "chapter":
                        reference_info.append("前面几张是我的个人IP形象极客狐DevFox，来自pictures目录的common子目录和chapter子目录")
                    elif image_type == "cover":
                        reference_info.append("前面几张是我的个人IP形象极客狐DevFox，来自pictures目录的common子目录和cover子目录")
                    else:
                        reference_info.append(f"前面几张是我的个人IP形象极客狐DevFox，来自pictures目录")

            # 2. 不再参考之前生成的章节图片，只使用IP形象保持一致性
            # 这样可以避免参考图片过多导致人物形象分散

            # 3. 构建提示词（章节标题放在"详细呈现以下内容："之后）
            # 格式：请你生成一幅手绘画风格的图片，详细呈现以下内容：{章节标题}\n\n{章节内容}
            if chapter_title:
                # 如果有章节标题，插入到固定格式中
                full_prompt = f"详细呈现以下内容：{chapter_title}\n\n{prompt}"
            else:
                # 如果没有章节标题，使用原始prompt（应该已经包含固定前缀）
                # 去掉开头的"请你生成一幅手绘画风格的图片，"避免重复
                if prompt.startswith("请你生成一幅手绘画风格的图片，"):
                    full_prompt = prompt.replace("请你生成一幅手绘画风格的图片，", "")
                else:
                    full_prompt = prompt

            if reference_info:
                ref_text = "、".join(reference_info) + "。"
                text_prompt = (
                    f"{ref_text}"
                    f"要求：请你以这个IP形象为主人公，生成一幅手绘画风格的图片，{full_prompt}。"
                    f"图片风格：手绘画风格。"
                    f"图片宽高比要求：{aspect_ratio_prompts.get(aspect_ratio, aspect_ratio)}。"
                    f"重要：图片中可以出现标题和描述性文字，但不要出现'极客狐DevFox'、'DevFox灵狐'或其他名字标识。"
                )
            else:
                # 没有参考图片时的提示词
                text_prompt = (
                    f"请生成一张手绘画风格的图片。"
                    f"描述：{full_prompt}。"
                    f"图片宽高比要求：{aspect_ratio_prompts.get(aspect_ratio, aspect_ratio)}。"
                )

            contents.append(types.Part(text=text_prompt))

            # 配置响应为图片格式
            config = types.GenerateContentConfig(
                response_modalities=["IMAGE"]
            )

            # 调用 Gemini API
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config
            )

            # 检查响应中的图片
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]

                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    for part in candidate.content.parts:
                        if hasattr(part, 'inline_data'):
                            inline_data = part.inline_data

                            if hasattr(inline_data, 'data'):
                                raw_data = inline_data.data

                                # 处理数据
                                if isinstance(raw_data, bytes):
                                    image_data = raw_data
                                elif isinstance(raw_data, str):
                                    image_data = base64.b64decode(raw_data)
                                else:
                                    image_data = base64.b64decode(str(raw_data))

                                if image_data:
                                    return self._save_image(image_data, filename)

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "prompt": prompt
            }

        return {
            "success": False,
            "error": "无法生成图片",
            "prompt": prompt
        }

    def _save_image(self, image_data: bytes, filename: Optional[str] = None) -> dict:
        """保存图片并返回结果"""
        if filename:
            output_path = OUTPUT_DIR / filename
        else:
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = OUTPUT_DIR / f"image_{timestamp}.png"

        # 保存图片
        output_path.write_bytes(image_data)

        return {
            "success": True,
            "filename": output_path.name,
            "path": str(output_path.absolute()),
            "url": str(output_path.absolute())
        }

    def generate_batch(
        self,
        prompts: list[str],
        aspect_ratio: str = "3:4",
        use_ip: bool = True
    ) -> list[dict]:
        """
        批量生成图片

        Args:
            prompts: 提示词列表
            aspect_ratio: 宽高比
            use_ip: 是否使用 IP 形象

        Returns:
            生成结果列表
        """
        results = []
        for i, prompt in enumerate(prompts, 1):
            print(f"正在生成第 {i}/{len(prompts)} 张图片...")
            result = self.generate_image(prompt, aspect_ratio, use_ip=use_ip)
            results.append(result)
        return results


def main():
    """命令行入口"""
    if len(sys.argv) < 2:
        print("用法: python image_generator.py '<提示词>' [宽高比] [文件名] [输出目录] [图片类型] [章节标题]")
        print("示例(微信封面): python image_generator.py '内容' '21:9' 'cover.png' '/path/to/output' 'wechat'")
        print("示例(小红书封面): python image_generator.py '内容' '3:4' 'cover.png' '/path/to/output' 'cover'")
        print("示例(小红书章节): python image_generator.py '章节内容' '3:4' 'chapter_1.png' '/path/to/output' 'chapter' '第一章：为什么你需要Moltbot'")
        print("示例(Twitter推文): python image_generator.py '推文内容' '4:5' 'tweet.png' '/path/to/output' 'tweet'")
        print(f"\n支持的宽高比: {', '.join(ASPECT_RATIOS)}")
        print(f"支持的图片类型及IP目录:")
        print(f"  - wechat   (微信公众号封面图) → pictures/common/")
        print(f"  - cover    (小红书封面图)     → pictures/cover/ + pictures/common/")
        print(f"  - chapter  (小红书章节图)     → pictures/chapter/ + pictures/common/")
        print(f"  - tweet    (Twitter推文图)    → pictures/twitter/tweet/ + pictures/twitter/common/")
        print(f"\n参考策略: 只使用 pictures/ 目录下的IP形象作为参考")
        print(f"章节标题: 可选参数，用于在图片中显示章节信息")
        sys.exit(1)

    prompt = sys.argv[1]
    image_type = sys.argv[5] if len(sys.argv) > 5 else "chapter"

    # 根据图片类型设置默认宽高比
    default_aspect_ratios = {
        "wechat": "21:9",   # 微信公众号封面图默认21:9（接近微信官方2.35:1）
        "tweet": "4:5",     # Twitter推文图默认4:5
        "cover": "3:4",     # 小红书封面图默认3:4
        "chapter": "3:4"    # 小红书章节图默认3:4
    }
    default_aspect_ratio = default_aspect_ratios.get(image_type, "3:4")

    aspect_ratio = sys.argv[2] if len(sys.argv) > 2 else default_aspect_ratio
    filename = sys.argv[3] if len(sys.argv) > 3 else None
    output_dir = sys.argv[4] if len(sys.argv) > 4 else None
    chapter_title = sys.argv[6] if len(sys.argv) > 6 else None

    # 更新输出目录（如果指定）
    global OUTPUT_DIR
    if output_dir:
        OUTPUT_DIR = Path(output_dir)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generator = ImageGenerator()
    result = generator.generate_image(prompt, aspect_ratio, filename, use_ip=True, image_type=image_type, chapter_title=chapter_title)

    if result.get("success"):
        print(f"✓ 图片生成成功: {result['path']}")
        print(f"✓ 使用 pictures/ 目录下的IP形象作为参考，保持人物风格一致")
        if chapter_title:
            print(f"✓ 章节标题: {chapter_title}")
    else:
        print(f"✗ 图片生成失败: {result.get('error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
