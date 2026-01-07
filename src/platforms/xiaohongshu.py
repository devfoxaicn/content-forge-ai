"""
小红书平台适配器
适配小红书的内容格式、风格、emoji使用等
"""

from typing import Dict, Any, List
from .base import BasePlatformAdapter, PlatformContent


class XiaohongshuAdapter(BasePlatformAdapter):
    """小红书平台适配器"""

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        super().__init__(config, prompts)
        self.platform_name = "小红书"

        # 小红书特定配置
        self.use_emojis = config.get("use_emojis", True)
        self.max_length = config.get("max_length", 1000)
        self.image_required = config.get("image_required", True)

        # 小红书常用emoji
        self.emoji_list = [
            "🔥", "✨", "💡", "🚀", "💪", "🎯", "⭐", "👍",
            "📌", "💻", "🤖", "🔧", "📚", "🎁", "🔔", "💬"
        ]

    def adapt_content(self, raw_content: Dict[str, Any]) -> PlatformContent:
        """
        适配内容为小红书格式

        特点：
        - 标题emoji + 吸引眼球
        - 内容分段 + emoji点缀
        - hashtag标签
        - 干货风格
        """
        title = raw_content.get("title", "")
        body = raw_content.get("body", "")
        keywords = raw_content.get("keywords", [])
        images = raw_content.get("images", [])

        # 格式化标题
        formatted_title = self._format_xiaohongshu_title(title)

        # 格式化内容
        formatted_content = self._format_xiaohongshu_content(body, keywords)

        # 生成标签
        tags = self._generate_hashtags(keywords)

        return PlatformContent(
            title=formatted_title,
            content=formatted_content,
            images=images,
            tags=tags,
            metadata={
                "platform": "xiaohongshu",
                "image_required": self.image_required,
                "use_emojis": self.use_emojis
            }
        )

    def _format_xiaohongshu_title(self, title: str) -> str:
        """格式化小红书标题"""
        if not self.use_emojis:
            return title

        # 添加emoji到标题
        emoji = self.emoji_list[0]  # 🔥
        return f"{emoji} {title}"

    def _format_xiaohongshu_content(self, content: str, keywords: List[str]) -> str:
        """格式化小红书内容"""
        if not self.use_emojis:
            return content

        # 分段处理
        paragraphs = content.split("\n\n")
        formatted_paragraphs = []

        for i, para in enumerate(paragraphs):
            if not para.strip():
                continue

            # 为每个段落添加emoji
            emoji_index = min(i + 1, len(self.emoji_list) - 1)
            emoji = self.emoji_list[emoji_index]

            # 确保段落不为空
            if para.strip():
                formatted_paragraphs.append(f"{emoji} {para.strip()}")

        # 用空行连接
        formatted_content = "\n\n".join(formatted_paragraphs)

        # 添加结尾
        if formatted_content:
            formatted_content += f"\n\n{self.emoji_list[2]} 觉得有用的话，记得点赞收藏哦～"

        return formatted_content

    def _generate_hashtags(self, keywords: List[str]) -> List[str]:
        """生成小红书标签"""
        # 常用标签
        common_tags = ["#AI工具", "#干货分享", "#效率提升"]

        # 基于关键词生成标签
        keyword_tags = [f"#{kw}" for kw in keywords[:5]]

        return common_tags + keyword_tags

    def validate_content(self, content: PlatformContent) -> bool:
        """验证小红书内容"""
        # 检查标题
        if not content.title or len(content.title) == 0:
            self.log("标题不能为空", "ERROR")
            return False

        # 检查内容长度
        if len(content.content) > self.max_length:
            self.log(f"内容长度超出限制（{len(content.content)} > {self.max_length}）", "WARNING")
            # 可以截断或返回False
            # return False

        # 检查图片
        if self.image_required and not content.images:
            self.log("小红书需要配图", "WARNING")
            # 不强制阻塞，仅警告

        # 检查标签
        if not content.tags:
            self.log("建议添加标签", "WARNING")

        return True

    def get_publish_method(self) -> str:
        """获取发布方式"""
        return self.config.get("publish_method", "mcp")

    def get_max_length(self) -> int:
        """获取最大长度"""
        return self.max_length

    def get_required_elements(self) -> List[str]:
        """获取必需元素"""
        elements = ["title", "content"]
        if self.image_required:
            elements.append("images")
        return elements
