"""
知乎平台适配器
适配知乎的专业、深度、Markdown格式等
"""

from typing import Dict, Any, List
from .base import BasePlatformAdapter, PlatformContent


class ZhihuAdapter(BasePlatformAdapter):
    """知乎平台适配器"""

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        super().__init__(config, prompts)
        self.platform_name = "知乎"

        # 知乎特定配置
        self.use_markdown = config.get("use_markdown", True)
        self.max_length = config.get("max_length", 10000)
        self.image_required = config.get("image_required", False)
        self.style = config.get("style", "professional")  # professional, casual

    def adapt_content(self, raw_content: Dict[str, Any]) -> PlatformContent:
        """
        适配内容为知乎格式

        特点：
        - 专业、深度
        - Markdown格式
        - 结构化（标题、列表、引用）
        - 无emoji或少量使用
        - 鼓励思考和讨论
        """
        title = raw_content.get("title", "")
        body = raw_content.get("body", "")
        keywords = raw_content.get("keywords", [])
        images = raw_content.get("images", [])

        # 格式化标题
        formatted_title = self._format_zhihu_title(title)

        # 格式化内容
        formatted_content = self._format_zhihu_content(body, keywords)

        # 生成标签（知乎用话题而非hashtag）
        tags = self._generate_topics(keywords)

        return PlatformContent(
            title=formatted_title,
            content=formatted_content,
            images=images,
            tags=tags,
            metadata={
                "platform": "zhihu",
                "use_markdown": self.use_markdown,
                "style": self.style
            }
        )

    def _format_zhihu_title(self, title: str) -> str:
        """格式化知乎标题"""
        # 知乎标题更直接、专业
        # 移除emoji
        import re
        title_without_emoji = re.sub(r'[^\w\s\u4e00-\u9fff]+', '', title)
        return title_without_emoji.strip() or title

    def _format_zhihu_content(self, content: str, keywords: List[str]) -> str:
        """格式化知乎内容为Markdown"""
        if not self.use_markdown:
            return content

        # 分段处理
        paragraphs = content.split("\n\n")
        formatted_paragraphs = []

        for para in paragraphs:
            if not para.strip():
                continue

            # 添加适当的Markdown格式
            formatted_para = para.strip()

            # 如果是列表项，确保有-
            if formatted_para.startswith("•"):
                formatted_para = "- " + formatted_para[1:].strip()

            formatted_paragraphs.append(formatted_para)

        # 用空行连接
        formatted_content = "\n\n".join(formatted_paragraphs)

        # 添加结尾引导
        if formatted_content:
            formatted_content += "\n\n---\n\n"
            formatted_content += "**欢迎在评论区分享你的看法和经验！**\n\n"
            formatted_content += "如果你觉得这篇文章有帮助，请点赞关注，我会持续分享AI相关的干货内容～"

        return formatted_content

    def _generate_topics(self, keywords: List[str]) -> List[str]:
        """生成知乎话题"""
        # 直接使用关键词作为话题
        return keywords[:5]

    def validate_content(self, content: PlatformContent) -> bool:
        """验证知乎内容"""
        # 检查标题
        if not content.title or len(content.title) < 5:
            self.log("知乎标题太短（建议至少5个字）", "WARNING")

        # 检查内容长度
        if len(content.content) < 100:
            self.log("知乎内容太短（建议至少100字）", "WARNING")
            return False

        if len(content.content) > self.max_length:
            self.log(f"内容长度超出限制（{len(content.content)} > {self.max_length}）", "WARNING")

        # 检查深度（简单判断）
        if content.content.count("\n") < 3:
            self.log("知乎内容建议分段，提升可读性", "WARNING")

        return True

    def get_publish_method(self) -> str:
        """获取发布方式"""
        return self.config.get("publish_method", "api")

    def get_max_length(self) -> int:
        """获取最大长度"""
        return self.max_length

    def get_required_elements(self) -> List[str]:
        """获取必需元素"""
        return ["title", "content"]
