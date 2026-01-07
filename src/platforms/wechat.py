"""
微信公众号平台适配器
适配微信公众号的HTML格式、排版等
"""

from typing import Dict, Any, List
from .base import BasePlatformAdapter, PlatformContent


class WechatAdapter(BasePlatformAdapter):
    """微信公众号平台适配器"""

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        super().__init__(config, prompts)
        self.platform_name = "微信公众号"

        # 微信特定配置
        self.use_html = config.get("use_html", True)
        self.max_length = config.get("max_length", 5000)
        self.image_required = config.get("image_required", True)
        self.style = config.get("style", "professional")

    def adapt_content(self, raw_content: Dict[str, Any]) -> PlatformContent:
        """
        适配内容为微信公众号格式

        特点：
        - HTML富文本格式
        - 排版精美（引用、加粗、分割线）
        - 封面图必需
        - 引导关注
        - 转发鼓励
        """
        title = raw_content.get("title", "")
        body = raw_content.get("body", "")
        keywords = raw_content.get("keywords", [])
        images = raw_content.get("images", [])

        # 格式化标题
        formatted_title = self._format_wechat_title(title)

        # 格式化内容
        formatted_content = self._format_wechat_content(body, keywords)

        # 生成标签（公众号用标签）
        tags = keywords[:5]

        return PlatformContent(
            title=formatted_title,
            content=formatted_content,
            images=images,
            tags=tags,
            metadata={
                "platform": "wechat",
                "use_html": self.use_html,
                "style": self.style,
                "need_cover_image": self.image_required
            }
        )

    def _format_wechat_title(self, title: str) -> str:
        """格式化微信公众号标题"""
        # 微信标题可以直接使用，但要避免emoji
        import re
        title_clean = re.sub(r'[^\w\s\u4e00-\u9fff，。！？、；：""''（）【】]+', '', title)
        return title_clean.strip() or title

    def _format_wechat_content(self, content: str, keywords: List[str]) -> str:
        """格式化微信公众号内容为HTML"""
        if not self.use_html:
            return content

        # 分段处理
        paragraphs = content.split("\n\n")
        html_paragraphs = []

        # 添加开头
        html_paragraphs.append('<section style="font-size: 16px; line-height: 1.8; color: #333;">')

        for para in paragraphs:
            if not para.strip():
                continue

            # HTML段落
            formatted_para = para.strip()

            # 如果是强调内容，使用strong标签
            if "重要" in formatted_para or "注意" in formatted_para or "关键" in formatted_para:
                formatted_para = f'<strong>{formatted_para}</strong>'

            html_paragraphs.append(f'<p style="margin: 15px 0; text-indent: 2em;">{formatted_para}</p>')

        # 添加结尾
        html_paragraphs.append('<hr style="border: none; border-top: 1px solid #eee; margin: 30px 0;">')
        html_paragraphs.append('<p style="text-align: center; color: #888; font-size: 14px; margin: 20px 0;">')
        html_paragraphs.append('--- END ---')
        html_paragraphs.append('</p>')
        html_paragraphs.append('<p style="margin: 20px 0; padding: 15px; background: #f5f5f5; border-radius: 5px; text-align: center;">')
        html_paragraphs.append('<strong>欢迎关注我的公众号，获取更多AI干货！</strong>')
        html_paragraphs.append('</p>')
        html_paragraphs.append('</section>')

        return "\n".join(html_paragraphs)

    def validate_content(self, content: PlatformContent) -> bool:
        """验证微信公众号内容"""
        # 检查标题
        if not content.title or len(content.title) < 10:
            self.log("公众号标题建议至少10个字", "WARNING")

        # 检查内容长度
        if len(content.content) < 300:
            self.log("公众号内容太短（建议至少300字）", "WARNING")
            return False

        if len(content.content) > self.max_length:
            self.log(f"内容长度超出限制", "WARNING")

        # 检查封面图
        if self.image_required and not content.images:
            self.log("公众号需要封面图", "ERROR")
            return False

        return True

    def get_publish_method(self) -> str:
        """获取发布方式"""
        return self.config.get("publish_method", "api")

    def get_max_length(self) -> int:
        """获取最大长度"""
        return self.max_length

    def get_required_elements(self) -> List[str]:
        """获取必需元素"""
        elements = ["title", "content"]
        if self.image_required:
            elements.append("images")
        return elements
