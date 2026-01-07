"""
平台适配器基类
所有平台适配器都继承此类
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class PlatformContent:
    """平台内容格式"""
    title: str
    content: str
    images: List[str]
    tags: List[str]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "title": self.title,
            "content": self.content,
            "images": self.images,
            "tags": self.tags,
            "metadata": self.metadata
        }


class BasePlatformAdapter(ABC):
    """平台适配器基类"""

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        """
        初始化平台适配器

        Args:
            config: 平台配置
            prompts: 提示词配置
        """
        self.config = config
        self.prompts = prompts
        self.platform_name = self.__class__.__name__.replace("Adapter", "")

    @abstractmethod
    def adapt_content(self, raw_content: Dict[str, Any]) -> PlatformContent:
        """
        将通用内容适配为平台特定格式

        Args:
            raw_content: 原始内容（包含title, body, keywords等）

        Returns:
            PlatformContent: 平台特定格式的内容
        """
        pass

    @abstractmethod
    def validate_content(self, content: PlatformContent) -> bool:
        """
        验证内容是否符合平台要求

        Args:
            content: 平台内容

        Returns:
            bool: 是否通过验证
        """
        pass

    @abstractmethod
    def get_publish_method(self) -> str:
        """
        获取发布方式

        Returns:
            str: 发布方式（mcp, api, manual）
        """
        pass

    def format_title(self, title: str) -> str:
        """
        格式化标题（可被子类覆盖）

        Args:
            title: 原始标题

        Returns:
            str: 格式化后的标题
        """
        return title

    def format_content(self, content: str) -> str:
        """
        格式化内容（可被子类覆盖）

        Args:
            content: 原始内容

        Returns:
            str: 格式化后的内容
        """
        return content

    def add_platform_specific_elements(self, content: PlatformContent) -> PlatformContent:
        """
        添加平台特定元素（emoji、hashtag等）

        Args:
            content: 平台内容

        Returns:
            PlatformContent: 添加了平台元素的内容
        """
        return content

    def get_max_length(self) -> int:
        """
        获取平台最大内容长度

        Returns:
            int: 最大字符数
        """
        return 10000

    def get_required_elements(self) -> List[str]:
        """
        获取平台必需元素

        Returns:
            List[str]: 必需元素列表（如: ["title", "content", "images"]）
        """
        return ["title", "content"]

    def log(self, message: str, level: str = "INFO"):
        """
        日志输出

        Args:
            message: 日志消息
            level: 日志级别
        """
        from loguru import logger
        log_func = getattr(logger, level.lower(), logger.info)
        log_func(f"[{self.platform_name}] {message}")
