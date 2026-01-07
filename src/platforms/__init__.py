"""
平台适配器模块
支持多平台内容发布
"""

from .base import BasePlatformAdapter, PlatformContent
from .xiaohongshu import XiaohongshuAdapter
from .zhihu import ZhihuAdapter
from .wechat import WechatAdapter

__all__ = [
    "BasePlatformAdapter",
    "PlatformContent",
    "XiaohongshuAdapter",
    "ZhihuAdapter",
    "WechatAdapter"
]
