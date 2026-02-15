"""
统一存储管理系统 v2.0

支持两种内容模式：
1. Daily Mode: 每日热点内容
2. Series Mode: 100期技术博客系列

存储结构：
data/
├── daily/             # 每日热点模式
│   └── YYYYMMDD/
│       ├── raw/
│       └── digest/
│
├── series/            # 100期技术博客系列
│   ├── LLM_series/    # LLM系列分类
│   │   └── {series_id}/   # 如 series_1_llm_foundation
│   │       ├── episode_{episode_number}/
│   │       │   ├── episode_metadata.json
│   │       │   └── epXXX_标题_article.md
│   │       └── series_metadata.json
│   └── ML_series/      # ML系列分类
│       └── {series_id}/   # 如 ml_series_1_ml_foundation
│           ├── episode_{episode_number}/
│           │   ├── episode_metadata.json
│           │   └── epXXX_标题_article.md
│           └── series_metadata.json
│
└── archive/           # 归档内容
"""

import os
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Literal
from abc import ABC, abstractmethod


def _get_now_beijing_time():
    """获取北京时间（兼容 TZ 环境变量）

    优先级：
    1. 使用 TZ 环境变量（如果设置）
    2. 否则使用 UTC+8（北京时间）

    Returns:
        datetime: 当前时间
    """
    tz_str = os.environ.get('TZ')

    if tz_str:
        # 设置时区（仅限 Unix/Linux）
        try:
            # 尝试使用 zoneinfo（Python 3.9+）
            try:
                from zoneinfo import ZoneInfo
                return datetime.now(ZoneInfo(tz_str))
            except ImportError:
                # Python 3.8 或更早版本，使用时区偏移
                if tz_str == 'Asia/Shanghai':
                    # 北京时间 UTC+8
                    return datetime.now(timezone.utc) + timedelta(hours=8)
                else:
                    # 其他时区，返回 UTC（简化处理）
                    return datetime.now(timezone.utc)
        except Exception:
            # 出错时返回 UTC+8
            return datetime.now(timezone.utc) + timedelta(hours=8)
    else:
        # 没有设置 TZ 环境变量，默认使用 UTC+8
        return datetime.now(timezone.utc) + timedelta(hours=8)


class BaseStorage(ABC):
    """存储系统基类"""

    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def get_root_dir(self) -> Path:
        """获取根目录"""
        pass

    def _create_subdirs(self, parent_dir: Path) -> None:
        """创建标准子目录"""
        subdirs = ["raw", "digest", "longform", "xiaohongshu", "twitter"]
        for subdir in subdirs:
            (parent_dir / subdir).mkdir(exist_ok=True)

    def get_path(self, category: str, filename: str) -> Path:
        """获取文件完整路径"""
        category_dir = self.get_root_dir() / category
        category_dir.mkdir(parents=True, exist_ok=True)
        return category_dir / filename

    def save_json(self, category: str, filename: str, data: dict, indent: int = 2) -> Path:
        """保存JSON文件"""
        filepath = self.get_path(category, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        return filepath

    def save_markdown(self, category: str, filename: str, content: str) -> Path:
        """保存Markdown文件"""
        filepath = self.get_path(category, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return filepath

    def save_text(self, category: str, filename: str, content: str) -> Path:
        """保存文本文件"""
        filepath = self.get_path(category, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return filepath

    def list_files(self, category: Optional[str] = None) -> list:
        """列出目录中的文件"""
        if category:
            target_dir = self.get_root_dir() / category
        else:
            target_dir = self.get_root_dir()

        if not target_dir.exists():
            return []
        return list(target_dir.glob("*"))


class DailyStorage(BaseStorage):
    """每日热点模式存储

    路径：data/daily/YYYYMMDD/
    只创建 raw 和 digest 目录
    """

    def __init__(self, date: Optional[str] = None, base_dir: str = "data"):
        """
        初始化每日存储

        Args:
            date: 日期字符串 (YYYYMMDD)，默认为今天
            base_dir: 基础存储目录
        """
        super().__init__(base_dir)

        if date:
            self.date_str = date
        else:
            self.date_str = _get_now_beijing_time().strftime("%Y%m%d")

        # 创建 daily/{日期} 目录
        self.daily_dir = self.base_dir / "daily" / self.date_str
        self.daily_dir.mkdir(parents=True, exist_ok=True)

        # 只创建 raw 和 digest 目录
        self._create_daily_subdirs(self.daily_dir)

    def get_root_dir(self) -> Path:
        return self.daily_dir

    def get_date_dir(self) -> Path:
        """获取日期目录路径"""
        return self.daily_dir

    def get_date_string(self) -> str:
        return self.date_str

    def _create_daily_subdirs(self, parent_dir: Path) -> None:
        """创建Daily模式专用子目录（只创建raw和digest）"""
        subdirs = ["raw", "digest"]
        for subdir in subdirs:
            (parent_dir / subdir).mkdir(exist_ok=True)


class SeriesStorage(BaseStorage):
    """系列内容存储（100期技术博客）

    目标路径结构：
    data/series/LLM_series/series_1_llm_foundation/episode_001/

    文件命名: epXXX_标题_article.md (如 ep001_transformer架构深度解析_article.md)
    文件直接保存在episode目录下，不创建子目录

    完整示例：
    data/series/LLM_series/series_1_llm_foundation/episode_001/
    ├── ep001_transformer架构深度解析_article.md
    └── episode_metadata.json

    data/series/LLM_series/series_1_llm_foundation/
    └── series_metadata.json
    """

    def __init__(
        self,
        series_id: str,
        episode_number: int,
        base_dir: str = "data"
    ):
        """
        初始化系列存储

        Args:
            series_id: 系列ID (如 "series_1", "ml_series_1")
            episode_number: 集数编号
            base_dir: 基础存储目录
        """
        super().__init__(base_dir)

        self.series_id = series_id
        self.episode_number = episode_number
        self.episode_str = f"episode_{episode_number:03d}"

        # 导入SeriesPathManager获取分类
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root))
        from src.utils.series_manager import SeriesPathManager

        # 获取系列分类
        category = SeriesPathManager.get_series_category(series_id)

        # 获取系列目录名
        series_dir_name = SeriesPathManager.get_series_directory_name(series_id)

        # 创建分类目录/系列目录/episode目录
        self.episode_dir = (
            self.base_dir / "series" / category / series_dir_name / self.episode_str
        )
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        # 不创建任何子目录

    def get_root_dir(self) -> Path:
        return self.episode_dir

    def get_series_id(self) -> str:
        return self.series_id

    def get_episode_number(self) -> int:
        return self.episode_number

    def get_episode_dir(self) -> Path:
        return self.episode_dir

    def _create_subdirs(self, parent_dir: Path) -> None:
        """不创建子目录（覆盖父类方法）"""
        pass

    def save_article(
        self,
        content: str,
        title: str = "",
        filename: str = ""
    ) -> Path:
        """
        保存文章到episode目录

        Args:
            content: 文章内容
            title: 文章标题（用于生成文件名）
            filename: 自定义文件名（优先级高于title）

        Returns:
            文件路径
        """
        # 确定文件名
        if filename:
            filepath = self.episode_dir / filename
        elif title:
            # 使用TopicFormatter生成文件名
            from src.utils.series_manager import TopicFormatter
            # 构造临时topic对象用于文件名生成
            topic_temp = {"episode": self.episode_number, "title": title}
            filename = TopicFormatter.generate_markdown_filename(topic_temp, "article")
            filepath = self.episode_dir / filename
        else:
            # 默认文件名
            filepath = self.episode_dir / f"ep{self.episode_number:03d}_article.md"

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return filepath

    def save_episode_metadata(self, metadata: dict) -> Path:
        """保存单集元数据"""
        return self.save_json("", "episode_metadata.json", metadata)

    def save_series_metadata(self, metadata: dict) -> Path:
        """保存系列元数据（存放在系列根目录）"""
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root))
        from src.utils.series_manager import SeriesPathManager

        category = SeriesPathManager.get_series_category(self.series_id)
        series_dir_name = SeriesPathManager.get_series_directory_name(self.series_id)

        series_root = self.base_dir / "series" / category / series_dir_name
        series_root.mkdir(parents=True, exist_ok=True)
        filepath = series_root / "series_metadata.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return filepath


class TopicStorage(BaseStorage):
    """Topic 模式存储

    路径结构：
    data/topic/YYYYMMDD/{topic_name}/
    ├── research_report.md   # 调研报告（含事前核查）
    ├── article.md           # 长文本文章
    ├── fact_check.md        # 事后核查报告
    ├── platforms/           # 多平台适配
    │   ├── xiaohongshu.md
    │   ├── weixin.md
    │   └── zhihu.md
    └── images/              # 配图
        ├── cover.png
        └── chapter_*.png
    """

    def __init__(
        self,
        topic_name: str,
        date_str: Optional[str] = None,
        base_dir: str = "data"
    ):
        """
        初始化 Topic 存储

        Args:
            topic_name: 话题名称
            date_str: 日期字符串 (YYYYMMDD)，默认为今天
            base_dir: 基础存储目录
        """
        super().__init__(base_dir)

        # 清理话题名称（用于目录名）
        self.topic_name = self._sanitize_name(topic_name)

        if date_str:
            self.date_str = date_str
        else:
            self.date_str = _get_now_beijing_time().strftime("%Y%m%d")

        # 创建 topic/{日期}/{话题} 目录
        self.topic_dir = self.base_dir / "topic" / self.date_str / self.topic_name
        self.topic_dir.mkdir(parents=True, exist_ok=True)

        # 创建子目录
        self._create_topic_subdirs(self.topic_dir)

    def _sanitize_name(self, name: str) -> str:
        """清理名称，只保留安全字符"""
        import re
        # 移除或替换不安全的字符
        name = re.sub(r'[^\w\u4e00-\u9fff\-]', '_', name)
        # 移除连续的下划线
        name = re.sub(r'_+', '_', name)
        # 移除首尾的下划线
        name = name.strip('_')
        # 限制长度
        return name[:50] if name else "untitled"

    def _create_topic_subdirs(self, parent_dir: Path) -> None:
        """创建 Topic 模式专用子目录"""
        subdirs = ["platforms", "images"]
        for subdir in subdirs:
            (parent_dir / subdir).mkdir(exist_ok=True)

    def get_root_dir(self) -> Path:
        return self.topic_dir

    def get_topic_dir(self) -> Path:
        """获取话题目录路径"""
        return self.topic_dir

    def get_topic_name(self) -> str:
        return self.topic_name

    def get_date_string(self) -> str:
        return self.date_str

    def save_article(self, content: str) -> Path:
        """保存长文本文章"""
        filepath = self.topic_dir / "article.md"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return filepath

    def save_research_report(self, content: str) -> Path:
        """保存调研报告（含事前核查）"""
        filepath = self.topic_dir / "research_report.md"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return filepath

    def save_fact_check(self, content: str) -> Path:
        """保存事后核查报告"""
        filepath = self.topic_dir / "fact_check.md"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return filepath

    def save_platform_content(self, platform: str, content: str) -> Path:
        """保存平台适配内容"""
        platforms_dir = self.topic_dir / "platforms"
        platforms_dir.mkdir(exist_ok=True)
        filepath = platforms_dir / f"{platform}.md"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return filepath

    def save_image(self, image_data: bytes, filename: str) -> Path:
        """保存图片"""
        images_dir = self.topic_dir / "images"
        images_dir.mkdir(exist_ok=True)
        filepath = images_dir / filename
        with open(filepath, 'wb') as f:
            f.write(image_data)
        return filepath

    def get_image_path(self, filename: str) -> Path:
        """获取图片路径"""
        return self.topic_dir / "images" / filename

    def save_metadata(self, metadata: dict) -> Path:
        """保存话题元数据"""
        filepath = self.topic_dir / "metadata.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return filepath


class StorageFactory:
    """存储工厂类 - 统一创建存储实例"""

    @staticmethod
    def create_daily(
        date: Optional[str] = None,
        base_dir: str = "data"
    ) -> DailyStorage:
        """创建每日热点存储"""
        return DailyStorage(date, base_dir)

    @staticmethod
    def create_series(
        series_id: str,
        episode_number: int,
        base_dir: str = "data"
    ) -> SeriesStorage:
        """创建系列内容存储"""
        return SeriesStorage(series_id, episode_number, base_dir)

    @staticmethod
    def create_topic(
        topic_name: str,
        date_str: Optional[str] = None,
        base_dir: str = "data"
    ) -> TopicStorage:
        """创建 Topic 模式存储"""
        return TopicStorage(topic_name, date_str, base_dir)

    @staticmethod
    def create_storage(
        mode: Literal["daily", "series", "topic"],
        **kwargs
    ) -> BaseStorage:
        """
        通用存储创建方法

        Args:
            mode: 存储模式 ("daily", "series", "topic")
            **kwargs: 模式特定参数
                - daily: date (可选)
                - series: series_id, episode_number
                - topic: topic_name, date_str (可选)

        Returns:
            BaseStorage: 对应的存储实例
        """
        if mode == "daily":
            return StorageFactory.create_daily(
                date=kwargs.get("date"),
                base_dir=kwargs.get("base_dir", "data")
            )
        elif mode == "series":
            return StorageFactory.create_series(
                series_id=kwargs["series_id"],
                episode_number=kwargs["episode_number"],
                base_dir=kwargs.get("base_dir", "data")
            )
        elif mode == "topic":
            return StorageFactory.create_topic(
                topic_name=kwargs["topic_name"],
                date_str=kwargs.get("date_str"),
                base_dir=kwargs.get("base_dir", "data")
            )
        else:
            raise ValueError(f"Unknown storage mode: {mode}")


# 兼容旧版本的函数
def get_storage(base_dir: str = "data") -> DailyStorage:
    """获取每日存储实例（兼容旧版）"""
    return StorageFactory.create_daily(base_dir=base_dir)


# 导出
__all__ = [
    "BaseStorage",
    "DailyStorage",
    "SeriesStorage",
    "TopicStorage",
    "StorageFactory",
    "get_storage",
]
