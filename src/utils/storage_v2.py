"""
统一存储管理系统 v2.0

支持四种内容模式：
1. Daily Mode: 每日热点内容
2. Series Mode: 100期技术博客系列
3. Custom Mode: 用户自定义内容生成
4. Refine Mode: 内容精炼（多平台转换）

存储结构：
data/
├── daily/             # 每日热点模式
│   └── YYYYMMDD/
│       ├── raw/
│       ├── digest/
│       ├── longform/
│       ├── xiaohongshu/
│       └── twitter/
│
├── series/            # 100期技术博客系列
│   └── {series_id}/   # 如 series_1_llm_foundation
│       ├── episode_{episode_number}/
│       │   ├── raw/
│       │   ├── digest/
│       │   ├── longform/
│       │   ├── xiaohongshu/
│       │   └── twitter/
│       └── metadata.json
│
├── custom/            # 用户自定义内容
│   └── {timestamp}_{topic}/
│       └── article.md  # 直接在根目录，不创建子目录
│
├── refine/            # 内容精炼
│   └── {source_name}/
│       ├── raw/
│       ├── wechat/
│       ├── xiaohongshu/
│       └── twitter/
│
└── archive/           # 归档内容
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Literal
from abc import ABC, abstractmethod


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
            self.date_str = datetime.now().strftime("%Y%m%d")

        # 创建 daily/{日期} 目录
        self.daily_dir = self.base_dir / "daily" / self.date_str
        self.daily_dir.mkdir(parents=True, exist_ok=True)
        self._create_subdirs(self.daily_dir)

    def get_root_dir(self) -> Path:
        return self.daily_dir

    def get_date_string(self) -> str:
        return self.date_str


class SeriesStorage(BaseStorage):
    """系列内容存储（100期技术博客）

    路径：data/series/{series_id}/episode_{episode_number}/
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
            series_id: 系列ID (如 "series_1_llm_foundation")
            episode_number: 集数编号
            base_dir: 基础存储目录
        """
        super().__init__(base_dir)

        self.series_id = series_id
        self.episode_number = episode_number
        self.episode_str = f"episode_{episode_number:03d}"

        # 创建 series/{series_id}/episode_{xxx} 目录
        self.episode_dir = (
            self.base_dir / "series" / self.series_id / self.episode_str
        )
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self._create_subdirs(self.episode_dir)

    def get_root_dir(self) -> Path:
        return self.episode_dir

    def get_series_id(self) -> str:
        return self.series_id

    def get_episode_number(self) -> int:
        return self.episode_number

    def get_episode_dir(self) -> Path:
        return self.episode_dir

    def save_episode_metadata(self, metadata: dict) -> Path:
        """保存单集元数据"""
        return self.save_json("", "episode_metadata.json", metadata)

    def save_series_metadata(self, metadata: dict) -> Path:
        """保存系列元数据（存放在系列根目录）"""
        series_root = self.base_dir / "series" / self.series_id
        series_root.mkdir(parents=True, exist_ok=True)
        filepath = series_root / "series_metadata.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return filepath


class CustomStorage(BaseStorage):
    """自定义内容存储

    路径：data/custom/{storage_id}/
    storage_id 格式：YYYYMMDD_HHMMSS_topic

    注意：Custom模式直接在根目录保存文件，不创建子目录
    """

    def __init__(
        self,
        storage_id: str,
        base_dir: str = "data"
    ):
        """
        初始化自定义内容存储

        Args:
            storage_id: 存储ID，格式为 YYYYMMDD_HHMMSS_topic
            base_dir: 基础存储目录
        """
        super().__init__(base_dir)

        self.storage_id = storage_id

        # 创建 custom/{storage_id} 目录（不创建子目录）
        self.custom_dir = self.base_dir / "custom" / storage_id
        self.custom_dir.mkdir(parents=True, exist_ok=True)
        # Custom模式不需要创建任何子目录

    def get_root_dir(self) -> Path:
        return self.custom_dir

    def get_storage_id(self) -> str:
        return self.storage_id


class RefineStorage(BaseStorage):
    """内容精炼存储

    路径：data/refine/{source_name}/

    注意：Refine 模式需要支持 wechat 目录（HTML 格式）
    """

    def __init__(
        self,
        source_name: str,
        base_dir: str = "data"
    ):
        """
        初始化精炼内容存储

        Args:
            source_name: 源文件名称（不含扩展名）
            base_dir: 基础存储目录
        """
        super().__init__(base_dir)

        self.source_name = source_name

        # 创建 refine/{source_name} 目录
        self.refine_dir = self.base_dir / "refine" / source_name
        self.refine_dir.mkdir(parents=True, exist_ok=True)

        # Refine 模式需要额外的 wechat 目录
        self._create_refine_subdirs(self.refine_dir)

    def _create_refine_subdirs(self, parent_dir: Path) -> None:
        """创建精炼模式专用子目录"""
        # 标准子目录
        subdirs = ["raw", "xiaohongshu", "twitter", "wechat"]
        for subdir in subdirs:
            (parent_dir / subdir).mkdir(exist_ok=True)

    def get_root_dir(self) -> Path:
        return self.refine_dir

    def get_source_name(self) -> str:
        return self.source_name


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
    def create_custom(
        storage_id: str,
        base_dir: str = "data"
    ) -> "CustomStorage":
        """创建自定义内容存储"""
        return CustomStorage(storage_id, base_dir)

    @staticmethod
    def create_refine(
        source_name: str,
        base_dir: str = "data"
    ) -> "RefineStorage":
        """创建精炼内容存储"""
        return RefineStorage(source_name, base_dir)

    @staticmethod
    def create_storage(
        mode: Literal["daily", "series", "custom", "refine"],
        **kwargs
    ) -> BaseStorage:
        """
        通用存储创建方法

        Args:
            mode: 存储模式 ("daily", "series", "custom", "refine")
            **kwargs: 模式特定参数
                - daily: date (可选)
                - series: series_id, episode_number
                - custom: storage_id
                - refine: source_name

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
        elif mode == "custom":
            return StorageFactory.create_custom(
                storage_id=kwargs["storage_id"],
                base_dir=kwargs.get("base_dir", "data")
            )
        elif mode == "refine":
            return StorageFactory.create_refine(
                source_name=kwargs["source_name"],
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
    "CustomStorage",
    "RefineStorage",
    "StorageFactory",
    "get_storage",
]
