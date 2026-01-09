"""
数据存储工具 (v2 - 向后兼容)

处理按日期分层的文件存储

注意：v2.4版本后，BatchStorage已移除。
如需批量生成内容，请使用100期系列模式。
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import json


class DailyStorage:
    """按日期分层的存储管理器

    v2.4更新：存储路径从 data/YYYYMMDD/ 改为 data/daily/YYYYMMDD/
    """

    def __init__(self, base_dir: str = "data"):
        """
        初始化存储管理器

        Args:
            base_dir: 基础存储目录
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # 获取或创建今天的日期目录
        self.date_dir = self._get_or_create_date_dir()

    def _get_or_create_date_dir(self) -> Path:
        """获取或创建日期目录"""
        today = datetime.now()
        date_str = today.strftime("%Y%m%d")

        # v2.4: 使用 daily/ 子目录
        date_dir = self.base_dir / "daily" / date_str

        # 创建日期目录和子目录
        date_dir.mkdir(parents=True, exist_ok=True)

        # 创建所有子目录（5个主目录）
        subdirs = ["raw", "digest", "longform", "xiaohongshu", "twitter"]
        for subdir in subdirs:
            (date_dir / subdir).mkdir(exist_ok=True)

        return date_dir

    def get_path(self, category: str, filename: str, create_dir: bool = True) -> Path:
        """
        获取文件完整路径

        Args:
            category: 子目录名称（raw/digest/longform/xiaohongshu）
            filename: 文件名
            create_dir: 是否创建目录

        Returns:
            Path: 文件完整路径
        """
        category_dir = self.date_dir / category
        if create_dir:
            category_dir.mkdir(parents=True, exist_ok=True)
        return category_dir / filename

    def save_json(self, category: str, filename: str, data: dict, indent: int = 2) -> Path:
        """
        保存JSON文件

        Args:
            category: 子目录名称
            filename: 文件名
            data: 要保存的数据
            indent: JSON缩进

        Returns:
            Path: 保存的文件路径
        """
        filepath = self.get_path(category, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        return filepath

    def save_markdown(self, category: str, filename: str, content: str) -> Path:
        """
        保存Markdown文件

        Args:
            category: 子目录名称
            filename: 文件名
            content: 文件内容

        Returns:
            Path: 保存的文件路径
        """
        filepath = self.get_path(category, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return filepath

    def save_text(self, category: str, filename: str, content: str) -> Path:
        """
        保存文本文件

        Args:
            category: 子目录名称
            filename: 文件名
            content: 文件内容

        Returns:
            Path: 保存的文件路径
        """
        filepath = self.get_path(category, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return filepath

    def get_base_dir(self) -> Path:
        """获取基础目录"""
        return self.base_dir

    def get_date_dir(self) -> Path:
        """获取日期目录"""
        return self.date_dir

    def get_date_string(self) -> str:
        """获取日期字符串"""
        return self.date_dir.name

    def list_files(self, category: Optional[str] = None) -> list:
        """
        列出目录中的文件

        Args:
            category: 子目录名称，如果为None则列出所有文件

        Returns:
            list: 文件路径列表
        """
        if category:
            target_dir = self.date_dir / category
        else:
            target_dir = self.date_dir

        if not target_dir.exists():
            return []

        return list(target_dir.glob("*"))

    def get_latest_file(self, category: str, pattern: str = "*.json") -> Optional[Path]:
        """
        获取指定类别中最新的文件

        Args:
            category: 子目录名称
            pattern: 文件匹配模式

        Returns:
            Optional[Path]: 最新的文件路径，如果没有则返回None
        """
        files = self.list_files(category)
        matching_files = [f for f in files if f.match(pattern)]

        if not matching_files:
            return None

        # 按修改时间排序，返回最新的
        return max(matching_files, key=lambda f: f.stat().st_mtime)


# BatchStorage 已在 v2.4 中移除
# 如需批量生成内容，请使用 SeriesStorage（100期技术博客系列模式）
class BatchStorage:
    """已弃用：BatchStorage 在 v2.4 中已移除

    请使用 src/utils/storage_v2.py 中的 SeriesStorage 代替。
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "BatchStorage 已在 v2.4 中移除。"
            "请使用 src/series_orchestrator.py 生成100期技术博客系列内容，"
            "或使用 src/utils/storage_v2.py 中的 SeriesStorage。"
        )


def get_storage(base_dir: str = "data") -> DailyStorage:
    """
    获取存储管理器实例

    Args:
        base_dir: 基础存储目录

    Returns:
        DailyStorage: 存储管理器实例
    """
    return DailyStorage(base_dir)
