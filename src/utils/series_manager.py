"""
100期技术博客系列管理器

管理100期技术博客的元数据、进度追踪、存储组织
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class SeriesMetadata:
    """系列元数据类"""

    def __init__(self, metadata_path: str = "config/blog_topics_100_complete.json"):
        """
        初始化系列元数据

        Args:
            metadata_path: 元数据JSON文件路径
        """
        self.metadata_path = Path(metadata_path)
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> dict:
        """加载元数据文件"""
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_batch_info(self) -> dict:
        """获取批次信息"""
        return self.metadata.get("batch_info", {})

    def get_series_list(self) -> List[dict]:
        """获取所有系列列表"""
        return self.metadata.get("series", [])

    def get_series_by_id(self, series_id: str) -> Optional[dict]:
        """根据ID获取系列信息"""
        for series in self.get_series_list():
            if series["id"] == series_id:
                return series
        return None

    def get_all_topics(self) -> List[dict]:
        """获取所有话题"""
        return self.metadata.get("topics", [])

    def get_topic_by_episode(self, episode: int) -> Optional[dict]:
        """根据集数获取话题"""
        for topic in self.get_all_topics():
            if topic["episode"] == episode:
                return topic
        return None

    def get_topic_by_id(self, topic_id: str) -> Optional[dict]:
        """根据ID获取话题"""
        for topic in self.get_all_topics():
            if topic["id"] == topic_id:
                return topic
        return None

    def get_topics_by_series(self, series_id: str) -> List[dict]:
        """获取指定系列的所有话题"""
        return [
            topic for topic in self.get_all_topics()
            if topic["series_id"] == series_id
        ]

    def get_pending_topics(self) -> List[dict]:
        """获取待生成的话题"""
        return [
            topic for topic in self.get_all_topics()
            if topic.get("status") == "pending"
        ]

    def get_completed_topics(self) -> List[dict]:
        """获取已完成的话题"""
        return [
            topic for topic in self.get_all_topics()
            if topic.get("status") == "completed"
        ]

    def update_topic_status(
        self,
        topic_id: str,
        status: str,
        completed_at: Optional[str] = None
    ) -> None:
        """
        更新话题状态

        Args:
            topic_id: 话题ID
            status: 新状态 (pending/generating/completed/failed)
            completed_at: 完成时间（可选）
        """
        for topic in self.metadata["topics"]:
            if topic["id"] == topic_id:
                topic["status"] = status
                if completed_at:
                    topic["completed_at"] = completed_at
                elif status == "completed":
                    topic["completed_at"] = datetime.now().strftime("%Y-%m-%d")
                break

        self._save_metadata()

    def _save_metadata(self) -> None:
        """保存元数据到文件"""
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def get_progress_summary(self) -> dict:
        """获取进度摘要"""
        total = len(self.get_all_topics())
        completed = len(self.get_completed_topics())
        pending = len(self.get_pending_topics())
        generating = total - completed - pending

        return {
            "total": total,
            "completed": completed,
            "pending": pending,
            "generating": generating,
            "completion_rate": f"{completed / total * 100:.1f}%" if total > 0 else "0%"
        }

    def get_series_summary(self) -> List[dict]:
        """获取各系列摘要"""
        summary = []

        for series in self.get_series_list():
            topics = self.get_topics_by_series(series["id"])
            completed = sum(1 for t in topics if t.get("status") == "completed")

            summary.append({
                "series_id": series["id"],
                "series_name": series["name"],
                "total_episodes": series["topic_count"],
                "completed_episodes": completed,
                "completion_rate": f"{completed / series['topic_count'] * 100:.1f}%"
            })

        return summary


class SeriesPathManager:
    """系列路径管理器"""

    # 系列ID到友好名称的映射（与实际文件夹名称一致）
    SERIES_NAME_MAP = {
        # LLM系列
        "series_1": "llm_foundation",
        "series_2": "rag_technique",
        "series_3": "agent_development",
        "series_4": "prompt_engineering",
        "series_5": "model_deployment",
        "series_6": "multimodal_frontier",
        "series_7": "ai_coding_tools",
        "series_8": "ai_data_engineering",
        "series_9": "ai_applications",
        "series_10": "ai_infrastructure",
        # ML系列（机器学习与深度学习）
        "ml_series_1": "ml_foundation",
        "ml_series_2": "deep_learning_foundation",
        "ml_series_3": "computer_vision",
        "ml_series_4": "natural_language_processing",
        "ml_series_5": "reinforcement_learning",
        "ml_series_6": "recommendation_systems",
        "ml_series_7": "model_optimization",
        "ml_series_8": "traditional_ml",
        "ml_series_9": "feature_engineering",
        "ml_series_10": "advanced_ml_topics",
    }

    @classmethod
    def get_series_category(cls, series_id: str) -> str:
        """获取系列分类：LLM 或 ML"""
        if series_id.startswith("ml_series_"):
            return "ML_series"
        return "LLM_series"

    @classmethod
    def get_series_directory_name(cls, series_id: str) -> str:
        """获取系列目录名称"""
        base_name = cls.SERIES_NAME_MAP.get(series_id, series_id)
        return f"{series_id}_{base_name}"

    @classmethod
    def get_episode_directory_name(cls, episode_number: int) -> str:
        """获取单集目录名称"""
        return f"episode_{episode_number:03d}"

    @classmethod
    def get_full_series_path(cls, base_dir: str = "data", series_id: str = None) -> Path:
        """
        获取系列完整路径

        路径格式：data/series/{category}/{series_directory}/
        例如：data/series/LLM_series/series_1_llm_foundation/
        """
        if series_id:
            category = cls.get_series_category(series_id)
            series_dir = cls.get_series_directory_name(series_id)
            return Path(base_dir) / "series" / category / series_dir
        return Path(base_dir) / "series"

    @classmethod
    def get_full_episode_path(
        cls,
        base_dir: str,
        series_id: str,
        episode_number: int
    ) -> Path:
        """
        获取单集完整路径

        路径格式：data/series/{category}/{series_directory}/episode_{XXX}/
        例如：data/series/LLM_series/series_1_llm_foundation/episode_001/
        """
        series_path = cls.get_full_series_path(base_dir, series_id)
        episode_dir = cls.get_episode_directory_name(episode_number)
        return series_path / episode_dir


class TopicFormatter:
    """话题格式化工具"""

    @staticmethod
    def format_topic_slug(title: str) -> str:
        """将标题转换为URL友好的slug"""
        # 移除特殊字符，替换为下划线
        slug = title.lower()
        # 替换空格和特殊字符
        for char in [' ', ':', '、', '（', '）', '：', '·', '？', '！', '，']:
            slug = slug.replace(char, '_')
        # 移除多余的下划线
        while '__' in slug:
            slug = slug.replace('__', '_')
        # 移除首尾下划线
        slug = slug.strip('_')
        return slug

    @staticmethod
    def generate_filename_prefix(topic: dict) -> str:
        """生成文件名前缀"""
        episode = topic.get("episode", 0)
        title = topic.get("title", "")
        slug = TopicFormatter.format_topic_slug(title)
        return f"ep{episode:03d}_{slug}"

    @staticmethod
    def generate_markdown_filename(topic: dict, content_type: str = "article") -> str:
        """生成Markdown文件名"""
        prefix = TopicFormatter.generate_filename_prefix(topic)
        suffix_map = {
            "article": "article",
            "digest": "digest",
            "note": "note",
            "twitter": "twitter"
        }
        suffix = suffix_map.get(content_type, "content")
        return f"{prefix}_{suffix}.md"

    @staticmethod
    def format_topic_summary(topic: dict) -> str:
        """格式化话题摘要（用于日志输出）"""
        episode = topic.get("episode", 0)
        title = topic.get("title", "")
        status = topic.get("status", "pending")
        series_id = topic.get("series_id", "")

        status_emoji = {
            "pending": "⏳",
            "generating": "🔄",
            "completed": "✅",
            "failed": "❌"
        }.get(status, "📝")

        return f"{status_emoji} Episode {episode:03d} | {title} [{series_id}]"


# 便捷函数
def get_series_metadata(metadata_path: str = "config/blog_topics_100_complete.json") -> SeriesMetadata:
    """获取系列元数据管理器"""
    return SeriesMetadata(metadata_path)


def print_progress_summary(metadata_path: str = "config/blog_topics_100_complete.json") -> None:
    """打印进度摘要"""
    manager = SeriesMetadata(metadata_path)

    print("\n" + "=" * 60)
    print("📊 100期技术博客生成进度")
    print("=" * 60)

    # 总体进度
    summary = manager.get_progress_summary()
    print(f"\n总体进度：{summary['completed']}/{summary['total']} ({summary['completion_rate']})")
    print(f"  待生成：{summary['pending']} | 生成中：{summary['generating']} | 已完成：{summary['completed']}")

    # 系列进度
    print("\n各系列进度：")
    for series_summary in manager.get_series_summary():
        print(f"  {series_summary['series_name']}: {series_summary['completed_episodes']}/{series_summary['total_episodes']} ({series_summary['completion_rate']})")

    print("\n" + "=" * 60 + "\n")


# 导出
__all__ = [
    "SeriesMetadata",
    "SeriesPathManager",
    "TopicFormatter",
    "get_series_metadata",
    "print_progress_summary",
]
