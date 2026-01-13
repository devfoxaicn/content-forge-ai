"""
100期技术博客系列生成协调器

支持100期技术博客的系列化生成，使用新的存储结构
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.storage_v2 import SeriesStorage, StorageFactory
from src.utils.series_manager import (
    SeriesMetadata,
    SeriesPathManager,
    TopicFormatter,
    get_series_metadata,
    print_progress_summary
)
from src.state import create_initial_state, update_state
from src.agents.base import BaseAgent


logger = logging.getLogger(__name__)


class SeriesOrchestrator:
    """100期技术博客系列生成协调器"""

    def __init__(
        self,
        config_path: str = "config/blog_topics_100_complete.json",
        config: Optional[Dict] = None,
        prompts: Optional[Dict] = None
    ):
        """
        初始化系列协调器

        Args:
            config_path: 100期配置文件路径
            config: 全局配置
            prompts: 提示词配置
        """
        self.config_path = config_path
        self.series_metadata = get_series_metadata(config_path)
        self.config = config or {}
        self.prompts = prompts or {}

        # 初始化agents
        self.agents = self._init_agents()

        logger.info(f"SeriesOrchestrator initialized with {config_path}")

    def _init_agents(self) -> Dict[str, BaseAgent]:
        """初始化所有Agent"""
        from src.agents.longform_generator import LongFormGeneratorAgent
        from src.agents.code_review_agent import CodeReviewAgent
        from src.agents.fact_check_agent import FactCheckAgent
        from src.agents.xiaohongshu_refiner import XiaohongshuRefinerAgent
        from src.agents.twitter_generator import TwitterGeneratorAgent
        from src.agents.title_optimizer import TitleOptimizerAgent
        from src.agents.image_generator import ImageGeneratorAgent
        from src.agents.quality_evaluator import QualityEvaluatorAgent

        agents = {}

        # 获取agent配置
        agents_config = self.config.get("agents", {})

        # 初始化各个agent
        agent_classes = {
            "longform_generator": LongFormGeneratorAgent,
            "code_review_agent": CodeReviewAgent,
            "fact_check_agent": FactCheckAgent,
            "xiaohongshu_refiner": XiaohongshuRefinerAgent,
            "twitter_generator": TwitterGeneratorAgent,
            "title_optimizer": TitleOptimizerAgent,
            "image_generator": ImageGeneratorAgent,
            "quality_evaluator": QualityEvaluatorAgent,
        }

        for agent_name, agent_class in agent_classes.items():
            agent_cfg = agents_config.get(agent_name, {})
            if agent_cfg.get("enabled", True):
                try:
                    agents[agent_name] = agent_class(
                        config=agent_cfg,
                        prompts=self.prompts
                    )
                    logger.info(f"Initialized agent: {agent_name}")
                except Exception as e:
                    logger.error(f"Failed to initialize {agent_name}: {e}")

        return agents

    def _generate_series_metadata(self, series_id: str) -> dict:
        """
        生成系列专用的元数据（只包含该系列的信息）

        Args:
            series_id: 系列 ID（如 series_1）

        Returns:
            系列元数据字典
        """
        from datetime import datetime

        # 获取系列信息
        series_info = self.series_metadata.get_series_by_id(series_id)
        if not series_info:
            logger.warning(f"Series {series_id} not found, using default info")
            series_info = {
                "id": series_id,
                "name": series_id,
                "description": "",
                "topic_count": 0,
                "difficulty": "未知",
                "priority": 0
            }

        # 获取该系列的所有话题
        topics = self.series_metadata.get_topics_by_series(series_id)

        # 统计完成情况
        completed_episodes = sum(1 for t in topics if t.get("status") == "completed")
        total_estimated_words = sum(t.get("estimated_words", 0) for t in topics)

        # 构建系列元数据
        metadata = {
            "series_info": {
                **series_info,
                "status": "completed" if completed_episodes == len(topics) else "in_progress"
            },
            "topics": topics,
            "statistics": {
                "total_episodes": len(topics),
                "completed_episodes": completed_episodes,
                "total_estimated_words": total_estimated_words,
                "completion_rate": f"{completed_episodes / len(topics) * 100:.1f}" if len(topics) > 0 else "0%",
                "start_date": min((t.get("completed_at") for t in topics if t.get("completed_at")), default=None),
                "end_date": max((t.get("completed_at") for t in topics if t.get("completed_at")), default=None)
            }
        }

        return metadata

    def generate_episode(
        self,
        episode_number: int,
        storage: Optional[SeriesStorage] = None
    ) -> Dict[str, Any]:
        """
        生成单集内容

        Args:
            episode_number: 集数编号（1-100）
            storage: 存储实例（可选）

        Returns:
            生成结果状态字典
        """
        # 获取话题信息
        topic = self.series_metadata.get_topic_by_episode(episode_number)
        if not topic:
            raise ValueError(f"Episode {episode_number} not found in metadata")

        series_id = topic["series_id"]

        # 使用 SeriesPathManager 转换 series_id 为目录名
        series_dir_name = SeriesPathManager.get_series_directory_name(series_id)

        # 创建存储实例
        if storage is None:
            storage = StorageFactory.create_series(
                series_id=series_dir_name,  # 使用转换后的目录名
                episode_number=episode_number
            )

        logger.info(f"Generating Episode {episode_number}: {topic['title']}")

        # 保存元数据
        storage.save_episode_metadata(topic)
        # 保存系列专用的元数据（只包含该系列的信息）
        series_metadata = self._generate_series_metadata(series_id)
        storage.save_series_metadata(series_metadata)

        # 创建初始状态
        state = create_initial_state(
            topic=topic["title"],
            target_audience="技术从业者",
            content_type="技术干货"
        )

        # 添加话题信息到状态
        # 同时设置 current_topic 和 selected_ai_topic（后者是LongFormGeneratorAgent期望的字段）
        state = update_state(state, {
            "current_topic": topic,
            "selected_ai_topic": {  # 兼容LongFormGeneratorAgent
                "title": topic["title"],
                "description": topic.get("description", ""),
                "source": f"series_{series_id}_episode_{episode_number}",
                "url": "",
                "tags": topic.get("keywords", []),
                "key_points": [topic.get("description", "")]
            },
            "series_id": series_id,
            "episode_number": episode_number
        })

        # 执行工作流
        try:
            state = self._execute_workflow(state, storage)

            # 更新状态为已完成
            self.series_metadata.update_topic_status(
                topic["id"],
                "completed"
            )

            logger.info(f"✅ Episode {episode_number} completed successfully")

        except Exception as e:
            logger.error(f"❌ Episode {episode_number} failed: {e}")
            self.series_metadata.update_topic_status(
                topic["id"],
                "failed"
            )
            state = update_state(state, {
                "error_message": str(e),
                "current_step": "episode_failed"
            })

        return state

    def _execute_workflow(
        self,
        state: Dict[str, Any],
        storage: SeriesStorage
    ) -> Dict[str, Any]:
        """执行内容生成工作流"""
        import time

        # 在每个agent调用之间添加延迟，避免API并发限制
        def _call_agent_safely(agent_name: str, state: Dict[str, Any]) -> Dict[str, Any]:
            """安全调用agent，处理异常"""
            try:
                logger.info(f"[{agent_name}] 开始执行...")
                result = self.agents[agent_name].execute(state)
                logger.info(f"[{agent_name}] 执行完成")
                # 添加延迟避免API并发
                time.sleep(2)
                return result
            except Exception as e:
                logger.error(f"[{agent_name}] 执行失败: {e}")
                # 失败时也添加延迟
                time.sleep(2)
                return state

        # 1. 长文本生成（最重要，必须成功）
        if "longform_generator" in self.agents:
            state = _call_agent_safely("longform_generator", state)
            # 保存长文本（字段名：longform_article）
            if "longform_article" in state:
                topic = state["current_topic"]
                article = state["longform_article"]
                # 构建Markdown内容
                md_content = f"""# {article['title']}

{article.get('full_content', '')}

---
**元数据**:
- 字数: {article.get('word_count', 0)}
- 阅读时间: {article.get('reading_time', 'N/A')}
- 标签: {', '.join(article.get('tags', []))}
- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
                filename = TopicFormatter.generate_markdown_filename(topic, "article")
                storage.save_markdown("longform", filename, md_content)
                logger.info("Saved longform article")

        # 2. 代码审查
        if "code_review_agent" in self.agents:
            state = _call_agent_safely("code_review_agent", state)

        # 3. 事实核查
        if "fact_check_agent" in self.agents:
            state = _call_agent_safely("fact_check_agent", state)

        # 4. 小红书精炼
        if "xiaohongshu_refiner" in self.agents:
            state = _call_agent_safely("xiaohongshu_refiner", state)
            # 保存小红书笔记（字段名：xiaohongshu_note）
            if "xiaohongshu_note" in state:
                topic = state["current_topic"]
                note = state["xiaohongshu_note"]
                # 构建Markdown内容
                md_content = f"""# {note['title']}

{note.get('intro', '')}

{note.get('body', '')}

{note.get('ending', '')}

---
**标签**: {' '.join(note.get('hashtags', []))}
**字数**: {note.get('word_count', 0)}
**压缩率**: {note.get('compression_ratio', 'N/A')}
"""
                filename = TopicFormatter.generate_markdown_filename(topic, "note")
                storage.save_markdown("xiaohongshu", filename, md_content)
                logger.info("Saved Xiaohongshu note")

        # 5. Twitter生成
        if "twitter_generator" in self.agents:
            state = _call_agent_safely("twitter_generator", state)
            # 保存Twitter帖子（字段名：twitter_post）
            if "twitter_post" in state:
                topic = state["current_topic"]
                twitter = state["twitter_post"]
                # 构建Markdown内容
                tweets_md = "\n\n".join([
                    f"### Tweet {i+1}\n\n{tweet}"
                    for i, tweet in enumerate(twitter.get('tweets', []))
                ])
                md_content = f"""# Twitter Thread

**原文章**: {twitter.get('original_article_title', 'N/A')}
**推文数量**: {twitter.get('tweet_count', 0)}
**总字符数**: {twitter.get('total_characters', 0)}
**风格**: {twitter.get('style', 'N/A')}

---

{tweets_md}

---
**话题标签**: {' '.join(twitter.get('hashtags', []))}
**是否Thread**: {'是' if twitter.get('is_thread') else '否'}
"""
                filename = TopicFormatter.generate_markdown_filename(topic, "twitter")
                storage.save_markdown("twitter", filename, md_content)
                logger.info("Saved Twitter thread")

        # 6. 标题优化
        if "title_optimizer" in self.agents:
            state = _call_agent_safely("title_optimizer", state)

        # 7. 配图生成
        if "image_generator" in self.agents:
            state = _call_agent_safely("image_generator", state)
            # 保存配图提示词（image_prompts是列表，需要转换为字符串）
            if "image_prompts" in state and state["image_prompts"]:
                topic = state["current_topic"]
                prefix = TopicFormatter.generate_filename_prefix(topic)
                # 将列表转换为格式化的字符串
                prompts_text = "\n\n".join([
                    f"图片 {i+1}:\n{prompt}"
                    for i, prompt in enumerate(state["image_prompts"])
                ])
                storage.save_text("xiaohongshu", f"prompts_{prefix}.txt", prompts_text)
                logger.info("Saved image prompts")

        # 8. 质量评估
        if "quality_evaluator" in self.agents:
            state = _call_agent_safely("quality_evaluator", state)

        return state

    def generate_series(
        self,
        series_id: str,
        start_episode: int = 1,
        end_episode: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        生成整个系列

        Args:
            series_id: 系列ID
            start_episode: 起始集数
            end_episode: 结束集数（可选，默认为系列末尾）

        Returns:
            所有集的生成结果列表
        """
        series = self.series_metadata.get_series_by_id(series_id)
        if not series:
            raise ValueError(f"Series {series_id} not found")

        if end_episode is None:
            # 获取该系列的最后一集
            topics = self.series_metadata.get_topics_by_series(series_id)
            if topics:
                end_episode = max(t["episode"] for t in topics)
            else:
                end_episode = start_episode

        logger.info(f"Generating series {series_id}: episodes {start_episode}-{end_episode}")

        results = []
        for ep in range(start_episode, end_episode + 1):
            try:
                result = self.generate_episode(ep)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to generate episode {ep}: {e}")
                results.append({"error": str(e), "episode": ep})

        return results

    def generate_all(
        self,
        start_episode: int = 1,
        end_episode: int = 100,
        continue_on_error: bool = True
    ) -> Dict[str, Any]:
        """
        生成全部100期内容

        Args:
            start_episode: 起始集数
            end_episode: 结束集数
            continue_on_error: 出错是否继续

        Returns:
            生成统计信息
        """
        logger.info(f"Generating all episodes: {start_episode}-{end_episode}")

        stats = {
            "total": end_episode - start_episode + 1,
            "success": 0,
            "failed": 0,
            "skipped": 0
        }

        for ep in range(start_episode, end_episode + 1):
            topic = self.series_metadata.get_topic_by_episode(ep)

            # 跳过已完成的话题
            if topic and topic.get("status") == "completed":
                logger.info(f"⏭️  Episode {ep} already completed, skipping")
                stats["skipped"] += 1
                continue

            try:
                self.generate_episode(ep)
                stats["success"] += 1
            except Exception as e:
                logger.error(f"❌ Failed to generate episode {ep}: {e}")
                stats["failed"] += 1
                if not continue_on_error:
                    break

        # 打印最终进度
        print_progress_summary(self.config_path)

        return stats


def main():
    """命令行入口"""
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="100期技术博客系列生成器")
    parser.add_argument("--config", default="config/config.yaml", help="全局配置文件")
    parser.add_argument("--series-config", default="config/blog_topics_100_complete.json", help="100期配置文件")
    parser.add_argument("--episode", type=int, help="生成指定集数")
    parser.add_argument("--series", help="生成指定系列")
    parser.add_argument("--start", type=int, default=1, help="起始集数")
    parser.add_argument("--end", type=int, default=100, help="结束集数")
    parser.add_argument("--all", action="store_true", help="生成全部100期")
    parser.add_argument("--progress", action="store_true", help="仅显示进度")

    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 创建协调器
    orchestrator = SeriesOrchestrator(
        config_path=args.series_config,
        config=config
    )

    # 仅显示进度
    if args.progress:
        print_progress_summary(args.series_config)
        return

    # 生成指定集数
    if args.episode:
        orchestrator.generate_episode(args.episode)
        return

    # 生成指定系列
    if args.series:
        orchestrator.generate_series(args.series)
        return

    # 生成全部
    if args.all:
        orchestrator.generate_all(
            start_episode=args.start,
            end_episode=args.end
        )
        return

    # 默认：显示帮助
    parser.print_help()


if __name__ == "__main__":
    main()
