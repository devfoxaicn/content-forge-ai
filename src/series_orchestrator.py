"""
100期技术博客系列生成协调器

支持100期技术博客的系列化生成，使用新的存储结构
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.storage_v2 import SeriesStorage, StorageFactory
from src.utils.series_manager import (
    SeriesMetadata,
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

        # 创建存储实例
        if storage is None:
            storage = StorageFactory.create_series(
                series_id=series_id,
                episode_number=episode_number
            )

        logger.info(f"Generating Episode {episode_number}: {topic['title']}")

        # 保存元数据
        storage.save_episode_metadata(topic)
        storage.save_series_metadata(self.series_metadata.get_batch_info())

        # 创建初始状态
        state = create_initial_state(
            topic=topic["title"],
            target_audience="技术从业者",
            content_type="技术干货"
        )

        # 添加话题信息到状态
        state = update_state(state, {
            "current_topic": topic,
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

        # 1. 长文本生成
        if "longform_generator" in self.agents:
            state = self.agents["longform_generator"].execute(state)
            # 保存长文本
            if "longform_content" in state:
                topic = state["current_topic"]
                filename = TopicFormatter.generate_markdown_filename(topic, "article")
                storage.save_markdown("longform", filename, state["longform_content"])
                logger.info("Saved longform article")

        # 2. 代码审查
        if "code_review_agent" in self.agents:
            state = self.agents["code_review_agent"].execute(state)

        # 3. 事实核查
        if "fact_check_agent" in self.agents:
            state = self.agents["fact_check_agent"].execute(state)

        # 4. 小红书精炼
        if "xiaohongshu_refiner" in self.agents:
            state = self.agents["xiaohongshu_refiner"].execute(state)
            if "xiaohongshu_content" in state:
                topic = state["current_topic"]
                filename = TopicFormatter.generate_markdown_filename(topic, "note")
                storage.save_markdown("xiaohongshu", filename, state["xiaohongshu_content"])
                logger.info("Saved Xiaohongshu note")

        # 5. Twitter生成
        if "twitter_generator" in self.agents:
            state = self.agents["twitter_generator"].execute(state)
            if "twitter_content" in state:
                topic = state["current_topic"]
                filename = TopicFormatter.generate_markdown_filename(topic, "twitter")
                storage.save_markdown("twitter", filename, state["twitter_content"])
                logger.info("Saved Twitter thread")

        # 6. 标题优化
        if "title_optimizer" in self.agents:
            state = self.agents["title_optimizer"].execute(state)

        # 7. 配图生成
        if "image_generator" in self.agents:
            state = self.agents["image_generator"].execute(state)
            if "image_prompts" in state:
                topic = state["current_topic"]
                prefix = TopicFormatter.generate_filename_prefix(topic)
                storage.save_text("xiaohongshu", f"prompts_{prefix}.txt", state["image_prompts"])
                logger.info("Saved image prompts")

        # 8. 质量评估
        if "quality_evaluator" in self.agents:
            state = self.agents["quality_evaluator"].execute(state)

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
