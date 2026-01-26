"""
100期技术博客系列生成协调器

只生成长文本技术博客，不生成社交内容
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
        """初始化Agent（优化版Series模式：只保留 Research + LongForm）"""
        from src.agents.longform_generator import LongFormGeneratorAgent
        from src.agents.research_agent import ResearchAgent

        agents = {}

        # 获取agent配置
        agents_config = self.config.get("agents", {})

        # 确保配置中包含 mode: "series"
        full_config = {**self.config, "mode": "series"}

        # 初始化研究Agent（优先于长文本生成）
        if agents_config.get("research_agent", {}).get("enabled", True):
            try:
                agents["research_agent"] = ResearchAgent(
                    config={**agents_config.get("research_agent", {}), "mode": "series"},
                    prompts=self.prompts
                )
                logger.info("Initialized agent: research_agent")
            except Exception as e:
                logger.warning(f"Failed to initialize research_agent: {e}")

        # 初始化长文本生成Agent（必需）
        if agents_config.get("longform_generator", {}).get("enabled", True):
            try:
                # 传递完整配置（包含mode: "series"）
                longform_config = {**agents_config.get("longform_generator", {}), "mode": "series"}
                agents["longform_generator"] = LongFormGeneratorAgent(
                    config=full_config,  # 传递完整配置，包含 mode: "series"
                    prompts=self.prompts
                )
                logger.info("Initialized agent: longform_generator")
            except Exception as e:
                logger.error(f"Failed to initialize longform_generator: {e}")
                raise

        return agents

    def _generate_series_metadata(self, series_id: str) -> dict:
        """
        生成系列专用的元数据（只包含该系列的信息）

        目标格式与 data/series/LLM_series/{series_directory}/series_metadata.json 一致

        Args:
            series_id: 系列 ID（如 series_1）

        Returns:
            系列元数据字典
        """
        from datetime import datetime

        # 获取系列完整目录名（如 series_1_llm_foundation）
        series_dir_name = SeriesPathManager.get_series_directory_name(series_id)

        # 获取系列信息
        series_info = self.series_metadata.get_series_by_id(series_id)
        if not series_info:
            logger.warning(f"Series {series_id} not found, using default info")
            series_info = {
                "id": series_dir_name,  # 使用完整目录名
                "name": series_id,
                "description": "",
                "topic_count": 0,
                "difficulty": "未知",
                "priority": 0
            }
        else:
            # 覆盖 id 为完整目录名（与目标格式一致）
            series_info = {**series_info, "id": series_dir_name}

        # 获取该系列的所有话题
        topics = self.series_metadata.get_topics_by_series(series_id)

        # 统计完成情况
        completed_episodes = sum(1 for t in topics if t.get("status") == "completed")
        total_estimated_words = sum(t.get("estimated_words", 0) for t in topics)

        # 获取日期范围
        completed_dates = [t.get("completed_at") for t in topics if t.get("completed_at")]
        start_date = min(completed_dates) if completed_dates else None
        end_date = max(completed_dates) if completed_dates else None

        # 构建系列元数据（与目标格式一致）
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
                "completion_rate": round(completed_episodes / len(topics) * 100, 1) if len(topics) > 0 else 0,
                "start_date": start_date,
                "end_date": end_date
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

        # 创建存储实例（直接使用series_id，SeriesStorage内部会处理目录名转换）
        if storage is None:
            storage = StorageFactory.create_series(
                series_id=series_id,
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
        """执行内容生成工作流（优化版：只有 Research + LongForm）"""
        import time

        def _call_agent_safely(agent_name: str, state: Dict[str, Any]) -> Dict[str, Any]:
            """安全调用agent，处理异常"""
            try:
                logger.info(f"[{agent_name}] 开始执行...")
                result = self.agents[agent_name].execute(state)
                logger.info(f"[{agent_name}] 执行完成")
                time.sleep(2)
                return result
            except Exception as e:
                logger.error(f"[{agent_name}] 执行失败: {e}")
                time.sleep(2)
                return state

        # 第一步：网络搜索研究（如果启用了research_agent）
        if "research_agent" in self.agents:
            state = _call_agent_safely("research_agent", state)
            sources_count = len(state.get('research_data', {}).get('sources', []))
            logger.info(f"✅ 研究完成，获取到 {sources_count} 个资料来源")

        # 第二步：长文本生成（必需）
        if "longform_generator" in self.agents:
            state = _call_agent_safely("longform_generator", state)

        # 保存文章到episode目录（文件名格式：epXXX_标题_article.md）
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

            # 使用标题生成文件名（格式：epXXX_标题_article.md）
            saved_path = storage.save_article(md_content, title=article['title'])
            logger.info(f"✅ 文章已保存: {saved_path.name}")

        return state

    def _format_quality_report(self, quality_report: Dict[str, Any]) -> str:
        """格式化质量报告为Markdown"""
        overall_score = quality_report.get("overall_score", 0)
        dimension_scores = quality_report.get("dimension_scores", {})
        improvements = quality_report.get("improvements", [])
        metadata = quality_report.get("metadata", {})

        md_lines = [
            "# 质量评估报告",
            "",
            f"## 总体评分",
            f"**{overall_score:.1f}/10**",
            "",
            f"状态: {'✅ 达标' if quality_report.get('meets_threshold') else '⚠️ 需改进'}",
            ""
        ]

        # 分维度评分
        md_lines.extend([
            "## 分维度评分",
            ""
        ])

        dimension_names = {
            "structure": "结构",
            "depth": "深度",
            "accuracy": "准确性",
            "readability": "可读性",
            "visual": "可视化",
            "timeliness": "时效性"
        }

        for dim, data in dimension_scores.items():
            score = data.get("score", 0)
            weight = data.get("weight", 0)
            feedback_list = data.get("feedback", [])
            md_lines.extend([
                f"### {dimension_names.get(dim, dim)} ({weight*100:.0f}%)",
                f"**评分**: {score:.1f}/10",
                ""
            ])
            if feedback_list:
                md_lines.append("反馈:")
                for fb in feedback_list:
                    md_lines.append(f"- {fb}")
                md_lines.append("")

        # 改进建议
        if improvements:
            md_lines.extend([
                "## 改进建议",
                ""
            ])
            for imp in improvements[:10]:
                md_lines.append(f"- {imp}")
            md_lines.append("")

        # 元数据
        md_lines.extend([
            "## 元数据",
            "",
            f"- 评估时间: {metadata.get('evaluated_at', 'N/A')}",
            f"- 文章长度: {metadata.get('article_length', 0)} 字符",
            f"- 代码块: {metadata.get('code_blocks', 0)} 个",
            f"- 事实声明: {metadata.get('fact_claims', 0)} 个"
        ])

        return "\n".join(md_lines)

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
