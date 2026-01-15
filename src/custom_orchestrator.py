"""
自定义内容生成协调器

根据用户指定的主题/关键词，生成长文本技术文章
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.storage_v2 import StorageFactory
from src.state import create_initial_state, update_state
from src.agents.base import BaseAgent


logger = logging.getLogger(__name__)


class CustomContentOrchestrator:
    """自定义内容生成协调器"""

    def __init__(
        self,
        config: Optional[Dict] = None,
        prompts: Optional[Dict] = None
    ):
        """
        初始化自定义内容协调器

        Args:
            config: 全局配置
            prompts: 提示词配置
        """
        self.config = config or {}
        self.prompts = prompts or {}

        # 初始化agents
        self.agents = self._init_agents()

        logger.info("CustomContentOrchestrator initialized")

    def _init_agents(self) -> Dict[str, BaseAgent]:
        """初始化所有Agent"""
        from src.agents.research_agent import ResearchAgent
        from src.agents.longform_generator import LongFormGeneratorAgent
        from src.agents.xiaohongshu_refiner import XiaohongshuRefinerAgent
        from src.agents.twitter_generator import TwitterGeneratorAgent
        from src.agents.title_optimizer import TitleOptimizerAgent
        from src.agents.image_generator import ImageGeneratorAgent

        agents = {}

        # 获取agent配置
        agents_config = self.config.get("agents", {})

        # 初始化各个agent
        agent_classes = {
            "research_agent": ResearchAgent,
            "longform_generator": LongFormGeneratorAgent,
            "xiaohongshu_refiner": XiaohongshuRefinerAgent,
            "twitter_generator": TwitterGeneratorAgent,
            "title_optimizer": TitleOptimizerAgent,
            "image_generator": ImageGeneratorAgent,
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

    def run(
        self,
        topic: str,
        prompt: Optional[str] = None,
        target_audience: str = "技术从业者",
        content_type: str = "技术干货",
        words: Optional[int] = None,
        style: str = "technical",
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行自定义内容生成

        Args:
            topic: 主题/关键词
            prompt: 详细要求描述
            target_audience: 目标受众
            content_type: 内容类型
            words: 目标字数
            style: 文章风格 (technical/practical/tutorial)
            **kwargs: 其他参数

        Returns:
            Dict[str, Any]: 执行结果
        """
        import time

        # 1. 构建初始状态
        state = self._build_initial_state(
            topic=topic,
            prompt=prompt,
            target_audience=target_audience,
            content_type=content_type,
            words=words,
            style=style
        )

        # 2. 创建存储实例
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        topic_slug = topic.lower().replace(" ", "_").replace("/", "_")[:50]
        storage_id = f"{timestamp}_{topic_slug}"
        storage = StorageFactory.create_custom(storage_id)

        # 3. 执行工作流
        try:
            state = self._execute_workflow(state, storage)
            logger.info(f"✅ Custom content generation completed")
        except Exception as e:
            logger.error(f"❌ Custom content generation failed: {e}")
            state = update_state(state, {
                "error_message": str(e),
                "current_step": "custom_generation_failed"
            })

        return state

    def _build_initial_state(
        self,
        topic: str,
        prompt: Optional[str],
        target_audience: str,
        content_type: str,
        words: Optional[int],
        style: str
    ) -> Dict[str, Any]:
        """构建初始工作流状态"""
        state = create_initial_state(
            topic=topic,
            target_audience=target_audience,
            content_type=content_type
        )

        # 设置选中的AI主题
        selected_ai_topic = {
            "title": topic,
            "description": prompt or f"关于{topic}的深度技术分析",
            "source": "custom",
            "url": "",
            "tags": [topic],
            "key_points": []
        }

        if prompt:
            selected_ai_topic["description"] = prompt
            selected_ai_topic["key_points"] = [prompt]

        # 添加自定义参数
        updates = {
            "selected_ai_topic": selected_ai_topic,
            "custom_prompt": prompt,
            "target_words": words,
            "article_style": style
        }

        return update_state(state, updates)

    def _execute_workflow(
        self,
        state: Dict[str, Any],
        storage
    ) -> Dict[str, Any]:
        """执行内容生成工作流"""
        import time

        def _call_agent_safely(agent_name: str, state: Dict[str, Any]) -> Dict[str, Any]:
            """安全调用agent"""
            try:
                logger.info(f"[{agent_name}] 开始执行...")
                result = self.agents[agent_name].execute(state)
                logger.info(f"[{agent_name}] 执行完成")
                time.sleep(2)  # 避免API并发限制
                return result
            except Exception as e:
                logger.error(f"[{agent_name}] 执行失败: {e}")
                time.sleep(2)
                return state

        # 0. 深度研究（为长文本生成提供背景资料）
        if "research_agent" in self.agents:
            state = _call_agent_safely("research_agent", state)

        # 1. 长文本生成（核心）
        if "longform_generator" in self.agents:
            state = _call_agent_safely("longform_generator", state)

            # 保存长文本
            if "longform_article" in state:
                article = state["longform_article"]
                md_content = f"""# {article['title']}

{article.get('full_content', '')}

---
**元数据**:
- 字数: {article.get('word_count', 0)}
- 阅读时间: {article.get('reading_time', 'N/A')}
- 标签: {', '.join(article.get('tags', []))}
- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
                storage.save_markdown("longform", "article.md", md_content)
                logger.info("Saved longform article")

        # 2. 小红书精炼
        if "xiaohongshu_refiner" in self.agents:
            state = _call_agent_safely("xiaohongshu_refiner", state)
            if "xiaohongshu_note" in state:
                note = state["xiaohongshu_note"]
                md_content = f"""# {note['title']}

{note.get('intro', '')}

{note.get('body', '')}

{note.get('ending', '')}

---
**标签**: {' '.join(note.get('hashtags', []))}
**字数**: {note.get('word_count', 0)}
**压缩率**: {note.get('compression_ratio', 'N/A')}
"""
                storage.save_markdown("xiaohongshu", "note.md", md_content)
                logger.info("Saved Xiaohongshu note")

        # 3. Twitter生成
        if "twitter_generator" in self.agents:
            state = _call_agent_safely("twitter_generator", state)
            if "twitter_post" in state:
                twitter = state["twitter_post"]
                tweets_md = "\n\n".join([
                    f"### Tweet {i+1}\n\n{tweet}"
                    for i, tweet in enumerate(twitter.get('tweets', []))
                ])
                md_content = f"""# Twitter Thread

**原文章**: {twitter.get('original_article_title', 'N/A')}
**推文数量**: {twitter.get('tweet_count', 0)}
**总字符数**: {twitter.get('total_characters', 0)}

---

{tweets_md}

---
**话题标签**: {' '.join(twitter.get('hashtags', []))}
"""
                storage.save_markdown("twitter", "thread.md", md_content)
                logger.info("Saved Twitter thread")

        # 4. 标题优化
        if "title_optimizer" in self.agents:
            state = _call_agent_safely("title_optimizer", state)

        # 5. 配图生成
        if "image_generator" in self.agents:
            state = _call_agent_safely("image_generator", state)
            if "image_prompts" in state and state["image_prompts"]:
                prompts_text = "\n\n".join([
                    f"图片 {i+1}:\n{prompt}"
                    for i, prompt in enumerate(state["image_prompts"])
                ])
                storage.save_text("xiaohongshu", "prompts.txt", prompts_text)
                logger.info("Saved image prompts")

        return state


def main():
    """命令行入口（用于测试）"""
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="自定义内容生成器")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--topic", required=True, help="主题/关键词")
    parser.add_argument("--prompt", help="详细要求描述")
    parser.add_argument("--audience", default="技术从业者", help="目标受众")
    parser.add_argument("--words", type=int, help="目标字数")
    parser.add_argument("--style", default="technical", help="文章风格 (technical/practical/tutorial)")

    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 创建协调器
    orchestrator = CustomContentOrchestrator(config=config)

    # 执行
    result = orchestrator.run(
        topic=args.topic,
        prompt=args.prompt,
        target_audience=args.audience,
        words=args.words,
        style=args.style
    )

    print(f"\n✅ 生成完成！")
    print(f"输出目录: data/custom/")


if __name__ == "__main__":
    main()
