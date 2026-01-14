"""
内容精炼协调器

基于已有高质量文本，精炼出可直接复制粘贴的多平台内容
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import re

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.storage_v2 import StorageFactory
from src.state import create_initial_state, update_state
from src.agents.base import BaseAgent


logger = logging.getLogger(__name__)


class RefineOrchestrator:
    """内容精炼协调器"""

    def __init__(
        self,
        config: Optional[Dict] = None,
        prompts: Optional[Dict] = None
    ):
        """
        初始化精炼内容协调器

        Args:
            config: 全局配置
            prompts: 提示词配置
        """
        self.config = config or {}
        self.prompts = prompts or {}

        # 初始化agents
        self.agents = self._init_agents()

        logger.info("RefineOrchestrator initialized")

    def _init_agents(self) -> Dict[str, BaseAgent]:
        """初始化所有Agent"""
        from src.agents.xiaohongshu_refiner import XiaohongshuRefinerAgent
        from src.agents.twitter_generator import TwitterGeneratorAgent
        from src.agents.wechat_generator import WechatGeneratorAgent

        agents = {}

        # 获取agent配置
        agents_config = self.config.get("agents", {})

        # 初始化各个agent
        agent_classes = {
            "xiaohongshu_refiner": XiaohongshuRefinerAgent,
            "twitter_generator": TwitterGeneratorAgent,
            "wechat_generator": WechatGeneratorAgent,
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
        input_source: str,
        platforms: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行内容精炼

        Args:
            input_source: 输入文件路径
            platforms: 目标平台列表 [wechat, xiaohongshu, twitter]
            **kwargs: 其他参数

        Returns:
            Dict[str, Any]: 执行结果
        """
        import time

        # 默认平台
        if platforms is None:
            platforms = ["wechat", "xiaohongshu", "twitter"]

        # 1. 读取输入内容
        longform_article = self._load_input_content(input_source)

        # 2. 确定存储名称（使用文件名）
        source_name = Path(input_source).stem
        storage = StorageFactory.create_refine(source_name)

        # 3. 保存原始输入
        storage.save_text("raw", "input.md", longform_article['full_content'])

        # 4. 构建状态
        state = {
            "longform_article": longform_article,
            "target_platforms": platforms
        }

        # 5. 为每个平台精炼内容
        try:
            for platform in platforms:
                state = self._refine_for_platform(state, platform, storage)

            logger.info(f"✅ Content refinement completed for platforms: {', '.join(platforms)}")
        except Exception as e:
            logger.error(f"❌ Content refinement failed: {e}")
            state = update_state(state, {
                "error_message": str(e),
                "current_step": "refine_failed"
            })

        return state

    def _load_input_content(self, input_source: str) -> Dict[str, Any]:
        """
        加载输入内容

        Args:
            input_source: 输入文件路径

        Returns:
            Dict[str, Any]: 长文本文章格式
        """
        input_path = Path(input_source)

        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_source}")

        # 读取文件内容
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()

        logger.info(f"读取输入文件: {input_source} ({len(content)} 字符)")

        # 提取标题（第一个 # 标题或文件名）
        title = input_path.stem.replace('_', ' ').replace('-', ' ').title()
        for line in content.split('\n')[:20]:  # 只检查前20行
            line = line.strip()
            if line.startswith('# '):
                title = line[2:].strip()
                break

        # 构建长文本文章格式
        return {
            "title": title,
            "full_content": content,
            "word_count": len(content),
            "tags": [],
            "source": input_source
        }

    def _refine_for_platform(
        self,
        state: Dict[str, Any],
        platform: str,
        storage
    ) -> Dict[str, Any]:
        """
        为指定平台精炼内容

        Args:
            state: 当前状态
            platform: 目标平台
            storage: 存储实例

        Returns:
            Dict[str, Any]: 更新后的状态
        """
        import time

        def _call_agent_safely(agent_name: str, state: Dict[str, Any]) -> Dict[str, Any]:
            """安全调用agent"""
            try:
                logger.info(f"[{platform} - {agent_name}] 开始执行...")
                result = self.agents[agent_name].execute(state)
                logger.info(f"[{platform} - {agent_name}] 执行完成")
                time.sleep(2)  # 避免API并发限制
                return result
            except Exception as e:
                logger.error(f"[{platform} - {agent_name}] 执行失败: {e}")
                time.sleep(2)
                return state

        # 根据平台选择对应的 agent
        if platform == "wechat":
            if "wechat_generator" in self.agents:
                state = _call_agent_safely("wechat_generator", state)

                # 保存微信文章（HTML 格式）
                if "wechat_article" in state:
                    wechat = state["wechat_article"]
                    storage.save_text("wechat", "article.html", wechat["html_content"])
                    storage.save_text("wechat", "cover_prompts.txt", wechat["cover_image_prompt"])
                    logger.info(f"Saved WeChat article to {storage.get_path('wechat', 'article.html')}")

        elif platform == "xiaohongshu":
            if "xiaohongshu_refiner" in self.agents:
                state = _call_agent_safely("xiaohongshu_refiner", state)

                # 保存小红书笔记
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
                    logger.info(f"Saved Xiaohongshu note to {storage.get_path('xiaohongshu', 'note.md')}")

        elif platform == "twitter":
            if "twitter_generator" in self.agents:
                state = _call_agent_safely("twitter_generator", state)

                # 保存 Twitter Thread
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
                    logger.info(f"Saved Twitter thread to {storage.get_path('twitter', 'thread.md')}")

        return state


def main():
    """命令行入口（用于测试）"""
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="内容精炼器")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--input", required=True, help="输入文件路径")
    parser.add_argument("--platforms", nargs="+", choices=["wechat", "xiaohongshu", "twitter"])

    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 创建协调器
    orchestrator = RefineOrchestrator(config=config)

    # 执行
    result = orchestrator.run(
        input_source=args.input,
        platforms=args.platforms
    )

    print(f"\n✅ 精炼完成！")
    print(f"输出目录: data/refine/{Path(args.input).stem}/")


if __name__ == "__main__":
    main()
