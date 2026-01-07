"""
自动化内容生成工作流
整合AI热点分析、长文本生成、小红书精炼和发布的完整流程
"""

import os
import sys
import yaml
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# LangGraph imports
from langgraph.graph import StateGraph, END

# 本地imports
from src.state import create_initial_state, update_state, add_agent_to_order, calculate_execution_time
from src.agents.ai_trend_analyzer_real import RealAITrendAnalyzerAgent
from src.agents.longform_generator import LongFormGeneratorAgent
from src.agents.xiaohongshu_refiner import XiaohongshuRefinerAgent
from src.agents.twitter_generator import TwitterGeneratorAgent
from src.agents.title_optimizer import TitleOptimizerAgent
from src.agents.image_advisor import ImageAdvisorAgent
from src.agents.image_generator import ImageGeneratorAgent
from src.agents.quality_evaluator import QualityEvaluatorAgent
from src.agents.publisher import PublisherAgent
from src.agents.trends_digest_agent import TrendsDigestAgent
from src.utils.storage import get_storage

# 日志配置
from loguru import logger


class AutoContentOrchestrator:
    """自动化内容生成协调器 - 新工作流"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化自动化协调器

        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.prompts = self._load_prompts()
        self._setup_logging()

        # 初始化存储管理器
        self.storage = get_storage(self.config.get("storage", {}).get("base_dir", "data"))

        # 初始化Agent
        self.agents = self._init_agents()

        # 构建工作流
        self.workflow = self._build_workflow()

        logger.info("自动化内容生成协调器初始化完成（新工作流）")
        logger.info(f"数据存储目录: {self.storage.get_date_dir()}")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"配置文件加载成功: {config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"配置文件不存在: {config_path}")
            sys.exit(1)
        except yaml.YAMLError as e:
            logger.error(f"配置文件格式错误: {e}")
            sys.exit(1)

    def _load_prompts(self) -> Dict[str, Any]:
        """加载提示词配置"""
        prompts_file = self.config.get("prompts", {}).get("template_file", "config/prompts.yaml")
        try:
            with open(prompts_file, 'r', encoding='utf-8') as f:
                prompts = yaml.safe_load(f)
            logger.info(f"提示词文件加载成功: {prompts_file}")
            return {"prompts": prompts}
        except FileNotFoundError:
            logger.warning(f"提示词文件不存在: {prompts_file}，使用默认配置")
            return {"prompts": {}}
        except yaml.YAMLError as e:
            logger.warning(f"提示词文件格式错误: {e}，使用默认配置")
            return {"prompts": {}}

    def _setup_logging(self):
        """配置日志（按日期分层存储）"""
        log_config = self.config.get("logging", {})

        # 日志级别
        level = log_config.get("level", "INFO")
        logger.remove()
        logger.add(sys.stderr, level=level)

        # 文件日志 - 使用日期目录
        if log_config.get("file", {}).get("enabled", True):
            from datetime import datetime
            date_str = datetime.now().strftime("%Y%m%d")
            log_dir = f"logs/{date_str}"
            os.makedirs(log_dir, exist_ok=True)

            log_file = os.path.join(log_dir, "app.log")
            logger.add(
                log_file,
                rotation=log_config.get("file", {}).get("rotation", "100 MB"),
                retention=log_config.get("file", {}).get("retention", "30 days"),
                level=level
            )

    def _init_agents(self) -> Dict[str, Any]:
        """初始化所有Agent"""
        agents = {}
        agents_config = self.config.get("agents", {})

        # AI热点分析Agent（真实API版本）
        if agents_config.get("ai_trend_analyzer", {}).get("enabled", True):
            agents["ai_trend_analyzer"] = RealAITrendAnalyzerAgent(self.config, self.prompts)

        # 热点汇总Agent（新增）
        if agents_config.get("trends_digest", {}).get("enabled", True):
            agents["trends_digest"] = TrendsDigestAgent(self.config, self.prompts)

        # 长文本生成Agent
        if agents_config.get("longform_generator", {}).get("enabled", True):
            agents["longform_generator"] = LongFormGeneratorAgent(self.config, self.prompts)

        # 小红书笔记精炼Agent
        if agents_config.get("xiaohongshu_refiner", {}).get("enabled", True):
            agents["xiaohongshu_refiner"] = XiaohongshuRefinerAgent(self.config, self.prompts)

        # Twitter帖子生成Agent（新增）
        if agents_config.get("twitter_generator", {}).get("enabled", True):
            agents["twitter_generator"] = TwitterGeneratorAgent(self.config, self.prompts)

        # 标题优化Agent
        if agents_config.get("title_optimizer", {}).get("enabled", True):
            agents["title_optimizer"] = TitleOptimizerAgent(self.config, self.prompts)

        # 图像建议Agent
        if agents_config.get("image_advisor", {}).get("enabled", True):
            agents["image_advisor"] = ImageAdvisorAgent(self.config, self.prompts)

        # 图片生成Agent
        if agents_config.get("image_generator", {}).get("enabled", True):
            agents["image_generator"] = ImageGeneratorAgent(self.config, self.prompts)

        # 质量评估Agent
        if agents_config.get("quality_evaluator", {}).get("enabled", True):
            agents["quality_evaluator"] = QualityEvaluatorAgent(self.config, self.prompts)

        # 发布Agent
        if agents_config.get("publisher", {}).get("enabled", True):
            agents["publisher"] = PublisherAgent(self.config, self.prompts)

        logger.info(f"已初始化 {len(agents)} 个Agent: {list(agents.keys())}")
        return agents

    def _build_workflow(self) -> StateGraph:
        """构建自动化工作流"""
        workflow = StateGraph(dict)

        # 添加Agent节点
        for agent_name, agent in self.agents.items():
            workflow.add_node(agent_name, self._create_agent_node(agent))

        # 定义执行顺序：
        # AI热点分析 → 热点汇总 → 长文本生成 → 小红书精炼 → Twitter → ...
        # 全部顺序执行，避免并发冲突
        if "ai_trend_analyzer" in self.agents:
            workflow.set_entry_point("ai_trend_analyzer")

            # 热点汇总Agent
            last_node = "ai_trend_analyzer"
            if "trends_digest" in self.agents:
                workflow.add_edge(last_node, "trends_digest")
                last_node = "trends_digest"

            # 长文本生成流程
            if "longform_generator" in self.agents:
                workflow.add_edge(last_node, "longform_generator")
                last_node = "longform_generator"

                # 顺序执行：长文本 -> 小红书 -> Twitter -> 标题优化
                # 避免并发更新state导致的冲突
                has_xiaohongshu = "xiaohongshu_refiner" in self.agents
                has_twitter = "twitter_generator" in self.agents

                if has_xiaohongshu:
                    workflow.add_edge(last_node, "xiaohongshu_refiner")
                    last_node = "xiaohongshu_refiner"

                if has_twitter:
                    workflow.add_edge(last_node, "twitter_generator")
                    last_node = "twitter_generator"

                # 标题优化跟在最后
                if "title_optimizer" in self.agents:
                    workflow.add_edge(last_node, "title_optimizer")
                    last_node = "title_optimizer"

                if "image_advisor" in self.agents:
                    workflow.add_edge(last_node, "image_advisor")
                    last_node = "image_advisor"

                if "image_generator" in self.agents:
                    workflow.add_edge(last_node, "image_generator")
                    last_node = "image_generator"

                if "quality_evaluator" in self.agents:
                    workflow.add_edge(last_node, "quality_evaluator")
                    last_node = "quality_evaluator"

                if "publisher" in self.agents:
                    workflow.add_edge(last_node, "publisher")
                    workflow.add_edge("publisher", END)
                else:
                    workflow.add_edge(last_node, END)
            else:
                workflow.add_edge(last_node, END)

        return workflow.compile()

    def _create_agent_node(self, agent):
        """创建Agent节点函数"""
        def node_function(state):
            logger.info(f"执行Agent: {agent.name}")
            try:
                result = agent.execute(state)
                return add_agent_to_order(result, agent.name)
            except Exception as e:
                logger.error(f"Agent {agent.name} 执行失败: {e}")
                return update_state(state, {
                    "error_message": str(e),
                    "current_step": f"{agent.name}_failed"
                })
        return node_function

    def run(self, topic: str = None, target_audience: str = "技术从业者",
            content_type: str = "干货分享", keywords: list = None) -> Dict[str, Any]:
        """
        运行完整自动化工作流

        Args:
            topic: 内容主题标识（可选，用于文件命名，留空则基于实时热点自动生成）
            target_audience: 目标受众
            content_type: 内容类型
            keywords: 关键词列表

        Returns:
            Dict[str, Any]: 最终输出
        """
        # 如果没有提供topic，使用auto作为标识（实际内容基于实时热点）
        if topic is None:
            topic = "auto"
            logger.info("开始执行自动化内容生产流程（基于实时热点）")
        else:
            logger.info(f"开始执行自动化内容生产流程: {topic}")

        # 创建初始状态
        state = create_initial_state(
            topic=topic,
            target_audience=target_audience,
            content_type=content_type,
            keywords=keywords,
            config=self.config
        )

        # 执行工作流
        try:
            result = self.workflow.invoke(state)
            result = calculate_execution_time(result)

            # 保存输出
            self._save_output(result)

            # 打印结果摘要
            self._print_summary(result)

            logger.success(f"自动化内容生产完成！耗时: {result.get('execution_time', 0):.2f}秒")
            return result

        except Exception as e:
            logger.error(f"工作流执行失败: {e}")
            raise

    def _save_output(self, state: Dict[str, Any]):
        """保存输出结果到按日期分层的目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 确定文件标识符
        topic = state.get("topic", "unknown")
        if topic == "auto":
            # 如果是auto模式，使用实际的热点标题
            selected_topic = state.get("selected_ai_topic", {})
            topic = selected_topic.get("title", "auto")

        # 清理文件名（移除特殊字符，限制长度）
        topic = topic.replace(" ", "_").replace("/", "_").replace("\\", "_")[:30]

        # 1. 保存完整工作流输出（JSON格式）
        filename = f"workflow_{topic}_{timestamp}.json"
        output_data = {
            "workflow": "auto_v2",
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "topic": state.get("topic"),
                "selected_ai_topic": state.get("selected_ai_topic", {}).get("title"),
                "execution_time": state.get("execution_time"),
                "agent_execution_order": state.get("agent_execution_order", [])
            },
            "content": {
                "ai_hot_topics": state.get("ai_hot_topics"),
                "longform_article": {
                    "title": state.get("longform_article", {}).get("title"),
                    "word_count": state.get("longform_article", {}).get("word_count"),
                    "reading_time": state.get("longform_article", {}).get("reading_time")
                },
                "xiaohongshu_note": {
                    "title": state.get("xiaohongshu_note", {}).get("title"),
                    "word_count": state.get("xiaohongshu_note", {}).get("word_count"),
                    "compression_ratio": state.get("xiaohongshu_note", {}).get("compression_ratio")
                },
                "optimized_titles": state.get("optimized_titles"),
                "recommended_title": state.get("recommended_title"),
                "image_suggestions": state.get("image_suggestions"),
                "generated_images": state.get("generated_images", []),
                "image_prompts": state.get("image_prompts", []),
                "quality_report": state.get("quality_report")
            },
            "publish": {
                "published": state.get("published", False),
                "publish_result": state.get("publish_result")
            },
            "status": "success" if not state.get("error_message") else "failed"
        }

        # 不再保存完整工作流JSON文件（精简输出）

        # 2. 保存长文本文章（只保存Markdown格式）
        if state.get("longform_article"):
            article = state["longform_article"]
            md_filename = f"article_{topic}_{timestamp}.md"
            md_content = f"""# {article['title']}

{article.get('full_content', '')}

---
**元数据**:
- 字数: {article.get('word_count', 0)}
- 阅读时间: {article.get('reading_time', 'N/A')}
- 来源热点: {article.get('source_topic', 'N/A')}
- 标签: {', '.join(article.get('tags', []))}
- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            md_file = self.storage.save_markdown("longform", md_filename, md_content)
            logger.info(f"长文本Markdown已保存: {md_file}")

        # 3. 保存小红书笔记（只保存Markdown格式）
        if state.get("xiaohongshu_note"):
            note = state["xiaohongshu_note"]
            note_md_filename = f"note_{topic}_{timestamp}.md"
            note_md_content = f"""# {note['title']}

{note.get('intro', '')}

{note.get('body', '')}

{note.get('ending', '')}

---
**标签**: {' '.join(note.get('hashtags', []))}
**字数**: {note.get('word_count', 0)}
**压缩率**: {note.get('compression_ratio', 'N/A')}
"""
            note_md_file = self.storage.save_markdown("xiaohongshu", note_md_filename, note_md_content)
            logger.info(f"小红书笔记Markdown已保存: {note_md_file}")

        # 4. 保存Twitter帖子（只保存Markdown格式）
        if state.get("twitter_post"):
            twitter_post = state["twitter_post"]
            # 保存Markdown格式（用于阅读）
            twitter_md_filename = f"twitter_{topic}_{timestamp}.md"
            twitter_md_content = f"""# Twitter Thread

**原文章**: {twitter_post.get('original_article_title', 'N/A')}
**推文数量**: {twitter_post.get('tweet_count', 0)}
**总字符数**: {twitter_post.get('total_characters', 0)}
**风格**: {twitter_post.get('style', 'N/A')}

---

{self._format_twitter_thread(twitter_post.get('tweets', []))}

---
**话题标签**: {' '.join(twitter_post.get('hashtags', []))}
**是否Thread**: {'是' if twitter_post.get('is_thread') else '否'}
"""
            twitter_md_file = self.storage.save_markdown("twitter", twitter_md_filename, twitter_md_content)
            logger.info(f"Twitter帖子Markdown已保存: {twitter_md_file}")

            # 保存图片提示词到twitter目录（Twitter配图）
            if state.get("image_prompts"):
                twitter_prompts_filename = f"prompts_{topic}_{timestamp}.txt"
                prompts_content = "\n\n".join([
                    f"Tweet {i+1} 配图:\n{prompt}"
                    for i, prompt in enumerate(state["image_prompts"][:twitter_post.get('tweet_count', 1)])
                ])
                twitter_prompts_file = self.storage.save_text("twitter", twitter_prompts_filename, prompts_content)
                logger.info(f"Twitter配图提示词已保存: {twitter_prompts_file}")

        # 5. 保存图片提示词到对应目录（小红书）
        if state.get("image_prompts"):
            # 提示词通常与小红书笔记关联，保存到xiaohongshu目录
            prompts_filename = f"prompts_{topic}_{timestamp}.txt"
            prompts_content = "\n\n".join([
                f"图片 {i+1}:\n{prompt}"
                for i, prompt in enumerate(state["image_prompts"])
            ])
            prompts_file = self.storage.save_text("xiaohongshu", prompts_filename, prompts_content)
            logger.info(f"图片提示词已保存: {prompts_file}")

        # 5. 保存热点简报（如果有）
        if state.get("trends_digest"):
            self._save_digest(state)

        logger.success(f"所有内容已保存到日期目录: {self.storage.get_date_dir()}")

    def _format_twitter_thread(self, tweets: list) -> str:
        """格式化Twitter thread为Markdown"""
        formatted_tweets = []
        for i, tweet in enumerate(tweets, 1):
            formatted_tweets.append(f"### Tweet {i}\n\n{tweet}\n")
        return "\n".join(formatted_tweets)

    def _save_digest(self, state: Dict[str, Any]):
        """保存热点简报到digest目录"""
        try:
            digest = state.get("trends_digest")
            if not digest:
                return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"digest_{digest.get('issue_number', timestamp)}"

            # 保存Markdown格式（主要格式）
            md_filename = f"{base_filename}.md"
            md_file = self.storage.save_markdown("digest", md_filename, digest.get('full_content', ''))
            logger.info(f"热点简报Markdown已保存: {md_file}")

            # 保存JSON格式（用于网站API）
            json_filename = f"{base_filename}.json"
            digest_data = {
                "metadata": {
                    "title": digest.get("title"),
                    "subtitle": digest.get("subtitle"),
                    "issue_number": digest.get("issue_number"),
                    "publish_date": digest.get("publish_date"),
                    "generated_at": datetime.now().isoformat(),
                    "word_count": digest.get("word_count"),
                    "reading_time": digest.get("reading_time"),
                    "total_topics": digest.get("total_topics"),
                    "style": digest.get("style")
                },
                "topics": digest.get("topics", []),
                "summary_analysis": digest.get("summary_analysis"),
                "sources": digest.get("sources")
            }
            json_file = self.storage.save_json("digest", json_filename, digest_data)
            logger.success(f"热点简报已保存: {md_file} (MD) + {json_file} (JSON)")

        except Exception as e:
            logger.error(f"保存热点简报失败: {e}")

    def _print_summary(self, state: Dict[str, Any]):
        """打印结果摘要"""
        print("\n" + "="*60)
        print("📝 自动化内容生成完成（新工作流）")
        print("="*60)

        print(f"主题领域: {state.get('topic', 'N/A')}")
        print(f"AI热点话题: {state.get('selected_ai_topic', {}).get('title', 'N/A')}")
        print(f"技术文章字数: {state.get('longform_article', {}).get('word_count', 'N/A')}")
        print(f"小红书笔记字数: {state.get('xiaohongshu_note', {}).get('word_count', 'N/A')}")
        print(f"内容压缩率: {state.get('xiaohongshu_note', {}).get('compression_ratio', 'N/A')}")

        # Twitter帖子信息（新增）
        if state.get('twitter_post'):
            twitter = state['twitter_post']
            print(f"\n🐦 Twitter帖子: {twitter.get('tweet_count', 0)} 条推文")
            print(f"   总字符数: {twitter.get('total_characters', 0)}")
            print(f"   平均字符: {twitter.get('average_characters', 0)} / 条")
            print(f"   风格: {twitter.get('style', 'N/A')}")
            print(f"   形式: {'Thread' if twitter.get('is_thread') else '单条推文'}")

        # 热点简报信息（新增）
        if state.get('trends_digest'):
            digest = state['trends_digest']
            print(f"\n📰 热点简报: {digest.get('title', 'N/A')}")
            print(f"   期号: #{digest.get('issue_number', 'N/A')}")
            print(f"   热点数量: {digest.get('total_topics', 0)} 个")
            print(f"   字数: {digest.get('word_count', 0)} 字")
            print(f"   阅读时间: {digest.get('reading_time', 'N/A')}")

        # 图片生成信息
        generated_images = state.get('generated_images', [])
        if generated_images:
            print(f"生成图片: {len(generated_images)} 张")
            for i, img in enumerate(generated_images, 1):
                if img.get('local_path'):
                    print(f"  图片{i}: {img['local_path']}")
                elif img.get('url'):
                    print(f"  图片{i}: {img['url']}")
                else:
                    print(f"  图片{i}: 提示词已保存")
        elif state.get('image_prompts'):
            print(f"图片提示词: {len(state.get('image_prompts', []))} 个（已保存）")

        print(f"质量评分: {state.get('quality_report', {}).get('overall_score', 'N/A')}/10")
        print(f"是否发布: {'是' if state.get('published') else '否（已保存为草稿）'}")
        print(f"执行耗时: {state.get('execution_time', 0):.2f}秒")

        if state.get('recommended_title'):
            print(f"\n推荐标题: {state['recommended_title']}")

        print("="*60 + "\n")
