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
from src.agents.concurrent_fetch_agent import ConcurrentFetchAgent  # v11.0: 并发数据获取
from src.agents.time_weight_agent import TimeWeightAgent  # v11.0: 时效性加权
from src.agents.auto_fact_check_agent import AutoFactCheckAgent  # v11.0: 轻量级事实核查
from src.agents.content_enhancer_agent import ContentEnhancerAgent  # v11.0: 内容增强
from src.agents.translation_refiner_agent import TranslationRefinerAgent  # v11.0: 翻译精炼
from src.agents.trend_categorizer_agent import TrendCategorizerAgent
from src.agents.news_scoring_agent import NewsScoringAgent
from src.agents.world_class_digest_agent_v8 import WorldClassDigestAgentV9  # v9.0: 6分类系统
from src.utils.storage_v2 import StorageFactory
from src.utils.github_publisher import GitHubPublisher

# 日志配置
from loguru import logger


class AutoContentOrchestrator:
    """自动化内容生成协调器 - 新工作流"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化自动化协调器（Auto模式：每日热点简报）

        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.prompts = self._load_prompts()
        self._setup_logging()

        # 初始化存储管理器（Auto模式：每日热点）
        self.storage = StorageFactory.create_daily(
            base_dir=self.config.get("storage", {}).get("base_dir", "data")
        )

        # 初始化Agent
        self.agents = self._init_agents()

        # 构建工作流
        self.workflow = self._build_workflow()

        logger.info("自动化内容生成协调器初始化完成（Auto模式：每日热点简报）")
        logger.info(f"数据存储目录: {self.storage.get_root_dir()}")

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
        """初始化所有Agent（Auto模式 v11.0：并发数据获取 + 时效性加权 + 6分类系统 + 质量保证）"""
        agents = {}
        agents_config = self.config.get("agents", {})

        # ========== 数据获取层 ==========
        # v11.0: 优先使用并发数据获取Agent
        if agents_config.get("concurrent_fetch", {}).get("enabled", False):
            agents["concurrent_fetch"] = ConcurrentFetchAgent(self.config, self.prompts)
            logger.info("使用 ConcurrentFetchAgent（并发模式）")
        # 降级到同步模式（向后兼容）
        elif agents_config.get("ai_trend_analyzer", {}).get("enabled", True):
            agents["ai_trend_analyzer"] = RealAITrendAnalyzerAgent(self.config, self.prompts)
            logger.info("使用 RealAITrendAnalyzerAgent（同步模式）")

        # v11.0: 时效性智能加权Agent
        if agents_config.get("time_weight", {}).get("enabled", False):
            agents["time_weight"] = TimeWeightAgent(self.config, self.prompts)
            logger.info("使用 TimeWeightAgent（时效性加权）")

        # ========== 分类评分层 ==========
        # 2. 热点分类Agent
        if agents_config.get("trends_digest", {}).get("enabled", True):
            agents["trend_categorizer"] = TrendCategorizerAgent(self.config, self.prompts)

        # 3. 新闻评分Agent
        if agents_config.get("news_scoring", {}).get("enabled", True):
            agents["news_scoring"] = NewsScoringAgent(self.config, self.prompts)

        # ========== 质量保证层 ==========
        # v11.0: 轻量级事实核查Agent
        if agents_config.get("auto_fact_check", {}).get("enabled", False):
            agents["auto_fact_check"] = AutoFactCheckAgent(self.config, self.prompts)
            logger.info("使用 AutoFactCheckAgent（事实核查）")

        # v11.0: 内容增强Agent
        if agents_config.get("content_enhancer", {}).get("enabled", False):
            agents["content_enhancer"] = ContentEnhancerAgent(self.config, self.prompts)
            logger.info("使用 ContentEnhancerAgent（内容增强）")

        # v11.0: 翻译精炼Agent
        if agents_config.get("translation_refiner", {}).get("enabled", False):
            agents["translation_refiner"] = TranslationRefinerAgent(self.config, self.prompts)
            logger.info("使用 TranslationRefinerAgent（翻译精炼）")

        # ========== 输出生成层 ==========
        # 4. 世界顶级中文简报Agent v9.0（6分类系统 + 30数据源 + Top5截取）
        if agents_config.get("world_class_digest", {}).get("enabled", True):
            agents["world_class_digest"] = WorldClassDigestAgentV9(self.config, self.prompts)

        # 注意：Auto模式下不初始化长文本、小红书、Twitter等Agent
        # 如需生成完整内容，请使用 Series 模式

        logger.info(f"Auto模式 v11.0 已初始化 {len(agents)} 个Agent: {list(agents.keys())}")
        return agents

    def _build_workflow(self) -> StateGraph:
        """构建自动化工作流（Auto模式 v11.0：完整质量保证流程）"""
        workflow = StateGraph(dict)

        # 添加Agent节点
        for agent_name, agent in self.agents.items():
            workflow.add_node(agent_name, self._create_agent_node(agent))

        # v11.0: 定义执行顺序（完整工作流）
        # concurrent_fetch → time_weight → trend_categorizer → news_scoring
        # → auto_fact_check → content_enhancer → translation_refiner → world_class_digest → END

        # 确定入口点（并发或同步）
        if "concurrent_fetch" in self.agents:
            workflow.set_entry_point("concurrent_fetch")
            last_node = "concurrent_fetch"
            logger.info("工作流入口: concurrent_fetch (并发模式)")
        elif "ai_trend_analyzer" in self.agents:
            workflow.set_entry_point("ai_trend_analyzer")
            last_node = "ai_trend_analyzer"
            logger.info("工作流入口: ai_trend_analyzer (同步模式)")
        else:
            # 如果没有数据获取Agent，直接从后续流程开始
            if "time_weight" in self.agents:
                workflow.set_entry_point("time_weight")
                last_node = "time_weight"
            elif "trend_categorizer" in self.agents:
                workflow.set_entry_point("trend_categorizer")
                last_node = "trend_categorizer"
            else:
                logger.warning("没有可用的数据获取或分类Agent")
                return workflow.compile()

        # 时效性智能加权Agent
        if "time_weight" in self.agents:
            workflow.add_edge(last_node, "time_weight")
            last_node = "time_weight"
            logger.info("工作流: 添加时效性加权")

        # 热点分类Agent
        if "trend_categorizer" in self.agents:
            workflow.add_edge(last_node, "trend_categorizer")
            last_node = "trend_categorizer"

        # 新闻评分Agent
        if "news_scoring" in self.agents:
            workflow.add_edge(last_node, "news_scoring")
            last_node = "news_scoring"

        # v11.0: 质量保证层
        # 轻量级事实核查Agent
        if "auto_fact_check" in self.agents:
            workflow.add_edge(last_node, "auto_fact_check")
            last_node = "auto_fact_check"
            logger.info("工作流: 添加事实核查")

        # 内容增强Agent
        if "content_enhancer" in self.agents:
            workflow.add_edge(last_node, "content_enhancer")
            last_node = "content_enhancer"
            logger.info("工作流: 添加内容增强")

        # 翻译精炼Agent
        if "translation_refiner" in self.agents:
            workflow.add_edge(last_node, "translation_refiner")
            last_node = "translation_refiner"
            logger.info("工作流: 添加翻译精炼")

        # 世界顶级中文简报Agent
        if "world_class_digest" in self.agents:
            workflow.add_edge(last_node, "world_class_digest")
            last_node = "world_class_digest"

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
            content_type: str = "干货分享", keywords: list = None,
            user_provided_topic: dict = None) -> Dict[str, Any]:
        """
        运行完整自动化工作流

        Args:
            topic: 内容主题标识（可选，用于文件命名，留空则基于实时热点自动生成）
            target_audience: 目标受众
            content_type: 内容类型
            keywords: 关键词列表
            user_provided_topic: 用户指定的完整话题数据（包含title, description, keywords等），
                               如果提供则跳过AI热点分析，直接使用该话题

        Returns:
            Dict[str, Any]: 最终输出
        """
        # 判断是否为用户指定话题模式
        is_user_topic_mode = user_provided_topic is not None

        if is_user_topic_mode:
            # 用户指定话题模式
            topic = topic or user_provided_topic.get("title", "user_topic")
            logger.info(f"🎯 用户指定话题模式: {topic}")
        elif topic is None:
            topic = "auto"
            logger.info("📡 开始执行自动化内容生产流程（基于实时热点）")
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

        # 如果是用户指定话题模式，设置选中的话题，跳过AI热点分析
        if is_user_topic_mode:
            state["selected_ai_topic"] = {
                "title": user_provided_topic.get("title", topic),
                "description": user_provided_topic.get("description", ""),
                "source": "user_provided",
                "url": "",
                "tags": user_provided_topic.get("keywords", []),
                "key_points": [user_provided_topic.get("description", "")]
            }
            state["ai_hot_topics"] = [state["selected_ai_topic"]]
            state["current_step"] = "user_topic_set"
            logger.info(f"✅ 已设置用户指定话题，跳过AI热点分析")

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
        """保存输出结果到按日期分层的目录（Auto模式 v3.0：原始数据+分类简报）"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. 保存AI热点原始数据（JSON格式）- 包含trends_by_source
        if state.get("trends_by_source"):
            trends_by_source = state["trends_by_source"]
            raw_filename = f"raw_topics_{timestamp}.json"
            raw_data = {
                "fetched_at": datetime.now().isoformat(),
                "total_topics": state.get("total_trends_count", 0),
                "data_sources": list(trends_by_source.keys()),
                "trends_by_source": trends_by_source,
                # 兼容旧格式
                "topics": state.get("ai_hot_topics", [])
            }
            raw_file = self.storage.save_json("raw", raw_filename, raw_data)
            logger.info(f"AI热点原始数据已保存: {raw_file}")

        # 2. 保存热点简报（兼容新旧字段）
        digest = state.get("trends_digest") or state.get("news_digest")
        if digest:
            self._save_digest(state, digest)

        logger.success(f"Auto模式内容已保存到: {self.storage.get_date_dir()}")

    def _format_twitter_thread(self, tweets: list) -> str:
        """格式化Twitter thread为Markdown"""
        formatted_tweets = []
        for i, tweet in enumerate(tweets, 1):
            formatted_tweets.append(f"### Tweet {i}\n\n{tweet}\n")
        return "\n".join(formatted_tweets)

    def _save_digest(self, state: Dict[str, Any], digest: Dict[str, Any]):
        """保存热点简报到digest目录（v7.0：支持增强JSON格式）"""
        try:
            if not digest:
                return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            issue_number = digest.get('issue_number', timestamp)
            base_filename = f"digest_{issue_number}"

            # 兼容 v7.0 新格式和旧格式
            markdown_content = digest.get('markdown_content') or digest.get('full_content', '')

            # 保存Markdown格式（主要格式，符合aibook要求）
            md_filename = f"{base_filename}.md"
            md_file = self.storage.save_markdown("digest", md_filename, markdown_content)
            logger.info(f"热点简报Markdown已保存: {md_file}")

            # 保存JSON格式（用于网站API）
            json_filename = f"{base_filename}.json"

            # v7.0 格式：digest 本身就是完整的JSON数据
            # 只需要确保没有 markdown_content 字段的重复（已经在外层了）
            json_data_to_save = dict(digest)

            # 如果是旧格式，转换为新格式
            if "editors_pick" not in json_data_to_save and "categories" not in json_data_to_save:
                # 旧格式转换
                json_data_to_save = {
                    "metadata": {
                        "title": digest.get("title"),
                        "subtitle": digest.get("subtitle"),
                        "issue_number": issue_number,
                        "publish_date": digest.get("publish_date"),
                        "generated_at": datetime.now().isoformat(),
                        "word_count": digest.get("word_count"),
                        "reading_time": digest.get("reading_time"),
                        "total_topics": digest.get("total_topics"),
                        "version": digest.get("version", "v3.0")
                    },
                    "editors_pick": [],
                    "categories": [],
                    "core_insights": [],
                    "trending_topics": [],
                    "sources": digest.get("sources", []),
                    "topics": digest.get("topics", []),
                    "summary_analysis": digest.get("summary_analysis"),
                    "markdown_content": markdown_content
                }

            json_file = self.storage.save_json("digest", json_filename, json_data_to_save)
            logger.success(f"热点简报已保存: {md_file} (MD) + {json_file} (JSON)")

            # ========== GitHub自动发布 ==========
            self._publish_to_github(md_file, json_file, digest)

        except Exception as e:
            logger.error(f"保存热点简报失败: {e}")

    def _publish_to_github(self, md_file: str, json_file: str, digest: Dict[str, Any]):
        """发布简报到GitHub"""
        try:
            # 检查是否启用GitHub发布
            enable_github_publish = self.config.get("agents", {}).get("ai_trend_analyzer", {}).get("github_publish", False)
            if not enable_github_publish:
                logger.info("GitHub发布功能未启用，跳过自动推送")
                return

            logger.info("开始发布简报到GitHub...")

            try:
                publisher = GitHubPublisher()

                # 检查Git状态
                status = publisher.check_git_status()
                logger.info(f"当前分支: {status['branch']}, 有更改: {status['has_changes']}")

                # 发布简报
                success = publisher.publish_daily_digest(
                    digest_file=md_file,
                    json_file=json_file
                )

                if success:
                    logger.success(f"✅ 简报已成功推送到GitHub: {digest.get('title')}")
                else:
                    logger.warning("⚠️ GitHub推送失败，但简报已保存到本地")

            except Exception as e:
                logger.warning(f"GitHub发布功能不可用或失败: {e}")
                logger.info("简报已保存到本地，可手动提交到GitHub")

        except Exception as e:
            logger.error(f"GitHub发布失败: {e}")

    def _print_summary(self, state: Dict[str, Any]):
        """打印结果摘要"""
        print("\n" + "="*60)
        print("📝 Auto模式 v8.0 - 世界顶级AI新闻简报生成完成")
        print("="*60)

        # 热点简报信息
        digest = state.get('news_digest')
        if digest:
            print(f"\n📰 热点简报: {digest.get('title', 'N/A')}")
            print(f"   期号: #{digest.get('issue_number', 'N/A')}")
            print(f"   热点数量: {digest.get('total_topics', 0)} 个")
            print(f"   版本: {digest.get('version', 'v4.0')}")
            print(f"   字数: {digest.get('word_count', 0)} 字")
            print(f"   阅读时间: {digest.get('reading_time', 'N/A')}")

        # AI热点信息
        total_count = state.get('total_trends_count', 0)
        if total_count > 0:
            print(f"\n🔥 获取到 {total_count} 个AI热点（按数据源汇总）")

            # 打印数据源统计
            trends_by_source = state.get('trends_by_source', {})
            if trends_by_source:
                print("\n📊 数据源统计:")
                for source, items in trends_by_source.items():
                    if items:
                        print(f"   {source}: {len(items)} 条")

        print(f"\n⏱️  执行耗时: {state.get('execution_time', 0):.2f}秒")
        print(f"📁 存储位置: {self.storage.get_date_dir()}")
        print("\n💡 提示：如需生成完整内容，请使用 Custom、Refine 或 Series 模式")
        print("="*60 + "\n")
