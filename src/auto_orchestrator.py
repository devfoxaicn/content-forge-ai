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
from src.agents.trends_digest_agent import TrendsDigestAgent
from src.utils.storage_v2 import StorageFactory

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
        """初始化所有Agent（Auto模式：只生成简报）"""
        agents = {}
        agents_config = self.config.get("agents", {})

        # Auto模式只初始化简报相关的Agent
        # AI热点分析Agent（真实API版本）
        if agents_config.get("ai_trend_analyzer", {}).get("enabled", True):
            agents["ai_trend_analyzer"] = RealAITrendAnalyzerAgent(self.config, self.prompts)

        # 热点汇总Agent
        if agents_config.get("trends_digest", {}).get("enabled", True):
            agents["trends_digest"] = TrendsDigestAgent(self.config, self.prompts)

        # 注意：Auto模式下不初始化长文本、小红书、Twitter等Agent
        # 如需生成完整内容，请使用 Custom、Refine 或 Series 模式

        logger.info(f"Auto模式已初始化 {len(agents)} 个Agent: {list(agents.keys())}")
        return agents

    def _build_workflow(self) -> StateGraph:
        """构建自动化工作流（Auto模式：简报生成）"""
        workflow = StateGraph(dict)

        # 添加Agent节点
        for agent_name, agent in self.agents.items():
            workflow.add_node(agent_name, self._create_agent_node(agent))

        # 定义执行顺序：AI热点分析 → 热点汇总 → END
        if "ai_trend_analyzer" in self.agents:
            workflow.set_entry_point("ai_trend_analyzer")

            # 热点汇总Agent
            last_node = "ai_trend_analyzer"
            if "trends_digest" in self.agents:
                workflow.add_edge(last_node, "trends_digest")
                last_node = "trends_digest"

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
        """保存输出结果到按日期分层的目录（Auto模式：只保存原始数据和简报）"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. 保存AI热点原始数据（JSON格式）
        if state.get("ai_hot_topics"):
            topics = state["ai_hot_topics"]
            raw_filename = f"raw_topics_{timestamp}.json"
            raw_data = {
                "fetched_at": datetime.now().isoformat(),
                "total_topics": len(topics),
                "topics": topics
            }
            raw_file = self.storage.save_json("raw", raw_filename, raw_data)
            logger.info(f"AI热点原始数据已保存: {raw_file}")

        # 2. 保存热点简报
        if state.get("trends_digest"):
            self._save_digest(state)

        logger.success(f"Auto模式内容已保存到: {self.storage.get_date_dir()}")

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
        print("📝 Auto模式 - 热点简报生成完成")
        print("="*60)

        # 热点简报信息
        if state.get('trends_digest'):
            digest = state['trends_digest']
            print(f"\n📰 热点简报: {digest.get('title', 'N/A')}")
            print(f"   期号: #{digest.get('issue_number', 'N/A')}")
            print(f"   热点数量: {digest.get('total_topics', 0)} 个")
            print(f"   字数: {digest.get('word_count', 0)} 字")
            print(f"   阅读时间: {digest.get('reading_time', 'N/A')}")

        # AI热点信息
        hot_topics = state.get('ai_hot_topics', [])
        if hot_topics:
            print(f"\n🔥 获取到 {len(hot_topics)} 个AI热点")

        print(f"\n⏱️  执行耗时: {state.get('execution_time', 0):.2f}秒")
        print(f"📁 存储位置: {self.storage.get_date_dir()}")
        print("\n💡 提示：如需生成完整内容，请使用 Custom、Refine 或 Series 模式")
        print("="*60 + "\n")
