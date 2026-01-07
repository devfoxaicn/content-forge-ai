"""
测试AI热点分析Agent（真实API版本）
单独测试热点获取功能
"""

import sys
import yaml
import argparse
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from ..src.agents.ai_trend_analyzer_real import RealAITrendAnalyzerAgent
from loguru import logger


def load_config(config_path: str = "config/config.yaml") -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def test_ai_trends(topic: str = "AI工具", show_details: bool = True):
    """
    测试AI热点分析

    Args:
        topic: 主题领域（AI工具、大模型应用、效率提升）
        show_details: 是否显示详细信息
    """
    logger.info(f"开始测试AI热点分析Agent")
    logger.info(f"主题领域: {topic}")

    # 加载配置
    config = load_config()
    prompts = {"prompts": {}}

    # 创建Agent
    agent = RealAITrendAnalyzerAgent(config, prompts)

    # 创建初始状态
    state = {
        "topic": topic,
        "target_audience": "技术从业者",
        "content_type": "干货分享",
        "keywords": []
    }

    # 执行分析
    logger.success("=" * 60)
    result = agent.execute(state)

    # 检查结果
    if "error_message" in result:
        logger.error(f"❌ 分析失败: {result['error_message']}")
        return

    # 显示结果
    hot_topics = result.get("ai_hot_topics", [])
    selected_topic = result.get("selected_ai_topic", {})

    logger.success(f"✅ 成功获取 {len(hot_topics)} 个热点话题")
    logger.success(f"✅ 选择话题: {selected_topic.get('title', 'N/A')}")
    logger.success("=" * 60)

    if show_details:
        print("\n" + "=" * 80)
        print("📊 热点话题详情")
        print("=" * 80)

        for i, topic_data in enumerate(hot_topics, 1):
            print(f"\n【热点 {i}】")
            print(f"标题: {topic_data.get('title', 'N/A')}")
            print(f"来源: {topic_data.get('source', 'N/A')}")
            print(f"时间: {topic_data.get('timestamp', 'N/A')}")
            print(f"热度评分: {topic_data.get('heat_score', 0)}")
            print(f"描述: {topic_data.get('description', 'N/A')[:150]}...")
            print(f"URL: {topic_data.get('url', 'N/A')}")

            metrics = topic_data.get('metrics', {})
            if metrics:
                print(f"指标: {metrics}")

            tags = topic_data.get('tags', [])
            if tags:
                print(f"标签: {', '.join(tags)}")

        print("\n" + "=" * 80)
        print("📊 数据源统计")
        print("=" * 80)

        # 统计各数据源数量
        source_counts = {}
        for topic_data in hot_topics:
            source = topic_data.get('source', 'Unknown')
            source_counts[source] = source_counts.get(source, 0) + 1

        for source, count in source_counts.items():
            print(f"  {source}: {count} 条")

        print("=" * 80)


def test_single_source(source: str):
    """测试单个数据源"""
    logger.info(f"测试单个数据源: {source}")

    config = load_config()

    # 临时配置：只启用一个数据源
    sources_map = {
        "hackernews": ["hackernews"],
        "arxiv": ["arxiv"],
        "github": ["github"],
        "reddit": ["reddit"],
        "huggingface": ["huggingface"],
        "stackoverflow": ["stackoverflow"],
        "kaggle": ["kaggle"],
        "newsapi": ["newsapi"],
        "devto": ["devto"],
        "pypi": ["pypi"],
        "github_topics": ["github_topics"]
    }

    if source not in sources_map:
        logger.error(f"未知的数据源: {source}")
        logger.info("可用数据源: hackernews, arxiv, github, reddit, huggingface, stackoverflow, kaggle, newsapi, devto, pypi, github_topics")
        return

    config["agents"]["ai_trend_analyzer"]["sources"] = sources_map[source]

    # 创建Agent并测试
    prompts = {"prompts": {}}
    agent = RealAITrendAnalyzerAgent(config, prompts)

    state = {
        "topic": "AI工具",
        "target_audience": "技术从业者",
        "content_type": "干货分享",
        "keywords": []
    }

    result = agent.execute(state)

    if "error_message" in result:
        logger.error(f"❌ {source} 数据源测试失败: {result['error_message']}")
    else:
        hot_topics = result.get("ai_hot_topics", [])
        logger.success(f"✅ {source} 数据源测试成功，获取 {len(hot_topics)} 条热点")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="测试AI热点分析Agent")
    parser.add_argument("--topic", type=str, default="AI工具",
                       choices=["AI工具", "大模型应用", "效率提升"],
                       help="主题领域")
    parser.add_argument("--source", type=str, default=None,
                       choices=["hackernews", "arxiv", "github", "reddit", "huggingface", "stackoverflow", "kaggle", "newsapi", "devto", "pypi", "github_topics"],
                       help="测试单个数据源")
    parser.add_argument("--brief", action="store_true",
                       help="简要输出（不显示详细信息）")

    args = parser.parse_args()

    # 配置日志
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    if args.source:
        # 测试单个数据源
        test_single_source(args.source)
    else:
        # 测试完整功能
        test_ai_trends(args.topic, show_details=not args.brief)


if __name__ == "__main__":
    main()
