#!/usr/bin/env python3
"""
快速启动脚本 - 支持用户指定话题或自动筛选热门话题
"""

import os
import sys
import yaml
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

from loguru import logger
from src.auto_orchestrator import AutoContentOrchestrator


def print_banner():
    """打印欢迎横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                                ║
║           ContentForge AI v2.2 - 内容快速启动                ║
║                                                                ║
║    支持用户指定话题 | 自动筛选AI热点 | 全流程内容生成         ║
║                                                                ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def load_topics():
    """加载预设话题配置"""
    topics_file = project_root / "config" / "topics.yaml"
    try:
        with open(topics_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config.get("topics", []), config.get("categories", {})
    except Exception as e:
        logger.warning(f"无法加载话题配置: {e}")
        return [], {}


def display_topics_by_category(topics, categories):
    """按分类显示话题"""
    print("\n" + "="*60)
    print("📚 热门技术话题分类")
    print("="*60)

    category_num = 1
    category_map = {}

    # 显示分类
    for cat_name, topic_ids in categories.items():
        print(f"\n【{category_num}】{cat_name.upper()}")
        category_map[category_num] = (cat_name, topic_ids)

        for topic_id in topic_ids:
            topic = next((t for t in topics if t.get("id") == topic_id), None)
            if topic:
                print(f"   • {topic.get('title')}")
                print(f"     {topic.get('description', '')[:60]}...")

        category_num += 1

    print(f"\n【{category_num}】🔍 自定义话题")
    category_map[category_num] = ("custom", None)

    print(f"\n【0】📡 自动筛选AI热门话题")
    category_map[0] = ("auto", None)

    return category_map


def select_category_topics(topics, category_name, topic_ids):
    """选择分类下的具体话题"""
    if not topic_ids:
        return None

    print(f"\n--- {category_name.upper()} 分类下的话题 ---")
    topic_options = []
    for idx, topic_id in enumerate(topic_ids, 1):
        topic = next((t for t in topics if t.get("id") == topic_id), None)
        if topic:
            topic_options.append(topic)
            print(f"[{idx}] {topic.get('title')}")
            print(f"    {topic.get('description')}")

    while True:
        choice = input("\n请选择话题编号 (0返回): ").strip()
        if choice == "0":
            return None
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(topic_options):
                return topic_options[idx]
        except ValueError:
            pass
        print("❌ 无效选择，请重试")


def get_custom_topic():
    """获取自定义话题"""
    print("\n" + "="*60)
    print("📝 自定义话题输入")
    print("="*60)

    title = input("\n请输入话题标题: ").strip()
    if not title:
        return None

    description = input("请输入话题描述 (可选): ").strip()
    if not description:
        description = title

    keywords_input = input("请输入关键词 (用逗号分隔，可选): ").strip()
    keywords = [k.strip() for k in keywords_input.split(",")] if keywords_input else [title]

    return {
        "title": title,
        "description": description,
        "keywords": keywords
    }


def run_workflow(user_topic=None):
    """运行工作流"""
    print("\n" + "="*60)
    print("🚀 启动内容生成工作流")
    print("="*60)

    # 初始化orchestrator
    logger.info("初始化自动化内容生成系统...")
    orchestrator = AutoContentOrchestrator()

    # 执行工作流
    try:
        if user_topic:
            logger.info(f"🎯 使用用户指定话题: {user_topic['title']}")
            result = orchestrator.run(
                topic=user_topic['title'],
                user_provided_topic=user_topic
            )
        else:
            logger.info("📡 使用AI自动筛选热门话题")
            result = orchestrator.run()

        # 打印结果
        print("\n" + "="*60)
        print("✅ 内容生成完成!")
        print("="*60)

        selected_topic = result.get("selected_ai_topic", {})
        print(f"\n📌 话题: {selected_topic.get('title', 'N/A')}")
        print(f"📄 长文章: {result.get('longform_article', {}).get('word_count', 0)} 字")
        print(f"📱 小红书: {result.get('xiaohongshu_note', {}).get('word_count', 0)} 字")

        if result.get('twitter_post'):
            print(f"🐦 Twitter: {result.get('twitter_post', {}).get('tweet_count', 0)} 条推文")

        print(f"\n⏱️ 耗时: {result.get('execution_time', 0):.2f} 秒")
        print(f"📁 保存位置: data/{result.get('start_time', '')[:10].replace('-', '')}/")

        return result

    except Exception as e:
        logger.error(f"工作流执行失败: {e}")
        print(f"\n❌ 执行失败: {e}")
        return None


def main():
    """主函数"""
    print_banner()

    # 加载话题配置
    topics, categories = load_topics()

    if not topics:
        print("⚠️ 警告: 无法加载预设话题，仅支持自定义或自动模式")

    # 显示分类并选择
    category_map = display_topics_by_category(topics, categories)

    user_topic = None
    mode = "unknown"

    while mode == "unknown":
        choice = input("\n请选择模式编号: ").strip()

        try:
            choice_num = int(choice)

            if choice_num == 0:
                # 自动模式
                mode = "auto"
                print("\n📡 模式: 自动筛选AI热门话题")

            elif choice_num == len(category_map) - 1:
                # 自定义模式
                user_topic = get_custom_topic()
                if user_topic:
                    mode = "custom"
                    print(f"\n📝 自定义话题: {user_topic['title']}")
                else:
                    print("⚠️ 话题输入为空，请重新选择")

            elif choice_num in category_map:
                # 预设分类模式
                cat_name, topic_ids = category_map[choice_num]
                if topic_ids:
                    selected_topic = select_category_topics(topics, cat_name, topic_ids)
                    if selected_topic:
                        user_topic = selected_topic
                        mode = "preset"
                        print(f"\n📚 预设话题: {selected_topic['title']}")
                    else:
                        print("返回分类选择")
                else:
                    print("⚠️ 该分类暂无话题")
            else:
                print("❌ 无效选择，请重试")

        except ValueError:
            print("❌ 请输入数字编号")

    # 确认开始
    print("\n" + "="*60)
    if user_topic:
        print(f"🎯 即将生成内容: {user_topic['title']}")
        print(f"📋 描述: {user_topic.get('description', '')[:80]}")
    else:
        print("📡 将自动筛选最新AI热门话题并生成内容")

    print("="*60)

    confirm = input("\n确认开始生成? (y/n): ").strip().lower()
    if confirm not in ['y', 'yes', '是', '确认']:
        print("已取消")
        return

    # 运行工作流
    run_workflow(user_topic)

    print("\n感谢使用 ContentForge AI!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断")
    except Exception as e:
        logger.error(f"程序异常: {e}")
        print(f"\n❌ 程序异常: {e}")
