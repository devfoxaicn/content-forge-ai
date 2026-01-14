"""
ContentForge AI - 统一主入口
支持四种模式：
1. auto模式 - 基于AI热点的自动内容生成
2. series模式 - 100期技术博客系列生成
3. custom模式 - 根据用户给定的关键词/要求产出高质量长文本
4. refine模式 - 根据已有高质量文本精炼出可直接复制粘贴的多平台内容
"""

import os
import sys
import argparse
from typing import Dict, Any

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

# 日志配置
from loguru import logger

# 导入协调器
from src.auto_orchestrator import AutoContentOrchestrator
from src.series_orchestrator import SeriesOrchestrator


def run_auto_mode(args):
    """运行自动模式（基于AI热点）"""
    logger.info("🚀 启动自动内容生成模式（基于AI热点）...")

    # 初始化自动工作流协调器
    orchestrator = AutoContentOrchestrator()

    # 执行工作流
    logger.info("开始执行自动化内容生成流程...")
    result = orchestrator.run(
        topic=args.topic,
        target_audience=args.audience,
        content_type=args.type,
        keywords=args.keywords if args.keywords else None
    )

    # 打印结果摘要
    logger.success("="*50)
    logger.success("📝 生成完成")
    logger.success(f"耗时: {result.get('execution_time', 0):.2f}秒")
    logger.success(f"执行Agent: {result.get('agent_execution_order', [])}")
    logger.success("="*50)

    return result


def run_series_mode(args):
    """运行系列模式（100期技术博客）"""
    import yaml

    logger.info("🚀 启动系列内容生成模式（100期技术博客）...")

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
        from src.utils.series_manager import print_progress_summary
        print_progress_summary(args.series_config)
        return

    # 生成指定集数
    if args.episode:
        logger.info(f"生成第 {args.episode} 期...")
        result = orchestrator.generate_episode(args.episode)
        logger.success(f"✅ 第 {args.episode} 期生成完成")
        return result

    # 生成指定系列
    if args.series:
        logger.info(f"生成系列 {args.series}...")
        results = orchestrator.generate_series(args.series)
        logger.success(f"✅ 系列 {args.series} 生成完成")
        return results

    # 生成全部
    if args.all:
        logger.info(f"生成第 {args.start}-{args.end} 期...")
        stats = orchestrator.generate_all(
            start_episode=args.start,
            end_episode=args.end
        )
        logger.success(f"✅ 批量生成完成: {stats}")
        return stats

    # 默认显示进度
    from src.utils.series_manager import print_progress_summary
    print_progress_summary(args.series_config)


def run_custom_mode(args):
    """运行自定义内容生成模式"""
    import yaml

    logger.info("🚀 启动自定义内容生成模式...")

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 创建协调器
    from src.custom_orchestrator import CustomContentOrchestrator
    orchestrator = CustomContentOrchestrator(config=config)

    # 执行
    result = orchestrator.run(
        topic=args.topic,
        prompt=args.prompt,
        target_audience=args.audience,
        words=args.words,
        style=args.style
    )

    logger.success("="*50)
    logger.success("✅ 自定义内容生成完成")
    logger.success(f"耗时: {result.get('execution_time', 0):.2f}秒")
    logger.success("="*50)

    return result


def run_refine_mode(args):
    """运行内容精炼模式"""
    import yaml

    logger.info("🚀 启动内容精炼模式...")

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 创建协调器
    from src.refine_orchestrator import RefineOrchestrator
    orchestrator = RefineOrchestrator(config=config)

    # 确定平台列表
    platforms = args.platforms or ["wechat", "xiaohongshu", "twitter"]

    # 执行
    result = orchestrator.run(
        input_source=args.input,
        platforms=platforms
    )

    logger.success("="*50)
    logger.success("✅ 内容精炼完成")
    logger.success(f"输出平台: {', '.join(platforms)}")
    logger.success("="*50)

    return result


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="ContentForge AI - 多平台内容自动化生产系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  自动模式（基于AI热点）:
    python src/main.py --mode auto --once
    python src/main.py --mode auto --topic "AI技术"

  系列模式（100期技术博客）:
    python src/main.py --mode series --progress
    python src/main.py --mode series --episode 1
    python src/main.py --mode series --series series_1
    python src/main.py --mode series --all --start 1 --end 10

  自定义模式（根据关键词生成长文本）:
    python src/main.py --mode custom --topic "RAG技术原理与实战"
    python src/main.py --mode custom --topic "RAG技术" --prompt "详细介绍架构和实战"

  精炼模式（已有文本精炼为多平台内容）:
    python src/main.py --mode refine --input article.md
    python src/main.py --mode refine --input article.md --platforms wechat xiaohongshu
        """
    )

    # 模式选择
    parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "series", "custom", "refine"],
        default="auto",
        help="运行模式"
    )

    # ===== 自动模式参数 =====
    parser.add_argument("--topic", type=str, default=None, help="内容主题标识（可选，用于文件命名）")
    parser.add_argument("--audience", type=str, default="技术从业者", help="目标受众")
    parser.add_argument("--type", type=str, default="干货分享", help="内容类型")
    parser.add_argument("--keywords", type=str, nargs="*", default=[], help="关键词列表")
    parser.add_argument("--once", action="store_true", help="立即生成一次")

    # ===== 系列模式参数 =====
    parser.add_argument("--config", default="config/config.yaml", help="全局配置文件")
    parser.add_argument("--series-config", default="config/blog_topics_100_complete.json", help="100期配置文件")
    parser.add_argument("--episode", type=int, help="生成指定集数")
    parser.add_argument("--series", help="生成指定系列（如 series_1）")
    parser.add_argument("--start", type=int, default=1, help="起始集数")
    parser.add_argument("--end", type=int, default=100, help="结束集数")
    parser.add_argument("--all", action="store_true", help="生成全部指定范围")
    parser.add_argument("--progress", action="store_true", help="仅显示进度")

    # ===== 自定义模式参数 =====
    parser.add_argument("--prompt", help="详细内容要求描述")
    parser.add_argument("--words", type=int, help="目标字数")
    parser.add_argument("--style", choices=["technical", "practical", "tutorial"],
                       help="文章风格")

    # ===== 精炼模式参数 =====
    parser.add_argument("--input", help="输入文件路径（Refine模式必需）")
    parser.add_argument("--platforms", nargs="+", choices=["wechat", "xiaohongshu", "twitter"],
                       help="目标平台")

    args = parser.parse_args()

    try:
        if args.mode == "auto":
            return run_auto_mode(args)
        elif args.mode == "series":
            return run_series_mode(args)
        elif args.mode == "custom":
            return run_custom_mode(args)
        elif args.mode == "refine":
            return run_refine_mode(args)
    except KeyboardInterrupt:
        logger.warning("用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"执行失败: {e}")
        raise


if __name__ == "__main__":
    main()
