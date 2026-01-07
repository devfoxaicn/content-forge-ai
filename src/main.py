"""
小红书AI内容自动化生产系统
主入口程序
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

# 导入新的自动工作流
from src.auto_orchestrator import AutoContentOrchestrator


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="小红书AI内容自动化生产系统")
    parser.add_argument("--topic", type=str, default=None, help="内容主题标识（可选，用于文件命名，留空则基于实时热点自动生成）")
    parser.add_argument("--audience", type=str, default="技术从业者", help="目标受众")
    parser.add_argument("--type", type=str, default="干货分享", help="内容类型")
    parser.add_argument("--keywords", type=str, nargs="*", default=[], help="关键词列表")
    parser.add_argument("--once", action="store_true", help="立即生成一次")

    args = parser.parse_args()

    # 初始化自动工作流协调器
    logger.info("初始化自动化内容生成系统...")
    orchestrator = AutoContentOrchestrator()

    # 执行工作流
    if args.once or True:
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


if __name__ == "__main__":
    main()
