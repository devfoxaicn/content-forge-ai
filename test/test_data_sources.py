"""
数据源测试脚本 - 验证新集成的数据源是否正常工作
"""
import sys
import os
from datetime import datetime

# 设置PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from loguru import logger

# 加载环境变量
load_dotenv()


def test_time_filter():
    """测试时间过滤器"""
    from src.utils.time_filter import TimeFilter

    logger.info("=" * 50)
    logger.info("测试时间过滤器")
    logger.info("=" * 50)

    # 创建24小时过滤器
    time_filter = TimeFilter(hours=24)

    # 测试各种时间格式
    test_cases = [
        datetime.now().isoformat(),
        "2026-02-01 12:00:00",
        "2026-02-01T12:00:00Z",
        datetime.now().timestamp(),
    ]

    for i, ts in enumerate(test_cases, 1):
        result = time_filter.is_within_time_window(ts)
        logger.info(f"测试 {i}: {ts} -> {'✅ 通过' if result else '❌ 失败'}")

    logger.success("时间过滤器测试完成")


def test_data_sources():
    """测试各个数据源"""
    from src.data_sources.manager import create_data_source_manager

    logger.info("=" * 50)
    logger.info("测试数据源")
    logger.info("=" * 50)

    # 配置
    config = {
        "api_keys": {
            "semantic_scholar": os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
            "openalex_email": os.getenv("OPENALEX_EMAIL"),
            "huggingface": os.getenv("HUGGINGFACE_TOKEN"),
            "github": os.getenv("GITHUB_TOKEN"),
            "product_hunt": os.getenv("PRODUCT_HUNT_API_KEY"),
            "reddit_client_id": os.getenv("REDDIT_CLIENT_ID"),
            "reddit_client_secret": os.getenv("REDDIT_CLIENT_SECRET"),
        }
    }

    # 创建管理器
    manager = create_data_source_manager(config)

    # 测试各个分类的数据获取
    categories = [
        ("academic_frontier", "学术前沿", manager.get_academic_frontier_papers),
        ("dev_tools", "开发工具", manager.get_dev_tools_updates),
        ("ai_agent", "AI Agent", manager.get_ai_agent_projects),
    ]

    for cat_id, cat_name, fetch_func in categories:
        logger.info(f"\n{'=' * 20} {cat_name} ({cat_id}) {'=' * 20}")

        try:
            items = fetch_func(limit=3)

            if items:
                logger.success(f"✅ {cat_name}: 获取到 {len(items)} 条数据")

                # 显示前3条
                for i, item in enumerate(items[:3], 1):
                    logger.info(f"  {i}. {item.get('title', 'N/A')}")
                    logger.info(f"     来源: {item.get('source', 'N/A')}")
                    logger.info(f"     时间: {item.get('published_at', 'N/A')}")
            else:
                logger.warning(f"⚠️ {cat_name}: 未获取到数据")

        except Exception as e:
            logger.error(f"❌ {cat_name}: 测试失败 - {e}")

    logger.success("\n数据源测试完成")


def main():
    """主函数"""
    logger.info("开始数据源测试...")
    logger.info(f"测试时间: {datetime.now().isoformat()}")

    # 测试时间过滤器
    test_time_filter()

    # 测试数据源
    test_data_sources()

    logger.success("=" * 50)
    logger.success("所有测试完成")
    logger.success("=" * 50)


if __name__ == "__main__":
    main()
