"""
测试新的AI热点数据源（产品+新闻+学术导向）
"""

import sys
import os

# 设置项目路径
sys.path.insert(0, '/Users/z/Documents/work/content-forge-ai')

from src.agents.ai_trend_analyzer_real import RealAITrendAnalyzerAgent
import yaml


def load_config():
    """加载配置文件"""
    with open('/Users/z/Documents/work/content-forge-ai/config/config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_prompts():
    """加载提示词"""
    return {}  # 简化测试，不加载prompts


def test_new_sources():
    """测试新的数据源配置"""
    print("=" * 60)
    print("测试新AI热点数据源（产品+新闻+学术）")
    print("=" * 60)

    # 加载配置
    config = load_config()

    # 设置环境变量以避免API检查失败
    os.environ.setdefault('ZHIPUAI_API_KEY', 'test_key')

    prompts = load_prompts()

    # 创建agent
    agent = RealAITrendAnalyzerAgent(config, prompts)

    print(f"\n数据源配置:")
    for source, enabled in agent.sources.items():
        status = "✅" if enabled else "❌"
        print(f"  {status} {source}")

    print(f"\n配置参数:")
    print(f"  max_trends: {agent.max_trends}")
    print(f"  min_score: {agent.min_score}")

    print("\n" + "=" * 60)
    print("开始测试数据源...")
    print("=" * 60)

    # 测试获取热点
    try:
        trends = agent._get_real_ai_trends()

        print(f"\n成功获取 {len(trends)} 个热点话题")
        print("=" * 60)

        # 显示前10个热点
        print("\n前10个热点:")
        for i, trend in enumerate(trends[:10], 1):
            print(f"\n{i}. [{trend['source']}] {trend['title']}")
            print(f"   热度: {trend['heat_score']} | {trend['timestamp']}")
            desc = trend.get('description', '')[:100]
            if desc:
                print(f"   描述: {desc}...")
            print(f"   标签: {', '.join(trend.get('tags', []))}")

        # 统计数据源分布
        print("\n" + "=" * 60)
        print("数据源分布:")
        print("=" * 60)

        source_count = {}
        for trend in trends:
            source = trend['source']
            source_count[source] = source_count.get(source, 0) + 1

        for source, count in sorted(source_count.items(), key=lambda x: x[1], reverse=True):
            print(f"  {source}: {count} 条")

        print("\n✅ 测试完成!")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_new_sources()
