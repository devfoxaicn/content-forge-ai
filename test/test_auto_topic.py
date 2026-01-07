"""
测试自动话题功能
验证系统可以在不指定topic的情况下运行
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from ..src.auto_orchestrator import AutoContentOrchestrator
from loguru import logger


def test_auto_topic():
    """测试自动话题模式"""
    print("="*60)
    print("测试自动话题模式（不指定topic）")
    print("="*60)

    # 1. 测试不指定topic
    print("\n[测试1] 不指定topic，系统自动从热点生成内容")
    orchestrator = AutoContentOrchestrator()

    try:
        # 不传递topic参数
        result = orchestrator.run()

        # 检查结果
        print("\n[验证] 检查结果...")
        assert result.get("topic") == "auto", "topic应该是'auto'"
        assert result.get("selected_ai_topic"), "应该有选中的热点话题"
        assert result.get("longform_article"), "应该生成长文本文章"

        print(f"✓ Topic标识: {result.get('topic')}")
        print(f"✓ 选中的热点: {result.get('selected_ai_topic', {}).get('title')}")
        print(f"✓ 文章标题: {result.get('longform_article', {}).get('title')}")
        print(f"✓ 文章字数: {result.get('longform_article', {}).get('word_count')}")

        print("\n✓ 测试1通过！")

    except Exception as e:
        print(f"\n✗ 测试1失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 2. 测试指定topic
    print("\n[测试2] 指定topic作为文件标识")
    try:
        result = orchestrator.run(topic="AI技术实战")

        # 检查结果
        print("\n[验证] 检查结果...")
        assert result.get("topic") == "AI技术实战", "topic应该是'AI技术实战'"
        assert result.get("selected_ai_topic"), "应该有选中的热点话题"

        print(f"✓ Topic标识: {result.get('topic')}")
        print(f"✓ 选中的热点: {result.get('selected_ai_topic', {}).get('title')}")

        print("\n✓ 测试2通过！")

    except Exception as e:
        print(f"\n✗ 测试2失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "="*60)
    print("✓ 所有测试通过！")
    print("="*60)

    print("\n说明:")
    print("- 不指定topic时，系统自动从11个数据源获取热点")
    print("- 长文本和小红书笔记基于实际热点生成")
    print("- 文件名使用实际热点标题，而不是固定的topic")
    print("- 指定topic时，仅作为文件命名标识")

    return True


if __name__ == "__main__":
    success = test_auto_topic()
    sys.exit(0 if success else 1)
