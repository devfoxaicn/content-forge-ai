"""
快速验证topic参数处理逻辑
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from ..src.state import create_initial_state


def test_topic_logic():
    """测试topic处理逻辑"""
    print("="*60)
    print("测试topic参数处理逻辑")
    print("="*60)

    # 测试1: 不指定topic
    print("\n[测试1] 不指定topic")
    state = create_initial_state()
    print(f"✓ Topic默认值: {state['topic']}")
    assert state['topic'] == "auto", "topic应该是'auto'"
    print("✓ 测试1通过")

    # 测试2: 指定topic
    print("\n[测试2] 指定topic为'AI技术'")
    state = create_initial_state(topic="AI技术")
    print(f"✓ Topic值: {state['topic']}")
    assert state['topic'] == "AI技术", "topic应该是'AI技术'"
    print("✓ 测试2通过")

    # 测试3: topic为None
    print("\n[测试3] topic显式传递None")
    state = create_initial_state(topic=None)
    print(f"✓ Topic值: {state['topic']}")
    assert state['topic'] == "auto", "topic为None时应该是'auto'"
    print("✓ 测试3通过")

    # 测试4: 模拟文件名生成逻辑
    print("\n[测试4] 测试文件名生成逻辑")
    from datetime import datetime

    # 模拟auto模式
    topic = "auto"
    selected_topic = {"title": "GPT-5发布，性能提升300%"}
    if topic == "auto":
        topic = selected_topic.get("title", "auto")

    # 清理文件名
    topic = topic.replace(" ", "_").replace("/", "_").replace("\\", "_")[:30]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"workflow_{topic}_{timestamp}.json"

    print(f"✓ 生成的文件名: {filename}")
    assert "GPT-5发布" in filename, "文件名应该包含热点标题"
    print("✓ 测试4通过")

    print("\n" + "="*60)
    print("✓ 所有逻辑测试通过！")
    print("="*60)

    return True


if __name__ == "__main__":
    try:
        success = test_topic_logic()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
