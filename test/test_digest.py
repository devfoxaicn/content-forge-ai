"""
测试热点汇总Agent
单独测试热点简报生成功能
"""

import os
import sys
import yaml
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from ..src.agents.trends_digest_agent import TrendsDigestAgent
from ..src.agents.ai_trend_analyzer_real import RealAITrendAnalyzerAgent


def load_config(config_path="config/config.yaml"):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_prompts(prompts_path="config/prompts.yaml"):
    """加载提示词文件"""
    with open(prompts_path, 'r', encoding='utf-8') as f:
        return {"prompts": yaml.safe_load(f)}


def test_digest_generation():
    """测试热点简报生成"""
    print("="*60)
    print("测试热点汇总Agent")
    print("="*60)

    # 1. 加载配置
    print("\n[1/4] 加载配置...")
    config = load_config()
    prompts = load_prompts()
    print("✓ 配置加载成功")

    # 2. 初始化Agent
    print("\n[2/4] 初始化Agent...")
    ai_analyzer = RealAITrendAnalyzerAgent(config, prompts)
    digest_agent = TrendsDigestAgent(config, prompts)
    print("✓ Agent初始化完成")

    # 3. 获取AI热点
    print("\n[3/4] 获取AI热点（可能需要10-20秒）...")
    try:
        state = {
            "topic": "AI技术",
            "target_audience": "技术从业者",
            "content_type": "技术分享"
        }

        # 获取热点
        hot_topics = ai_analyzer._get_real_ai_trends("AI技术")
        print(f"✓ 获取到 {len(hot_topics)} 个热点话题")

        # 更新state
        state["ai_hot_topics"] = hot_topics

        # 显示前3个热点
        print("\n前3个热点:")
        for i, topic in enumerate(hot_topics[:3], 1):
            print(f"  {i}. {topic['title']}")
            print(f"     来源: {topic['source']} | 热度: {topic.get('heat_score', 0)}")

    except Exception as e:
        print(f"✗ 获取热点失败: {e}")
        return

    # 4. 生成简报
    print("\n[4/4] 生成热点简报（可能需要5-10秒）...")
    try:
        result = digest_agent.execute(state)

        if result.get("trends_digest"):
            digest = result["trends_digest"]
            print("✓ 简报生成成功！")
            print(f"\n📰 {digest['title']}")
            print(f"   {digest['subtitle']}")
            print(f"   热点数量: {digest['total_topics']}")
            print(f"   字数: {digest['word_count']}")
            print(f"   阅读时间: {digest['reading_time']}")

            # 保存简报
            output_dir = config.get("agents", {}).get("trends_digest", {}).get("output_dir", "data/digest")
            os.makedirs(output_dir, exist_ok=True)

            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            md_file = f"{output_dir}/test_digest_{timestamp}.md"

            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(digest['full_content'])

            print(f"\n✓ 简报已保存到: {md_file}")

            # 显示简报预览（前500字）
            print("\n📄 简报预览（前500字）:")
            print("-" * 60)
            print(digest['full_content'][:500] + "...")
            print("-" * 60)

        else:
            print("✗ 简报生成失败")

    except Exception as e:
        print(f"✗ 简报生成失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_digest_generation()
