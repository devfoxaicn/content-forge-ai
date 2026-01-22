#!/usr/bin/env python3
"""
AI Daily Digest - 一键生成简报并提交GitHub

自动执行完整流程：
1. 从14个顶级AI媒体源获取最新资讯
2. 按分类组织热点
3. 使用LLM批量翻译生成高质量中文简报
4. 保存到 data/daily/YYYYMMDD/digest/ 目录
5. 自动提交并推送到GitHub

环境要求：
- 虚拟环境: /Users/z/Documents/work/content-forge-ai/venv
- Python 3.8+
- 依赖: pip install -r requirements.txt
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 检查虚拟环境
VENV_PATH = PROJECT_ROOT / "venv"
IN_VENV = sys.prefix == str(VENV_PATH) or hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

from src.auto_orchestrator import AutoContentOrchestrator
from src.utils.github_publisher import GitHubPublisher
from loguru import logger


def main():
    """主函数：执行完整流程"""

    # 检查虚拟环境
    if not IN_VENV:
        print("⚠️  警告：未检测到虚拟环境")
        print(f"   建议使用虚拟环境: {VENV_PATH}")
        print(f"   激活方式: source {VENV_PATH}/bin/activate")
        print()
    else:
        print(f"📦 虚拟环境: {sys.prefix}")
        print()

    print("=" * 60)
    print("🚀 AI Daily Digest - 一键生成简报并提交GitHub")
    print("=" * 60)
    print()

    # Step 1: 生成简报
    print("📡 Step 1/3: 生成AI新闻简报...")
    print("-" * 60)

    orchestrator = AutoContentOrchestrator()
    result = orchestrator.run()

    if result.get("error_message"):
        print(f"❌ 简报生成失败: {result['error_message']}")
        return 1

    digest = result.get("news_digest")
    if not digest:
        print("❌ 未生成简报内容")
        return 1

    print(f"✅ 简报生成成功!")
    print(f"   - 标题: {digest.get('title')}")
    print(f"   - 热点数: {digest.get('total_topics')} 条")
    print(f"   - 字数: {digest.get('word_count')} 字")
    print(f"   - 版本: {digest.get('version')}")
    print()

    # Step 2: 提交到Git
    print("📝 Step 2/3: 提交到Git...")
    print("-" * 60)

    try:
        publisher = GitHubPublisher()

        # 检查Git状态
        status = publisher.check_git_status()
        print(f"   当前分支: {status['branch']}")
        print(f"   有更改: {status['has_changes']}")

        # 获取简报文件路径
        today = datetime.now().strftime("%Y%m%d")
        digest_dir = PROJECT_ROOT / "data" / "daily" / today / "digest"
        md_file = digest_dir / f"digest_{today}.md"
        json_file = digest_dir / f"digest_{today}.json"

        if not md_file.exists():
            print(f"❌ 简报文件不存在: {md_file}")
            return 1

        # 添加并提交
        commit_success = publisher.publish_daily_digest(
            digest_file=str(md_file),
            json_file=str(json_file) if json_file.exists() else None
        )

        if not commit_success:
            print("❌ Git提交失败")
            return 1

        print("✅ Git提交成功!")
        print()

    except Exception as e:
        print(f"❌ Git操作失败: {e}")
        print("   简报已生成，请手动提交")
        return 1

    # Step 3: 完成
    print("🎉 Step 3/3: 完成!")
    print("-" * 60)
    print()
    print("📊 简报信息:")
    print(f"   📰 标题: {digest.get('title')}")
    print(f"   📅 日期: {digest.get('publish_date')}")
    print(f"   🔗 Issue: #{digest.get('issue_number')}")
    print(f"   📊 热点: {digest.get('total_topics')} 条")
    print(f"   📝 字数: {digest.get('word_count')} 字")
    print(f"   ⏱️  阅读: {digest.get('reading_time')}")
    print()
    print(f"📁 文件位置: {digest_dir}")
    print(f"🔗 GitHub: https://github.com/devfoxaicn/content-forge-ai")
    print()
    print("=" * 60)
    print("✨ 全部完成！简报已生成并提交到GitHub")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
