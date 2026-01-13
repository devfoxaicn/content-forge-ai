#!/usr/bin/env python3
"""检查100期内容完成情况"""

import os
from pathlib import Path
from collections import defaultdict

# 定义系列编号映射
SERIES_MAPPING = {
    "series_1_llm_foundation": (1, 10, "LLM原理基础"),
    "series_2_rag_technique": (11, 18, "RAG技术实战"),
    "series_3_agent_development": (19, 26, "Agent智能体开发"),
    "series_4_prompt_engineering": (27, 32, "提示工程"),
    "series_5_model_deployment": (33, 40, "模型部署与优化"),
    "series_6_multimodal_frontier": (41, 50, "多模态与前沿技术"),
    "series_7_ai_coding_tools": (51, 60, "AI编程与开发工具"),
    "series_8_ai_data_engineering": (61, 70, "AI数据处理与工程"),
    "series_9_ai_applications": (71, 85, "AI应用场景实战"),
    "series_10_ai_infrastructure": (86, 100, "AI基础设施与架构"),
}

def check_episode_content(episode_path):
    """检查单个episode的内容完整性"""
    required_content = {
        "longform": {"min_size": 5000, "pattern": "*.md"},  # 长文本至少5000字节
        "xiaohongshu": {"min_size": 1000, "pattern": "*.md"},
        "twitter": {"min_size": 500, "pattern": "*.md"}
    }

    if not episode_path.exists():
        return {"exists": False, "complete": False, "issues": ["目录不存在"]}

    content_status = {
        "exists": True,
        "complete": True,
        "issues": [],
        "word_count": 0,
        "file_size": 0
    }

    for content_type, requirements in required_content.items():
        content_dir = episode_path / content_type

        if not content_dir.exists():
            content_status["complete"] = False
            content_status["issues"].append(f"{content_type}目录不存在")
            continue

        # 查找markdown文件
        md_files = list(content_dir.glob(requirements["pattern"]))

        if not md_files:
            content_status["complete"] = False
            content_status["issues"].append(f"{content_type}目录为空")
            continue

        # 检查文件大小和内容
        for md_file in md_files:
            try:
                size = md_file.stat().st_size
                content = md_file.read_text(encoding='utf-8', errors='ignore')

                if content_type == "longform":
                    # 统计中文字数（排除markdown语法）
                    import re
                    text_only = re.sub(r'[#*\-\[\]()`]|http\S+', '', content)
                    word_count = len(text_only.replace('\n', '').replace(' ', ''))
                    content_status["word_count"] = word_count
                    content_status["file_size"] = size

                    # 检查是否为虚拟内容/占位符
                    if size < requirements["min_size"]:
                        content_status["complete"] = False
                        content_status["issues"].append(f"{content_type}内容过小 ({size}字节)")
                    elif word_count < 3000:
                        content_status["complete"] = False
                        content_status["issues"].append(f"{content_type}字数不足 ({word_count}字)")
                    elif "待生成" in content or "TODO" in content or "占位符" in content:
                        content_status["complete"] = False
                        content_status["issues"].append(f"{content_type}包含占位符文本")

            except Exception as e:
                content_status["complete"] = False
                content_status["issues"].append(f"{content_type}读取失败: {e}")

    return content_status

def main():
    series_dir = Path("/Users/z/Documents/work/content-forge-ai/data/series")

    print("=" * 80)
    print("100期内容完成情况报告")
    print("=" * 80)

    total_expected = 100
    total_found = 0
    total_complete = 0

    series_summary = []

    for series_name, (start_ep, end_ep, description) in SERIES_MAPPING.items():
        series_path = series_dir / series_name

        if not series_path.exists():
            print(f"\n❌ {series_name} ({description})")
            print(f"   预期: Episode {start_ep:03d}-{end_ep:03d} (共{end_ep-start_ep+1}集)")
            print(f"   状态: 系列目录不存在")
            continue

        # 查找所有episode文件夹
        episode_dirs = sorted([d for d in series_path.iterdir() if d.is_dir() and d.name.startswith("episode_")])

        # 提取episode编号
        episode_numbers = []
        for ep_dir in episode_dirs:
            try:
                num = int(ep_dir.name.split("_")[1])
                episode_numbers.append(num)
            except (IndexError, ValueError):
                pass

        episode_numbers.sort()

        # 检查每个episode的内容完整性
        complete_episodes = 0
        incomplete_episodes = []
        total_words = 0
        word_stats = []

        for ep_num in episode_numbers:
            ep_path = series_path / f"episode_{ep_num:03d}"
            status = check_episode_content(ep_path)
            if status["exists"] and status["complete"]:
                complete_episodes += 1
                total_words += status["word_count"]
                word_stats.append((ep_num, status["word_count"], status["file_size"]))
            elif status["exists"]:
                incomplete_episodes.append((ep_num, status["issues"]))

        total_found += len(episode_numbers)
        total_complete += complete_episodes

        # 显示系列信息
        expected_count = end_ep - start_ep + 1
        actual_count = len(episode_numbers)
        avg_words = total_words // complete_episodes if complete_episodes > 0 else 0

        print(f"\n{'✅' if actual_count == expected_count and complete_episodes == actual_count else '⚠️'}  {series_name} ({description})")
        print(f"   预期: Episode {start_ep:03d}-{end_ep:03d} (共{expected_count}集)")
        print(f"   实际: {len(episode_numbers)}集 | 高质量: {complete_episodes}集 | 平均字数: {avg_words}字")

        if episode_numbers:
            print(f"   编号: {min(episode_numbers):03d}-{max(episode_numbers):03d}")

        if incomplete_episodes:
            print(f"   ⚠️  内容质量问题:")
            for ep_num, issues in incomplete_episodes[:5]:
                print(f"      Episode {ep_num:03d}: {', '.join(issues)}")
            if len(incomplete_episodes) > 5:
                print(f"      ... 还有 {len(incomplete_episodes) - 5} 集有问题")

        # 显示字数统计
        if word_stats:
            min_words = min(ws[1] for ws in word_stats)
            max_words = max(ws[1] for ws in word_stats)
            print(f"   字数范围: {min_words}-{max_words} 字")

        series_summary.append({
            "name": series_name,
            "description": description,
            "expected": expected_count,
            "actual": actual_count,
            "complete": complete_episodes
        })

    # 总体统计
    print("\n" + "=" * 80)
    print("总体统计（高质量长文本标准：>3000字）")
    print("=" * 80)
    print(f"总预期: {total_expected}集")
    print(f"已生成: {total_found}集")
    print(f"高质量: {total_complete}集 ({total_complete/total_expected*100:.1f}%)")
    print(f"低质量: {total_found - total_complete}集")
    print(f"缺失: {total_expected - total_found}集")

    # 检查编号是否正确
    print("\n" + "=" * 80)
    print("编号检查")
    print("=" * 80)

    all_series_dir = Path("/Users/z/Documents/work/content-forge-ai/data/series")
    all_episodes = []

    for series_path in all_series_dir.iterdir():
        if not series_path.is_dir() or series_path.name.startswith("."):
            continue

        for ep_dir in series_path.iterdir():
            if ep_dir.is_dir() and ep_dir.name.startswith("episode_"):
                try:
                    num = int(ep_dir.name.split("_")[1])
                    all_episodes.append((num, series_path.name, ep_dir.name))
                except (IndexError, ValueError):
                    pass

    all_episodes.sort()

    # 检查重复编号
    episode_counts = defaultdict(list)
    for num, series, ep_dir in all_episodes:
        episode_counts[num].append((series, ep_dir))

    duplicates = {num: locs for num, locs in episode_counts.items() if len(locs) > 1}
    if duplicates:
        print("⚠️  发现重复的episode编号:")
        for num, locs in sorted(duplicates.items()):
            print(f"   Episode {num:03d}:")
            for series, ep_dir in locs:
                print(f"      - {series}/{ep_dir}")
    else:
        print("✅ 没有重复的episode编号")

    # 检查缺失编号
    missing_numbers = []
    for i in range(1, 101):
        if i not in episode_counts:
            missing_numbers.append(i)

    if missing_numbers:
        print(f"\n⚠️  缺失的episode编号 (共{len(missing_numbers)}个):")
        for num in missing_numbers:
            print(f"   Episode {num:03d}")
    else:
        print("\n✅ 所有episode编号(1-100)都已生成")

if __name__ == "__main__":
    main()
