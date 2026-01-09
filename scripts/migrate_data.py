#!/usr/bin/env python3
"""
数据迁移脚本

将现有数据从旧存储结构迁移到新结构：
- 旧：data/YYYYMMDD/ -> data/daily/YYYYMMDD/
- 旧：data/batch/ -> 保持不变
- 新增：data/series/ 用于100期技术博客
"""

import shutil
from pathlib import Path
from datetime import datetime
import sys

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def migrate_daily_data(base_dir: str = "data", dry_run: bool = False) -> dict:
    """
    迁移每日热点数据

    Args:
        base_dir: 基础数据目录
        dry_run: 是否为演练模式（不实际移动文件）

    Returns:
        dict: 迁移结果统计
    """
    base_path = Path(base_dir)
    source_dir = base_path
    target_dir = base_path / "daily"

    result = {
        "success_count": 0,
        "skip_count": 0,
        "error_count": 0,
        "migrated_dirs": []
    }

    print("\n" + "=" * 60)
    print("📦 迁移每日热点数据")
    print("=" * 60)

    # 遍历data目录下的日期文件夹
    for item in source_dir.iterdir():
        if not item.is_dir():
            continue

        # 跳过已存在的目录（daily, batch, series等）
        if item.name in ["daily", "batch", "series", "archive"]:
            continue

        # 检查是否为日期格式目录（YYYYMMDD）
        if not item.name.isdigit() or len(item.name) != 8:
            print(f"⏭️  跳过非日期目录: {item.name}")
            result["skip_count"] += 1
            continue

        # 目标路径
        target_path = target_dir / item.name

        # 检查目标是否已存在
        if target_path.exists():
            print(f"⏭️  目标已存在，跳过: {item.name}")
            result["skip_count"] += 1
            continue

        if dry_run:
            print(f"🔍 [演练] 将迁移: {item.name} -> daily/{item.name}")
            result["success_count"] += 1
            result["migrated_dirs"].append(item.name)
        else:
            try:
                print(f"📦 迁移中: {item.name} -> daily/{item.name}")
                shutil.move(str(item), str(target_path))
                result["success_count"] += 1
                result["migrated_dirs"].append(item.name)
                print(f"✅ 成功: {item.name}")
            except Exception as e:
                print(f"❌ 失败: {item.name} - {e}")
                result["error_count"] += 1

    return result


def create_series_structure(base_dir: str = "data", dry_run: bool = False) -> dict:
    """
    创建100期技术博客的目录结构

    Args:
        base_dir: 基础数据目录
        dry_run: 是否为演练模式

    Returns:
        dict: 创建结果统计
    """
    series_config = [
        ("series_1", "llm_foundation", 10),
        ("series_2", "rag_techniques", 8),
        ("series_3", "agent_development", 8),
        ("series_4", "prompt_engineering", 6),
        ("series_5", "model_deployment", 8),
        ("series_6", "multimodal_ai", 10),
        ("series_7", "ai_coding_tools", 10),
        ("series_8", "ai_data_engineering", 10),
        ("series_9", "ai_applications", 15),
        ("series_10", "ai_infrastructure", 15),
    ]

    base_path = Path(base_dir)
    series_root = base_path / "series"

    result = {
        "series_count": 0,
        "episode_count": 0,
        "created_dirs": []
    }

    print("\n" + "=" * 60)
    print("🗂️  创建100期技术博客目录结构")
    print("=" * 60)

    if dry_run:
        print("🔍 [演练模式] 将创建以下目录结构：\n")

    for series_id, series_name, episode_count in series_config:
        series_dir_name = f"{series_id}_{series_name}"
        series_path = series_root / series_dir_name

        if dry_run:
            print(f"  📁 {series_path}")
            for ep in range(1, episode_count + 1):
                ep_dir = f"episode_{ep:03d}"
                print(f"    📁 {ep_dir}/")
            result["series_count"] += 1
            result["episode_count"] += episode_count
        else:
            try:
                # 创建系列目录
                series_path.mkdir(parents=True, exist_ok=True)

                # 创建单集目录
                for ep in range(1, episode_count + 1):
                    ep_dir = series_path / f"episode_{ep:03d}"
                    ep_dir.mkdir(exist_ok=True)

                    # 创建子目录
                    subdirs = ["raw", "digest", "longform", "xiaohongshu", "twitter"]
                    for subdir in subdirs:
                        (ep_dir / subdir).mkdir(exist_ok=True)

                    result["episode_count"] += 1

                result["series_count"] += 1
                result["created_dirs"].append(series_dir_name)
                print(f"✅ 创建系列: {series_dir_name} ({episode_count}集)")

            except Exception as e:
                print(f"❌ 创建失败: {series_dir_name} - {e}")

    return result


def verify_structure(base_dir: str = "data") -> dict:
    """
    验证新存储结构

    Args:
        base_dir: 基础数据目录

    Returns:
        dict: 验证结果
    """
    base_path = Path(base_dir)

    result = {
        "daily_exists": False,
        "daily_count": 0,
        "batch_exists": False,
        "batch_count": 0,
        "series_exists": False,
        "series_count": 0,
        "structure_valid": False
    }

    print("\n" + "=" * 60)
    print("🔍 验证存储结构")
    print("=" * 60)

    # 检查 daily 目录
    daily_dir = base_path / "daily"
    if daily_dir.exists():
        result["daily_exists"] = True
        result["daily_count"] = len([d for d in daily_dir.iterdir() if d.is_dir()])
        print(f"✅ daily/ 目录存在 ({result['daily_count']} 个日期)")

    # 检查 batch 目录
    batch_dir = base_path / "batch"
    if batch_dir.exists():
        result["batch_exists"] = True
        result["batch_count"] = len([d for d in batch_dir.iterdir() if d.is_dir()])
        print(f"✅ batch/ 目录存在 ({result['batch_count']} 个批次)")

    # 检查 series 目录
    series_dir = base_path / "series"
    if series_dir.exists():
        result["series_exists"] = True
        series_list = [d for d in series_dir.iterdir() if d.is_dir()]
        result["series_count"] = len(series_list)
        print(f"✅ series/ 目录存在 ({result['series_count']} 个系列)")

    # 验证结构完整性
    result["structure_valid"] = all([
        result["daily_exists"],
        result["batch_exists"],
        result["series_exists"]
    ])

    if result["structure_valid"]:
        print("\n✅ 存储结构验证通过")
    else:
        print("\n⚠️  存储结构不完整，请检查")

    return result


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="数据迁移脚本")
    parser.add_argument("--dry-run", action="store_true", help="演练模式，不实际执行")
    parser.add_argument("--verify", action="store_true", help="仅验证结构")
    parser.add_argument("--base-dir", default="data", help="基础数据目录")

    args = parser.parse_args()

    if args.verify:
        # 仅验证模式
        verify_structure(args.base_dir)
        return

    print("\n🚀 开始数据迁移")
    print(f"基础目录: {args.base_dir}")
    if args.dry_run:
        print("⚠️  演练模式：不会实际修改文件")

    # 1. 迁移每日数据
    daily_result = migrate_daily_data(args.base_dir, args.dry_run)

    # 2. 创建系列目录结构
    series_result = create_series_structure(args.base_dir, args.dry_run)

    # 3. 验证结果（如果不是演练模式）
    if not args.dry_run:
        verify_result = verify_structure(args.base_dir)
    else:
        verify_result = None

    # 打印摘要
    print("\n" + "=" * 60)
    print("📊 迁移摘要")
    print("=" * 60)

    if not args.dry_run:
        print(f"\n每日数据迁移:")
        print(f"  成功: {daily_result['success_count']}")
        print(f"  跳过: {daily_result['skip_count']}")
        print(f"  失败: {daily_result['error_count']}")

        print(f"\n系列结构创建:")
        print(f"  系列数: {series_result['series_count']}")
        print(f"  集数: {series_result['episode_count']}")

        if verify_result:
            print(f"\n结构验证:")
            print(f"  通过: {'✅' if verify_result['structure_valid'] else '❌'}")
    else:
        print("\n🔍 演练模式完成，以上为将要执行的操作")
        print("   去掉 --dry-run 参数以实际执行")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
