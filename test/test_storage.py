"""
测试新的存储结构
验证按日期分层的文件存储
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from ..src.utils.storage import get_storage


def test_storage():
    """测试存储功能"""
    print("="*60)
    print("测试新的按日期分层存储结构")
    print("="*60)

    # 1. 创建存储管理器
    print("\n[1/6] 创建存储管理器...")
    storage = get_storage("data")
    print(f"✓ 存储目录: {storage.get_date_dir()}")
    print(f"✓ 日期字符串: {storage.get_date_string()}")

    # 2. 检查子目录
    print("\n[2/6] 检查子目录...")
    subdirs = ["raw", "digest", "longform", "xiaohongshu"]
    for subdir in subdirs:
        subdir_path = storage.get_path(subdir, "", create_dir=False)
        exists = subdir_path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {subdir}/")

    # 3. 测试保存JSON
    print("\n[3/6] 测试保存JSON文件...")
    test_data = {
        "title": "测试文章",
        "content": "这是一篇测试文章",
        "timestamp": "2026-01-07T12:00:00"
    }
    json_file = storage.save_json("longform", "test_article.json", test_data)
    print(f"✓ JSON文件已保存: {json_file}")
    print(f"  文件大小: {json_file.stat().st_size} bytes")

    # 4. 测试保存Markdown
    print("\n[4/6] 测试保存Markdown文件...")
    md_content = """# 测试标题

这是一篇测试的Markdown文件。

## 子标题

内容...
"""
    md_file = storage.save_markdown("digest", "test_digest.md", md_content)
    print(f"✓ Markdown文件已保存: {md_file}")
    print(f"  文件大小: {md_file.stat().st_size} bytes")

    # 5. 测试保存文本
    print("\n[5/6] 测试保存文本文件...")
    text_content = "这是文本文件内容\n第二行内容"
    text_file = storage.save_text("xiaohongshu", "test_prompts.txt", text_content)
    print(f"✓ 文本文件已保存: {text_file}")
    print(f"  文件大小: {text_file.stat().st_size} bytes")

    # 6. 列出文件
    print("\n[6/6] 列出各目录的文件...")
    for category in ["longform", "digest", "xiaohongshu"]:
        files = storage.list_files(category)
        print(f"\n  {category}/ ({len(files)} 个文件):")
        for file in files:
            print(f"    - {file.name}")

    # 获取最新文件
    print("\n[额外] 测试获取最新文件...")
    latest_json = storage.get_latest_file("longform", "*.json")
    if latest_json:
        print(f"✓ 最新JSON文件: {latest_json.name}")

    # 读取并验证
    print("\n[验证] 读取并验证保存的文件...")
    import json
    with open(latest_json, 'r', encoding='utf-8') as f:
        loaded_data = json.load(f)
        if loaded_data == test_data:
            print("✓ JSON内容验证成功")
        else:
            print("✗ JSON内容不匹配")

    print("\n" + "="*60)
    print("✓ 存储结构测试完成！")
    print("="*60)

    # 显示完整目录结构
    print("\n完整目录结构:")
    print_directory_tree(storage.get_date_dir())

    print(f"\n所有数据已保存到: {storage.get_date_dir()}")


def print_directory_tree(path, prefix="", max_depth=3, current_depth=0):
    """打印目录树"""
    if current_depth > max_depth:
        return

    try:
        entries = sorted(path.iterdir(), key=lambda e: (not e.is_dir(), e.name))
        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            current_prefix = "└── " if is_last else "├── "
            print(f"{prefix}{current_prefix}{entry.name}")

            if entry.is_dir() and current_depth < max_depth:
                next_prefix = prefix + ("    " if is_last else "│   ")
                print_directory_tree(entry, next_prefix, max_depth, current_depth + 1)
    except PermissionError:
        pass


if __name__ == "__main__":
    test_storage()
