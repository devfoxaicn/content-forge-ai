#!/usr/bin/env python3
"""
测试小红书短笔记Agent
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import yaml
from src.agents.xiaohongshu_short_refiner import XiaohongshuShortRefinerAgent
from src.state import create_initial_state

# 加载配置
with open("config/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

with open("config/prompts.yaml", "r", encoding="utf-8") as f:
    prompts = yaml.safe_load(f)

# 读取长笔记内容
long_note_path = "data/refine/20260116_Claude_Cowork入门指南/xiaohongshu/note_long.md"
with open(long_note_path, "r", encoding="utf-8") as f:
    long_note_content = f.read()

# 提取标题和内容
lines = long_note_content.split('\n')
title = lines[0].replace('# ', '').strip()
content = '\n'.join(lines[1:])

print(f"标题: {title}")
print(f"长笔记字数: {len(content)} 字符")
print("-" * 50)

# 初始化短笔记Agent
agent_config = config.get("agents", {}).get("xiaohongshu_short_refiner", {})
agent = XiaohongshuShortRefinerAgent(config=agent_config, prompts=prompts)

# 构建状态
state = create_initial_state()
state["xiaohongshu_long_note"] = {
    "title": title,
    "full_content": content,
    "word_count": len(content)
}

# 执行Agent
print("\n开始生成短笔记...")
result = agent.execute(state)

# 获取结果
if "xiaohongshu_short_note" in result:
    short_note = result["xiaohongshu_short_note"]
    print(f"\n生成成功!")
    print(f"字数: {short_note.get('word_count', 0)} 字符")
    print(f"压缩率: {short_note.get('compression_ratio', 'N/A')}")
    print("-" * 50)
    print("\n内容预览:")
    print(short_note.get('full_content', '')[:500] + "...")

    # 保存到文件
    output_path = "data/refine/20260116_Claude_Cowork入门指南/xiaohongshu/note_short.md"
    md_content = f"""{short_note.get('full_content', '')}

---
**标签**: {' '.join(short_note.get('hashtags', []))}
**字数**: {short_note.get('word_count', 0)}
**类型**: 短笔记（爆款风格）
**压缩率**: {short_note.get('compression_ratio', 'N/A')}
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print(f"\n已保存到: {output_path}")
else:
    print("生成失败!")
    if "error_message" in result:
        print(f"错误: {result['error_message']}")
