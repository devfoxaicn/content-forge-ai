---
name: de-ai-humanizer
description: 四层去AI化处理，让内容更自然、有人情味，降低AI痕迹评分
triggers: ["去AI化", "人性化", "去AI腔", "内容自然化", "AI痕迹", "AI味"]
---

# 四层去AI化处理器 (De-AI Humanizer)

## 概述

将AI生成的内容转化为更自然、更有人情味的表达，通过四层处理降低"AI味"评分。

## 调用 Python 模块

```python
from src.utils.deai import DeAIAgent, humanize_content, detect_ai_traces

# 方式1: 使用类（完整控制）
agent = DeAIAgent()
humanized, report = agent.humanize(
    content,
    intensity=0.5,          # 处理强度 0.0-1.0
    platform="weixin",      # 目标平台
    enable_colloquial=True, # 启用口语化
    enable_emotional=True,  # 启用情感注入
    enable_particles=True   # 启用语气词
)

print(f"AI味评分: {report.original_ai_score}% → {report.final_ai_score}%")
print(f"连接词替换: {len(report.connector_replacements)} 处")
print(f"口语化替换: {len(report.colloquial_replacements)} 处")

# 方式2: 使用便捷函数
humanized, report = humanize_content(content, platform="xiaohongshu")

# 方式3: 检测AI痕迹
traces = detect_ai_traces(content)
for trace in traces:
    print(f"发现: {trace['text']} (出现 {trace['count']} 次)")
```

---

## 四层处理流程

### Layer 1: 连接词替换

将AI痕迹明显的连接词替换为自然表达：

| AI连接词 | 人性化替代词 |
|---------|-------------|
| 首先 | 说起来、其实吧、老实说、一开始 |
| 其次 | 还有一点、另外啊、对了、而且 |
| 综上所述 | 说到底、归根结底、简单说 |
| 然而 | 不过、但是、可、话说回来 |
| 值得注意的是 | 有意思的是、你知道吗、重点来了 |

### Layer 2: 口语化转换

将正式词汇转为口语化表达：

| 正式词 | 口语化替代 |
|-------|-----------|
| 认为 | 觉得、感觉、寻思、想 |
| 使用 | 用、整、搞、弄 |
| 实现 | 搞定、弄好、做到 |
| 非常 | 特别、超、老、贼 |
| 进行 | 做、搞、弄、整 |

### Layer 3: 情感注入

添加个人感受和情绪词：

```python
# 情感词类型
EMOTION_WORDS = {
    "positive_surprise": ["真的没想到", "最让我惊喜的是", "太赞了"],
    "question": ["你有多久没...了？", "你猜怎么着？"],
    "exclamation": ["这个真的太...了！", "关键是"],
    "story": ["我之前也是", "后来我发现", "让我印象深的是"],
    "empathy": ["这真不是你一个人的问题", "我也一样"],
}
```

### Layer 4: 语气词添加

添加"吧、呢、啊"等语气词：

| 语气词 | 情感 | 用法 |
|-------|-----|------|
| 啊 | 感叹 | 强调、惊讶 |
| 呢 | 亲切 | 轻松、延续 |
| 吧 | 缓和 | 推测、建议 |
| 哦 | 提醒 | 恍然大悟 |
| 嘛 | 解释 | 明显、解释 |

---

## 平台配置

不同平台有不同的去AI化强度：

| 平台 | intensity | colloquial | emotional | 目标AI味 |
|------|-----------|------------|-----------|---------|
| 微信公众号 | 0.7 | ✅ | ✅ | ≤15% |
| 小红书 | 0.3 | ✅ | ❌ | ≤30% |
| Twitter/X | 0.2 | ❌ | ❌ | ≤40% |
| 知乎 | 0.4 | ✅ | ❌ | ≤20% |

### 使用示例

```python
# 微信公众号：高强度去AI化
humanized, report = agent.humanize(content, platform="weixin", intensity=0.7)

# 小红书：适度口语化，不添加情感
humanized, report = agent.humanize(content, platform="xiaohongshu", intensity=0.3)

# Twitter：保持专业，轻度处理
humanized, report = agent.humanize(content, platform="twitter", intensity=0.2)
```

---

## AI味评分计算

```python
# 评分公式
ai_score = (AI连接词字数 * 2 + 禁忌标题字数 * 4) / 总字数 * 100

# 评分标准
# ≤15%  优秀 - 内容自然，AI痕迹很少
# 15-30% 良好 - 轻微AI痕迹，可接受
# >30%  需改进 - AI痕迹明显，建议处理
```

---

## 小标题验证

检查并建议小标题格式：

```python
# 禁忌格式
forbidden = ["第一章", "1.", "一、", "第一部分", "第一节"]

# 推荐格式
recommended = {
    "question": "提问式：为什么需要关注XXX？",
    "number": "数字式：3个关键步骤掌握XXX",
    "contrast": "对比式：传统方法 vs 创新方案",
    "golden": "金句式：破解XXX的核心秘密",
    "benefit": "利益式：如何用XXX提升效率",
    "painpoint": "痛点式：别再为XXX烦恼了",
}

# 使用验证
issues = agent.validate_subheadings(content)
suggestions = agent.suggest_subheadings("AI写作", count=5)
```

---

## 完整示例

```python
from src.utils.deai import DeAIAgent

# 原始内容（AI痕迹明显）
content = """
首先，我们需要了解AI的基本原理。其次，AI的应用非常广泛。
综上所述，AI是一个值得深入研究的领域。
"""

# 创建处理器
agent = DeAIAgent()

# 检测AI痕迹
traces = agent.detect_ai_patterns(content)
# 输出: [
#   {"type": "ai_connector", "text": "首先", "count": 1},
#   {"type": "ai_connector", "text": "其次", "count": 1},
#   {"type": "ai_connector", "text": "综上所述", "count": 1}
# ]

# 去AI化处理
humanized, report = agent.humanize(content, platform="weixin")

print(f"原始AI味: {report.original_ai_score}%")
print(f"处理后AI味: {report.final_ai_score}%")
print(f"替换详情: {report.connector_replacements}")

# 输出示例:
# 原始AI味: 8.5%
# 处理后AI味: 2.1%
# 替换详情: [
#   {"original": "首先", "replacement": "说起来"},
#   {"original": "其次", "replacement": "还有一点"},
#   {"original": "综上所述", "replacement": "说到底"}
# ]
```

---

## 注意事项

1. **处理强度**: 建议从 0.3-0.5 开始，根据效果调整
2. **平台适配**: 不同平台有不同的受众和风格要求
3. **多次处理**: 可以多次调用以进一步降低AI味
4. **人工审核**: 建议处理后人工审核，确保语义正确
5. **保留专业术语**: 技术类文章应保留必要的专业表达

---

**Created**: 2026-02-14
**Version**: 1.0
**Dependencies**: `src/utils/deai/`
