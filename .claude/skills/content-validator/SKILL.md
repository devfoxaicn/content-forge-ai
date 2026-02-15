---
name: content-validator
description: 三层内容验证系统，确保内容质量符合平台要求
triggers: ["验证内容", "内容检查", "平台验证", "发布前检查", "字数检查"]
---

# 三层内容验证系统 (Content Validator)

## 概述

对内容进行三层验证，确保符合平台要求和质量标准。

## 调用 Python 模块

```python
from src.utils.validation import ValidationSystem, validate_content

# 方式1: 使用类（完整控制）
validator = ValidationSystem()
passed, results = validator.validate_all(
    content,
    platform="weixin",
    title="文章标题",
    images=["cover.png"]
)

if passed:
    print("✅ 所有验证通过")
else:
    for layer_name, result in results.items():
        if not result.passed:
            print(f"❌ {layer_name}: {result.issues}")

# 方式2: 使用便捷函数
passed, results = validate_content(content, platform="xiaohongshu")

# 获取验证摘要
summary = validator.get_validation_summary(results)
print(f"通过率: {summary['pass_rate']:.1f}%")
print(f"错误: {summary['errors']}")
print(f"警告: {summary['warnings']}")
```

---

## 三层验证架构

### Layer 1: 长文本验证

验证文章的基本结构：

| 检查项 | 标准 | 级别 |
|-------|------|------|
| 字数范围 | 3000-8000字（推荐3500-4000） | ERROR/WARNING |
| 章节结构 | 3-12章（推荐5-8章） | ERROR/WARNING |
| AI连接词密度 | ≤3%（超过5%为错误） | ERROR/WARNING |
| 标题长度 | 5-64字 | ERROR/WARNING |

```python
# Layer 1 验证结果示例
results["layer1"] = ValidationResult(
    layer="长文本验证",
    passed=True,
    checks=[
        CheckResult(name="字数检查", passed=True, level=INFO, message="字数合格：3800字"),
        CheckResult(name="章节结构", passed=True, level=INFO, message="章节结构良好：7章"),
        CheckResult(name="AI连接词密度", passed=True, level=INFO, message="AI连接词密度合格：1.2%"),
    ]
)
```

### Layer 2: 平台内容验证

根据不同平台的要求进行验证：

#### 平台限制配置

| 平台 | 标题长度 | 正文字数 | 特殊要求 |
|------|---------|---------|---------|
| 微信公众号 | 5-64字 | 3500-8000字 | 建议有摘要/导语 |
| 小红书 | - | 100-1000字 | 建议使用emoji |
| Twitter/X | - | 10-280字 | 简洁直接 |
| 知乎 | 5-50字 | 500-100000字 | 深度长文 |

```python
# 平台配置
PLATFORM_LIMITS = {
    "xiaohongshu": {
        "title_max": 20,
        "body_max": 1000,
        "body_min": 100,
    },
    "twitter": {
        "body_max": 280,
        "body_min": 10,
    },
    "weixin": {
        "title_max": 64,
        "body_min": 3500,
        "body_max": 8000,
    },
    "zhihu": {
        "title_max": 50,
        "body_min": 500,
    },
}
```

### Layer 3: 发布前验证

检查图片、链接等资源：

| 检查项 | 说明 |
|-------|------|
| 图片引用 | 内容中引用的图片是否存在 |
| 链接有效性 | 外部链接格式是否正确 |
| 资源完整性 | 所有必要资源是否准备就绪 |

---

## 验证级别

```python
class ValidationLevel(Enum):
    ERROR = "error"      # 必须修复，否则无法通过
    WARNING = "warning"  # 建议修复，不影响通过
    INFO = "info"        # 提示信息
```

### 处理逻辑

```python
# ERROR: 阻止发布
if check.level == ValidationLevel.ERROR and not check.passed:
    result.passed = False
    result.issues.append(check.message)

# WARNING: 仅提示
elif check.level == ValidationLevel.WARNING and not check.passed:
    result.suggestions.append(check.message)
```

---

## 使用示例

### 完整验证流程

```python
from src.utils.validation import ValidationSystem

validator = ValidationSystem()

# 验证微信文章
content = open("article.md").read()
passed, results = validator.validate_all(
    content=content,
    platform="weixin",
    title="AI技术深度解析",
    images=["cover.png", "diagram.png"]
)

# 打印结果
print(f"验证结果: {'通过' if passed else '未通过'}")

for layer_name, result in results.items():
    print(f"\n{result.layer}:")
    for check in result.checks:
        status = "✅" if check.passed else "❌"
        print(f"  {status} {check.name}: {check.message}")
```

### 平台特定验证

```python
# 小红书验证（重点：字数和emoji）
passed, results = validator.validate_all(content, platform="xiaohongshu")

# Twitter验证（重点：280字符限制）
passed, results = validator.validate_all(content, platform="twitter")

# 知乎验证（重点：深度和专业性）
passed, results = validator.validate_all(content, platform="zhihu")
```

### 自定义平台限制

```python
# 添加自定义平台
custom_limits = {
    "my_blog": {
        "title_min": 10,
        "title_max": 100,
        "body_min": 1000,
        "body_max": 20000,
    }
}

validator = ValidationSystem(custom_limits=custom_limits)
passed, results = validator.validate_all(content, platform="my_blog")
```

---

## 验证报告示例

```python
# 获取验证摘要
summary = validator.get_validation_summary(results)

# 输出:
{
    "total_checks": 12,
    "passed_checks": 10,
    "pass_rate": 83.3,
    "all_passed": False,
    "errors": [
        "[长文本验证] 字数不足：当前2800字，最低要求3000字"
    ],
    "warnings": [
        "[平台验证(xiaohongshu)] emoji使用较少：2个，小红书推荐使用emoji",
        "[平台验证(weixin)] 建议添加文章摘要或导语"
    ]
}
```

---

## 与去AI化模块联动

```python
from src.utils.deai import DeAIAgent
from src.utils.validation import ValidationSystem

# 1. 先验证
validator = ValidationSystem()
passed, results = validator.validate_all(content, platform="weixin")

# 2. 如果AI连接词密度过高，进行去AI化
ai_check = next(
    (c for c in results["layer1"].checks if c.name == "AI连接词密度"),
    None
)

if ai_check and ai_check.details.get("density", 0) > 3:
    agent = DeAIAgent()
    content, report = agent.humanize(content, platform="weixin")
    print(f"已进行去AI化处理: {report.original_ai_score}% → {report.final_ai_score}%")

    # 3. 重新验证
    passed, results = validator.validate_all(content, platform="weixin")
```

---

## 注意事项

1. **验证顺序**: 建议先 Layer 1 → Layer 2 → Layer 3
2. **平台优先**: Layer 2 的平台限制会覆盖 Layer 1 的通用标准
3. **图片检查**: Layer 3 需要提供图片列表才能完整验证
4. **错误处理**: ERROR 级别的问题必须修复，WARNING 级别可酌情处理
5. **多次验证**: 内容修改后应重新验证

---

**Created**: 2026-02-14
**Version**: 1.0
**Dependencies**: `src/utils/validation/`, `src/utils/deai/`
