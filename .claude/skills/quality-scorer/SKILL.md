---
name: quality-scorer
description: 5维度质量评分和爆款标题生成，评估内容整体质量
triggers: ["质量评分", "内容评分", "爆款标题", "标题生成", "质量评估"]
---

# 质量评分与标题生成 (Quality Scorer)

## 概述

5维度质量评分系统 + 8大爆款标题公式，全面评估和优化内容质量。

## 调用 Python 模块

```python
from src.utils.quality import QualityScorer, TitleGenerator, score_content, generate_titles

# ===== 质量评分 =====
scorer = QualityScorer()
report = scorer.score(
    content,
    title="文章标题",
    fact_report={"confidence": 0.9, "issues": []}
)

print(f"总分: {report.total_score}")
print(f"等级: {report.grade.value}")  # excellent/good/needs_improvement
print(f"优势: {report.strengths}")
print(f"劣势: {report.weaknesses}")

# ===== 标题生成 =====
generator = TitleGenerator()
titles = generator.generate(
    topic="AI写作技巧",
    content=content,
    count=5,
    platform="weixin"
)

for t in titles:
    print(f"[{t.formula_name}] {t.title} (推荐指数: {t.score})")
```

---

## 5维度质量评分

### 评分维度与权重

| 维度 | 权重 | 说明 | 评分标准 |
|------|------|------|---------|
| 字数符合度 | 20% | 是否符合3500-4000字标准 | 3500-4000字满分，<3000字扣分 |
| 去AI化程度 | 20% | AI味评分是否低于15% | ≤15%满分，每超1%扣2分 |
| 事实准确性 | 30% | 事实核查报告置信度 | 基于事实核查报告评分 |
| 吸引力指数 | 15% | 标题和内容的吸引力 | 标题+开头+章节结构 |
| 格式规范度 | 15% | Markdown格式、章节结构 | 标题层级、代码块、列表 |

### 质量等级

```python
class QualityGrade(Enum):
    EXCELLENT = "excellent"           # ≥85分 - 优秀
    GOOD = "good"                     # 70-84分 - 良好
    NEEDS_IMPROVEMENT = "needs_improvement"  # <70分 - 需改进
```

### 评分示例

```python
from src.utils.quality import QualityScorer

scorer = QualityScorer()
report = scorer.score(content, title="AI技术深度解析")

# 详细维度评分
for name, dim in report.dimensions.items():
    print(f"{dim.name}: {dim.raw_score}分 × {dim.weight} = {dim.weighted_score}分")

# 输出示例:
# 字数符合度: 95分 × 0.2 = 19.0分
# 去AI化程度: 85分 × 0.2 = 17.0分
# 事实准确性: 80分 × 0.3 = 24.0分
# 吸引力指数: 70分 × 0.15 = 10.5分
# 格式规范度: 90分 × 0.15 = 13.5分
# ─────────────────────────────
# 总分: 84.0分 (good)
```

---

## 8大爆款标题公式

### 公式一览

| 公式 | 模板 | 示例 |
|------|------|------|
| 数字+利益型 | [数字]+[动词]+[结果] | 7天涨粉3000，我做对了这3件事 |
| 提问好奇型 | [为什么/如何]+[痛点]+[暗示] | 为什么你的文章没人看？试试这个方法 |
| 对比反差型 | [A vs B]+[意想不到的结果] | 月薪3千和3万的新媒体人，差距就在这 |
| 警告紧迫型 | [警告]+[痛点]+[紧迫感] | 别再用AI写文章了！平台开始大规模封号 |
| 故事情感型 | [人物]+[困境]+[转折] | 失业3个月后，我靠写作月入2万 |
| 金句观点型 | [短句]+[情感/价值观] | 写作改变命运，这句话不是鸡汤 |
| 教程承诺型 | [动词]+[对象]+[时间承诺] | 10分钟学会爆款标题，新手也能上手 |
| 悬念省略型 | [陈述]+[暗示/省略] | 研究了100篇10w+，发现一个规律... |

### 使用示例

```python
from src.utils.quality import TitleGenerator, TitleFormula

generator = TitleGenerator()

# 生成标题
titles = generator.generate(
    topic="RAG技术",
    content=content,
    count=5,
    platform="weixin"
)

for t in titles:
    print(f"[{t.formula_name}] {t.title}")
    print(f"  模板: {t.template}")
    print(f"  推荐指数: {t.score}")
```

### 分析现有标题

```python
# 分析标题特征
analysis = generator.analyze_title("7天学会AI写作，效率翻倍")

# 输出:
{
    "title": "7天学会AI写作，效率翻倍",
    "length": 13,
    "matched_formulas": [
        {"formula": "number_benefit", "name": "数字+利益型"}
    ],
    "score": 90
}
```

### 获取公式示例

```python
# 获取特定公式示例
examples = generator.get_formula_examples(TitleFormula.NUMBER_BENEFIT)
# {"name": "数字+利益型", "template": "...", "examples": [...]}

# 获取所有公式
all_formulas = generator.get_formula_examples()
```

---

## 平台标题长度限制

| 平台 | 最小长度 | 最大长度 |
|------|---------|---------|
| 微信公众号 | 5 | 64 |
| 小红书 | 8 | 20 |
| 知乎 | 5 | 50 |
| Twitter | 5 | 70 |

```python
# 生成符合平台限制的标题
titles = generator.generate(
    topic="AI写作",
    platform="xiaohongshu"  # 限制8-20字
)
```

---

## 完整工作流示例

```python
from src.utils.deai import DeAIAgent
from src.utils.validation import ValidationSystem
from src.utils.quality import QualityScorer, TitleGenerator

# 1. 原始内容
content = open("article.md").read()

# 2. 去AI化处理
deai_agent = DeAIAgent()
content, deai_report = deai_agent.humanize(content, platform="weixin")
print(f"去AI化: {deai_report.original_ai_score}% → {deai_report.final_ai_score}%")

# 3. 内容验证
validator = ValidationSystem()
passed, validation_results = validator.validate_all(content, platform="weixin")
print(f"验证: {'通过' if passed else '未通过'}")

# 4. 质量评分
scorer = QualityScorer()
quality_report = scorer.score(content, title="文章标题")
print(f"质量评分: {quality_report.total_score}分 ({quality_report.grade.value})")

# 5. 生成标题建议
title_gen = TitleGenerator()
titles = title_gen.generate(topic="主题", content=content, count=5)
print("\n推荐标题:")
for i, t in enumerate(titles, 1):
    print(f"{i}. [{t.formula_name}] {t.title}")

# 6. 判断是否可以发布
if passed and quality_report.is_good:
    print("\n✅ 内容已达到发布标准")
else:
    print("\n❌ 内容需要改进:")
    print(f"  - 验证问题: {validation_results['layer1'].issues}")
    print(f"  - 质量建议: {quality_report.suggestions}")
```

---

## 评分标准详解

### 字数符合度 (20%)

```python
# 评分规则
if 3500 <= word_count <= 4000:
    raw_score = 100  # 满分
elif word_count < 3000:
    raw_score = (word_count / 3000) * 80  # 按比例扣分
elif word_count > 8000:
    excess = word_count - 8000
    raw_score = 100 - min(20, excess / 100)  # 每100字扣1分，最多扣20分
else:
    raw_score = 90  # 3500以下但3000以上
```

### 去AI化程度 (20%)

```python
# AI味评分越低越好
if ai_score <= 15:
    raw_score = 100  # 目标达成
else:
    raw_score = max(0, 100 - (ai_score - 15) * 2)  # 每超1%扣2分
```

### 事实准确性 (30%)

```python
# 基于事实核查报告
if fact_report:
    confidence = fact_report.get("confidence", 0.8)
    issues = fact_report.get("issues", [])
    raw_score = confidence * 100 - len(issues) * 5
else:
    # 无报告时使用启发式评估
    raw_score = heuristic_accuracy_check(content)  # 基础80分
```

### 吸引力指数 (15%)

```python
score = 70  # 基础分

# 标题检查
if 10 <= len(title) <= 30: score += 10
if has_number(title): score += 5
if has_question(title): score += 5

# 开头检查
if has_story_hook(first_para): score += 5
if has_question_hook(first_para): score += 5

# 章节结构
if subheadings >= 5: score += 5
```

### 格式规范度 (15%)

```python
score = 80  # 基础分

# 标题层级
if h2_count >= 5: score += 5

# 代码块
if has_code_blocks: score += 5

# 列表结构
if list_items >= 3: score += 3

# 段落过长
if long_paragraphs > 30%: score -= 10

# 空行过多
if has_excess_blank_lines: score -= 5
```

---

## 注意事项

1. **权重可调整**: 可以通过 `custom_weights` 参数自定义权重
2. **无事实核查**: 无事实核查报告时会使用启发式评估（基础80分）
3. **标题生成**: 生成的标题是建议性的，需要人工审核
4. **多次评分**: 内容修改后应重新评分
5. **平台适配**: 标题生成时注意平台长度限制

---

**Created**: 2026-02-14
**Version**: 1.0
**Dependencies**: `src/utils/quality/`, `src/utils/deai/`
