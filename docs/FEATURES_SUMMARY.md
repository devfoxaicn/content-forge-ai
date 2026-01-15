# ContentForge AI v2.7 - 功能汇总

## 项目概述

**ContentForge AI** 是一个基于 LangChain/LangGraph 的自动化内容生产系统，支持四种模式和完整的质量保证流程。

**版本**: v2.7
**更新日期**: 2026-01-15
**核心特点**: 多模式内容生成 + 7层质量保证体系

---

## 一、四种运行模式

| 模式 | 用途 | 触发方式 | 输出 |
|------|------|----------|------|
| **Auto Mode** | AI热点追踪与简报 | 定时/crontab | 热点数据 + 趋势简报 |
| **Series Mode** | 100期技术博客系列 | 手动执行 | 35000字深度长文 + 质量报告 |
| **Custom Mode** | 用户自定义主题 | 手动执行 | 按需生成的长文（深度研究+长文生成） |
| **Refine Mode** | 多平台内容精炼 | 手动执行 | 微信HTML + 小红书 + Twitter |

---

## 二、完整Agent列表（共16个）

### 2.1 核心生成Agent（3个）

| Agent | 文件 | 功能 | 状态 |
|-------|------|------|------|
| **AITrendAnalyzerAgent** | `ai_trend_analyzer_real.py` | 从7个数据源获取AI热点 | ✅ 原有 |
| **LongFormGeneratorAgent** | `longform_generator.py` | 分阶段生成35000字深度长文 | ✅ 原有 |
| **ResearchAgent** | `research_agent.py` | 网络搜索获取研究资料 | ✅ 增强版 |

### 2.2 质量保证Agent - Phase 1（3个）

| Agent | 文件 | 功能 | 检查项 | 状态 |
|-------|------|------|--------|------|
| **CodeReviewAgent** | `code_review_agent.py` | 代码审查 | • 语法正确性<br>• 最佳实践<br>• 安全性<br>• 可读性 | ✅ **NEW** |
| **FactCheckAgent** | `fact_check_agent.py` | 事实核查 | • 版本号验证<br>• 性能数据<br>• 技术规格<br>• 准确性评分 | ✅ **NEW** |
| **QualityEvaluatorAgent** | `quality_evaluator_agent.py` | 质量评估 | • 结构(15%)<br>• 深度(25%)<br>• 准确性(20%)<br>• 可读性(15%)<br>• 可视化(10%)<br>• 时效性(15%) | ✅ **NEW** |

### 2.3 增强Agent - Phase 2（2个）

| Agent | 文件 | 功能 | 输出 | 状态 |
|-------|------|------|------|------|
| **ConsistencyCheckerAgent** | `consistency_checker_agent.py` | 一致性检查 | • 术语一致性<br>• 引用完整性<br>• 数据一致性<br>• 格式一致性<br>• 术语表 | ✅ **NEW** |
| **VisualizationGeneratorAgent** | `visualization_generator_agent.py` | 可视化生成 | • 架构图<br>• 流程图<br>• 序列图<br>• 类图<br>• Mermaid代码 | ✅ **NEW** |

### 2.4 高级Agent - Phase 3（1个）

| Agent | 文件 | 功能 | 输出 | 状态 |
|-------|------|------|------|------|
| **CitationFormatterAgent** | `citation_formatter_agent.py` | 引用格式化 | • GB/T 7714格式<br>• APA格式<br>• IEEE格式<br>• 引用完整性验证<br>• 参考文献列表生成 | ✅ **NEW** |

### 2.5 社交内容Agent（2个）

| Agent | 文件 | 功能 | 状态 |
|-------|------|------|------|
| **XiaohongshuRefinerAgent** | `xiaohongshu_refiner.py` | 精炼为小红书笔记 | ✅ 原有 |
| **TwitterGeneratorAgent** | `twitter_generator.py` | 生成Twitter Thread | ✅ 原有 |

### 2.6 辅助Agent（2个）

| Agent | 文件 | 功能 | 状态 |
|-------|------|------|------|
| **TitleOptimizerAgent** | `title_optimizer.py` | 标题优化 | ✅ 原有 |
| **ImageGeneratorAgent** | `image_generator.py` | 配图提示词生成 | ✅ 原有 |

---

## 三、Series模式完整工作流

```
┌─────────────────────────────────────────────────────────────┐
│             Series模式质量保证工作流（8层）                 │
└─────────────────────────────────────────────────────────────┘

第1层：ResearchAgent
  ├─ Tavily搜索（≤10次调用控制成本）
  ├─ zhipuAI联网搜索（年包，无额外成本）
  └─ 输出：结构化研究数据 + 资料来源

第2层：LongFormGeneratorAgent
  ├─ 第一阶段：生成大纲（10-12章节）
  ├─ 第二阶段：逐节展开（context window保持连贯）
  └─ 第三阶段：生成总结

第3层：CodeReviewAgent ✨ NEW
  ├─ 提取所有代码块
  ├─ Python：ast模块语法检查
  ├─ 其他语言：基础检查
  ├─ 最佳实践验证
  └─ 输出：代码审查报告 + 改进建议

第4层：FactCheckAgent ✨ NEW
  ├─ 提取事实性声明（版本号、性能数据、技术规格等）
  ├─ zhipuAI内置知识验证（免费）
  ├─ Tavily关键验证（≤10次/文章）
  └─ 输出：准确率评分 + 高风险声明列表

第5层：QualityEvaluatorAgent ✨ NEW
  ├─ 6维度多指标评分
  ├─ 计算加权总分
  ├─ 生成改进建议
  └─ 输出：质量报告 + 是否达标判定

第6层：ConsistencyCheckerAgent ✨ NEW
  ├─ 术语翻译一致性检查（Transformer、Embedding等）
  ├─ 引用完整性验证
  ├─ 数据单位一致性
  └─ 输出：术语表 + 问题列表

第7层：VisualizationGeneratorAgent ✨ NEW
  ├─ 识别可视化机会（架构、流程、交互等）
  ├─ 生成Mermaid图表代码
  ├─ 支持类型：flowchart, sequence, graph, class, pie
  └─ 输出：可插入文章的图表

第8层：CitationFormatterAgent ✨ NEW
  ├─ 识别文内引用（[1], [Author et al., 2023]等）
  ├─ 提取并格式化参考文献
  ├─ 支持格式：GB/T 7714、APA、IEEE
  ├─ 验证引用完整性
  └─ 输出：格式化参考文献列表 + 验证报告

最终输出：
  ├─ 长文本文章（30000-40000字）
  ├─ 代码审查报告
  ├─ 事实核查报告
  ├─ 质量评估报告
  ├─ 一致性检查报告
  ├─ Mermaid图表代码
  ├─ 引用格式化报告 ✨ NEW
  └─ 元数据文件
```

---

## 四、质量标准体系

### 4.1 6维度评分标准

| 维度 | 权重 | 检查项 | 目标分数 |
|------|------|--------|----------|
| **结构** | 15% | 层次清晰、逻辑连贯、章节平衡 | ≥8/10 |
| **深度** | 25% | 技术深度、数据支撑、代码示例 | ≥8/10 |
| **准确性** | 20% | 代码正确率、事实准确率 | ≥9/10 |
| **可读性** | 15% | 段落长度、句子结构、术语解释 | ≥8/10 |
| **可视化** | 10% | 图表数量、图表质量 | ≥7/10 |
| **时效性** | 15% | 信息最新、实践验证、前瞻性 | ≥8/10 |

### 4.2 质量目标

| 阶段 | 总分 | 代码通过率 | 事实错误率 | 术语一致性 | 图表数量 |
|------|------|-----------|-----------|-----------|---------|
| **Phase 1** | 7.0/10 | >95% | <1% | - | - |
| **Phase 2** | 8.0/10 | >95% | <1% | >98% | ≥3 |
| **目标** | 9.0/10 | 100% | <0.5% | >99% | ≥5 |

---

## 五、成本控制策略

### 5.1 API使用策略

| API | 用途 | 频率限制 | 成本 |
|-----|------|----------|------|
| **zhipuai glm-4.7** | 主要LLM调用 | 年包无限制 | **￥0**（已付费） |
| **Tavily** | 关键声明验证 | ≤10次/文章 | ≤￥5/文章 |
| **Python ast** | 代码语法检查 | 无限制 | **￥0** |
| **Python内置** | 各种检查功能 | 无限制 | **￥0** |

### 5.2 月度成本估算

- **固定成本**: ￥0（zhipuai年包）
- **变动成本**: ￥0-500/月
- **单篇文章**: ≤￥5（假设10次Tavily调用）

---

## 六、关键文件结构

```
src/
├── agents/
│   ├── base.py                          # Agent基类
│   ├── ai_trend_analyzer_real.py        # AI热点分析（7数据源）
│   ├── longform_generator.py            # 长文本生成（分阶段）
│   ├── research_agent.py                 # 网络搜索研究 ✨ 增强版
│   ├── code_review_agent.py              # 代码审查 ✨ NEW
│   ├── fact_check_agent.py               # 事实核查 ✨ NEW
│   ├── quality_evaluator_agent.py        # 质量评估 ✨ NEW
│   ├── consistency_checker_agent.py      # 一致性检查 ✨ NEW
│   ├── visualization_generator_agent.py   # 可视化生成 ✨ NEW
│   ├── xiaohongshu_refiner.py            # 小红书精炼
│   ├── twitter_generator.py               # Twitter生成
│   ├── title_optimizer.py                # 标题优化
│   └── image_generator.py                 # 配图提示词
├── orchestrators/
│   ├── auto_orchestrator.py              # Auto模式协调器
│   ├── series_orchestrator.py            # Series模式协调器 ✨ 增强
│   ├── custom_orchestrator.py             # Custom模式协调器 ✨ 增强
│   └── refine_orchestrator.py             # Refine模式协调器
├── main.py                               # 统一入口
└── state.py                              # 状态定义

config/
├── config.yaml                            # 主配置 ✨ 更新
├── prompts.yaml                           # 提示词模板 ✨ 更新
└── blog_topics_100_complete.json         # 100期内容规划

data/
├── daily/                                 # Auto模式输出
├── series/                               # Series模式输出
│   └── series_X_*/
│       └── episode_XXX/
│           ├── longform/
│           ├── quality/                   # 质量报告 ✨ NEW
│           └── episode_metadata.json
└── custom/                               # Custom模式输出
```

---

## 七、使用示例

### 7.1 生成一篇带质量保证的文章

```bash
# 设置环境变量
export PYTHONPATH=/Users/z/Documents/work/content-forge-ai

# Series模式生成Episode 1（包含7层质量保证）
PYTHONPATH=/Users/z/Documents/work/content-forge-ai ./venv/bin/python src/main.py \
  --mode series \
  --episode 1
```

**输出位置**：
```
data/series/series_1_llm_foundation/episode_001/
├── longform/
│   └── ep001_llm_transformer_attention_mechanism_article.md
├── quality/
│   └── quality_report.md                   # 质量评估报告
└── episode_metadata.json
```

### 7.2 Custom模式生成（仅生成长文本）

```bash
PYTHONPATH=/Users/z/Documents/work/content-forge-ai ./venv/bin/python src/main.py \
  --mode custom \
  --topic "RAG技术原理" \
  --prompt "详细介绍架构和实战案例" \
  --words 35000
```

**输出**：自动调用ResearchAgent（Tavily搜索）→生成长文本文章

**输出位置**：
```
data/custom/YYYYMMDD_HHMMSS_rag技术原理/
└── longform/
    └── article.md
```

**注意**：Custom模式不包含质量保证Agent（代码审查、事实核查等），如需高质量内容请使用Series模式

---

## 八、版本历史

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| **v2.5** | 2026-01-13 | 四模式重构、代码清理 |
| **v2.6** | 2026-01-14 | 100期技术博客系列完成 |
| **v2.7** | 2026-01-15 | **质量保证系统** - Phase 1完成 |

---

## 九、质量保证系统亮点

### 9.1 世界级标准对标

ContentForge AI v2.7的质量保证系统对齐世界级技术文章的10大标准：

1. ✅ **清晰的层次结构** - QualityEvaluatorAgent检查结构
2. ✅ **深度与广度平衡** - 多维度评分确保深度
3. ✅ **技术准确性** - CodeReviewAgent + FactCheckAgent
4. ✅ **代码示例标准** - CodeReviewAgent验证
5. ✅ **可视化呈现** - VisualizationGeneratorAgent生成图表
6. ✅ **可读性优化** - QualityEvaluatorAgent可读性评分
7. ✅ **专业写作规范** - ConsistencyCheckerAgent格式检查
8. ✅ **与时俱进** - QualityEvaluatorAgent时效性评分
9. ✅ **质量控制机制** - 7层质量保证流程
10. ✅ **数据驱动** - 质量分数可量化、可追踪

### 9.2 零成本高质量

- **纯Python实现** - CodeReviewAgent使用ast模块，无需额外工具
- **智能降级** - FactCheckAgent优先使用zhipuAI（年包）
- **限制调用** - Tavily限制≤10次/文章，控制成本
- **自动化** - 完全自动质量检查，无需人工审核

---

## 十、总结

ContentForge AI v2.7 现已具备：

✅ **4种运行模式** - Auto/Series/Custom/Refine
✅ **15个Agent** - 覆盖内容生成、质量保证、社交内容
✅ **7层质量保证** - 从研究到可视化的完整流程
✅ **6维度评分** - 结构、深度、准确性、可读性、可视化、时效性
✅ **零额外成本** - 主要使用zhipuai年包 + Python内置工具
✅ **世界级标准** - 对齐顶级技术博客的10大标准

**可生成高质量技术文章**：30000-40000字深度长文 + 代码示例 + Mermaid图表 + 质量报告
