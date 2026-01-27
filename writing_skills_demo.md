# 写作技能生态系统使用指南

## 📦 已安装的技能总览

### 🎯 核心写作技能
| 技能 | 用途 | 适用场景 |
|------|------|----------|
| **scriptwriting** | 中文剧本创作 | 漫剧剧本、短视频剧本 |
| **platform-adaptation** | 多平台内容适配 | 小红书、公众号、知乎、番茄小说、抖音、B站 |
| **writing-clearly-and-concisely** | 清晰简洁写作 | 技术文档、教程 |
| **copywriting** | 营销文案写作 | 产品介绍、营销页面 |
| **copy-editing** | 文章编辑润色 | 内容优化、校对 |
| **content-research-writer** | 长文研究与写作 | 深度文章、技术博客 |
| **social-content** | 社交媒体内容 | 各平台短内容 |
| **marketing-psychology** | 营销心理学 | 用户心理、转化优化 |
| **skill-creator** | 创建自定义技能 | 扩展写作能力 |

### 🏰 Novel-OS 小说创作系统
- **位置**: `~/.novel-os/`
- **用途**: 结构化小说创作工作流
- **三层上下文**: 写作标准 → 小说设定 → 手稿细节

---

## 🎬 场景 1: 漫剧剧本创作

### 激活技能
```
请使用 scriptwriting 技能帮我创作一个漫剧剧本
```

### 示例对话
```
你：我想写一个现代都市题材的漫剧剧本，主角是程序员，讲述他意外获得能够预测未来的AI系统。

Claude 会自动：
1. 使用 scriptwriting 技能的格式标准
2. 创建场景标题（第X场）
3. 按照中文剧本格式编写对话
4. 包含人物、地点、神态描述
```

### 输出示例
```
第1场
地点：内景 程序员李明的出租屋 夜
人物：李明（25岁，程序员）、AI助手（画外音）

李明：（疲惫地揉着眼睛）又是凌晨两点...
（电脑屏幕突然闪烁，一行字浮现：明天上午10:14，你会收到一封改变命运的邮件）

AI助手：（冷静地）李明，这不是幻觉。我是来自未来的你。
```

---

## 📱 场景 2: 多平台内容适配

### 激活技能
```
请使用 platform-adaptation 技能帮我改编这篇文章到小红书
```

### 平台特性对照表

| 平台 | 内容类型 | 最佳长度 | 关键特征 |
|------|----------|----------|----------|
| **小红书** | 生活化、视觉、个人 | 800-2000字 | Emoji丰富、亲切口语、标题党、干货标签 |
| **微信公众号** | 深度、专业 | 2000-5000字 | 正式、权威、结构完整、可引用 |
| **知乎** | 专业、分析 | 1500-4000字 | 逻辑严密、数据支撑、专业术语 |
| **番茄小说** | 小说连载 | 2000-3000字/章 | 快节奏、强冲突、悬念结尾 |
| **抖音/B站** | 视频脚本 | 1-3分钟 | 口语化、快节奏、强hook |
| **网站/博客** | 技术深度 | 3000-8000字 | 完整、专业、代码示例 |

### 示例：同一内容的多平台改编

**原始内容**：一篇技术博客《RAG技术原理与实战》

#### 小红书版本特点
```markdown
# 🤖 RAG技术太绝了！扒一皮大模型知识检索

姐妹们！今天给大家分享一个超火的AI技术～
（emoji: 📚✨💡🔥）

## 1️⃣ 什么是RAG？
简单说就是给大模型外挂大脑！
（口语化表达）
...
## 2️⃣ 核心原理（超简单）
...
## 3️⃣ 实战手把手
...
## 📚 标签
#AI技术 #RAG #大模型 #干货分享
```

#### 微信公众号版本特点
```markdown
# RAG技术深度解析：从原理到生产级实践

## 摘要
本文深入探讨检索增强生成（RAG）技术的核心原理...

## 一、RAG技术概述
### 1.1 背景与挑战
（专业术语、数据引用）
...
## 二、核心架构设计
...
## 参考文献
[1] Lewis et al. (2020). Retrieval-Augmented Generation...
```

---

## 🛠️ 场景 3: 技术博客/教程创作

### 推荐技能组合
```
使用 content-research-writer + writing-clearly-and-concisely
```

### 工作流程
```
1. content-research-writer: 收集资料、生成大纲
2. writing-clearly-and-concisely: 确保表达清晰简洁
3. copy-editing: 最终润色
```

### 示例对话
```
你：帮我写一篇关于"LangGraph多Agent系统"的技术教程

Claude 会：
1. 使用 content-research-writer 进行背景研究
2. 创建详细大纲
3. 逐节撰写内容（含代码示例）
4. 应用 writing-clearly-and-concisely 规则优化表达
5. 使用 copy-editing 进行七轮编辑检查
```

---

## 🏆 场景 4: 小说创作（使用 Novel-OS）

### Novel-OS 三层工作流

#### 第1层：写作标准（设置一次，到处使用）
```bash
# 位置
~/.novel-os/standards/writing-style.md
~/.novel-os/standards/narrative-techniques.md

# 内容定义
- 叙事视角：第三人称限制视角
- 文字风格：简洁有力，避免形容词堆砌
- 对话风格：自然口语，每个人物有独特声音
```

#### 第2层：小说设定（项目特定）
```bash
# 位置
your-novel-project/.novel-os/novel/

# 包含文件
- premise.md: 故事前提
- writing-plan.md: 写作计划
- decisions.md: 创意决策记录
- writing-style.md: 这本书特有的风格
```

#### 第3层：手稿细节
```bash
# 位置
your-novel-project/.novel-os/manuscripts/YYYY-MM-DD-story/

# 包含文件
- story-outline.md: 完整故事大纲
- character-profiles.md: 人物档案
- tasks.md: 逐场写作任务
```

### 使用示例
```
你：我想开始写一部科幻小说，设定是2077年，人类实现了意识上传

Claude 会：
1. 使用 Novel-OS 的 plan-novel 工作流
2. 创建小说项目结构
3. 生成完整的故事前提
4. 制定写作计划（分阶段目标）
5. 创建人物档案和世界观设定
```

---

## 🔄 技能协同使用示例

### 完整内容生产流程

```
主题：《AI写作工具实战指南》

Step 1: 研究与大纲
→ content-research-writer（收集资料）
→ 生成详细大纲

Step 2: 长文撰写
→ content-research-writer（逐节撰写）
→ writing-clearly-and-concisely（优化表达）
→ 添加代码示例和实战案例

Step 3: 编辑润色
→ copy-editing（七轮编辑）
→ fact-check（技术准确性）

Step 4: 多平台改编
→ platform-adaptation（同时生成）
  - 小红书版本（800字，emoji丰富）
  - 微信公众号版本（3000字，深度分析）
  - 知乎版本（2000字，专业讨论）
  - 抖音脚本版本（2分钟，口语化）

Step 5: 营销优化（可选）
→ marketing-psychology（标题优化）
→ copywriting（营销文案）
```

---

## 🎯 快速开始指南

### 写小说
```
"使用 Novel-OS 帮我规划一部都市悬疑小说"
```

### 写剧本
```
"使用 scriptwriting 技能创作一个三分钟短剧剧本"
```
- **核心优势**: 格式标准化，人物系统完整，对话技巧专业
- **适用场景**: 短视频剧本、漫剧剧本、微电影剧本

### 写技术教程
```
"使用 content-research-writer 和 writing-clearly-and-concisely 帮我写一篇LangChain教程"
```
- **核心优势**: 研究深入，逻辑清晰，表达准确
- **适用场景**: 技术博客、实战教程、深度文章

### 适配到小红书
```
"使用 platform-adaptation 技能把这篇文章改编成小红书笔记"
```
- **核心优势**: 算法优化，格式适配，标题党技巧
- **适用场景**: 所有内容平台之间的转换

### 优化营销文案
```
"使用 copywriting 和 marketing-psychology 优化这个产品页面"
```
- **核心优势**: 转化导向，心理触发点，A/B测试框架
- **适用场景**: 落地页、产品介绍、广告文案

---

## 📊 技能选择决策树

```
开始写作
    │
    ├─ 写小说？
    │   └─→ Novel-OS (完整系统)
    │
    ├─ 写剧本？
    │   └─→ scriptwriting (中文剧本格式)
    │
    ├─ 写技术内容？
    │   ├─ 深度教程 → content-research-writer
    │   └─ 简单说明 → writing-clearly-and-concisely
    │
    ├─ 改编到平台？
    │   └─→ platform-adaptation (所有中文平台)
    │
    ├─ 营销文案？
    │   ├─ 产品文案 → copywriting
    │   └─ 心理优化 → marketing-psychology
    │
    └─ 社交媒体？
        └─→ social-content (全平台策略)
```

---

## 💡 高级技巧

### 1. 组合技能
```
"使用 content-research-writer 研究并撰写文章，
然后用 copy-editing 编辑，
最后用 platform-adaptation 改编到小红书和公众号"
```

### 2. 自定义风格
```
"在编写时，请参考我~/.novel-os/standards/writing-style.md中定义的风格"
```

### 3. 批量生产
```
"帮我为同一主题生成5个小红书版本、3个公众号版本、2个知乎版本"
```

---

## 🎉 现在你可以

✅ 使用 Novel-OS 创作结构化小说
✅ 使用 scriptwriting 创作专业剧本
✅ 使用 platform-adaptation 一键适配所有中文平台
✅ 使用 content-research-writer 撰写深度技术文章
✅ 使用 copywriting 提升转化率
✅ 使用 marketing-psychology 优化用户心理体验

**开始你的创作之旅吧！** 🚀

---

*生成时间: 2026-01-27*
*系统版本: ContentForge AI Writing Skills Ecosystem v1.0*
