# 写作技能速查表

## 🎯 技能触发词

直接在对话中提及这些关键词，Claude会自动激活对应技能：

| 触发词 | 技能名称 |
|--------|----------|
| "剧本" "script" "screenplay" | **scriptwriting** |
| "小红书" "公众号" "知乎" "平台适配" | **platform-adaptation** |
| "小说" "novel" "fiction" | **Novel-OS** |
| "清晰写作" "简洁" "writing clearly" | **writing-clearly-and-concisely** |
| "研究" "research" "长文" "深度文章" | **content-research-writer** |
| "文案" "copywriting" "转化" | **copywriting** |
| "编辑" "edit" "润色" | **copy-editing** |
| "社交媒体" "social" "viral" | **social-content** |
| "心理" "psychology" "用户行为" | **marketing-psychology** |

---

## 📱 平台适配速查

### 小红书
- **长度**: 800-2000字
- **风格**: 亲切、emoji丰富、口语化
- **结构**: ## 1️⃣ 2️⃣ 3️⃣ 编号章节
- **关键词**: "太绝了" "扒一皮" "姐妹们" "干货"
- **标签**: #标签1 #标签2 #标签3

### 微信公众号
- **长度**: 2000-5000字
- **风格**: 专业、权威、正式
- **结构**: 完整摘要、参考文献
- **特点**: 可引用、适合转发

### 知乎
- **长度**: 1500-4000字
- **风格**: 专业、数据支撑
- **结构**: 逻辑严密、专业术语
- **特点**: 引用论文、数据图表

### 番茄小说
- **长度**: 2000-3000字/章
- **风格**: 快节奏、强冲突
- **结构**: 悬念结尾、情绪起伏
- **特点**: 爽文路线、代入感强

---

## 🎬 剧本格式速查

### 场景标题
```
第X场
地点：内景/外景 具体地点 - 时间
人物：角色A、角色B
```

### 对话格式
```
角色名：（神态/动作）对话内容
（动作或反应描述）
```

### 转场
```
→ 切至
→ 淡出
→ 闪回
```

---

## 📝 小说创作检查清单

### Novel-OS 三层检查
- [ ] **Layer 1**: 写作标准已定义 (`~/.novel-os/standards/`)
- [ ] **Layer 2**: 小说设定已完成 (`.novel-os/novel/`)
- [ ] **Layer 3**: 手稿大纲已创建 (`.novel-os/manuscripts/`)

### 人物档案检查
- [ ] 外貌特征
- [ ] 性格特点
- [ ] 背景故事
- [ ] 对话风格（独特声音）
- [ ] 人物弧光

### 场景检查
- [ ] 场景目标
- [ ] 冲突点
- [ ] 人物变化
- [ ] 情绪曲线

---

## 🔄 内容生产工作流

```
┌─────────────────┐
│  原始想法/主题  │
└────────┬────────┘
         │
    ┌────▼────┐
    │ 研究阶段 │ ← content-research-writer
    └────┬────┘
         │
    ┌────▼────┐
    │ 创作阶段 │ ← writing-clearly-and-concisely
    └────┬────┘
         │
    ┌────▼────┐
    │ 编辑阶段 │ ← copy-editing
    └────┬────┘
         │
    ┌────▼────┐
    │ 适配阶段 │ ← platform-adaptation
    └────┬────┘
         │
    ┌────▼────┐
    │ 多平台发布 │
    │  ├─ 小红书 │
    │  ├─ 公众号 │
    │  ├─ 知乎  │
    │  └─ 小说  │
    └─────────┘
```

---

## 💬 示例对话模板

### 创作剧本
```
你：使用 scriptwriting 技能，创作一个3分钟的都市爱情短剧，
男主是程序员，女主是设计师，两人因为AI配对软件相遇。

Claude会：
1. 按照中文剧本格式
2. 创建3-5个场景
3. 包含人物、对话、动作
4. 控制时长在3分钟
```

### 适配平台
```
你：使用 platform-adaptation 技能，把这篇技术博客改编成：
1. 小红书版本（1000字，emoji丰富）
2. 公众号版本（深度分析）
3. 知乎版本（专业讨论）

Claude会：
1. 分析原文核心内容
2. 按平台特性重组
3. 优化标题和开头
4. 添加平台特定元素
```

### 撰写技术文章
```
你：使用 content-research-writer 撰写一篇《LangGraph实战》教程，
目标读者是Python开发者，包含完整代码示例。

Claude会：
1. 研究最新资料
2. 创建详细大纲
3. 逐节撰写内容
4. 添加代码示例
5. 确保技术准确性
```

---

## 🎯 快速决策

**我想...** → **使用...**

写小说 → Novel-OS
写剧本 → scriptwriting
写技术教程 → content-research-writer + writing-clearly-and-concisely
写营销文案 → copywriting + marketing-psychology
适配小红书 → platform-adaptation
适配公众号 → platform-adaptation
编辑润色 → copy-editing
社交媒体 → social-content

---

**Created**: 2026-01-27
**System**: ContentForge AI Writing Skills Ecosystem
