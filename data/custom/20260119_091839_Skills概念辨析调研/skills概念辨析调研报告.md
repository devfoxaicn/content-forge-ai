# Skills 概念辨析调研报告

**调研日期**: 2025年1月19日
**调研主题**: Skills 与易混淆概念的辨析
**调研目的**: 明确 Skills 在 AI Agent 体系中的定位，厘清与其他相关概念的区别

---

## 一、核心概念对比表

| 概念 | 本质定位 | 主要作用 | 典型场景 |
|------|---------|---------|---------|
| **Skills (技能)** | 知识封装 | 教AI"懂什么" - 领域专业知识 | 代码规范、业务流程、最佳实践 |
| **Tools (工具)** | 功能封装 | 让AI"能做什么" - 执行操作 | 查天气、发邮件、搜索API |
| **Functions (函数调用)** | 基础API能力 | 结构化外部函数调用 | API开发、系统集成 |
| **Plugins (插件)** | UI层扩展 | ChatGPT界面功能增强 | ChatGPT产品内使用 |
| **MCP** | 标准化协议 | 工具和数据接入标准 | 跨平台工具对接 |
| **Prompts (提示词)** | 交互指令 | 一次性任务描述 | 临时指令输入 |

---

## 二、详细辨析

### 2.1 Skills vs Tools (工具)

这是最常被混淆的一对概念。

| 维度 | Skills (技能) | Tools (工具) |
|------|--------------|-------------|
| **本质** | 知识封装 | 执行动作 |
| **核心能力** | 教AI"懂什么" | 让AI"能做什么" |
| **实现方式** | 高级Prompt + 上下文扩展 | 函数调用/API执行 |
| **技术基础** | 提示词工程 | 函数编程 |
| **典型例子** | 代码规范、业务流程知识、写作风格 | 查天气、发邮件、数据库查询 |
| **调用方式** | AI自动判断是否使用 | 结构化函数调用 |
| **返回结果** | 增强的上下文理解 | 具体操作结果 |

**核心区别口诀**：
> Tools 是"手"，Skills 是"脑"
> —— Tools执行操作，Skills提供专业知识

**实战示例**：
```
场景：让AI处理代码审查

Tool方式：
- 提供代码执行环境
- 运行测试用例
- 执行静态分析工具

Skill方式：
- 教AI团队代码规范
- 传授最佳实践模式
- 识别常见反模式
```

---

### 2.2 Skills vs MCP (Model Context Protocol)

| 维度 | Skills | MCP |
|------|--------|-----|
| **定位** | 知识模块 | 标准化协议 |
| **组成** | 领域专家知识 | Tools、Prompts、Resources三种原语 |
| **类比** | 专业技能书 | USB-C接口标准 |
| **关系** | 可配合MCP使用 | 提供工具和数据接入 |
| **适用场景** | 封装领域知识 | 标准化系统对接 |

**核心区别**：
> MCP 是"接口标准"，Skills 是"专业知识包" —— 两者互补

**组合使用示例**：
```
一个办公自动化Agent可能需要：

MCP部分：
- 标准化接入飞书API
- 统一数据源格式
- 跨系统消息传递

Skills部分：
- 企业内部审批流程规范
- 业务术语定义
- 数据分析专业知识
```

---

### 2.3 Skills vs Plugins (插件)

| 维度 | Skills | Plugins |
|------|--------|---------|
| **使用范围** | 任何支持Skills的AI系统 | 仅限ChatGPT等特定产品 |
| **集成方式** | 提示词扩展 | UI界面集成 |
| **可见性** | 透明调用 | 用户可见安装 |
| **灵活性** | 高度灵活，跨平台 | 绑定特定产品 |
| **开发门槛** | 编写skill.md文件 | 需要开发完整插件应用 |
| **维护成本** | 低 | 高 |

**核心区别**：
> Skills 是通用能力，Plugins 是产品功能扩展

**演变趋势**：
```
Plugins (2023) → Tools (2024) → Skills (2025)

Plugins正逐渐被更通用的Skills和MCP方案取代
```

---

### 2.4 Skills vs Prompts (提示词)

| 维度 | Skills | Prompts |
|------|--------|---------|
| **复用性** | 模块化，可重复调用 | 通常一次性使用 |
| **结构** | 有标准的skill.md格式 | 自由文本 |
| **触发机制** | AI自动判断触发 | 手动输入 |
| **知识密度** | 高度浓缩领域知识 | 简单指令 |
| **维护性** | 版本管理、易于更新 | 难以复用和维护 |
| **Token效率** | 高（可节省98%Token） | 低（每次重复输入） |

**核心区别**：
> Skills 是"工程化的Prompt"，具备复用性和自动触发能力

**对比示例**：
```
Prompt方式（每次都要输入）：
"请按照以下规范写Python代码：使用类型注解、函数名用蛇形命名法、
类名用帕斯卡命名法、每个函数都要有docstring..."

Skills方式（自动调用）：
定义skill.md，AI在写Python代码时自动应用这些规范
```

---

## 三、概念关系图

```
                    AI Agent 能力扩展
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
    执行层              知识层              接口层
        │                  │                  │
   Tools/Functions      Skills             MCP
        │                  │                  │
   调用外部API        领域专业知识        标准化协议
   执行具体操作       教AI"懂什么"       统一工具接入
        │                  │
        └──────────┬───────┘
                   │
              Plugins (产品层)
              ChatGPT界面扩展
```

---

## 四、进化路径

```
LLM工具调用进化史：

2023年：Functions (OpenAI)
  └─ 基础函数调用能力

2024年：Tools & Plugins
  ├─ Tools: Functions的超集
  └─ Plugins: ChatGPT UI扩展

2025年：MCP & Skills
  ├─ MCP: 标准化协议
  └─ Skills: 知识封装

未来趋势：MCP + Skills = 完整Agent能力
```

---

## 五、实战场景对比

### 场景1：让AI帮写代码

| 方案 | 实现 | 适用情况 | 优势 | 劣势 |
|------|------|---------|------|------|
| **Tool** | 提供代码执行环境，让AI运行代码 | 需要实际执行验证 | 可真实运行 | 需要基础设施 |
| **Skill** | 教AI团队代码规范、最佳实践 | 保证代码风格一致性 | 高效复用、节省Token | 需要预先配置 |
| **Prompt** | 每次手动输入"请写Python函数" | 临时需求 | 灵活随意 | 重复劳动、Token浪费 |

### 场景2：让AI处理业务数据

| 方案 | 实现 | 适用情况 | 优势 |
|------|------|---------|------|
| **Tool** | 连接数据库API执行查询 | 需要实时数据 | 准确可靠 |
| **Skill** | 教AI业务逻辑、数据字典 | 理解业务含义 | 智能分析 |
| **MCP** | 标准化数据源接入 | 跨系统对接 | 统一标准 |

### 场景3：ChatGPT用户

| 方案 | 实现 | 适用情况 | 优势 |
|------|------|---------|------|
| **Plugin** | 安装官方插件，界面可见 | ChatGPT网页版用户 | 易于使用 |
| **Skill** | 通过Claude Code等工具使用 | 开发者/专业用户 | 跨平台通用 |

---

## 六、选择决策树

```
你的需求是什么？
│
├─ 需要调用外部系统/执行操作
│   └─ 使用 Tools/Functions
│
├─ 需要教AI领域知识/业务规范
│   └─ 使用 Skills
│
├─ 需要在ChatGPT界面使用
│   └─ 使用 Plugins
│
├─ 需要标准化对接多个工具
│   └─ 使用 MCP
│
└─ 临时一次性指令
    └─ 使用 Prompts
```

---

## 七、关键记忆点

### 7.1 核心区别口诀

```
Tools 是手 —— 执行操作
Skills 是脑 —— 专业知识
MCP 是线 —— 连接标准
Plugins 是脸 —— 产品界面
Prompts 是话 —— 临时指令
```

### 7.2 记忆技巧

| 概念 | 记忆关键词 |
|------|-----------|
| Skills | **知识**、规范、懂什么 |
| Tools | **操作**、API、能做什么 |
| MCP | **协议**、标准、USB |
| Plugins | **界面**、ChatGPT、安装 |
| Functions | **调用**、JSON、参数 |
| Prompts | **对话**、指令、一次性 |

### 7.3 组合使用示例

**一个完整的办公自动化Agent**：

```
需要的能力：

1. Tools（执行层）：
   - 调用飞书API发送消息
   - 查询数据库获取数据
   - 执行Excel操作

2. Skills（知识层）：
   - 企业内部审批流程规范
   - 业务术语和数据字典
   - 数据分析专业知识
   - 报告写作规范

3. MCP（协议层）：
   - 标准化接入多个数据源
   - 统一工具调用接口

4. Prompts（交互层）：
   - 用户描述具体任务需求
```

---

## 八、调研结论

### 8.1 核心发现

1. **Skills 不是 Tools 的替代品**，而是互补关系
   - Tools 解决"能做什么"（执行能力）
   - Skills 解决"懂什么"（知识能力）

2. **Skills 具有独特优势**
   - Token效率极高（可节省98%）
   - 无需编写代码
   - 跨平台通用

3. **MCP 是未来趋势**
   - 标准化协议促进生态发展
   - 与Skills形成完整解决方案

### 8.2 实践建议

1. **优先使用 Skills 的场景**：
   - 需要传递领域专业知识
   - 有明确的标准和规范
   - 需要频繁重复使用

2. **优先使用 Tools 的场景**：
   - 需要执行实际操作
   - 需要调用外部API
   - 需要获取实时数据

3. **最佳实践**：
   - Skills + Tools 组合使用
   - MCP 标准化工具接入
   - Prompts 作为补充交互方式

### 8.3 发展展望

```
当前阶段 (2025)：
Skills 作为新兴概念，正在快速发展

未来趋势：
- Skills 标准化（通用格式）
- Skills 市场（可交易的技能包）
- Skills + MCP 深度融合
- AI Agent "技能化"成为常态
```

---

## 九、参考资料

### 官方文档
- [Claude Skills - Claude Code Docs](https://code.claude.com/docs/en/skills)
- [MCP (Model Context Protocol) 官方文档](https://modelcontextprotocol.io/)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)

### 技术文章
- [深入解析Function Calling、MCP和Skills的本质差异](https://zhuanlan.zhihu.com/p/1992172117328950705)
- [LLM工具调用进化史：从Function Call到Agent Skills](https://zhuanlan.zhihu.com/p/1980618218222683718)
- [Claude Skills vs MCP: A Technical Comparison](https://intuitionlabs.ai/articles/claude-skills-vs-mcp)
- [别搞混了！MCP和Agent Skill到底有什么区别](https://juejin.cn/post/7584057497205817387)
- [十分钟搞清楚Agent、MCP和Skills的概念与区别](https://www.51cto.com/article/834160.html)
- [Agent Skills 与MCP：智能体能力扩展的两种范式](https://github.com/datawhalechina/hello-agents)
- [Claude Agent Skills第一性原理深度解析](https://m.aitntnews.com/newDetail.html?newId=20299)
- [从第一性原理深度拆解Claude Agent Skill](https://baoyu.io/translations/claude-skills-deep-dive)

### 视频教程
- [Agent Skills与MCP Tools，搞清它们的关系仅需5分钟](https://www.bilibili.com/video/BV1jgr2BtEa7/)

---

**报告编写**: AI调研助手
**最后更新**: 2025年1月19日
**版本**: v1.0
