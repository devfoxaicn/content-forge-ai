# GitHub Actions 验证清单

**日期**: 2026-01-27
**版本**: v8.0
**目的**: 确保明天（1月28日）的3次定时任务能够正确执行

---

## ✅ 已验证项目

### 1. 代码版本
- [x] GitHub上的最新提交: `39d7583` (v8.0优化版本)
- [x] 包含最新的prompts优化
- [x] 包含NewsScoringAgent v8.0增强
- [x] 包含WorldClassDigestAgent v8.0优化

### 2. Workflow配置
- [x] 文件路径: `.github/workflows/daily-digest.yml`
- [x] 定时任务配置正确:
  - 早上6点: `cron: '0 22 * * *'` (UTC 22:00 = 北京时间 06:00)
  - 中午12点: `cron: '0 4 * * *'` (UTC 04:00 = 北京时间 12:00)
  - 晚上6点: `cron: '0 10 * * *'` (UTC 10:00 = 北京时间 18:00)
- [x] 权限配置: `contents: write` (允许自动提交)
- [x] 超时时间: 90分钟
- [x] 支持手动触发: `workflow_dispatch`

### 3. 依赖配置
- [x] `requirements.txt` 包含所有必需依赖
- [x] Python版本: 3.10
- [x] LangChain & LangGraph版本使用范围配置

---

## ⚠️ 需要用户确认的项目

### GitHub Secrets配置

**请访问以下URL确认secrets已配置**:
```
https://github.com/devfoxaicn/content-forge-ai/settings/secrets/actions
```

**必需的Secrets**:

1. **ZHIPUAI_API_KEY** (必需)
   - 获取地址: https://open.bigmodel.cn/
   - 用途: 智谱AI GLM-4.7模型调用
   - 状态: ⚠️ 需要确认

2. **NEWSAPI_KEY** (可选)
   - 获取地址: https://newsapi.org/
   - 用途: NewsAPI数据源
   - 状态: ⚠️ 可选，但建议配置以获得更多数据源

3. **GITHUB_TOKEN** (自动提供)
   - GitHub自动提供，无需手动配置
   - 状态: ✅ 自动管理

---

## 📋 明天任务时间表

**2026年1月28日（明天）**:

| 时间（北京时间） | 时间（UTC） | 任务编号 | 预期内容 |
|-----------------|-------------|----------|----------|
| 06:00 | 22:00* | Run #19+ | AI每日热点简报（v8.0标准） |
| 12:00 | 04:00 | Run #20+ | AI每日热点简报（v8.0标准） |
| 18:00 | 10:00 | Run #21+ | AI每日热点简报（v8.0标准） |

*注：早上6点的任务实际上是在前一天UTC 22:00执行，即北京时间当天早上6点前。 |

---

## 🧪 测试方法

### 方法1: GitHub网页手动触发（推荐）

1. 访问: https://github.com/devfoxaicn/content-forge-ai/actions
2. 点击左侧 "AI Daily Digest Generator"
3. 点击右侧 "Run workflow" 按钮
4. 选择分支 "main"
5. 点击绿色 "Run workflow" 按钮
6. 等待执行完成（约5-10分钟）

### 方法2: GitHub CLI

```bash
# 安装GitHub CLI（如果未安装）
brew install gh

# 登录GitHub
gh auth login

# 触发workflow
gh workflow run daily-digest.yml
```

### 方法3: 本地测试

```bash
# 设置环境变量
export ZHIPUAI_API_KEY="your_api_key_here"
export NEWSAPI_KEY="your_newsapi_key_here"  # 可选
export PYTHONPATH=/Users/z/Documents/work/content-forge-ai

# 运行auto模式
python src/main.py --mode auto --once
```

---

## 📊 验证检查点

**任务执行后，请检查以下内容**:

### 1. GitHub Actions执行状态
- 访问: https://github.com/devfoxaicn/content-forge-ai/actions
- 确认状态: ✅ success (绿色) / ❌ failed (红色)

### 2. 生成的内容文件
- 路径: `data/daily/YYYYMMDD/digest/digest_YYYYMMDD_HHMMSS.md`
- 检查:
  - [ ] 文件是否生成
  - [ ] 内容是否包含"核心洞察"
  - [ ] 内容是否包含"深度观察"
  - [ ] 分类热点是否完整

### 3. 自动提交
- 检查: https://github.com/devfoxaicn/content-forge-ai/commits/main
- 确认:
  - [ ] 有新的自动提交 (作者: github-actions[bot])
  - [ ] 提交信息格式: "feat: AI内容自动生成 - YYYY-MM-DD"

### 4. v8.0内容质量检查
- [ ] 标题简洁有力（不超过30字）
- [ ] 专业术语保留正确（LLM、RAG、Agent等）
- [ ] 摘要精炼准确（60-100字）
- [ ] 核心洞察有深度（不泛泛而谈）
- [ ] 使用有力动词（"标志着"、"重塑了"、"颠覆了"）

---

## 🔧 故障排查

### 如果任务失败

1. **检查Actions日志**:
   - 访问具体的workflow run页面
   - 查看哪一步失败（Install dependencies / Run daily digest / Commit and push）

2. **常见问题**:

   **问题1: API Key错误**
   - 症状: 日志显示 "Authentication failed" 或 "Invalid API key"
   - 解决: 检查GitHub Secrets中的ZHIPUAI_API_KEY是否正确

   **问题2: 依赖安装失败**
   - 症状: Install dependencies步骤失败
   - 解决: 检查requirements.txt，确认依赖兼容性

   **问题3: 生成超时**
   - 症状: Run daily digest步骤超时（超过90分钟）
   - 解决: 考虑减少数据源或使用更快的模型（glm-4-flash）

   **问题4: 没有新数据提交**
   - 症状: 任务成功但无新commit
   - 解决: 可能数据源没有更新，或生成内容与之前相同

### 获取帮助

- 查看详细日志: GitHub Actions → 点击具体run → 查看各步骤日志
- 查看系统日志: `logs/$(date +%Y%m%d)/app.log`

---

## 📝 版本记录

- **v8.0** (2026-01-27):
  - 融合TechCrunch、The Verge、36氪、虎嗅写作风格
  - AI关键词识别系统
  - 技术趋势敏感度评分
  - 九维质量评估系统
  - 翻译质量提升（顶级科技媒体标准）

---

**维护者**: ContentForge AI Team
**文档版本**: v1.0
**最后更新**: 2026-01-27 23:30
