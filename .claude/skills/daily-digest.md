---
name: daily-digest
description: 一键生成AI新闻简报并提交到GitHub。自动从14个顶级AI媒体源获取最新资讯，使用LLM批量翻译生成高质量中文简报，然后自动提交到GitHub仓库。
---

# AI Daily Digest - 一键生成简报并提交GitHub

自动执行完整流程：
1. 从14个顶级AI媒体源获取最新资讯
2. 按分类组织热点（产业动态、学术前沿、技术创新、产品工具、行业应用）
3. 使用LLM批量翻译生成高质量中文简报
4. 保存到 `data/daily/YYYYMMDD/digest/` 目录
5. 自动提交并推送到GitHub

## 使用方法

### 方式一：直接运行脚本（推荐）

```bash
python scripts/daily_digest.py
```

或使用PYTHONPATH：

```bash
PYTHONPATH=/Users/z/Documents/work/content-forge-ai python scripts/daily_digest.py
```

### 方式二：使用main.py

```bash
python src/main.py --mode auto --once
```

然后在完成后手动提交：

```bash
git add -A
git commit -m "docs: AI每日热点"
git push origin main
```

## 输出

- **简报文件**: `data/daily/YYYYMMDD/digest/digest_YYYYMMDD.md`
- **元数据**: `data/daily/YYYYMMDD/digest/digest_YYYYMMDD.json`
- **字数**: 约13,000-15,000字
- **处理时间**: 约9-10分钟（含LLM翻译）

## 执行流程

```
Step 1/3: 生成AI新闻简报
  ↓
Step 2/3: 提交到Git
  ↓
Step 3/3: 完成
```

## 配置

相关配置在 `config/config.yaml`:

```yaml
agents:
  world_class_digest:
    translate_enabled: true   # 启用翻译
    batch_size: 5             # 批量处理大小
    max_items_per_category: 15  # 每个分类最多显示数量
```

## 环境变量

确保 `.env` 文件中包含：

```bash
ZHIPUAI_API_KEY=your_api_key_here
```

## GitHub

自动提交并推送到：https://github.com/devfoxaicn/content-forge-ai
