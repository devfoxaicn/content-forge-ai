# 内容版本管理规范

> **版本**: v1.0
> **创建日期**: 2026-02-13

本文档定义了 ContentForge AI 系列文章的版本管理规范。

---

## 📋 版本号格式

采用语义化版本（Semantic Versioning）：`MAJOR.MINOR.PATCH`

- **MAJOR**: 重大内容重构（如章节重组、核心观点修正）
- **MINOR**: 内容增强（如新增案例、补充代码示例）
- **PATCH**: 小幅修订（如错别字修正、链接更新）

**示例**：
- `1.0.0` - 初始版本
- `1.1.0` - 新增代码示例
- `1.1.1` - 修正错别字
- `2.0.0` - 章节重组

---

## 📝 Episode Metadata 结构

每篇文章的 `episode_metadata.json` 应包含版本信息：

```json
{
  "episode": 96,
  "title": "扩散模型Diffusion Models深入",
  "series_id": "ml_series_10",
  "version": "1.0.0",
  "status": "completed",
  "created_at": "2026-02-13T11:07:00+08:00",
  "updated_at": "2026-02-13T11:35:00+08:00",
  "word_count": 47021,
  "quality_score": 8.5,
  "changelog": [
    {
      "version": "1.0.0",
      "date": "2026-02-13",
      "changes": ["初始版本发布"],
      "author": "ContentForge AI"
    }
  ],
  "validation": {
    "code_review": true,
    "fact_check": true,
    "quality_evaluated": true
  },
  "references": {
    "papers": ["DDPM (2020)", "Stable Diffusion (2022)"],
    "github_repos": ["CompVis/stable-diffusion"],
    "documentation": ["https://huggingface.co/docs/diffusers"]
  }
}
```

---

## 🔄 更新流程

### 1. 内容更新

当需要更新文章时：

```bash
# 1. 更新文章内容
# 2. 更新 metadata
# 3. 添加 changelog 条目
```

### 2. Changelog 格式

```json
{
  "version": "1.1.0",
  "date": "2026-02-15",
  "changes": [
    "新增 DiT 架构章节",
    "补充 Sora 视频生成案例",
    "修正公式 (3.2) 系数"
  ],
  "author": "ContentForge AI"
}
```

### 3. 版本递增规则

| 变更类型 | 版本递增 | 示例 |
|---------|---------|------|
| 新增章节/案例 | MINOR | 1.0.0 → 1.1.0 |
| 修正错别字 | PATCH | 1.0.0 → 1.0.1 |
| 章节重组 | MAJOR | 1.0.0 → 2.0.0 |
| 代码示例更新 | MINOR | 1.0.0 → 1.1.0 |
| 链接/引用更新 | PATCH | 1.0.0 → 1.0.1 |

---

## 📊 Series Metadata 结构

每个系列的 `series_metadata.json` 应包含：

```json
{
  "series_id": "ml_series_10",
  "series_name": "高级机器学习专题系列",
  "version": "1.0.0",
  "total_episodes": 10,
  "completed_episodes": 10,
  "created_at": "2026-01-31",
  "updated_at": "2026-02-13",
  "episodes": [
    {
      "episode": 91,
      "title": "图神经网络GNN基础",
      "version": "1.0.0",
      "status": "completed",
      "completed_at": "2026-01-31"
    }
  ],
  "changelog": [
    {
      "version": "1.0.0",
      "date": "2026-02-13",
      "changes": ["系列全部完成"]
    }
  ]
}
```

---

## 🔍 版本查询 API

### Python 接口

```python
from src.utils.series_manager import SeriesVersionManager

# 初始化
vm = SeriesVersionManager("ml_series_10")

# 获取系列版本
print(vm.get_version())  # "1.0.0"

# 获取文章版本
print(vm.get_episode_version(96))  # "1.0.0"

# 获取更新历史
history = vm.get_changelog(96)
for entry in history:
    print(f"v{entry['version']}: {entry['changes']}")

# 更新版本
vm.update_version(96, "1.1.0", ["新增案例研究"])
```

### 命令行接口

```bash
# 查看系列版本
python src/main.py --mode series --version-info --series ml_series_10

# 查看文章版本
python src/main.py --mode series --version-info --episode 96

# 更新版本
python src/main.py --mode series --update-version --episode 96 --version 1.1.0 --note "新增案例"
```

---

## 📈 版本统计

### 按版本统计

```bash
# 查看所有 v1.x.x 版本
python scripts/version_stats.py --major 1

# 查看最近更新的文章
python scripts/version_stats.py --recent 7
```

### 版本分布报告

```markdown
## ML Series 版本分布

| 版本 | 文章数 | 占比 |
|------|-------|------|
| 1.0.x | 95 | 95% |
| 1.1.x | 5 | 5% |
| 2.0.x | 0 | 0% |

**最后更新**: 2026-02-13
```

---

## ⚠️ 注意事项

1. **向后兼容**: MINOR 和 PATCH 更新应保持向后兼容
2. **破坏性变更**: MAJOR 更新需要通知所有读者
3. **备份**: 更新前备份原版本
4. **审核**: 重大更新需要人工审核

---

## 📚 参考资料

- [Semantic Versioning 2.0.0](https://semver.org/)
- [Keep a Changelog](https://keepachangelog.com/)

---

**维护者**: ContentForge AI
**最后更新**: 2026-02-13
