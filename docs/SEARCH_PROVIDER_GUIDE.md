# 搜索提供商切换指南

## 🎯 使用场景建议

### 日常生成：zhipuAI（推荐）
**适合场景**：
- 日常批量生成技术博客
- 成本敏感型项目（已包含在年包中）
- 国内访问要求高的场景

**优势**：
- ✅ 无额外费用（年包）
- ✅ 配置简单，无需额外依赖
- ✅ 国内访问稳定

**配置方式**：
```yaml
# config/config.yaml
research_agent:
  search_provider: "zhipuai"
```

---

### 高质量需求：Tavily
**适合场景**：
- 需要最高质量搜索结果
- 重要主题/核心章节
- 需要精确引用和溯源
- 免费额度充足（1000次/月）

**优势**：
- ✅ 搜索结果更丰富
- ✅ 专业搜索引擎API
- ✅ 支持自定义域名过滤
- ✅ 免费额度：1000 credits/月

**配置方式**：
```yaml
# config/config.yaml
research_agent:
  search_provider: "tavily"
```

**前提条件**：
- 已安装：`pip install tavily-python` ✅
- 已配置：`.env` 中添加 `TAVILY_API_KEY=tvly-your-key-here`

---

## 🔄 快速切换命令

### 切换到 Tavily
```bash
# 方法1：手动编辑
vim config/config.yaml
# 修改 search_provider: "tavily"

# 方法2：使用sed命令
sed -i '' 's/search_provider: "zhipuai"/search_provider: "tavily"/' config/config.yaml
```

### 切换回 zhipuAI
```bash
# 方法1：手动编辑
vim config/config.yaml
# 修改 search_provider: "zhipuai"

# 方法2：使用sed命令
sed -i '' 's/search_provider: "tavily"/search_provider: "zhipuai"/' config/config.yaml
```

---

## 📊 使用成本对比

### 场景：100期技术博客项目

| 提供商 | 每期消耗 | 月生成10期 | 月生成50期 | 免费额度 |
|--------|----------|------------|------------|----------|
| **zhipuAI** | 0次调用 | 0元 | 0元 | 无限制（年包） |
| **Tavily** | 1 credit | 10 credits | 50 credits | 1000 credits/月 |

**结论**：
- zhipuAI：无额外成本（年包已覆盖）
- Tavily：免费额度绰绰有余，可生成1000期/月

---

## 🎬 实战建议

### 推荐工作流

1. **日常批量生成**：使用 `zhipuai`
   ```bash
   # 配置zhipuai，批量生成Series 1
   sed -i '' 's/search_provider: "tavily"/search_provider: "zhipuai"/' config/config.yaml
   PYTHONPATH=/Users/z/Documents/work/content-forge-ai python src/main.py --mode series --series series_1
   ```

2. **重点章节精修**：切换到 `tavily`
   ```bash
   # 配置tavily，重新生成核心章节
   sed -i '' 's/search_provider: "zhipuai"/search_provider: "tavily"/' config/config.yaml
   PYTHONPATH=/Users/z/Documents/work/content-forge-ai python src/main.py --mode series --episode 1
   ```

3. **生成后检查**：
   - 对比两种搜索结果的质量差异
   - 根据实际需求决定长期使用哪个

---

## 📋 快速检查脚本

```bash
# 检查当前配置
grep "search_provider" config/config.yaml

# 检查Tavily API Key
grep "TAVILY_API_KEY" .env

# 检查tavily-python是否安装
pip show tavily-python
```

---

## 🎯 推荐配置策略

### 方案A：成本优先（默认）
```yaml
research_agent:
  search_provider: "zhipuai"  # 日常使用
```
- 成本：0元
- 质量：⭐⭐⭐⭐
- 适合：批量生成

### 方案B：质量优先
```yaml
research_agent:
  search_provider: "tavily"  # 追求极致质量
```
- 成本：0元（免费额度内）
- 质量：⭐⭐⭐⭐⭐
- 适合：重点章节

### 方案C：混合使用（推荐）
- 日常生成：zhipuAI
- 核心章节：Tavily
- 根据需求灵活切换

---

## 💡 最佳实践

1. **初次使用**：先用zhipuAI生成一批，测试质量
2. **质量对比**：切换Tavily，生成同一期，对比差异
3. **长期选择**：根据对比结果，选择长期使用的方案
4. **灵活调整**：重要章节用Tavily，普通章节用zhipuAI

---

**最后更新**: 2026-01-15
