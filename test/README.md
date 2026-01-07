# 测试文件说明

本目录包含所有项目的测试文件。

## 测试文件列表

### 1. test_topic_logic.py
**功能**：测试topic参数处理逻辑
```bash
cd test
python test_topic_logic.py
```

### 2. test_storage.py
**功能**：测试存储结构（按日期分层的文件系统）
```bash
cd test
python test_storage.py
```

### 3. test_ai_trends.py
**功能**：测试AI热点获取（11个数据源）

```bash
# 测试所有数据源
cd test
python test_ai_trends.py --topic "AI"

# 测试单个数据源
python test_ai_trends.py --source hackernews
python test_ai_trends.py --source arxiv
python test_ai_trends.py --source huggingface

# 简洁输出
python test_ai_trends.py --brief
```

### 4. test_digest.py
**功能**：测试热点简报Agent
```bash
cd test
python test_digest.py
```

### 5. test_auto_topic.py
**功能**：测试自动话题模式（不指定topic）
```bash
cd test
python test_auto_topic.py
```

## 运行所有测试

```bash
# 从项目根目录运行
cd test
python test_topic_logic.py
python test_storage.py
python test_auto_topic.py
```

## 注意事项

- 所有测试文件都需要从项目根目录运行，或正确设置PYTHONPATH
- API相关测试（test_ai_trends.py）需要网络连接
- 部分测试需要配置API密钥（ZhipuAI、Gemini等）
