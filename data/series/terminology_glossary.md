# AI 系列文章术语对照表

> **版本**: v1.0
> **更新日期**: 2026-02-13
> **适用范围**: LLM Series (100期) + ML Series (100期)

本文档统一了全系列文章中的专业术语翻译，确保内容一致性。

---

## 📖 使用说明

1. **新文章生成时**：参考本表使用统一术语
2. **术语首次出现时**：使用"中文（English）"格式，后续可直接使用中文
3. **发现新术语时**：及时更新本表

---

## 🤖 LLM Series 术语表

### Transformer 架构

| 英文术语 | 中文翻译 | 首次出现 | 说明 |
|---------|---------|---------|------|
| Transformer | Transformer | Ep 001 | 保留英文，专有名词 |
| Self-Attention | 自注意力机制 | Ep 001 | |
| Multi-Head Attention | 多头注意力 | Ep 001 | |
| Position Encoding | 位置编码 | Ep 001 | |
| Positional Embedding | 位置嵌入 | Ep 001 | |
| Layer Normalization | 层归一化 | Ep 001 | |
| Feed-Forward Network | 前馈网络 | Ep 001 | |
| Residual Connection | 残差连接 | Ep 001 | |
| Encoder | 编码器 | Ep 001 | |
| Decoder | 解码器 | Ep 001 | |
| Cross-Attention | 交叉注意力 | Ep 001 | |

### Tokenization

| 英文术语 | 中文翻译 | 首次出现 | 说明 |
|---------|---------|---------|------|
| Tokenizer | 分词器 | Ep 002 | |
| Token | 词元/标记 | Ep 002 | 上下文选择 |
| BPE | 字节对编码 | Ep 002 | Byte-Pair Encoding |
| WordPiece | WordPiece | Ep 002 | 保留英文 |
| Unigram | Unigram | Ep 002 | 保留英文 |
| Vocabulary | 词汇表 | Ep 002 | |
| Vocabulary Size | 词汇表大小 | Ep 002 | |
| Special Token | 特殊标记 | Ep 002 | |

### 预训练

| 英文术语 | 中文翻译 | 首次出现 | 说明 |
|---------|---------|---------|------|
| Pretraining | 预训练 | Ep 003 | |
| MLM | 掩码语言建模 | Ep 003 | Masked Language Modeling |
| CLM | 因果语言建模 | Ep 003 | Causal Language Modeling |
| Next Token Prediction | 下一词预测 | Ep 003 | |
| Scaling Law | 缩放定律 | Ep 003 | |
| Emergent Ability | 涌现能力 | Ep 003 | |
| Foundation Model | 基础模型 | Ep 003 | |

### 对齐与微调

| 英文术语 | 中文翻译 | 首次出现 | 说明 |
|---------|---------|---------|------|
| SFT | 监督微调 | Ep 004 | Supervised Fine-Tuning |
| RLHF | 人类反馈强化学习 | Ep 004 | Reinforcement Learning from Human Feedback |
| DPO | 直接偏好优化 | Ep 004 | Direct Preference Optimization |
| Alignment | 对齐 | Ep 004 | |
| Instruction Tuning | 指令微调 | Ep 004 | |
| Reward Model | 奖励模型 | Ep 004 | |
| PPO | 近端策略优化 | Ep 004 | Proximal Policy Optimization |

### RAG

| 英文术语 | 中文翻译 | 首次出现 | 说明 |
|---------|---------|---------|------|
| RAG | 检索增强生成 | Ep 011 | Retrieval-Augmented Generation |
| Embedding | 嵌入 | Ep 011 | |
| Vector Database | 向量数据库 | Ep 011 | |
| Retrieval | 检索 | Ep 011 | |
| Chunking | 分块 | Ep 011 | |
| Context Window | 上下文窗口 | Ep 011 | |
| Semantic Search | 语义搜索 | Ep 011 | |

---

## 🧠 ML Series 术语表

### 机器学习基础

| 英文术语 | 中文翻译 | 首次出现 | 说明 |
|---------|---------|---------|------|
| Supervised Learning | 监督学习 | ML Ep 001 | |
| Unsupervised Learning | 无监督学习 | ML Ep 001 | |
| Reinforcement Learning | 强化学习 | ML Ep 001 | |
| Feature | 特征 | ML Ep 001 | |
| Label | 标签 | ML Ep 001 | |
| Training Set | 训练集 | ML Ep 001 | |
| Validation Set | 验证集 | ML Ep 001 | |
| Test Set | 测试集 | ML Ep 001 | |
| Overfitting | 过拟合 | ML Ep 001 | |
| Underfitting | 欠拟合 | ML Ep 001 | |
| Bias-Variance Tradeoff | 偏差-方差权衡 | ML Ep 001 | |

### 深度学习

| 英文术语 | 中文翻译 | 首次出现 | 说明 |
|---------|---------|---------|------|
| Neural Network | 神经网络 | ML Ep 011 | |
| CNN | 卷积神经网络 | ML Ep 012 | Convolutional Neural Network |
| RNN | 循环神经网络 | ML Ep 013 | Recurrent Neural Network |
| LSTM | 长短期记忆网络 | ML Ep 013 | Long Short-Term Memory |
| GRU | 门控循环单元 | ML Ep 013 | Gated Recurrent Unit |
| Activation Function | 激活函数 | ML Ep 011 | |
| Loss Function | 损失函数 | ML Ep 011 | |
| Backpropagation | 反向传播 | ML Ep 011 | |
| Gradient Descent | 梯度下降 | ML Ep 011 | |
| Learning Rate | 学习率 | ML Ep 011 | |
| Batch Size | 批次大小 | ML Ep 011 | |
| Epoch | 训练轮次 | ML Ep 011 | |

### 计算机视觉

| 英文术语 | 中文翻译 | 首次出现 | 说明 |
|---------|---------|---------|------|
| Image Classification | 图像分类 | ML Ep 022 | |
| Object Detection | 目标检测 | ML Ep 023 | |
| Semantic Segmentation | 语义分割 | ML Ep 024 | |
| Instance Segmentation | 实例分割 | ML Ep 024 | |
| Feature Map | 特征图 | ML Ep 021 | |
| Convolution | 卷积 | ML Ep 021 | |
| Pooling | 池化 | ML Ep 021 | |
| Backbone | 骨干网络 | ML Ep 023 | |

### NLP

| 英文术语 | 中文翻译 | 首次出现 | 说明 |
|---------|---------|---------|------|
| NLP | 自然语言处理 | ML Ep 031 | Natural Language Processing |
| Word Embedding | 词嵌入 | ML Ep 032 | |
| Language Model | 语言模型 | ML Ep 033 | |
| Text Classification | 文本分类 | ML Ep 035 | |
| Named Entity Recognition | 命名实体识别 | ML Ep 034 | |
| Sentiment Analysis | 情感分析 | ML Ep 035 | |
| Seq2Seq | 序列到序列 | ML Ep 036 | |
| Attention | 注意力机制 | ML Ep 036 | |

### 扩散模型

| 英文术语 | 中文翻译 | 首次出现 | 说明 |
|---------|---------|---------|------|
| Diffusion Model | 扩散模型 | ML Ep 096 | |
| Forward Diffusion | 前向扩散 | ML Ep 096 | |
| Reverse Denoising | 反向去噪 | ML Ep 096 | |
| DDPM | 去噪扩散概率模型 | ML Ep 096 | Denoising Diffusion Probabilistic Models |
| Score Matching | 分数匹配 | ML Ep 096 | |
| Latent Space | 潜空间 | ML Ep 096 | |
| VAE | 变分自编码器 | ML Ep 096 | Variational Autoencoder |
| CFG | 无分类器引导 | ML Ep 096 | Classifier-Free Guidance |
| U-Net | U-Net | ML Ep 096 | 保留英文 |
| Text-to-Image | 文生图 | ML Ep 096 | |
| Image-to-Image | 图生图 | ML Ep 096 | |

### 神经符号AI

| 英文术语 | 中文翻译 | 首次出现 | 说明 |
|---------|---------|---------|------|
| Neuro-Symbolic AI | 神经符号AI | ML Ep 097 | |
| Explainability | 可解释性 | ML Ep 097 | |
| LIME | LIME | ML Ep 097 | Local Interpretable Model-agnostic Explanations |
| SHAP | SHAP | ML Ep 097 | SHapley Additive exPlanations |
| Attention Visualization | 注意力可视化 | ML Ep 097 | |
| Causal Inference | 因果推断 | ML Ep 097 | |
| Counterfactual Explanation | 反事实解释 | ML Ep 097 | |
| Differentiable Logic | 可微分逻辑 | ML Ep 097 | |

---

## 📊 通用术语

| 英文术语 | 中文翻译 | 说明 |
|---------|---------|------|
| Inference | 推理 | |
| Fine-tuning | 微调 | |
| Zero-shot | 零样本 | |
| Few-shot | 少样本 | |
| Prompt | 提示词 | |
| Context | 上下文 | |
| Token | 词元/标记 | 上下文选择 |
| Temperature | 温度参数 | |
| Top-k Sampling | Top-k采样 | |
| Top-p Sampling | Top-p采样/核采样 | |
| Beam Search | 束搜索 | |
| Perplexity | 困惑度 | |
| BLEU | BLEU分数 | 保留英文 |
| F1 Score | F1分数 | |
| Accuracy | 准确率 | |
| Precision | 精确率 | |
| Recall | 召回率 | |
| AUC | AUC | 保留英文 |
| ROC | ROC曲线 | 保留英文 |

---

## 🔧 工具与框架

| 英文术语 | 中文翻译 | 说明 |
|---------|---------|------|
| PyTorch | PyTorch | 保留英文 |
| TensorFlow | TensorFlow | 保留英文 |
| Hugging Face | Hugging Face | 保留英文 |
| Transformers | Transformers库 | 保留英文 |
| CUDA | CUDA | 保留英文 |
| GPU | GPU | 保留英文 |
| TPU | TPU | 保留英文 |

---

## 📝 写作规范

1. **首次出现**：使用"中文（English）"格式
   - 例：扩散模型（Diffusion Model）是当前最流行的生成模型...

2. **后续出现**：直接使用中文
   - 例：扩散模型的训练过程包括...

3. **保留英文的情况**：
   - 专有名词（PyTorch, CUDA, U-Net）
   - 缩写（CNN, RNN, LSTM, BPE）
   - 无标准中文翻译的术语

4. **数学符号**：
   - 使用 LaTeX 格式：`$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$`
   - 向量使用粗体：$\mathbf{x}$

---

**维护者**: ContentForge AI
**最后更新**: 2026-02-13
