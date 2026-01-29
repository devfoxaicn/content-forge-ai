# 第53讲：深度学习推荐模型——从Wide&Deep到DLRM

## 课程概览

深度学习革命性地改变了推荐系统的技术架构。从Google的Wide&Deep到阿里巴巴的DeepFM，从YouTube的深度召回到Meta的DLRM，本讲将系统剖析现代推荐系统的深度学习模型演进。

**核心学习目标**：
- 掌握Wide&Deep的核心思想和变体
- 深入理解DeepFM、xDeepFM、DCN等模型
- 学习序列推荐：GRU4Rec、BERT4Rec、SASRec
- 理解多任务学习：ESMM、MMOE、PLE

---

## 一、Wide&Deep：记忆与泛化的平衡

### 1.1 核心思想

Google在2016年提出的Wide&Deep模型开创了推荐系统深度学习时代。

**关键洞察**：
- **Wide侧（线性模型）**：记忆历史共现模式
- **Deep侧（神经网络）**：泛化到未见过的特征组合

**架构图**：
```
特征输入
    │
    ├─────→ Wide侧 (线性模型)
    │         │
    │         └─────┐
    │               │
    └─────→ Deep侧 (DNN)  ─→  │  → 输出 (CTR预测)
                 │       拼接
                 └─────┘
```

### 1.2 模型实现

```python
import torch
import torch.nn as nn

class WideAndDeep(nn.Module):
    def __init__(self, wide_dim, deep_dim, hidden_layers=[256, 128, 64]):
        super(WideAndDeep, self).__init__()

        # Wide侧：线性模型
        self.wide_linear = nn.Linear(wide_dim, 1)

        # Deep侧：深度神经网络
        self.deep_embedding = nn.Linear(deep_dim, hidden_layers[0])
        self.deep_layers = nn.ModuleList()

        for i in range(len(hidden_layers) - 1):
            self.deep_layers.append(
                nn.Linear(hidden_layers[i], hidden_layers[i+1])
            )

        self.deep_output = nn.Linear(hidden_layers[-1], 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

        # 最终输出
        self.output = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wide_features, deep_features):
        # Wide侧
        wide_out = self.wide_linear(wide_features)

        # Deep侧
        deep_x = self.deep_embedding(deep_features)
        deep_x = self.relu(deep_x)
        deep_x = self.dropout(deep_x)

        for layer in self.deep_layers:
            deep_x = self.relu(layer(deep_x))
            deep_x = self.dropout(deep_x)

        deep_out = self.deep_output(deep_x)

        # 拼接
        concat = torch.cat([wide_out, deep_out], dim=1)
        out = self.sigmoid(self.output(concat))

        return out

# 使用示例
model = WideAndDeep(wide_dim=1000, deep_dim=512)
```

### 1.3 训练策略

**损失函数**：
```python
def weighted_binary_cross_entropy(pred, target, pos_weight=3.0):
    """
    加权BCE损失，解决正负样本不平衡
    """
    bce = nn.BCELoss(reduction='none')
    loss = bce(pred, target)

    # 正样本权重更高
    weight = torch.where(target == 1, pos_weight, 1.0)

    return (loss * weight).mean()

# 训练循环
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = weighted_binary_cross_entropy

for epoch in range(num_epochs):
    for batch in dataloader:
        wide_features, deep_features, labels = batch

        # 前向传播
        predictions = model(wide_features, deep_features)

        # 计算损失
        loss = criterion(predictions, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 二、DeepFM：因子分解机的深度演进

### 2.1 从FM到DeepFM

**因子分解机(FM)**的核心：
- 二阶特征交互的低秩表示
- 捕捉特征间的非线性关系

**DeepFM创新**：
- FM组件 + Deep组件
- 端到端训练，共享特征嵌入

### 2.2 模型架构

```python
class DeepFM(nn.Module):
    def __init__(self, field_dims, embed_dim=32, hidden_layers=[256, 128]):
        """
        field_dims: 每个特征域的维度列表
        embed_dim: 嵌入维度
        """
        super(DeepFM, self).__init__()

        self.num_fields = len(field_dims)

        # 特征嵌入
        self.feature_embeddings = nn.ModuleList([
            nn.Embedding(dim, embed_dim) for dim in field_dims
        ])

        # FM一阶项
        self.fm_linear = nn.ModuleList([
            nn.Linear(dim, 1) for dim in field_dims
        ])

        # FM二阶项（隐式）
        # 通过嵌入的内积实现

        # Deep组件
        deep_input_dim = self.num_fields * embed_dim
        self.deep_layers = nn.ModuleList()

        for i in range(len(hidden_layers)):
            if i == 0:
                self.deep_layers.append(nn.Linear(deep_input_dim, hidden_layers[i]))
            else:
                self.deep_layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))

        self.deep_output = nn.Linear(hidden_layers[-1], 1)

        # 最终输出
        self.output = nn.Linear(2, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        """
        X: (batch_size, num_fields)
        """
        batch_size = X.size(0)

        # 特征嵌入
        embeddings = []
        for i, (emb_layer, x) in enumerate(zip(self.feature_embeddings, X.t())):
            embeddings.append(emb_layer(x))

        embeddings = torch.stack(embeddings, dim=1)  # (B, F, E)

        # FM一阶
        fm_first_order = 0
        for i, linear in enumerate(self.fm_linear):
            fm_first_order += linear(X[:, i].float())

        # FM二阶
        sum_emb = torch.sum(embeddings, dim=1)  # (B, E)
        sum_of_square = torch.sum(embeddings ** 2, dim=1)  # (B, E)
        fm_second_order = 0.5 * torch.sum(sum_emb ** 2 - sum_of_square, dim=1, keepdim=True)

        # Deep组件
        deep_input = embeddings.view(batch_size, -1)
        deep_x = deep_input
        for layer in self.deep_layers:
            deep_x = self.relu(layer(deep_x))

        deep_out = self.deep_output(deep_x)

        # 组合
        fm_out = fm_first_order + fm_second_order
        concat = torch.cat([fm_out, deep_out], dim=1)
        output = self.sigmoid(self.output(concat))

        return output
```

### 2.3 训练技巧

```python
class FMDataLoader:
    """专门为FM设计的数据加载器"""

    def __init__(self, data, batch_size=1024):
        self.data = data
        self.batch_size = batch_size

    def __iter__(self):
        # 随机打乱
        indices = np.random.permutation(len(self.data))

        for i in range(0, len(self.data), self.batch_size):
            batch_indices = indices[i:i+self.batch_size]
            batch = [self.data[idx] for idx in batch_indices]

            # 转换为模型输入格式
            X = torch.LongTensor([item['features'] for item in batch])
            y = torch.FloatTensor([item['label'] for item in batch])

            yield X, y
```

---

## 三、xDeepFM与DCN：显式特征交叉

### 3.1 xDeepFM：CIN网络

**核心创新**：Compressed Interaction Network (CIN)

```python
class CIN(nn.Module):
    """
    Compressed Interaction Network
    显式建模高阶特征交互
    """
    def __init__(self, field_dim, embed_dim, hidden_layers=[32, 32]):
        super(CIN, self).__init__()

        self.field_dim = field_dim
        self.embed_dim = embed_dim

        # CIN层
        self.cin_layers = nn.ModuleList()
        prev_layer_dim = field_dim

        for layer_dim in hidden_layers:
            # 卷积操作
            self.cin_layers.append(
                nn.Conv1d(prev_layer_dim * field_dim, layer_dim * field_dim, 1)
            )
            prev_layer_dim = layer_dim

        # 输出
        self.output = nn.Linear(hidden_layers[-1], 1)

    def forward(self, embeddings):
        """
        embeddings: (batch_size, field_dim, embed_dim)
        """
        batch_size = embeddings.size(0)
        X0 = embeddings  # 第0层

        # 逐层计算
        for layer in self.cin_layers:
            # 外积：特征交互
            X_k = layer(X0)

        return output
```

### 3.2 DCN：深度交叉网络

**Cross Network的核心**：
- 显式学习有界阶特征交叉
- 高效且无需手动特征工程

```python
class CrossNetwork(nn.Module):
    def __init__(self, input_dim, num_layers=3):
        super(CrossNetwork, self).__init__()

        self.num_layers = num_layers
        self.cross_weights = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim))
            for _ in range(num_layers)
        ])
        self.cross_bias = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim))
            for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        x: (batch_size, input_dim)
        """
        x0 = x  # 保留原始特征

        for i in range(self.num_layers):
            # 交叉：x0 * (w * x + b)
            x = x0 * (x @ self.cross_weights[i].t() + self.cross_bias[i]) + x

        return x

class DCN(nn.Module):
    def __init__(self, input_dim, cross_layers=3, deep_layers=[256, 128]):
        super(DCN, self).__init__()

        # Cross Network
        self.cross_network = CrossNetwork(input_dim, cross_layers)

        # Deep Network
        self.deep_layers = nn.ModuleList()
        prev_dim = input_dim

        for dim in deep_layers:
            self.deep_layers.append(nn.Linear(prev_dim, dim))
            self.deep_layers.append(nn.ReLU())
            self.deep_layers.append(nn.Dropout(0.1))
            prev_dim = dim

        # 输出
        self.output = nn.Linear(input_dim + deep_layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Cross Network
        cross_out = self.cross_network(x)

        # Deep Network
        deep_x = x
        for layer in self.deep_layers:
            deep_x = layer(deep_x)

        # 拼接
        concat = torch.cat([cross_out, deep_x], dim=1)
        output = self.sigmoid(self.output(concat))

        return output
```

---

## 四、序列推荐模型

### 4.1 GRU4Rec：基于RNN的会话推荐

```python
class GRU4Rec(nn.Module):
    def __init__(self, num_items, embed_dim=128, hidden_dim=256):
        super(GRUU4Rec, self).__init__()

        # 物品嵌入
        self.item_embedding = nn.Embedding(num_items, embed_dim)

        # GRU层
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

        # 输出层
        self.output = nn.Linear(hidden_dim, num_items)

    def forward(self, item_sequence):
        """
        item_sequence: (batch_size, seq_len)
        """
        # 嵌入
        embedded = self.item_embedding(item_sequence)

        # GRU编码
        _, hidden = self.gru(embedded)
        hidden = hidden.squeeze(0)

        # 预测下一个物品
        logits = self.output(hidden)

        return logits

# 训练
model = GRU4Rec(num_items=10000)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for session in sessions:
    # 滑动窗口生成训练样本
    for i in range(1, len(session)):
        input_seq = session[:i]
        target_item = session[i]

        # 前向传播
        logits = model(input_seq)
        loss = F.cross_entropy(logits, target_item)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 4.2 SASRec：自注意力序列推荐

**核心创新**：使用Transformer处理序列

```python
class SASRec(nn.Module):
    def __init__(self, num_items, embed_dim=128, num_heads=4,
                 num_layers=2, max_len=50):
        super(SASRec, self).__init__()

        # 物品嵌入
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # 输出
        self.output = nn.Linear(embed_dim, num_items)

    def forward(self, item_seq, mask=None):
        """
        item_seq: (batch_size, seq_len)
        mask: (seq_len, seq_len) 注意力mask
        """
        seq_len = item_seq.size(1)

        # 物品嵌入 + 位置嵌入
        items = self.item_embedding(item_seq)
        positions = torch.arange(seq_len, device=item_seq.device)
        pos_emb = self.pos_embedding(positions)

        x = items + pos_emb  # (B, L, E)

        # Transformer编码
        x = x.transpose(0, 1)  # (L, B, E) for Transformer
        x = self.transformer(x, mask=mask)
        x = x.transpose(0, 1)  # (B, L, E)

        # 取最后一个位置
        last_hidden = x[:, -1, :]

        # 预测
        logits = self.output(last_hidden)

        return logits
```

### 4.3 BERT4Rec：双向编码

```python
class BERT4Rec(nn.Module):
    """双向Transformer用于序列推荐"""

    def __init__(self, num_items, embed_dim=128, num_heads=4,
                 num_layers=2, max_len=50, mask_prob=0.2):
        super(BERT4Rec, self).__init__()

        self.mask_prob = mask_prob
        self.num_items = num_items

        # 嵌入
        self.item_embedding = nn.Embedding(num_items + 1, embed_dim)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        self.segment_embedding = nn.Embedding(2, embed_dim)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # 输出
        self.output = nn.Linear(embed_dim, num_items + 1)

    def mask_item_sequence(self, item_seq):
        """随机mask物品（MLM风格）"""
        masked_seq = item_seq.clone()
        mask = torch.rand_like(item_seq.float()) < self.mask_prob

        # 用特殊token替换
        masked_seq[mask] = self.num_items  # MASK token

        return masked_seq, mask

    def forward(self, item_seq):
        batch_size, seq_len = item_seq.shape

        # Masking
        masked_seq, mask = self.mask_item_sequence(item_seq)

        # 嵌入
        items = self.item_embedding(masked_seq)
        positions = torch.arange(seq_len, device=item_seq.device)
        pos_emb = self.pos_embedding(positions)

        x = items + pos_emb

        # Transformer
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)

        # 预测被mask的位置
        logits = self.output(x[mask])

        return logits
```

---

## 五、多任务学习

### 5.1 MMOE：多门专家混合

```python
class MMOE(nn.Module):
    """
    Multi-gate Mixture-of-Experts
    Google提出的多任务学习框架
    """
    def __init__(self, input_dim, num_experts=4, num_tasks=2,
                 expert_hidden=[64, 32], tower_hidden=[32]):
        super(MMOE, self).__init__()

        self.num_experts = num_experts
        self.num_tasks = num_tasks

        # 专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_hidden[0]),
                nn.ReLU(),
                nn.Linear(expert_hidden[0], expert_hidden[-1])
            )
            for _ in range(num_experts)
        ])

        # 门控网络（每个任务一个）
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, num_experts),
                nn.Softmax(dim=1)
            )
            for _ in range(num_tasks)
        ])

        # 任务塔（每个任务一个）
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_hidden[-1], tower_hidden[0]),
                nn.ReLU(),
                nn.Linear(tower_hidden[0], 1),
                nn.Sigmoid()
            )
            for _ in range(num_tasks)
        ])

    def forward(self, x):
        # 专家输出
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))

        expert_outputs = torch.stack(expert_outputs, dim=1)  # (B, E, D)

        # 每个任务的门控 + 输出
        task_outputs = []
        for task_id in range(self.num_tasks):
            # 门控权重
            gate_weights = self.gates[task_id](x)  # (B, E)
            gate_weights = gate_weights.unsqueeze(2)  # (B, E, 1)

            # 加权组合专家输出
            expert_mix = torch.sum(expert_outputs * gate_weights, dim=1)  # (B, D)

            # 任务塔
            task_out = self.towers[task_id](expert_mix)
            task_outputs.append(task_out)

        return task_outputs

# 使用
model = MMOE(input_dim=100, num_experts=4, num_tasks=2)

# 训练
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for batch in dataloader:
    features, labels_task1, labels_task2 = batch

    # 多任务输出
    pred_task1, pred_task2 = model(features)

    # 多任务损失
    loss1 = F.binary_cross_entropy(pred_task1, labels_task1)
    loss2 = F.binary_cross_entropy(pred_task2, labels_task2)

    # 加权组合
    loss = 0.6 * loss1 + 0.4 * loss2

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 5.2 PLE：渐进分层提取

```python
class PLE(nn.Module):
    """
    Progressive Layered Extraction
    更细粒度的多任务学习
    """
    def __init__(self, input_dim, num_tasks=2, num_experts_per_task=3):
        super(PLE, self).__init__()

        # 任务特定专家
        self.task_experts = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(input_dim, 64)
                for _ in range(num_experts_per_task)
            ])
            for _ in range(num_tasks)
        ])

        # 共享专家
        self.shared_experts = nn.ModuleList([
            nn.Linear(input_dim, 64)
            for _ in range(num_experts_per_task)
        ])

        # 门控网络（更复杂）
        self.gates = nn.ModuleList([
            nn.Linear(input_dim, num_experts_per_task * 2)
            for _ in range(num_tasks)
        ])

        # 任务塔
        self.towers = nn.ModuleList([
            nn.Linear(64, 1)
            for _ in range(num_tasks)
        ])

    def forward(self, x):
        # 专家输出
        task_outputs = []

        for task_id in range(self.num_tasks):
            # 任务特定专家
            task_expert_outs = []
            for expert in self.task_experts[task_id]:
                task_expert_outs.append(expert(x))

            # 共享专家
            shared_expert_outs = []
            for expert in self.shared_experts:
                shared_expert_outs.append(expert(x))

            # 门控组合
            gate_logits = self.gates[task_id](x)
            gate_weights = F.softmax(gate_logits, dim=1)

            # 组合输出
            combined = self.combine_experts(task_expert_outs,
                                          shared_expert_outs,
                                          gate_weights)

            # 任务塔
            task_out = torch.sigmoid(self.towers[task_id](combined))
            task_outputs.append(task_out)

        return task_outputs
```

### 5.3 ESMM：完整空间多任务模型

**用于CVR预测，解决样本选择偏差**

```python
class ESMM(nn.Module):
    """
    Entire Space Multi-Task Model
    联合训练CTR和CVR，约束CVR = CVR / CTR
    """
    def __init__(self, input_dim, hidden_layers=[256, 128]):
        super(ESMM, self).__init__()

        # CTR塔
        self_ctr_layers = []
        prev_dim = input_dim
        for dim in hidden_layers:
            self.ctr_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = dim
        self.ctr_layers.append(nn.Linear(prev_dim, 1))
        self.ctr_tower = nn.Sequential(*self.ctr_layers)

        # CVR塔
        self.cvr_layers = []
        prev_dim = input_dim
        for dim in hidden_layers:
            self.cvr_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = dim
        self.cvr_layers.append(nn.Linear(prev_dim, 1))
        self.cvr_tower = nn.Sequential(*self.cvr_layers)

    def forward(self, x):
        # CTR预测
        ctr = torch.sigmoid(self.ctr_tower(x))

        # CVR预测
        cvr = torch.sigmoid(self.cvr_tower(x))

        # CTRCVR = CTR × CVR
        ctcvr = ctr * cvr

        return ctr, cvr, ctcvr

# 训练
def train_esmm(model, dataloader):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for batch in dataloader:
        features, click_label, conversion_label = batch

        # 前向传播
        ctr, cvr, ctcvr = model(features)

        # 损失函数
        # CTR任务
        ctr_loss = F.binary_cross_entropy(ctr, click_label)

        # CT-CVR任务（只在点击的样本上计算）
        ctcvr_loss = F.binary_cross_entropy(
            ctcvr[click_label == 1],
            conversion_label[click_label == 1]
        )

        # 总损失
        loss = ctr_loss + ctcvr_loss

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 六、工业实践案例

### 6.1 YouTube推荐架构

```python
class YouTubeRecommendation:
    """
    YouTube深度推荐系统
    两阶段：召回 + 排序
    """

    def __init__(self):
        # 召回模型
        self.retrieval_model = CandidateGenerationModel()

        # 排序模型
        self.ranking_model = RankingModel()

    def recommend(self, user_history):
        # 阶段1：召回（从百万视频筛选数百）
        candidates = self.retrieval_model.retrieve(user_history, top_k=500)

        # 阶段2：排序（对数百候选精排）
        scored_candidates = []
        for video in candidates:
            score = self.ranking_model.score(user_history, video)
            scored_candidates.append((video, score))

        # 选择Top-N
        ranked = sorted(scored_candidates, key=lambda x: x[1], reverse=True)
        return ranked[:10]


class CandidateGenerationModel(nn.Module):
    """YouTube召回模型"""

    def __init__(self, num_videos, embed_dim=128):
        super().__init__()

        # 视频嵌入
        self.video_embedding = nn.Embedding(num_videos, embed_dim)

        # 用户嵌入（从观看历史）
        self.user_embedding = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )

        # 输出层
        self.output = nn.Linear(embed_dim * 2, num_videos)

    def forward(self, watch_history):
        # 历史视频嵌入
        history_embeds = self.video_embedding(watch_history)

        # 聚合（平均池化）
        user_embed = torch.mean(history_embeds, dim=1)
        user_embed = self.user_embedding(user_embed)

        # 为每个视频计算得分
        all_scores = []
        for video_id in range(self.output.out_features):
            video_embed = self.video_embedding.weight[video_id]
            concat = torch.cat([user_embed, video_embed], dim=1)
            score = self.output(concat)
            all_scores.append(score)

        return torch.cat(all_scores, dim=1)


class RankingModel(nn.Module):
    """YouTube排序模型"""

    def __init__(self, num_features):
        super().__init__()

        # 深度网络
        self.layers = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # 宽度网络（重要特征）
        self.wide = nn.Linear(num_features, 1)

        # 输出
        self.output = nn.Linear(2, 1)

    def forward(self, features):
        # 深度侧
        deep_out = self.layers(features)

        # 宽度侧
        wide_out = self.wide(features)

        # 组合
        concat = torch.cat([deep_out, wide_out], dim=1)
        output = self.output(concat)

        return torch.sigmoid(output)
```

### 6.2 阿里巴巴推荐架构

```python
class DIN(nn.Module):
    """
    Deep Interest Network
    动态兴趣建模
    """
    def __init__(self, num_items, embed_dim=32, hidden_layers=[200, 80]):
        super().__init__()

        # 物品嵌入
        self.item_embedding = nn.Embedding(num_items, embed_dim)

        # 历史兴趣嵌入
        self.history_embedding = nn.Embedding(num_items, embed_dim)

        # 注意力机制（动态计算历史物品的重要性）
        self.attention = nn.Sequential(
            nn.Linear(embed_dim * 4, 36),
            nn.ReLU(),
            nn.Linear(36, 1)
        )

        # Deep网络
        deep_layers = []
        prev_dim = embed_dim * 2
        for dim in hidden_layers:
            deep_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = dim
        deep_layers.append(nn.Linear(prev_dim, 1))
        self.deep = nn.Sequential(*deep_layers)

    def forward(self, target_item, history_items, history_mask):
        """
        target_item: (B,)
        history_items: (B, H)
        history_mask: (B, H) 标记有效历史
        """
        # 目标物品嵌入
        target_emb = self.item_embedding(target_item)  # (B, E)

        # 历史物品嵌入
        history_embs = self.history_embedding(history_items)  # (B, H, E)

        # 注意力权重
        target_emb_expanded = target_emb.unsqueeze(1).expand_as(history_embs)
        concat = torch.cat([target_emb_expanded, history_embs,
                           target_emb_expanded * history_embs,
                           target_emb_expanded - history_embs], dim=2)

        attn_scores = self.attention(concat).squeeze(2)  # (B, H)

        # Mask掉无效历史
        attn_scores = attn_scores.masked_fill(history_mask == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=1)

        # 加权聚合历史兴趣
        weighted_history = torch.sum(history_embs * attn_weights.unsqueeze(2),
                                    dim=1)  # (B, E)

        # 拼接
        concat = torch.cat([target_emb, weighted_history], dim=1)

        # 深度网络
        output = torch.sigmoid(self.deep(concat))

        return output
```

---

## 七、训练优化技巧

### 7.1 学习率调度

```python
def get_scheduler(optimizer, warmup_steps=10000, total_steps=100000):
    """
    余弦退火 + Warmup
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / \
                       float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# 使用
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = get_scheduler(optimizer)

for step, batch in enumerate(dataloader):
    loss = train_step(model, batch)
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

### 7.2 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    features, labels = batch

    # 自动混合精度
    with autocast():
        predictions = model(features)
        loss = criterion(predictions, labels)

    # 反向传播（自动缩放）
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

### 7.3 分布式训练

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 初始化进程组
dist.init_process_group(backend='nccl')

# 包装模型
model = model.to(device)
model = DDP(model, device_ids=[local_rank])

# 训练
for batch in dataloader:
    features, labels = batch
    features, labels = features.to(device), labels.to(device)

    predictions = model(features)
    loss = criterion(predictions, labels)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

---

## 八、总结与展望

### 核心要点回顾

1. **Wide&Deep范式**：
   - Wide侧：记忆能力
   - Deep侧：泛化能力
   - 平衡关键

2. **特征交互建模**：
   - FM：隐式二阶
   - DeepFM：FM+Deep
   - xDeepFM：显式高阶
   - DCN：深度交叉

3. **序列建模**：
   - GRU4Rec：RNN方法
   - SASRec：自注意力
   - BERT4Rec：双向编码

4. **多任务学习**：
   - MMOE：门控专家
   - PLE：渐进分层
   - ESMM：空间约束

### 未来趋势

1. **大模型融合**：LLM4Rec
2. **图神经网络**：用户-物品图
3. **因果推断**：去偏学习
4. **在线学习**：实时适应

---

## 参考文献

**经典论文**：
- Cheng et al. (2016). "Wide & Deep Learning for Recommender Systems." DLRS.
- Guo et al. (2017). "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction." IJCAI.
- Zhou et al. (2018). "Deep Interest Network for Click-Through Rate Prediction." KDD.
- Sun et al. (2019). "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer." CIKM.

---

**下一讲预告**：我们将深入探讨召回算法与索引技术，学习如何从海量候选中快速筛选感兴趣的内容。敬请期待！
