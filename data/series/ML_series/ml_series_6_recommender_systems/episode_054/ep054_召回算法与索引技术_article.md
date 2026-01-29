# 第54讲：召回算法与索引技术
## 大规模推荐系统的基石

---

## 课程概览

在推荐系统的三层架构中，**召回层** 负责从海量候选池中快速筛选出用户可能感兴趣的候选集。本讲将深入探讨召回算法的核心技术与工程实践，包括双塔模型、深度语义匹配（DSSM）、近似最近邻搜索（Faiss/HNSW），以及亿级商品召回中的工程挑战。

**核心要点**：
- 召回层的定位与评估指标
- 双塔模型架构与训练技巧
- DSSM及其在召回中的应用
- 向量检索与Faiss实战
- HNSW索引优化
- 多路召回策略
- YouTube、阿里巴巴召回架构解析

---

## 一、召回层：推荐系统的漏斗入口

### 1.1 召回层的核心作用

在典型的推荐系统架构中：

```
用户请求 → 召回层(百万级) → 排序层(百级) → 重排层(十级) → 最终展示
```

**召回层的职责**：
1. **海量筛选**：从百万/亿级候选池筛选出数百个候选
2. **快速响应**：通常要求在50ms内完成
3. **覆盖多样性**：保留不同类型的候选
4. **泛化能力**：发现用户潜在兴趣

### 1.2 召回 vs 排序

| 维度 | 召回层 | 排序层 |
|------|--------|--------|
| 候选规模 | 百万→百 | 百→十 |
| 计算复杂度 | O(1)~O(log N) | O(N) |
| 模型复杂度 | 轻量 | 复杂 |
| 特征维度 | 稀疏 | 密集 |
| 评估指标 | Recall@K, Precision@K | AUC, NDCG |

### 1.3 召回评估指标

**离线指标**：
```python
import numpy as np
from sklearn.metrics import recall_score, precision_score

def recall_at_k(y_true, y_pred, k=100):
    """计算Recall@K"""
    return len(set(y_true) & set(y_pred[:k])) / len(y_true)

def precision_at_k(y_true, y_pred, k=100):
    """计算Precision@K"""
    return len(set(y_true) & set(y_pred[:k])) / k

def mean_reciprocal_rank(y_true, y_pred):
    """计算MRR"""
    for i, item in enumerate(y_pred):
        if item in y_true:
            return 1.0 / (i + 1)
    return 0.0

# 示例
ground_truth = [101, 205, 308, 412, 520]
predictions = [101, 150, 205, 308, 500, 600, 700]

print(f"Recall@10: {recall_at_k(ground_truth, predictions, 10):.3f}")
print(f"Precision@10: {precision_at_k(ground_truth, predictions, 10):.3f}")
print(f"MRR: {mean_reciprocal_rank(ground_truth, predictions):.3f}")
```

---

## 二、双塔模型：召回的深度学习范式

### 2.1 双塔模型架构

双塔模型（Two-Tower Model）是召回层最流行的深度学习架构：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoTowerModel(nn.Module):
    """双塔召回模型"""
    def __init__(self, 
                 user_feature_dim,
                 item_feature_dim,
                 embedding_dim=128,
                 hidden_layers=[256, 128]):
        super(TwoTowerModel, self).__init__()
        
        # User Tower
        self.user_tower = nn.Sequential(
            nn.Linear(user_feature_dim, hidden_layers[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_layers[1], embedding_dim)
        )
        
        # Item Tower
        self.item_tower = nn.Sequential(
            nn.Linear(item_feature_dim, hidden_layers[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_layers[1], embedding_dim)
        )
        
    def forward(self, user_features, item_features):
        """
        Args:
            user_features: [batch_size, user_feature_dim]
            item_features: [batch_size, item_feature_dim]
        Returns:
            similarity: [batch_size]
        """
        # 获取双塔embedding
        user_emb = self.user_tower(user_features)  # [B, D]
        item_emb = self.item_tower(item_features)  # [B, D]
        
        # 归一化
        user_emb = F.normalize(user_emb, dim=1)
        item_emb = F.normalize(item_emb, dim=1)
        
        # 计算相似度（内积）
        similarity = (user_emb * item_emb).sum(dim=1)
        
        return similarity
    
    def get_user_embedding(self, user_features):
        """获取用户embedding，用于索引"""
        return F.normalize(self.user_tower(user_features), dim=1)
    
    def get_item_embedding(self, item_features):
        """获取物品embedding，用于建库"""
        return F.normalize(self.item_tower(item_features), dim=1)

# 模型实例化
model = TwoTowerModel(
    user_feature_dim=50,
    item_feature_dim=100,
    embedding_dim=128
)

# 模拟数据
batch_size = 64
user_feat = torch.randn(batch_size, 50)
item_feat = torch.randn(batch_size, 100)

# 前向传播
scores = model(user_feat, item_feat)
print(f"Similarity scores shape: {scores.shape}")
print(f"Sample scores: {scores[:3]}")

# 获取独立的embeddings
user_emb = model.get_user_embedding(user_feat)
item_emb = model.get_item_embedding(item_feat)
print(f"User embedding shape: {user_emb.shape}")
print(f"Item embedding shape: {item_emb.shape}")
```

### 2.2 训练策略：Sampled Softmax

由于候选集规模巨大，直接计算所有物品的softmax不现实：

```python
class SampledSoftmaxLoss(nn.Module):
    """采样softmax损失"""
    def __init__(self, num_items, embedding_dim, num_samples=50):
        super(SampledSoftmaxLoss, self).__init__()
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_samples = num_samples
        
        # Item embeddings for negative sampling
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
    
    def forward(self, user_emb, positive_item_id, negative_item_ids=None):
        """
        Args:
            user_emb: [batch_size, embedding_dim]
            positive_item_id: [batch_size]
            negative_item_ids: [batch_size, num_negative_samples] or None
        Returns:
            loss: scalar
        """
        batch_size = user_emb.shape[0]
        
        if negative_item_ids is None:
            # 随机采样负样本
            negative_item_ids = torch.randint(
                0, self.num_items, 
                (batch_size, self.num_samples),
                device=user_emb.device
            )
        
        # 获取正样本embedding
        pos_item_emb = self.item_embeddings(positive_item_id)  # [B, D]
        
        # 获取负样本embedding
        neg_item_emb = self.item_embeddings(negative_item_ids)  # [B, K, D]
        
        # 计算正样本得分
        pos_scores = (user_emb * pos_item_emb).sum(dim=1, keepdim=True)  # [B, 1]
        
        # 计算负样本得分
        neg_scores = (user_emb.unsqueeze(1) * neg_item_emb).sum(dim=2)  # [B, K]
        
        # LogSumExp技巧
        all_scores = torch.cat([pos_scores, neg_scores], dim=1)  # [B, K+1]
        log_sum_exp = torch.logsumexp(all_scores, dim=1)  # [B]
        
        # 损失
        loss = torch.mean(log_sum_exp - pos_scores.squeeze(1))
        
        return loss

# 使用示例
loss_fn = SampledSoftmaxLoss(num_items=100000, embedding_dim=128, num_samples=100)

# 模拟训练数据
user_emb = F.normalize(torch.randn(64, 128), dim=1)
pos_item_ids = torch.randint(0, 100000, (64,))

# 计算损失
loss = loss_fn(user_emb, pos_item_ids)
print(f"Sampled Softmax Loss: {loss.item():.4f}")
```

### 2.3 双塔模型训练技巧

**1. 温度系数**：
```python
def similarity_with_temperature(user_emb, item_emb, temperature=0.05):
    """带温度系数的相似度计算"""
    scores = (user_emb * item_emb).sum(dim=1)
    return scores / temperature

# 温度系数的作用：放大预测差异
# temperature越小，预测越"自信"
```

**2. 难负样本挖掘**：
```python
def hard_negative_mining(model, user_emb, item_embeddings, k=10):
    """挖掘难负样本"""
    # 计算与所有item的相似度
    similarities = torch.mm(user_emb, item_embeddings.t())  # [B, N]
    
    # 找到最相似的负样本
    _, top_indices = torch.topk(similarities, k=k, dim=1)
    
    return top_indices

# 在训练循环中使用
def train_step_with_hard_negatives(model, batch, item_embeddings, optimizer):
    user_feat, pos_item_id = batch
    
    # 前向传播
    user_emb = model.get_user_embedding(user_feat)
    pos_item_emb = model.get_item_embedding(item_embeddings[pos_item_id])
    
    # 挖掘难负样本
    hard_neg_indices = hard_negative_mining(
        model, user_emb, item_embeddings.weight, k=5
    )
    
    # 计算损失（包含难负样本）
    neg_item_ids = torch.cat([
        torch.randint(0, 100000, (64, 45), device=user_emb.device),
        hard_neg_indices
    ], dim=1)
    
    loss = loss_fn(user_emb, pos_item_id, neg_item_ids)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()
```

---

## 三、DSSM：深度语义匹配模型

### 3.1 DSSM原理

DSSM (Deep Structured Semantic Models) 最早用于Web搜索，后来被广泛应用到推荐系统召回：

```python
class DSSM(nn.Module):
    """Deep Structured Semantic Model"""
    def __init__(self, vocab_size, embedding_dim=128, hidden_layers=[300, 128]):
        super(DSSM, self).__init__()
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 查询塔
        self.query_tower = nn.Sequential(
            nn.Linear(embedding_dim, hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU()
        )
        
        # 文档塔
        self.doc_tower = nn.Sequential(
            nn.Linear(embedding_dim, hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU()
        )
        
    def forward(self, query_ids, doc_ids):
        """
        Args:
            query_ids: [batch_size, seq_len]
            doc_ids: [batch_size, seq_len]
        """
        # 词平均池化
        query_emb = self.embedding(query_ids).mean(dim=1)  # [B, D]
        doc_emb = self.embedding(doc_ids).mean(dim=1)  # [B, D]
        
        # 通过塔结构
        query_vec = self.query_tower(query_emb)  # [B, H]
        doc_vec = self.doc_tower(doc_emb)  # [B, H]
        
        # 归一化
        query_vec = F.normalize(query_vec, dim=1)
        doc_vec = F.normalize(doc_vec, dim=1)
        
        # 余弦相似度
        similarity = (query_vec * doc_vec).sum(dim=1)
        
        return similarity

# 使用示例
dssm_model = DSSM(vocab_size=50000, embedding_dim=128)
query_input = torch.randint(0, 50000, (32, 20))  # 32个查询，每个20词
doc_input = torch.randint(0, 50000, (32, 50))    # 32个文档，每个50词

scores = dssm_model(query_input, doc_input)
print(f"DSSM scores: {scores.shape}")
```

### 3.2 YouTube召回架构

YouTube的召回系统是DSSM的经典应用：

```python
class YouTubeRetrievalModel(nn.Module):
    """YouTube DNN召回模型"""
    def __init__(self, 
                 num_users,
                 num_items,
                 user_feature_dim,
                 embedding_dim=64):
        super(YouTubeRetrievalModel, self).__init__()
        
        # 用户特征嵌入
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        
        # 用户侧网络
        self.user_net = nn.Sequential(
            nn.Linear(embedding_dim + user_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )
        
        # 物品嵌入矩阵（用于softmax）
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        
    def forward(self, user_ids, user_features, item_ids):
        """
        训练阶段：使用sampled softmax
        """
        # 用户侧
        user_emb = self.user_embedding(user_ids)
        user_input = torch.cat([user_emb, user_features], dim=1)
        user_vec = self.user_net(user_input)
        
        # 物品侧
        item_emb = self.item_embeddings(item_ids)
        
        # 相似度
        scores = (user_vec * item_emb).sum(dim=1)
        
        return scores
    
    def predict(self, user_ids, user_features, item_embeddings):
        """
        推理阶段：使用annoy/faiss进行近似最近邻搜索
        """
        user_emb = self.user_embedding(user_ids)
        user_input = torch.cat([user_emb, user_features], dim=1)
        user_vec = self.user_net(user_input)
        
        # 归一化
        user_vec = F.normalize(user_vec, dim=1)
        
        return user_vec

# YouTube模型的关键特点：
# 1. 使用用户历史观看序列作为输入
# 2. 使用sampled softmax处理大规模负样本
# 3. 输出embedding用于ANN检索
# 4. 强调示例权重平衡（避免热门物品主导）
```

---

## 四、向量检索：Faiss实战

### 4.1 Faiss基础使用

Faiss是Facebook开源的高效相似度搜索库：

```python
import numpy as np
import faiss

# 1. 创建索引
def create_index(embedding_dim, index_type='flat'):
    """
    Args:
        embedding_dim: embedding维度
        index_type: 'flat', 'ivf', 'hnsw'
    """
    if index_type == 'flat':
        # 精确搜索（基线）
        index = faiss.IndexFlatL2(embedding_dim)
    elif index_type == 'ivf':
        # IVF（倒排文件）
        quantizer = faiss.IndexFlatL2(embedding_dim)
        index = faiss.IndexIVFFlat(
            quantizer, 
            embedding_dim, 
            nlist=100  # 聚类中心数量
        )
    elif index_type == 'hnsw':
        # HNSW（层次化小世界图）
        index = faiss.IndexHNSWFlat(embedding_dim, M=16)
    
    return index

# 2. 构建物品索引库
def build_item_index(item_embeddings):
    """
    Args:
        item_embeddings: [num_items, embedding_dim] numpy array
    """
    num_items, embedding_dim = item_embeddings.shape
    
    # 使用IVF+PQ（乘积量化）加速
    quantizer = faiss.IndexFlatL2(embedding_dim)
    nlist = min(100, num_items // 10)  # 聚类中心数量
    
    index = faiss.IndexIVFPQ(
        quantizer,
        embedding_dim,
        nlist,
        64,    # PQ子向量数量
        8      # 每个子向量的bit数
    )
    
    # 训练索引
    index.train(item_embeddings.astype('float32'))
    
    # 添加向量
    index.add(item_embeddings.astype('float32'))
    
    return index

# 3. 检索Top-K
def retrieve_top_k(index, user_embedding, k=100):
    """
    Args:
        index: faiss索引
        user_embedding: [embedding_dim] or [batch_size, embedding_dim]
        k: 返回top-k结果
    Returns:
        distances, indices
    """
    user_embedding = np.array(user_embedding, dtype='float32')
    
    if user_embedding.ndim == 1:
        user_embedding = user_embedding.reshape(1, -1)
    
    distances, indices = index.search(user_embedding, k)
    
    return distances, indices

# 完整示例
if __name__ == "__main__":
    # 模拟数据：100万物品，128维embedding
    num_items = 1000000
    embedding_dim = 128
    
    print(f"Creating {num_items} item embeddings...")
    item_embeddings = np.random.randn(num_items, embedding_dim).astype('float32')
    
    # 构建索引
    print("Building index...")
    index = build_item_index(item_embeddings)
    
    # 模拟用户embedding
    user_embedding = np.random.randn(embedding_dim).astype('float32')
    
    # 检索
    print("Retrieving top 100 items...")
    distances, indices = retrieve_top_k(index, user_embedding, k=100)
    
    print(f"Top 10 item indices: {indices[0][:10]}")
    print(f"Top 10 distances: {distances[0][:10]}")
```

### 4.2 HNSW索引优化

HNSW (Hierarchical Navigable Small World) 是目前最快的ANN算法之一：

```python
def build_hnsw_index(embeddings, M=16, efConstruction=200):
    """
    构建HNSW索引
    
    Args:
        embeddings: [n, d] 向量矩阵
        M: 每个节点的连接数（越大越精确但越慢）
        efConstruction: 构建时的搜索范围（越大质量越好）
    """
    embedding_dim = embeddings.shape[1]
    
    # 创建HNSW索引
    index = faiss.IndexHNSWFlat(embedding_dim, M)
    index.hnsw.efConstruction = efConstruction
    index.hnsw.efSearch = 100  # 搜索时的范围
    
    # 添加向量
    index.add(embeddings.astype('float32'))
    
    return index

def compare_index_performance():
    """比较不同索引的性能"""
    import time
    
    # 测试数据
    n_items = 100000
    embedding_dim = 128
    embeddings = np.random.randn(n_items, embedding_dim).astype('float32')
    queries = np.random.randn(100, embedding_dim).astype('float32')
    
    # 测试不同索引
    indexes = {
        'Flat-L2': faiss.IndexFlatL2(embedding_dim),
        'IVF-PQ': None,
        'HNSW': None
    }
    
    # IVF-PQ需要训练
    quantizer = faiss.IndexFlatL2(embedding_dim)
    indexes['IVF-PQ'] = faiss.IndexIVFPQ(
        quantizer, embedding_dim, 100, 64, 8
    )
    indexes['IVF-PQ'].train(embeddings)
    
    # HNSW
    indexes['HNSW'] = faiss.IndexHNSWFlat(embedding_dim, 16)
    
    # 添加向量并测试
    results = {}
    
    for name, index in indexes.items():
        # 添加向量
        if name == 'IVF-PQ':
            index.add(embeddings)
        else:
            index.add(embeddings)
        
        # 测试检索速度
        start = time.time()
        _, _ = index.search(queries, k=100)
        elapsed = time.time() - start
        
        results[name] = {
            'time': elapsed * 1000,  # ms
            'qps': 100 / elapsed
        }
    
    # 打印结果
    print("\n索引性能对比:")
    print("-" * 50)
    for name, metrics in results.items():
        print(f"{name:15s}: {metrics['time']:6.2f}ms ({metrics['qps']:6.1f} QPS)")

# 运行性能测试
compare_index_performance()
```

---

## 五、多路召回策略

### 5.1 召回通路设计

工业界通常使用多路召回：

```python
class MultiPathRetriever:
    """多路召回器"""
    def __init__(self):
        self.retrievers = {}
    
    def add_retriever(self, name, retriever, weight=1.0):
        """添加召回通路"""
        self.retrievers[name] = {
            'model': retriever,
            'weight': weight
        }
    
    def retrieve(self, user_features, top_k=100):
        """多路召回"""
        all_candidates = {}
        
        for name, config in self.retrievers.items():
            # 单路召回
            candidates = config['model'].retrieve(
                user_features, 
                top_k=top_k
            )
            
            # 加权
            weight = config['weight']
            for item_id, score in candidates:
                if item_id not in all_candidates:
                    all_candidates[item_id] = 0
                all_candidates[item_id] += score * weight
        
        # 排序并返回top-k
        sorted_items = sorted(
            all_candidates.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k]
        
        return sorted_items

# 示例：定义不同的召回通路
class CollaborativeFilteringRetriever:
    """协同过滤召回"""
    def __init__(self, user_item_matrix):
        self.matrix = user_item_matrix
    
    def retrieve(self, user_id, top_k=100):
        # 基于用户历史行为召回相似用户的偏好物品
        pass

class ContentBasedRetriever:
    """内容召回"""
    def __init__(self, item_features):
        self.item_features = item_features
    
    def retrieve(self, user_profile, top_k=100):
        # 基于用户兴趣标签召回相关物品
        pass

class DeepLearningRetriever:
    """深度学习召回"""
    def __init__(self, model, item_index):
        self.model = model
        self.item_index = item_index
    
    def retrieve(self, user_features, top_k=100):
        # 使用双塔模型+Faiss召回
        user_emb = self.model.get_user_embedding(user_features)
        distances, indices = self.item_index.search(user_emb, top_k)
        return list(zip(indices[0], -distances[0]))

# 组合多路召回
multi_retriever = MultiPathRetriever()
multi_retriever.add_retriever('cf', CollaborativeFilteringRetriever(matrix), weight=0.3)
multi_retriever.add_retriever('content', ContentBasedRetriever(item_features), weight=0.3)
multi_retriever.add_retriever('dnn', DeepLearningRetriever(model, index), weight=0.4)

candidates = multi_retriever.retrieve(user_features, top_k=500)
```

### 5.2 召回融合策略

```python
def recall_fusion_strategies(candidates_list):
    """
    多路召回融合策略
    
    Args:
        candidates_list: [(path_name, [(item_id, score), ...]), ...]
    """
    strategies = {}
    
    # 1. 加权融合
    def weighted_fusion(candidates_list, weights=None):
        if weights is None:
            weights = {name: 1.0 for name, _ in candidates_list}
        
        fused = {}
        for name, candidates in candidates_list:
            weight = weights.get(name, 1.0)
            for item_id, score in candidates:
                if item_id not in fused:
                    fused[item_id] = 0
                fused[item_id] += score * weight
        
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)
    
    # 2. Rank融合（Reciprocal Rank Fusion）
    def rrf_fusion(candidates_list, k=60):
        """RRF: 不依赖分数，只依赖排序"""
        fused = {}
        for name, candidates in candidates_list:
            for rank, (item_id, _) in enumerate(candidates):
                if item_id not in fused:
                    fused[item_id] = 0
                fused[item_id] += 1.0 / (k + rank + 1)
        
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)
    
    # 3. 级联融合
    def cascade_fusion(candidates_list):
        """逐路过滤"""
        final_items = set()
        
        for name, candidates in candidates_list:
            # 从每路取top-k，然后合并
            final_items.update([item_id for item_id, _ in candidates[:100]])
        
        return list(final_items)
    
    strategies['weighted'] = weighted_fusion
    strategies['rrf'] = rrf_fusion
    strategies['cascade'] = cascade_fusion
    
    return strategies
```

---

## 六、工程实践与优化

### 6.1 倒排索引实现

```python
from collections import defaultdict

class InvertedIndex:
    """倒排索引实现"""
    def __init__(self):
        self.index = defaultdict(list)  # {feature: [(item_id, score), ...]}
        self.item_features = {}  # {item_id: {feature: value}}
    
    def add_item(self, item_id, features):
        """添加物品到索引"""
        self.item_features[item_id] = features
        
        for feature, value in features.items():
            self.index[feature].append((item_id, value))
    
    def retrieve(self, query_features, top_k=100):
        """基于特征检索"""
        candidates = defaultdict(float)
        
        for feature, query_value in query_features.items():
            if feature in self.index:
                for item_id, item_value in self.index[feature]:
                    # 计算匹配度
                    score = self._compute_score(query_value, item_value)
                    candidates[item_id] += score
        
        # 排序
        sorted_candidates = sorted(
            candidates.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k]
        
        return sorted_candidates
    
    def _compute_score(self, query_value, item_value):
        # 简单的匹配度计算
        if isinstance(query_value, set) and isinstance(item_value, set):
            return len(query_value & item_value) / len(query_value | item_value)
        return 1.0 if query_value == item_value else 0.0

# 使用示例
inverted_index = InvertedIndex()

# 添加物品
inverted_index.add_item(1, {'category': 'electronics', 'tags': {'tech', 'gadget'}})
inverted_index.add_item(2, {'category': 'electronics', 'tags': {'tech', 'audio'}})
inverted_index.add_item(3, {'category': 'books', 'tags': {'fiction'}})

# 检索
query = {'category': 'electronics', 'tags': {'tech'}}
results = inverted_index.retrieve(query, top_k=10)
```

### 6.2 实时更新索引

```python
class RealTimeIndexUpdater:
    """实时索引更新器"""
    def __init__(self, base_index, update_threshold=1000):
        self.base_index = base_index
        self.updates = []
        self.update_threshold = update_threshold
    
    def add_update(self, item_id, item_embedding):
        """添加更新到缓冲区"""
        self.updates.append((item_id, item_embedding))
        
        # 达到阈值后批量更新
        if len(self.updates) >= self.update_threshold:
            self.flush_updates()
    
    def flush_updates(self):
        """批量更新索引"""
        if not self.updates:
            return
        
        # 添加到主索引
        item_ids = np.array([item_id for item_id, _ in self.updates])
        embeddings = np.array([emb for _, emb in self.updates])
        
        self.base_index.add(embeddings.astype('float32'))
        
        # 清空缓冲区
        self.updates = []
        
        print(f"Flushed {len(item_ids)} updates to index")

# 使用增量索引更新
# 避免每次更新都重建整个索引
```

---

## 七、工业界案例：阿里巴巴召回架构

### 7.1 阿里召回体系

```python
class AlibabaRetrievalSystem:
    """阿里巴巴召回系统架构（简化版）"""
    def __init__(self):
        # 多路召回
        self.paths = {
            'user_cf': None,      # 用户协同过滤
            'item_cf': None,      # 物品协同过滤
            'content': None,      # 内容召回
            'dnn': None,          # 深度召回
            'sequence': None,     # 序列召回（ DIN/DSIN）
            'graph': None         # 图召回（EGES）
        }
    
    def retrieve(self, user_id, user_features, top_k=500):
        """多路召回"""
        all_candidates = {}
        
        # 1. 用户协同过滤
        cf_candidates = self._user_cf_retrieve(user_id, top_k=100)
        for item_id, score in cf_candidates:
            all_candidates[item_id] = all_candidates.get(item_id, 0) + score * 0.2
        
        # 2. 深度召回
        dnn_candidates = self._dnn_retrieve(user_features, top_k=200)
        for item_id, score in dnn_candidates:
            all_candidates[item_id] = all_candidates.get(item_id, 0) + score * 0.4
        
        # 3. 序列召回
        seq_candidates = self._sequence_retrieve(user_id, top_k=100)
        for item_id, score in seq_candidates:
            all_candidates[item_id] = all_candidates.get(item_id, 0) + score * 0.3
        
        # 4. 图召回
        graph_candidates = self._graph_retrieve(user_id, top_k=100)
        for item_id, score in graph_candidates:
            all_candidates[item_id] = all_candidates.get(item_id, 0) + score * 0.1
        
        # 排序返回
        sorted_candidates = sorted(
            all_candidates.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return sorted_candidates
    
    def _user_cf_retrieve(self, user_id, top_k=100):
        # 基于用户相似度召回
        pass
    
    def _dnn_retrieve(self, user_features, top_k=200):
        # 使用双塔模型+Faiss召回
        pass
    
    def _sequence_retrieve(self, user_id, top_k=100):
        # 基于用户历史序列召回
        pass
    
    def _graph_retrieve(self, user_id, top_k=100):
        # 基于物品关系图召回
        pass
```

---

## 总结

本讲深入探讨了推荐系统召回层的核心技术：

**核心要点回顾**：
1. **召回层定位**：海量筛选、快速响应、覆盖多样性
2. **双塔模型**：用户塔+物品塔，Sampled Softmax训练
3. **DSSM**：深度语义匹配，适用于文本召回
4. **向量检索**：Faiss/HNSW实现高效ANN搜索
5. **多路召回**：CF、内容、DNN、序列、图等多路融合
6. **工程优化**：倒排索引、实时更新、缓存策略

**实践建议**：
- 召回层核心是**速度与质量的平衡**
- 多路召回需要**调权融合**
- 定期**更新索引**保证时效性
- 监控**召回覆盖率**和**多样性**

---

## 参考资料

1. **论文**：
   - Covington et al. "Deep Neural Networks for YouTube Recommendations" (RecSys '16)
   - Huang et al. "Deep Structured Semantic Models" (CIKM '13)
   - Malkov & Yashunin. "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs" (2018)

2. **开源工具**：
   - Faiss: https://github.com/facebookresearch/faiss
   - Annoy: https://github.com/spotify/annoy
   - Hnswlib: https://github.com/nmslib/hnswlib

3. **工业实践**：
   - YouTube Recommendations: https://research.google/pubs/pub45530/
   - 阿里推荐技术：https://www.alibabacloud.com/zh/tech-guide/first-glance-at-personalized-recommendation

---

**第54讲完**。下讲将深入探讨排序学习（Learning to Rank）在推荐系统中的应用。
