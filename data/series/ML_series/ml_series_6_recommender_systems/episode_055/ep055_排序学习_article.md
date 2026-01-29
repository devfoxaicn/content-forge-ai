# 第55讲：排序学习
## 推荐系统的精准排序引擎

---

## 课程概览

排序学习是推荐系统的核心组件，负责从召回的候选集中精准排序，最终展示给用户。本讲将深入探讨LTR的三个范式（Pointwise、Pairwise、Listwise），经典算法（LambdaMART、LambdaRank），以及在搜索排名、信息流排序中的应用。

**核心要点**：
- LTR三个范式的对比与选择
- LambdaMART算法原理与实现
- 深度LTR模型：RankNet、LambdaRank
- 多目标排序与帕累托优化
- 在搜索、推荐中的工程实践

---

## 一、排序学习基础

### 1.1 什么是排序学习

排序学习的目标：学习一个排序函数 f(x, q)，使得相关文档排在前面。

```python
def ranking_model(score):
    """排序模型的核心"""
    if score > 0.8:
        return "Highly Relevant"
    elif score > 0.5:
        return "Relevant"
    else:
        return "Not Relevant"

# 传统机器学习 vs 排序学习
# 传统: 关注单样本的预测准确性
# LTR: 关注整个序列的顺序正确性
```

### 1.2 评估指标

```python
import numpy as np

def dcg(relevances, k=10):
    """Discounted Cumulative Gain"""
    relevances = np.array(relevances)[:k]
    discounts = np.log2(np.arange(len(relevances)) + 2)
    return np.sum(relevances / discounts)

def ndcg(relevances, k=10):
    """Normalized DCG"""
    actual_dcg = dcg(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)[:k]
    ideal_dcg = dcg(ideal_relevances, k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0

def mean_average_precision(relevances, k=10):
    """Mean Average Precision"""
    relevances = np.array(relevances)[:k]
    if relevances.sum() == 0:
        return 0.0
    
    precisions = []
    num_relevant = 0
    
    for i, rel in enumerate(relevances):
        if rel == 1:
            num_relevant += 1
            precisions.append(num_relevant / (i + 1))
    
    return np.mean(precisions) if precisions else 0.0

# 示例
relevances = [3, 2, 1, 0, 2, 1, 0, 0, 1, 0]
print(f"DCG@10: {dcg(relevances, 10):.3f}")
print(f"NDCG@10: {ndcg(relevances, 10):.3f}")
print(f"MAP@10: {mean_average_precision([1,1,1,0,0,0,0,0,0,0], 10):.3f}")
```

---

## 二、三个范式：Pointwise、Pairwise、Listwise

### 2.1 Pointwise方法

将排序问题转化为分类或回归问题：

```python
import torch
import torch.nn as nn

class PointwiseRanker(nn.Module):
    """Pointwise排序模型"""
    def __init__(self, feature_dim, hidden_dim=64):
        super(PointwiseRanker, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 输出相关性分数
        )
    
    def forward(self, features):
        return self.net(features)

# Pointwise的问题：
# 1. 忽略了文档间的相对顺序
# 2. 损失函数与排序指标不一致
```

### 2.2 Pairwise方法

考虑文档对的相对顺序：

```python
class RankNet(nn.Module):
    """RankNet: Pairwise排序模型"""
    def __init__(self, feature_dim, hidden_dim=64):
        super(RankNet, self).__init__()
        
        # 共享的打分网络
        self.scoring_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, doc_i_features, doc_j_features):
        """
        Args:
            doc_i_features: [batch_size, feature_dim]
            doc_j_features: [batch_size, feature_dim]
        Returns:
            prob_i_better: P(doc_i > doc_j)
        """
        # 计算两个文档的分数
        score_i = self.scoring_net(doc_i_features)  # [B, 1]
        score_j = self.scoring_net(doc_j_features)  # [B, 1]
        
        # 计算分数差
        score_diff = score_i - score_j  # [B, 1]
        
        # Sigmoid转换
        prob_i_better = torch.sigmoid(score_diff)  # [B, 1]
        
        return prob_i_better
    
    def forward_single(self, features):
        """对单个文档打分"""
        return self.scoring_net(features)

# RankNet损失函数
def ranknet_loss(prob_i_better, label):
    """
    Args:
        prob_i_better: P(doc_i > doc_j) 模型预测
        label: 1 if doc_i > doc_j, 0 if doc_i < doc_j, 0.5 if equal
    """
    return -label * torch.log(prob_i_better) - (1 - label) * torch.log(1 - prob_i_better)

# 使用示例
model = RankNet(feature_dim=50, hidden_dim=64)

# 模拟数据
batch_size = 32
doc_i = torch.randn(batch_size, 50)
doc_j = torch.randn(batch_size, 50)
labels = torch.randint(0, 2, (batch_size, 1)).float()  # 0 or 1

# 前向传播
prob = model(doc_i, doc_j)
loss = ranknet_loss(prob, labels)

print(f"RankNet Loss: {loss.item():.4f}")
```

### 2.3 Listwise方法

直接优化整个列表的排序指标：

```python
class ListWiseRanker(nn.Module):
    """ListWise排序模型：直接优化NDCG等指标"""
    def __init__(self, feature_dim, hidden_dim=64):
        super(ListWiseRanker, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, features_list):
        """
        Args:
            features_list: List of [feature_dim] tensors
        Returns:
            scores: List of scores
        """
        scores = []
        for features in features_list:
            score = self.net(features)
            scores.append(score)
        return torch.cat(scores)

# ListNet：使用Softmax作为概率分布
def listnet_loss(predicted_scores, true_scores):
    """
    ListNet使用Softmax构建概率分布
    
    Args:
        predicted_scores: [n_docs] 预测分数
        true_scores: [n_docs] 真实相关性分数
    """
    # Softmax归一化
    pred_probs = torch.softmax(predicted_scores, dim=0)
    true_probs = torch.softmax(true_scores, dim=0)
    
    # KL散度损失
    loss = torch.sum(true_probs * torch.log(true_probs / (pred_probs + 1e-10) + 1e-10))
    
    return loss
```

---

## 三、LambdaMART：最强大的LTR算法

### 3.1 Lambda梯度原理

LambdaMART的核心是通过Lambda梯度直接优化NDCG：

```python
def compute_lambda(doc_scores, true_relevances, rank=10):
    """
    计算Lambda梯度（用于LambdaMART）
    
    Args:
        doc_scores: [n_docs] 模型预测分数
        true_relevances: [n_docs] 真实相关性
        rank: 计算NDCG的位置
    """
    n_docs = len(doc_scores)
    lambdas = np.zeros(n_docs)
    
    # 获取排序
    sorted_indices = np.argsort(-doc_scores)
    
    for i in range(n_docs):
        for j in range(n_docs):
            if i == j:
                continue
            
            # 计算NDCG变化
            rel_i = true_relevances[i]
            rel_j = true_relevances[j]
            
            if rel_i == rel_j:
                continue  # 相关性相同，不更新
            
            # 计算交换后的NDCG变化
            rank_i = np.where(sorted_indices == i)[0][0]
            rank_j = np.where(sorted_indices == j)[0][0]
            
            # 当前NDCG贡献
            discount_i = 1.0 / np.log2(rank_i + 2)
            discount_j = 1.0 / np.log2(rank_j + 2)
            
            # 交换后的NDCG贡献
            new_discount_i = 1.0 / np.log2(rank_j + 2)
            new_discount_j = 1.0 / np.log2(rank_i + 2)
            
            # NDCG变化量
            delta_ndcg = (rel_i - rel_j) * (new_discount_i - discount_i) + \
                         (rel_j - rel_i) * (new_discount_j - discount_j)
            
            # Lambda梯度
            score_diff = doc_scores[i] - doc_scores[j]
            lambda_ij = delta_ndcg * (1.0 / (1.0 + np.exp(score_diff)))
            
            lambdas[i] += lambda_ij
            lambdas[j] -= lambda_ij
    
    return lambdas

# LambdaMART关键：直接优化NDCG梯度
```

### 3.2 MART（Multiple Additive Regression Trees）

```python
class MART:
    """MART: 梯度提升决策树"""
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = None
    
    def fit(self, X, y):
        """
        Args:
            X: [n_samples, n_features] 特征
            y: [n_samples] 目标值（可以是lambda梯度）
        """
        from sklearn.tree import DecisionTreeRegressor
        
        # 初始预测
        self.initial_prediction = np.mean(y)
        predictions = np.full(len(y), self.initial_prediction)
        
        for _ in range(self.n_estimators):
            # 计算负梯度（残差）
            residuals = y - predictions
            
            # 拟合残差
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            
            # 更新预测
            predictions += self.learning_rate * tree.predict(X)
            
            self.trees.append(tree)
    
    def predict(self, X):
        """预测"""
        predictions = np.full(len(X), self.initial_prediction)
        
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        
        return predictions

# LambdaMART = MART + Lambda梯度
class LambdaMART:
    """LambdaMART: Lambda梯度 + MART"""
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.mart = MART(n_estimators, learning_rate, max_depth)
    
    def fit(self, X, true_relevances, query_ids):
        """
        Args:
            X: [n_docs, n_features]
            true_relevances: [n_docs] 真实相关性
            query_ids: [n_docs] 查询ID（同一query的文档需要一起排序）
        """
        from sklearn.tree import DecisionTreeRegressor
        
        n_docs = len(X)
        
        # 初始预测
        predictions = np.zeros(n_docs)
        
        for iteration in range(self.n_estimators):
            # 计算Lambda梯度
            lambdas = np.zeros(n_docs)
            
            # 按query分组计算lambda
            unique_queries = np.unique(query_ids)
            for qid in unique_queries:
                mask = query_ids == qid
                q_predictions = predictions[mask]
                q_relevances = true_relevances[mask]
                
                # 计算该query的lambda梯度
                q_lambdas = compute_lambda(q_predictions, q_relevances)
                lambdas[mask] = q_lambdas
            
            # 用MART拟合lambda梯度
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, lambdas)
            
            # 更新预测
            predictions += self.learning_rate * tree.predict(X)
            
            self.mart.trees.append(tree)
            
            # 打印进度
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.n_estimators} completed")
    
    def predict(self, X):
        """预测分数"""
        predictions = np.zeros(len(X))
        
        for tree in self.mart.trees:
            predictions += self.learning_rate * tree.predict(X)
        
        return predictions

# 使用示例
if __name__ == "__main__":
    # 模拟数据
    n_docs = 1000
    n_features = 50
    n_queries = 50
    
    X = np.random.randn(n_docs, n_features)
    true_relevances = np.random.randint(0, 5, n_docs)
    query_ids = np.random.randint(0, n_queries, n_docs)
    
    # 训练LambdaMART
    lambdamart = LambdaMART(n_estimators=50, learning_rate=0.1)
    lambdamart.fit(X, true_relevances, query_ids)
    
    # 预测
    scores = lambdamart.predict(X)
    
    print(f"Predictions shape: {scores.shape}")
    print(f"Sample scores: {scores[:5]}")
```

---

## 四、深度LTR模型

### 4.1 深度RankNet

```python
import torch.nn as nn

class DeepRankNet(nn.Module):
    """深度RankNet：更复杂的网络结构"""
    def __init__(self, feature_dim, hidden_layers=[256, 128, 64]):
        super(DeepRankNet, self).__init__()
        
        # 构建打分网络
        layers = []
        input_dim = feature_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_layers[-1], 1))
        
        self.scoring_net = nn.Sequential(*layers)
    
    def forward(self, features):
        return self.scoring_net(features)
    
    def pairwise_loss(self, doc_i_features, doc_j_features, labels):
        """
        Args:
            doc_i_features: [B, D]
            doc_j_features: [B, D]
            labels: [B] 1 if i > j, 0 otherwise
        """
        score_i = self.forward(doc_i_features)
        score_j = self.forward(doc_j_features)
        
        score_diff = score_i - score_j
        prob_i_better = torch.sigmoid(score_diff)
        
        loss = nn.BCELoss()(prob_i_better.squeeze(), labels)
        
        return loss

# 训练示例
def train_deep_ranknet(model, train_loader, optimizer, epochs=10):
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        
        for batch in train_loader:
            doc_i, doc_j, labels = batch
            
            optimizer.zero_grad()
            loss = model.pairwise_loss(doc_i, doc_j, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
```

### 4.2 LambdaRank

```python
class LambdaRank(nn.Module):
    """LambdaRank: 直接优化NDCG的深度模型"""
    def __init__(self, feature_dim, hidden_dim=128):
        super(LambdaRank, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, features):
        return self.net(features)
    
    def lambda_loss(self, features, true_relevances):
        """
        Lambda损失：考虑NDCG变化
        
        Args:
            features: [n_docs, feature_dim]
            true_relevances: [n_docs]
        """
        # 计算预测分数
        scores = self.forward(features).squeeze()
        
        # 计算Lambda梯度
        lambdas = compute_lambda(
            scores.detach().numpy(),
            true_relevances.numpy()
        )
        lambdas = torch.tensor(lambdas, dtype=torch.float32)
        
        # MSE损失（使用lambda作为目标）
        loss = nn.MSELoss()(scores, lambdas)
        
        return loss
```

---

## 五、多目标排序

### 5.1 多目标学习

```python
class MultiObjectiveRanker(nn.Module):
    """多目标排序模型"""
    def __init__(self, feature_dim, hidden_dim=128, num_objectives=3):
        super(MultiObjectiveRanker, self).__init__()
        
        # 共享特征提取
        self.shared_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 多个任务头
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            ) for _ in range(num_objectives)
        ])
    
    def forward(self, features):
        """
        Returns:
            scores: [num_objectives, n_docs]
        """
        shared_features = self.shared_net(features)
        
        scores = []
        for head in self.task_heads:
            scores.append(head(shared_features))
        
        return torch.cat(scores, dim=1)  # [n_docs, num_objectives]
    
    def weighted_score(self, features, weights):
        """
        加权融合多目标分数
        
        Args:
            weights: [num_objectives] 各目标权重
        """
        scores = self.forward(features)  # [n_docs, num_objectives]
        
        weighted_score = (scores * weights).sum(dim=1)
        
        return weighted_score

# 帕累托最优
def pareto_optimal_solutions(scores_matrix):
    """
    找到帕累托最优解
    
    Args:
        scores_matrix: [n_solutions, n_objectives]
    Returns:
        pareto_indices: 帕累托最优解的索引
    """
    n_solutions = scores_matrix.shape[0]
    is_pareto = np.ones(n_solutions, dtype=bool)
    
    for i in range(n_solutions):
        for j in range(n_solutions):
            if i == j:
                continue
            
            # 如果j在所有目标上都>=i，且至少一个>i，则i不是帕累托最优
            if np.all(scores_matrix[j] >= scores_matrix[i]) and \
               np.any(scores_matrix[j] > scores_matrix[i]):
                is_pareto[i] = False
                break
    
    return np.where(is_pareto)[0]
```

### 5.2 多目标优化策略

```python
class MultiObjectiveOptimization:
    """多目标优化策略"""
    
    @staticmethod
    def weighted_sum(scores, weights):
        """加权求和"""
        return (scores * weights).sum(axis=1)
    
    @staticmethod
    def product_of_scores(scores, weights):
        """加权乘积"""
        weighted = scores ** weights
        return np.prod(weighted, axis=1)
    
    @staticmethod
    def constrained_optimization(scores, constraints):
        """
        约束优化
        
        Args:
            scores: [n_docs, n_objectives]
            constraints: {objective_id: min_value}
        """
        mask = np.ones(len(scores), dtype=bool)
        
        for obj_id, min_value in constraints.items():
            mask &= (scores[:, obj_id] >= min_value)
        
        return mask

# 使用示例
# scores: [CTR, CVR, dwell_time]
scores = np.array([
    [0.8, 0.3, 120],  # 方案1
    [0.6, 0.5, 180],  # 方案2
    [0.7, 0.4, 150],  # 方案3
])

# 加权融合
weights = np.array([0.5, 0.3, 0.2])
final_scores = MultiObjectiveOptimization.weighted_sum(scores, weights)
print(f"Final scores: {final_scores}")
```

---

## 六、工程实践

### 6.1 搜索排名pipeline

```python
class SearchRankingPipeline:
    """搜索排序pipeline"""
    def __init__(self, retrieval_model, ranking_model):
        self.retrieval_model = retrieval_model
        self.ranking_model = ranking_model
    
    def search(self, query, top_k=100):
        """搜索并排序"""
        # 1. 召回
        candidates = self.retrieval_model.retrieve(query, top_k=500)
        
        # 2. 特征提取
        features = self._extract_features(query, candidates)
        
        # 3. 排序
        scores = self.ranking_model.predict(features)
        
        # 4. 重新排序
        ranked_docs = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return ranked_docs
    
    def _extract_features(self, query, documents):
        """提取排序特征"""
        features = []
        
        for doc in documents:
            # 文本相似度
            text_sim = self._text_similarity(query, doc)
            
            # 页面质量
            page_quality = self._page_quality_score(doc)
            
            # 点击历史
            ctr = self._get_historical_ctr(doc)
            
            features.append([text_sim, page_quality, ctr])
        
        return np.array(features)
```

### 6.2 信息流排序

```python
class FeedRankingModel(nn.Module):
    """信息流排序模型"""
    def __init__(self, feature_dim, user_dim, item_dim):
        super(FeedRankingModel, self).__init__()
        
        # 用户特征
        self.user_net = nn.Sequential(
            nn.Linear(user_dim, 64),
            nn.ReLU()
        )
        
        # 物品特征
        self.item_net = nn.Sequential(
            nn.Linear(item_dim, 128),
            nn.ReLU()
        )
        
        # 交叉特征
        self.cross_net = nn.Sequential(
            nn.Linear(64 + 128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, user_features, item_features, context_features):
        """
        Args:
            user_features: [B, user_dim]
            item_features: [B, item_dim]
            context_features: [B, context_dim]
        """
        user_vec = self.user_net(user_features)
        item_vec = self.item_net(item_features)
        
        # 拼接
        combined = torch.cat([user_vec, item_vec, context_features], dim=1)
        
        # 排序分数
        scores = self.cross_net(combined)
        
        return scores
```

---

## 总结

本讲深入探讨了排序学习的核心技术：

**核心要点回顾**：
1. **三个范式**：Pointwise（单点）、Pairwise（成对）、Listwise（列表）
2. **LambdaMART**：通过Lambda梯度直接优化NDCG
3. **深度LTR**：RankNet、LambdaRank的深度学习版本
4. **多目标排序**：帕累托优化、加权融合
5. **工程实践**：搜索排名、信息流排序

**实践建议**：
- Pointwise最简单但效果有限
- Pairwise平衡效果与复杂度
- Listwise直接优化排序指标
- LambdaMART是业界最强传统算法
- 多目标排序需要业务经验调权

---

## 参考资料

1. **论文**：
   - Burges et al. "Learning to Rank using Gradient Descent" (ICML '05)
   - Burges et al. "From RankNet to LambdaRank to LambdaMART: An Overview" (2010)
   - Cao et al. "Learning to Rank: From Pairwise Approach to Listwise Approach" (ICML '07)

2. **开源工具**：
   - XGBoost: 支持LambdaMART
   - LightGBM: 高效的LTR实现
   - RankLib: Java实现的LTR算法库

---

**第55讲完**。下讲将深入探讨重排策略与多样性优化。
