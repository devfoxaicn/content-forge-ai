# 第52讲：协同过滤算法深入解析

## 课程概览

协同过滤是推荐系统中最经典、最广泛使用的算法之一。从Netflix Prize竞赛的辉煌到现代工业推荐系统的基础，协同过滤思想影响深远。本讲将系统剖析协同过滤的原理、变体和实践技巧。

**核心学习目标**：
- 理解用户-物品矩阵的稀疏性挑战
- 掌握User-based和Item-based CF的实现
- 深入理解矩阵分解（SVD、SVD++、ALS）
- 学习隐语义模型的原理与应用

---

## 一、协同过滤的核心思想

### 1.1 基本假设

**核心直觉**：
> "物以类聚，人以群分"

协同过滤基于以下观察：
1. **用户相似性**：相似用户喜欢相似物品
2. **物品相似性**：相似物品被相似用户喜欢
3. **历史行为预测**：用户过去的行为模式可预测未来行为

### 1.2 用户-物品矩阵

**矩阵表示**：

```
        物品1  物品2  物品3  物品4  物品5
用户1    5     3     ?     1     ?
用户2    4     ?     ?     1     ?
用户3    ?     3     ?     4     5
用户4    1     ?     5     4     ?
用户5    ?     1     5     ?     3
```

**核心挑战**：
- 矩阵极其稀疏（99%以上为空）
- 评分规模不一致（1-5分 vs 点击/未点击）
- 冷启动问题（新用户/新物品）
- 扩展性问题（用户数×物品数可能达到10^15）

**数据类型**：

| 类型 | 示例 | 特点 |
|-----|------|------|
| 显式反馈 | 评分(1-5星)、点赞 | 明确表达偏好 |
| 隐式反馈 | 点击、购买、停留时长 | 反馈噪声大，数据多 |

### 1.3 协同过滤分类

```
协同过滤
├─ 基于记忆 (Memory-based)
│  ├─ User-based CF
│  └─ Item-based CF
└─ 基于模型 (Model-based)
   ├─ 矩阵分解 (MF)
   ├─ 隐语义模型 (LFM)
   ├─ SVD, SVD++
   └─ 深度学习方法
```

---

## 二、User-based协同过滤

### 2.1 核心思想

**假设**：如果用户A在过去和用户B有相似的偏好，那么用户A在未来也会和用户B有相似偏好。

**预测公式**：

```
预测值 = 用户平均评分 + Σ(相似用户对物品的评分偏差 × 相似度) / Σ|相似度|
```

### 2.2 完整算法流程

**步骤1：计算用户相似度**

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def user_similarity_matrix(rating_matrix):
    """
    计算用户相似度矩阵

    参数:
        rating_matrix: 用户-物品评分矩阵 (m×n)
                     m个用户, n个物品

    返回:
        sim_matrix: 用户相似度矩阵 (m×m)
    """
    # 中心化：减去用户平均评分
    user_means = np.nanmean(rating_matrix, axis=1, keepdims=True)
    centered_ratings = rating_matrix - user_means

    # 将NaN替换为0
    centered_ratings = np.nan_to_num(centered_ratings)

    # 余弦相似度
    sim_matrix = cosine_similarity(centered_ratings)

    return sim_matrix

# 示例
R = np.array([
    [5, 3, np.nan, 1, np.nan],
    [4, np.nan, np.nan, 1, np.nan],
    [np.nan, 3, np.nan, 4, 5],
    [1, np.nan, 5, 4, np.nan],
    [np.nan, 1, 5, np.nan, 3]
])

sim = user_similarity_matrix(R)
print(sim)
```

**步骤2：找出K个最相似用户**

```python
def find_top_k_similar_users(user_id, sim_matrix, k=10):
    """
    找出最相似的K个用户

    参数:
        user_id: 目标用户ID
        sim_matrix: 用户相似度矩阵
        k: 返回前K个相似用户

    返回:
        top_k_users: (user_ids, similarities)
    """
    user_sims = sim_matrix[user_id]

    # 排除自己
    user_sims[user_id] = -1

    # 找出top-k
    top_k_indices = np.argsort(user_sims)[-k:][::-1]
    top_k_similarities = user_sims[top_k_indices]

    return top_k_indices, top_k_similarities
```

**步骤3：预测评分**

```python
def predict_rating_user_based(user_id, item_id, rating_matrix,
                               sim_matrix, k=10):
    """
    User-based协同过滤预测评分

    参数:
        user_id: 目标用户
        item_id: 目标物品
        rating_matrix: 评分矩阵
        sim_matrix: 用户相似度矩阵
        k: 使用k个最相似用户

    返回:
        predicted_rating: 预测评分
    """
    # 如果已经有评分，直接返回
    if not np.isnan(rating_matrix[user_id, item_id]):
        return rating_matrix[user_id, item_id]

    # 找出对物品item_id评分过的相似用户
    item_ratings = rating_matrix[:, item_id]
    rated_users = np.where(~np.isnan(item_ratings))[0]

    # 计算这些用户与目标用户的相似度
    similarities = sim_matrix[user_id, rated_users]

    # 选择top-k
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    top_k_users = rated_users[top_k_indices]
    top_k_sims = similarities[top_k_indices]

    # 加权平均预测
    user_mean = np.nanmean(rating_matrix[user_id])

    weighted_sum = 0
    sim_sum = 0

    for u, sim in zip(top_k_users, top_k_sims):
        u_mean = np.nanmean(rating_matrix[u])
        rating = rating_matrix[u, item_id]

        weighted_sum += sim * (rating - u_mean)
        sim_sum += abs(sim)

    if sim_sum == 0:
        return user_mean  # 回退到用户平均

    predicted = user_mean + weighted_sum / sim_sum

    # 限制在评分范围内
    predicted = max(1, min(5, predicted))

    return predicted
```

### 2.3 相似度度量方法

**1. 余弦相似度**：

```python
def cosine_similarity_user(u1_ratings, u2_ratings):
    """
    余弦相似度：测量两个用户评分向量的夹角
    """
    # 找出共同评分的物品
    common_items = ~(np.isnan(u1_ratings) | np.isnan(u2_ratings))

    if not np.any(common_items):
        return 0

    v1 = u1_ratings[common_items]
    v2 = u2_ratings[common_items]

    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0

    return dot_product / (norm1 * norm2)
```

**2. 皮尔逊相关系数**：

```python
def pearson_correlation(u1_ratings, u2_ratings):
    """
    皮尔逊相关系数：考虑评分偏置
    """
    # 找出共同评分的物品
    common_items = ~(np.isnan(u1_ratings) | np.isnan(u2_ratings))

    if np.sum(common_items) < 2:  # 至少2个共同物品
        return 0

    v1 = u1_ratings[common_items]
    v2 = u2_ratings[common_items]

    # 中心化
    v1_centered = v1 - np.mean(v1)
    v2_centered = v2 - np.mean(v2)

    # 计算相关系数
    numerator = np.dot(v1_centered, v2_centered)
    denominator = np.linalg.norm(v1_centered) * np.linalg.norm(v2_centered)

    if denominator == 0:
        return 0

    return numerator / denominator
```

**3. Jaccard相似度**（用于隐式反馈）：

```python
def jaccard_similarity(u1_items, u2_items):
    """
    Jaccard相似度：用于二值数据（点击/未点击）

    sim(A,B) = |A ∩ B| / |A ∪ B|
    """
    intersection = len(set(u1_items) & set(u2_items))
    union = len(set(u1_items) | set(u2_items))

    if union == 0:
        return 0

    return intersection / union
```

### 2.4 优缺点分析

**优点**：
- 直观易理解
- 适合用户数量相对稳定的场景
- 可以发现用户间的共同兴趣

**缺点**：
- 数据稀疏性：用户共同评分物品少
- 扩展性问题：用户数量大时计算成本高
- 冷启动：新用户无相似用户
- 灰羊问题（Gray Sheep）：难以找到相似用户的特殊用户

---

## 三、Item-based协同过滤

### 3.1 核心思想

**假设**：用户喜欢与历史上喜欢的物品相似的物品。

**优势**：
- 物品相似度相对稳定（用户兴趣变化快）
- 物品数量通常少于用户数量
- 可以预先计算物品相似度矩阵

### 3.2 算法实现

```python
def item_similarity_matrix(rating_matrix):
    """
    计算物品相似度矩阵

    参数:
        rating_matrix: 用户-物品评分矩阵 (m×n)

    返回:
        item_sim: 物品相似度矩阵 (n×n)
    """
    num_users, num_items = rating_matrix.shape

    # 中心化：减去物品平均评分
    item_means = np.nanmean(rating_matrix, axis=0, keepdims=True)
    centered_ratings = rating_matrix - item_means

    # 将NaN替换为0
    centered_ratings = np.nan_to_num(centered_ratings)

    # 计算物品间余弦相似度
    item_sim = cosine_similarity(centered_ratings.T)

    return item_sim

def predict_rating_item_based(user_id, item_id, rating_matrix,
                               item_sim_matrix, k=10):
    """
    Item-based协同过滤预测评分

    核心思想：用户对相似物品的评分可以预测对该物品的评分
    """
    # 如果已经有评分
    if not np.isnan(rating_matrix[user_id, item_id]):
        return rating_matrix[user_id, item_id]

    # 找出用户评过分的物品
    user_ratings = rating_matrix[user_id, :]
    rated_items = np.where(~np.isnan(user_ratings))[0]

    if len(rated_items) == 0:
        return np.nanmean(rating_matrix)  # 全局平均

    # 获取这些物品与目标物品的相似度
    similarities = item_sim_matrix[item_id, rated_items]

    # 选择top-k最相似物品
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    top_k_items = rated_items[top_k_indices]
    top_k_sims = similarities[top_k_indices]

    # 加权平均
    weighted_sum = 0
    sim_sum = 0

    for item, sim in zip(top_k_items, top_k_sims):
        rating = rating_matrix[user_id, item]

        if not np.isnan(rating) and sim > 0:
            weighted_sum += sim * rating
            sim_sum += abs(sim)

    if sim_sum == 0:
        return np.nanmean(rating_matrix[user_id])

    predicted = weighted_sum / sim_sum

    return max(1, min(5, predicted))
```

### 3.3 工业实践优化

**预计算策略**：

```python
def precompute_item_similarities(rating_matrix, top_k=100):
    """
    预计算物品相似度（离线批量计算）

    优化：
    1. 只保存top-k相似物品
    2. 使用稀疏矩阵存储
    3. 定期更新（如每天）
    """
    from scipy.sparse import csr_matrix
    from sklearn.neighbors import NearestNeighbors

    # 构建稀疏矩阵
    sparse_matrix = csr_matrix(np.nan_to_num(rating_matrix))

    # 使用近似最近邻（加快搜索）
    model_knn = NearestNeighbors(n_neighbors=top_k+1,
                                 algorithm='auto',
                                 metric='cosine')
    model_knn.fit(sparse_matrix.T)

    # 找出每个物品的top-k相似物品
    distances, indices = model_knn.kneighbors(sparse_matrix.T)

    # 转换为相似度（距离 -> 相似度）
    similarities = 1 - distances

    # 返回稀疏表示
    item_similarities = {}
    for i in range(len(indices)):
        # 排除自己
        neighbors = indices[i][1:]
        sims = similarities[i][1:]

        item_similarities[i] = list(zip(neighbors, sims))

    return item_similarities

# 使用示例
item_sims = precompute_item_similarities(R, top_k=50)

# 在线查询
def get_similar_items(item_id, item_sims, top_k=10):
    """在线获取相似物品"""
    if item_id not in item_sims:
        return []

    similar_items = item_sims[item_id][:top_k]
    return similar_items
```

---

## 四、矩阵分解

### 4.1 核心思想

**直觉**：
- 用户和物品都可以用低维隐向量表示
- 用户对物品的评分 = 用户向量与物品向量的内积
- 捕捉潜在因素（如电影类型、用户偏好）

**数学模型**：

```
R ≈ U × V^T

其中：
- R: 用户-物品评分矩阵 (m×n)
- U: 用户隐因子矩阵 (m×k)
- V: 物品隐因子矩阵 (n×k)
- k: 隐因子数量（通常k << min(m,n)）

预测：r̂_ui = u_i · v_j^T = Σ(u_if × v_jf)
```

### 4.2 SVD分解

**传统SVD**：

```python
import numpy as np
from scipy.linalg import svds

def svd_predict(rating_matrix, k=10):
    """
    奇异值分解(SVD)

    注意：传统SVD要求矩阵无缺失值
    实际应用需要先用均值/零填充缺失值
    """
    # 填充缺失值
    filled_matrix = np.nan_to_num(rating_matrix,
                                  nan=np.nanmean(rating_matrix))

    # 中心化
    user_means = np.mean(filled_matrix, axis=1, keepdims=True)
    centered_matrix = filled_matrix - user_means

    # SVD分解
    U, sigma, Vt = svds(centered_matrix, k=k)

    # 构造对角矩阵
    Sigma = np.diag(sigma)

    # 预测
    predicted = U.dot(Sigma).dot(Vt) + user_means

    return predicted, U, Sigma, Vt

# 使用
predicted, U, Sigma, Vt = svd_predict(R, k=3)
print("预测评分矩阵:")
print(predicted)
```

### 4.3 交替最小二乘（ALS）

**核心思想**：
- 固定用户矩阵U，优化物品矩阵V
- 固定物品矩阵V，优化用户矩阵U
- 交替迭代直至收敛

**数学推导**：

```
目标函数：
min Σ(r_ui - u_i·v_j^T)^2 + λ(||u_i||^2 + ||v_j||^2)
 u,v

ALS步骤：
1. 固定V，求解U：
   u_i = (V^TV + λI)^(-1) V^T r_i

2. 固定U，求解V：
   v_j = (U^TU + λI)^(-1) U^T r_j

3. 重复1-2直至收敛
```

**实现**：

```python
def als_matrix_factorization(R, k=10, lambda_reg=0.01,
                            n_iterations=20):
    """
    交替最小二乘(ALS)矩阵分解

    参数:
        R: 评分矩阵 (m×n)，缺失值为nan
        k: 隐因子数量
        lambda_reg: 正则化参数
        n_iterations: 迭代次数

    返回:
        U: 用户矩阵 (m×k)
        V: 物品矩阵 (n×k)
        predicted: 预测评分矩阵
    """
    # 初始化
    m, n = R.shape
    U = np.random.randn(m, k) * 0.01
    V = np.random.randn(n, k) * 0.01

    # 创建掩码（有效评分位置）
    mask = ~np.isnan(R)

    # ALS迭代
    for iteration in range(n_iterations):
        # 更新U（固定V）
        for i in range(m):
            # 获取用户i的有效评分
            valid_items = mask[i]
            if np.sum(valid_items) == 0:
                continue

            V_j = V[valid_items]
            r_i = R[i, valid_items]

            # 最小二乘求解
            A = V_j.T.dot(V_j) + lambda_reg * np.eye(k)
            b = V_j.T.dot(r_i)

            U[i] = np.linalg.solve(A, b)

        # 更新V（固定U）
        for j in range(n):
            # 获取物品j的有效评分
            valid_users = mask[:, j]
            if np.sum(valid_users) == 0:
                continue

            U_i = U[valid_users]
            r_j = R[valid_users, j]

            # 最小二乘求解
            A = U_i.T.dot(U_i) + lambda_reg * np.eye(k)
            b = U_i.T.dot(r_j)

            V[j] = np.linalg.solve(A, b)

        # 计算损失（可选，用于监控）
        if iteration % 5 == 0:
            predicted = U.dot(V.T)
            loss = np.nansum((R - predicted)**2)
            print(f"Iteration {iteration}, Loss: {loss:.2f}")

    # 最终预测
    predicted = U.dot(V.T)

    return U, V, predicted

# 使用
U, V, predicted = als_matrix_factorization(R, k=5, lambda_reg=0.1)
```

### 4.4 SVD++：增强隐式反馈

**动机**：
- 用户历史行为包含丰富信息
- 显式评分稀疏，隐式反馈丰富

**模型**：

```
预测：r̂_ui = μ + b_u + b_i + q_i^T (p_u + |N(u)|^(-1/2) Σ y_j)

其中：
- μ: 全局平均评分
- b_u, b_i: 用户和物品偏置
- p_u: 用户隐向量
- q_i: 物品隐向量
- N(u): 用户u反馈过的物品集合
- y_j: 隐式反馈的影响向量
```

**简化实现**：

```python
def svd_plus_plus(R, k=10, lambda_reg=0.01,
                  n_iterations=20):
    """
    SVD++：考虑隐式反馈

    简化版：只考虑用户历史物品的影响
    """
    m, n = R.shape

    # 初始化参数
    global_mean = np.nanmean(R)
    b_u = np.zeros(m)  # 用户偏置
    b_i = np.zeros(n)  # 物品偏置
    P = np.random.randn(m, k) * 0.01  # 用户因子
    Q = np.random.randn(n, k) * 0.01  # 物品因子
    Y = np.random.randn(n, k) * 0.01  # 隐式反馈因子

    mask = ~np.isnan(R)

    for iteration in range(n_iterations):
        # 更新用户参数
        for u in range(m):
            # 用户u评过分的物品
            rated_items = np.where(mask[u])[0]

            if len(rated_items) == 0:
                continue

            # 隐式反馈影响
            implicit_sum = np.sum(Y[rated_items], axis=0)
            sqrt_N = np.sqrt(len(rated_items))
            p_u_imp = P[u] + implicit_sum / sqrt_N

            # 梯度下降更新
            for i in rated_items:
                pred = global_mean + b_u[u] + b_i[i] + np.dot(Q[i], p_u_imp)
                error = R[u, i] - pred

                # 更新
                b_u[u] += lambda_reg * (error - lambda_reg * b_u[u])
                b_i[i] += lambda_reg * (error - lambda_reg * b_i[i])
                P[u] += lambda_reg * (error * Q[i] - lambda_reg * P[u])
                Q[i] += lambda_reg * (error * p_u_imp - lambda_reg * Q[i])

                # 更新Y
                Y[i] += lambda_reg * (error * Q[i] / sqrt_N - lambda_reg * Y[i])

        if iteration % 5 == 0:
            print(f"Iteration {iteration}")

    # 预测
    predicted = np.zeros((m, n))
    for u in range(m):
        rated_items = np.where(mask[u])[0]
        if len(rated_items) > 0:
            implicit_sum = np.sum(Y[rated_items], axis=0)
            sqrt_N = np.sqrt(len(rated_items))
            p_u_imp = P[u] + implicit_sum / sqrt_N
        else:
            p_u_imp = P[u]

        for i in range(n):
            predicted[u, i] = global_mean + b_u[u] + b_i[i] + \
                              np.dot(Q[i], p_u_imp)

    return predicted
```

---

## 五、隐语义模型（LFM）

### 5.1 核心思想

**动机**：
- 用户和物品之间存在隐含的主题/类别
- 这些隐含特征影响用户偏好

**示例**：
```
电影隐因子：
因子1: 动作/爱情（负值=动作，正值=爱情）
因子2: 经典/现代
因子3: 儿童/成人

用户隐向量：
用户A: [0.8, 0.3, -0.5] → 偏爱爱情、经典、成人电影
```

### 5.2 模型形式化

**目标函数**：

```
min Σ (r_ui - Σ p_uf × q_if)^2 + λ(Σ ||p_u||^2 + Σ ||q_i||^2)
 p,q

约束：
- p_uf: 用户u在隐因子f上的偏好
- q_if: 物品i在隐因子f上的属性
- λ: 正则化参数，防止过拟合
```

**随机梯度下降(SGD)求解**：

```python
def lfm_sgd(R, k=10, alpha=0.01, lambda_reg=0.01,
            n_iterations=20):
    """
    隐语义模型 - 随机梯度下降求解

    参数:
        R: 评分矩阵
        k: 隐因子数量
        alpha: 学习率
        lambda_reg: 正则化参数
        n_iterations: 迭代次数
    """
    m, n = R.shape

    # 初始化隐因子矩阵
    P = np.random.randn(m, k) * 0.01  # 用户因子
    Q = np.random.randn(n, k) * 0.01  # 物品因子

    # 获取有效评分
    users, items = np.where(~np.isnan(R))
    ratings = R[users, items]

    # SGD迭代
    for iteration in range(n_iterations):
        # 随机打乱
        indices = np.random.permutation(len(ratings))

        for idx in indices:
            u = users[idx]
            i = items[idx]
            r = ratings[idx]

            # 预测
            pred = np.dot(P[u], Q[i])
            error = r - pred

            # 梯度更新
            P[u] += alpha * (error * Q[i] - lambda_reg * P[u])
            Q[i] += alpha * (error * P[u] - lambda_reg * Q[i])

        # 学习率衰减
        alpha *= 0.9

        if iteration % 5 == 0:
            # 计算RMSE
            pred_all = P.dot(Q.T)
            mask = ~np.isnan(R)
            rmse = np.sqrt(np.mean((R[mask] - pred_all[mask])**2))
            print(f"Iteration {iteration}, RMSE: {rmse:.4f}")

    # 最终预测
    predicted = P.dot(Q.T)

    return P, Q, predicted

# 使用
P, Q, predicted = lfm_sgd(R, k=5, alpha=0.01)
```

---

## 六、实践中的优化技巧

### 6.1 处理隐式反馈

**加权ALS**：

```python
def weighted_als(R, k=10, alpha=40, lambda_reg=0.01):
    """
    加权ALS：处理隐式反馈

    核心思想：
    - 二值化：有交互=1，无交互=0
    - 置信度：c_ui = 1 + alpha × r_ui
    - 优化：min Σ c_ui (p_ui - 1)^2 + λ(...)
    """
    # 二值化
    binary_R = (~np.isnan(R)).astype(float)

    # 置信度（假设原评分反映偏好强度）
    R_filled = np.nan_to_num(R)
    confidence = 1 + alpha * R_filled

    # ALS迭代...
    # （类似前述ALS，但加入置信度权重）

    return predicted
```

### 6.2 偏置处理

**加入用户和物品偏置**：

```python
def biased_mf(R, k=10, lambda_reg=0.01):
    """
    加入偏置的矩阵分解

    预测：r̂_ui = μ + b_u + b_i + u_i·v_j
    """
    m, n = R.shape

    # 全局平均
    mu = np.nanmean(R)

    # 偏置初始化
    b_u = np.zeros(m)
    b_i = np.zeros(n)

    # 隐因子
    U = np.random.randn(m, k) * 0.01
    V = np.random.randn(n, k) * 0.01

    # 优化过程（类似ALS或SGD）
    # 在更新时同时优化偏置和隐因子

    return predicted, mu, b_u, b_i, U, V
```

### 6.3 冷启动处理

**新用户处理**：

```python
def cold_start_user_new(user_profile, item_features, V):
    """
    新用户冷启动

    方法：
    1. 基于用户属性（人口统计）找相似用户
    2. 基于初始少量行为快速适配
    3. 使用物品内容特征
    """
    # 方法1：相似用户平均
    similar_users = find_similar_users(user_profile)
    user_vector = np.mean(V[similar_users], axis=0)

    # 方法2：内容特征映射
    if item_features is not None:
        # 学习从内容特征到隐因子的映射
        user_vector = content_to_factor(user_profile)

    return user_vector
```

### 6.4 大规模优化

**分布式训练**：

```python
# Spark ALS示例
from pyspark.ml.recommendation import ALS

def spark_als(ratings_df, k=10, lambda_reg=0.01):
    """
    使用Spark进行大规模ALS训练

    参数:
        ratings_df: DataFrame[user_id, item_id, rating]
    """
    als = ALS(
        rank=k,
        regParam=lambda_reg,
        userCol="user_id",
        itemCol="item_id",
        ratingCol="rating",
        coldStartStrategy="drop"
    )

    model = als.fit(ratings_df)

    # 预测
    predictions = model.transform(ratings_df)

    return model, predictions
```

---

## 七、评估与选择

### 7.1 评估指标

**预测准确度**：

```python
def rmse(predicted, actual):
    """均方根误差"""
    mask = ~np.isnan(actual)
    return np.sqrt(np.mean((predicted[mask] - actual[mask])**2))

def mae(predicted, actual):
    """平均绝对误差"""
    mask = ~np.isnan(actual)
    return np.mean(np.abs(predicted[mask] - actual[mask]))
```

**Top-N推荐指标**：

```python
def precision_at_k(predicted, actual, k=10):
    """
    Precision@K

    参数:
        predicted: 预测的排序列表
        actual: 实际喜欢的物品集合
        k: Top-K
    """
    top_k = predicted[:k]
    hits = len(set(top_k) & set(actual))
    return hits / k

def recall_at_k(predicted, actual, k=10):
    """
    Recall@K
    """
    top_k = predicted[:k]
    hits = len(set(top_k) & set(actual))
    return hits / len(actual) if len(actual) > 0 else 0

def ndcg_at_k(predicted, actual, k=10):
    """
    NDCG@K
    """
    import math

    def dcg(rel):
        return sum((2**rel - 1) / math.log(i+2, 2)
                   for i, rel in enumerate(rel))

    # 相关度（二值）
    relevance = [1 if item in actual else 0 for item in predicted[:k]]

    # DCG
    dcg_score = dcg(relevance)

    # IDCG（理想排序）
    ideal_relevance = sorted(relevance, reverse=True)
    idcg_score = dcg(ideal_relevance)

    return dcg_score / idcg_score if idcg_score > 0 else 0
```

### 7.2 算法选择指南

```
场景              推荐算法
-------------------------
用户数 << 物品数    Item-based CF
用户数 >> 物品数    User-based CF
实时性要求高        Item-based CF（预计算相似度）
评分数据充足        矩阵分解
只有隐式反馈        加权ALS / SVD++
冷启动问题严重      内容推荐 + 混合推荐
需要可解释性        User-based / Item-based CF
```

---

## 八、总结与展望

### 核心要点

1. **协同过滤基础**：
   - User-based: 相似用户喜欢相似物品
   - Item-based: 相似物品被相似用户喜欢

2. **矩阵分解**：
   - SVD: 传统奇异值分解
   - ALS: 交替最小二乘
   - SVD++: 结合隐式反馈

3. **优化技巧**：
   - 偏置处理
   - 隐式反馈加权
   - 冷启动策略
   - 大规模分布式训练

### 未来方向

1. **深度协同过滤**：
   - 神经协同过滤(NCF)
   - 深度矩阵分解
   - 图神经网络(GNN)

2. **混合模型**：
   - CF + 内容特征
   - CF + 知识图谱
   - CF + 社交关系

3. **AutoML**：
   - 自动选择最佳算法
   - 自动超参数优化
   - 自动特征工程

---

## 参考文献

**经典论文**：
- Koren et al. (2009). "Matrix Factorization Techniques for Recommender Systems." IEEE Computer.
- Koren (2008). "Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model." KDD.
- Takács et al. (2009). "Scalable Collaborative Filtering Approaches for Large Recommender Systems." JMLR.

**推荐阅读**：
- Ricci et al. (2011). "Recommender Systems Handbook." Springer.
- Netflix Prize documentation

**开源库**：
- Surprise: Python推荐系统库
- LightFM: 混合推荐算法
- Spark MLlib: 分布式机器学习

---

**下一讲预告**：我们将深入探讨深度学习推荐模型，从Wide & Deep到DeepFM、xDeepFM，系统分析现代推荐系统的深度学习架构。敬请期待！
