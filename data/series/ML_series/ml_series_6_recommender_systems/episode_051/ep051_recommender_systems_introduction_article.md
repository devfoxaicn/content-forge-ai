# 第51讲：推荐系统导论——架构与评估

## 课程概览

推荐系统是现代互联网应用的核心组件，从电商、视频、音乐到社交媒体，推荐系统无处不在。本讲将带您深入推荐系统的世界，理解其核心价值、架构设计和评估方法。

**学习目标**：
- 理解推荐系统的核心价值和应用场景
- 掌握召回、排序、重排三层架构
- 学会离线评估指标（AUC、NDCG、MAP）
- 了解在线评估与A/B测试方法
- 理解推荐系统的技术挑战和解决方案

**核心关键词**：推荐系统架构、召回排序、AUC、NDCG、A/B测试、评估指标

---

## 一、推荐系统概述

### 1.1 什么是推荐系统

推荐系统是一种信息过滤系统，它预测用户对物品的偏好或评分，并向用户推荐其可能感兴趣的物品。与搜索引擎不同，推荐系统不需要用户主动查询，而是根据用户的历史行为和偏好主动推送内容。

**推荐系统的本质**：
- **个性化**：为不同用户提供定制化的推荐结果
- **自动化**：无需用户明确表达需求
- **规模化**：处理海量用户和物品
- **实时性**：快速响应用户行为变化

**核心价值**：
1. **提升用户体验**：减少信息过载，帮助用户发现感兴趣的内容
2. **增加业务收益**：提高点击率、转化率、用户停留时长
3. **优化资源分配**：将长尾物品推荐给潜在用户
4. **增强用户粘性**：通过个性化推荐提升用户满意度

### 1.2 推荐系统的应用场景

**电商推荐**（淘宝、京东、亚马逊）：
- "猜你喜欢"：基于用户浏览和购买历史推荐商品
- "购买了X的人还购买了Y"：关联商品推荐
- "为您推荐"：实时个性化推荐

**视频推荐**（YouTube、Netflix、抖音）：
- 首页推荐：根据观看历史推荐视频
- 相关推荐：当前视频的相关内容
- 热门推荐：结合流行度和个性化

**音乐推荐**（Spotify、网易云音乐）：
- 每日推荐：个性化歌单
- 相似歌曲/歌手推荐
- 基于场景的推荐（工作、运动、睡眠）

**社交推荐**（Twitter、Facebook、微博）：
- 关注推荐：推荐可能感兴趣的用户
- 内容推荐：推荐可能感兴趣的内容
- 好友推荐：基于社交关系和共同兴趣

**其他场景**：
- 新闻推荐（今日头条）
- 旅游推荐（携程、Airbnb）
- 求职推荐（LinkedIn、BOSS直聘）
- 外卖推荐（美团、饿了么）

### 1.3 推荐系统的分类

**基于推荐技术**：
1. **协同过滤（Collaborative Filtering）**
   - 基于用户（User-based）
   - 基于物品（Item-based）
   - 矩阵分解

2. **基于内容（Content-based）**
   - 物品内容特征
   - 用户画像匹配

3. **混合推荐（Hybrid）**
   - 组合多种技术
   - 优势互补

**基于应用场景**：
1. **离线推荐**：批量生成推荐结果
2. **实时推荐**：根据实时行为动态调整
3. **会话推荐**：基于当前会话的即时推荐

**基于推荐目标**：
1. **点击率（CTR）预测**
2. **转化率（CVR）预测**
3. **时长预测**
4. **满意度预测**

---

## 二、推荐系统架构

### 2.1 经典三层架构

现代工业级推荐系统通常采用**召回-排序-重排**三层架构，这种架构在效果和性能之间取得了良好平衡。

```
用户请求 → 召回层 → 排序层 → 重排层 → 最终推荐列表
           (百万级) (千级)    (百级)    (几十个)
```

**架构优势**：
- **性能优化**：逐层过滤，减少计算量
- **效果保证**：每层专注不同目标
- **灵活扩展**：可以独立优化各层
- **工程可行**：满足实时性要求

### 2.2 召回层（Recall）

**目标**：从海量物品库（百万到亿级）中快速筛选出候选集（几百到几千个）。

**核心要求**：
- **高召回率**：尽可能不漏掉用户感兴趣的物品
- **低延迟**：毫秒级响应
- **多路召回**：使用多种策略保证多样性

**常用召回策略**：

1. **协同过滤召回**
```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeFilteringRecall:
    """基于协同过滤的召回"""
    def __init__(self, user_item_matrix, top_k=100):
        self.user_item_matrix = user_item_matrix
        self.top_k = top_k
        # 计算用户相似度
        self.user_similarity = cosine_similarity(user_item_matrix)
        # 计算物品相似度
        self.item_similarity = cosine_similarity(user_item_matrix.T)

    def user_based_recall(self, user_id, n_items=100):
        """基于用户的召回"""
        # 找到相似用户
        similar_users = np.argsort(self.user_similarity[user_id])[-self.top_k:]

        # 聚合相似用户的物品
        scores = np.zeros(self.user_item_matrix.shape[1])
        for similar_user in similar_users:
            # 加权求和：相似度 * 评分
            scores += self.user_similarity[user_id][similar_user] * \
                     self.user_item_matrix[similar_user]

        # 排除已交互物品
        interacted_items = np.where(self.user_item_matrix[user_id] > 0)[0]
        scores[interacted_items] = -np.inf

        # 返回top-n物品
        top_items = np.argsort(scores)[-n_items:][::-1]
        return top_items, scores[top_items]

    def item_based_recall(self, user_id, n_items=100):
        """基于物品的召回"""
        # 获取用户已交互的物品
        interacted_items = np.where(self.user_item_matrix[user_id] > 0)[0]

        # 基于物品相似度计算推荐分数
        scores = np.zeros(self.user_item_matrix.shape[1])
        for item in interacted_items:
            # 相似物品 * 用户对该物品的评分
            scores += self.item_similarity[item] * \
                     self.user_item_matrix[user_id][item]

        # 排除已交互物品
        scores[interacted_items] = -np.inf

        # 返回top-n物品
        top_items = np.argsort(scores)[-n_items:][::-1]
        return top_items, scores[top_items]

# 示例使用
if __name__ == "__main__":
    # 创建示例用户-物品矩阵（100个用户，10000个物品）
    np.random.seed(42)
    user_item_matrix = np.random.randint(0, 6, size=(100, 10000))

    # 初始化召回器
    recall = CollaborativeFilteringRecall(user_item_matrix, top_k=50)

    # 为用户0召回物品
    user_id = 0
    items, scores = recall.user_based_recall(user_id, n_items=100)
    print(f"为用户{user_id}召回的物品ID: {items[:10]}")
    print(f"对应分数: {scores[:10]}")
```

2. **向量召回（双塔模型）**
```python
import tensorflow as tf
from tensorflow.keras import layers, Model

class TwoTowerModel(Model):
    """双塔模型：User Tower和Item Tower"""
    def __init__(self, user_feature_dim, item_feature_dim, embedding_dim=64):
        super(TwoTowerModel, self).__init__()
        # User Tower
        self.user_tower = tf.keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dense(embedding_dim)  # 用户embedding
        ])

        # Item Tower
        self.item_tower = tf.keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dense(embedding_dim)  # 物品embedding
        ])

    def call(self, inputs):
        user_features, item_features = inputs
        user_embedding = self.user_tower(user_features)
        item_embedding = self.item_tower(item_features)
        return user_embedding, item_embedding

    def compute_similarity(self, user_embedding, item_embedding):
        """计算相似度（内积）"""
        return tf.reduce_sum(user_embedding * item_embedding, axis=-1)

# 模型训练
def train_two_tower_model():
    # 示例数据
    num_users = 10000
    num_items = 50000
    user_feature_dim = 50
    item_feature_dim = 100
    embedding_dim = 64

    # 创建模型
    model = TwoTowerModel(user_feature_dim, item_feature_dim, embedding_dim)

    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['AUC']
    )

    # 模拟训练数据
    batch_size = 1024
    user_features = tf.random.normal((batch_size, user_feature_dim))
    item_features = tf.random.normal((batch_size, item_feature_dim))
    labels = tf.random.uniform((batch_size, 1), 0, 2, dtype=tf.int32)

    # 训练（实际使用真实数据）
    history = model.fit(
        [user_features, item_features],
        labels,
        epochs=10,
        batch_size=256,
        validation_split=0.2
    )

    return model

# 召回实现
def vector_recall(model, user_features, item_embeddings, top_k=100):
    """
    使用训练好的双塔模型进行向量召回

    Args:
        model: 训练好的双塔模型
        user_features: 用户特征
        item_embeddings: 预先计算的物品embedding矩阵
        top_k: 返回top-k物品

    Returns:
        top_items: 推荐物品ID
        scores: 推荐分数
    """
    # 获取用户embedding
    user_embedding = model.user_tower(user_features)

    # 计算与所有物品的相似度
    similarities = tf.matmul(user_embedding, item_embeddings, transpose_b=True)

    # 获取top-k
    top_scores, top_items = tf.nn.top_k(similarities[0], k=top_k)

    return top_items.numpy(), top_scores.numpy()

# 预计算所有物品的embedding
def precompute_item_embeddings(model, item_features):
    """预计算物品embedding用于召回"""
    item_embeddings = model.item_tower(item_features)
    # 归一化便于使用余弦相似度
    item_embeddings = tf.nn.l2_normalize(item_embeddings, axis=1)
    return item_embeddings
```

3. **多路召回融合**
```python
class MultiPathRecall:
    """多路召回策略"""
    def __init__(self):
        self.recalls = {
            'cf_recall': None,  # 协同过滤
            'content_recall': None,  # 内容召回
            'hot_recall': None,  # 热门召回
            'social_recall': None  # 社交召回
        }

    def add_recall(self, name, recall_func):
        """添加召回策略"""
        self.recalls[name] = recall_func

    def recall_and_fuse(self, user_id, n_items=500):
        """多路召回并融合"""
        all_items = {}
        weights = {
            'cf_recall': 0.5,
            'content_recall': 0.3,
            'hot_recall': 0.15,
            'social_recall': 0.05
        }

        # 多路召回
        for name, recall_func in self.recalls.items():
            if recall_func is not None:
                items, scores = recall_func(user_id, n_items*2)
                weight = weights.get(name, 0.1)

                # 加权融合
                for item, score in zip(items, scores):
                    if item not in all_items:
                        all_items[item] = 0
                    all_items[item] += weight * score

        # 排序并返回top-n
        sorted_items = sorted(all_items.items(), key=lambda x: x[1], reverse=True)
        top_items = [item for item, score in sorted_items[:n_items]]
        top_scores = [score for item, score in sorted_items[:n_items]]

        return top_items, top_scores

# 使用示例
def multipath_recall_example():
    # 创建多路召回器
    multi_recall = MultiPathRecall()

    # 添加不同召回策略（实际使用时传入真实函数）
    # multi_recall.add_recall('cf_recall', cf_recall_func)
    # multi_recall.add_recall('content_recall', content_recall_func)

    # 多路召回
    items, scores = multi_recall.recall_and_fuse(user_id=0, n_items=500)
    print(f"多路召回得到{len(items)}个候选物品")
```

### 2.3 排序层（Ranking）

**目标**：对召回的候选集（几百到几千个）进行精准排序，预测用户对每个物品的偏好程度。

**核心要求**：
- **准确性**：精准预测用户偏好
- **可解释性**：理解模型决策
- **特征工程**：充分利用多源特征

**排序模型特征**：
1. **用户特征**：年龄、性别、地域、历史行为、偏好标签
2. **物品特征**：类别、价格、品牌、发布时间、统计特征
3. **上下文特征**：时间、地点、设备、场景
4. **交叉特征**：用户×物品组合特征

**常用排序模型**：LR、GBDT、DeepFM、DCN、Wide&Deep

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

class WideAndDeep(Model):
    """Wide & Deep模型"""
    def __init__(self, wide_dim, deep_feature_dims, embedding_dim=16):
        super(WideAndDeep, self).__init__()
        # Wide部分：线性模型（记忆能力）
        self.wide = layers.Dense(1)

        # Deep部分：深度神经网络（泛化能力）
        self.embeddings = []
        self.deep_layers = tf.keras.Sequential([
            layers.Dense(1024, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(256, activation='relu'),
            layers.Dense(1)
        ])

        # 为每个稀疏特征创建embedding
        for dim in deep_feature_dims:
            self.embeddings.append(layers.Embedding(dim, embedding_dim))

    def call(self, inputs):
        # inputs: [wide_features, deep_features_sparse]
        wide_features, deep_features = inputs

        # Wide部分
        wide_output = self.wide(wide_features)

        # Deep部分
        embeddings = []
        for i, embedding in enumerate(self.embeddings):
            emb = embedding(deep_features[i])
            embeddings.append(emb)

        # 拼接所有embedding
        deep_input = tf.concat(embeddings, axis=-1)
        deep_output = self.deep_layers(deep_input)

        # 组合Wide和Deep
        output = wide_output + deep_output
        return tf.nn.sigmoid(output)

# 训练排序模型
def train_ranking_model():
    # 特征维度
    wide_dim = 1000  # Wide特征维度（交叉特征）
    deep_feature_dims = [10000, 5000, 1000]  # 三个稀疏特征的vocab size
    embedding_dim = 16

    # 创建模型
    model = WideAndDeep(wide_dim, deep_feature_dims, embedding_dim)

    # 编译
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.AUC(name='auc')]
    )

    # 模拟训练数据
    batch_size = 1024
    # Wide特征（已交叉）
    wide_features = tf.random.normal((batch_size, wide_dim))
    # Deep特征（稀疏特征索引）
    deep_features = [
        tf.random.uniform((batch_size, 1), 0, deep_feature_dims[0], dtype=tf.int32),
        tf.random.uniform((batch_size, 1), 0, deep_feature_dims[1], dtype=tf.int32),
        tf.random.uniform((batch_size, 1), 0, deep_feature_dims[2], dtype=tf.int32)
    ]
    # 标签
    labels = tf.random.uniform((batch_size, 1), 0, 2, dtype=tf.int32)

    # 训练
    model.fit(
        [wide_features, deep_features],
        labels,
        epochs=5,
        batch_size=256,
        validation_split=0.2
    )

    return model

# 排序实现
def ranking_stage(model, recall_items, user_features, item_features):
    """
    对召回的物品进行排序

    Args:
        model: 训练好的排序模型
        recall_items: 召回的物品ID列表
        user_features: 用户特征
        item_features: 物品特征字典 {item_id: features}

    Returns:
        ranked_items: 排序后的物品列表
        scores: 对应的预测分数
    """
    predictions = []

    # 批量预测
    for item_id in recall_items:
        # 构造特征（实际应用需要更复杂的特征工程）
        features = construct_ranking_features(user_features, item_features[item_id])
        score = model.predict(features, verbose=0)[0][0]
        predictions.append((item_id, score))

    # 按分数排序
    predictions.sort(key=lambda x: x[1], reverse=True)

    ranked_items = [item_id for item_id, score in predictions]
    scores = [score for item_id, score in predictions]

    return ranked_items, scores

def construct_ranking_features(user_features, item_features):
    """构造排序特征（简化版）"""
    # 实际应用中需要更复杂的特征工程
    wide_features = user_features['wide_features']
    deep_features = [
        user_features['user_id_feature'],
        item_features['item_id_feature'],
        item_features['category_feature']
    ]
    return [wide_features, deep_features]
```

### 2.4 重排层（Re-ranking）

**目标**：对排序后的列表进行调整，考虑多样性、业务规则等因素。

**核心要求**：
- **多样性**：避免推荐结果过于单一
- **新颖性**：推荐一些新物品
- **业务规则**：满足运营需求

**重排策略**：
1. **MMR（Maximal Marginal Relevance）**：平衡相关性和多样性
2. **打散策略**：按类别、来源等打散
3. **多臂老虎机**：探索-利用平衡
4. **业务规则**：加权、过滤、插入

```python
import numpy as np

class MMRReRanker:
    """MMR重排算法"""
    def __init__(self, lambda_param=0.7):
        """
        Args:
            lambda_param: 平衡参数
                          接近1：更重视相关性
                          接近0：更重视多样性
        """
        self.lambda_param = lambda_param

    def rerank(self, ranked_items, scores, item_features, top_k=50):
        """
        MMR重排

        Args:
            ranked_items: 排序后的物品列表
            scores: 对应的相关性分数
            item_features: 物品特征矩阵（用于计算相似度）
            top_k: 返回top-k

        Returns:
            reranked_items: 重排后的物品列表
        """
        selected_items = []
        remaining_items = list(ranked_items)
        scores_dict = dict(zip(ranked_items, scores))

        while len(selected_items) < top_k and remaining_items:
            # 计算每个剩余物品的MMR分数
            mmr_scores = []
            for item in remaining_items:
                # 相关性分数
                relevance = scores_dict[item]

                # 多样性分数：与已选物品的最大相似度
                if selected_items:
                    selected_indices = [ranked_items.index(i) for i in selected_items]
                    item_index = ranked_items.index(item)
                    similarities = [
                        item_features[item_index, si]
                        for si in selected_indices
                    ]
                    diversity = max(similarities)
                else:
                    diversity = 0

                # MMR分数
                mmr_score = self.lambda_param * relevance - (1 - self.lambda_param) * diversity
                mmr_scores.append((item, mmr_score))

            # 选择MMR分数最高的物品
            mmr_scores.sort(key=lambda x: x[1], reverse=True)
            best_item = mmr_scores[0][0]
            selected_items.append(best_item)
            remaining_items.remove(best_item)

        return selected_items

# 使用示例
def mmr_rerank_example():
    # 创建示例数据
    num_items = 100
    ranked_items = list(range(num_items))
    scores = np.random.rand(num_items)  # 相关性分数

    # 物品特征（用随机向量表示）
    item_features = np.random.rand(num_items, 64)
    # 归一化
    item_features = item_features / np.linalg.norm(item_features, axis=1, keepdims=True)

    # MMR重排
    reranker = MMRReRanker(lambda_param=0.7)
    reranked_items = reranker.rerank(ranked_items, scores, item_features, top_k=50)

    print(f"原始排序前10个: {ranked_items[:10]}")
    print(f"重排后前10个: {reranked_items[:10]}")

    return reranked_items

# 多样性打散策略
class DiversityReRanker:
    """基于类别的多样性重排"""
    def __init__(self, category_dict, max_same_category=3):
        """
        Args:
            category_dict: 物品到类别的映射 {item_id: category}
            max_same_category: 同类物品最大连续数
        """
        self.category_dict = category_dict
        self.max_same_category = max_same_category

    def rerank(self, ranked_items, top_k=50):
        """
        按类别打散重排

        Args:
            ranked_items: 排序后的物品列表
            top_k: 返回top-k

        Returns:
            reranked_items: 重排后的物品列表
        """
        reranked = []
        # 按类别分组
        category_items = {}
        for item in ranked_items:
            category = self.category_dict.get(item, 'unknown')
            if category not in category_items:
                category_items[category] = []
            category_items[category].append(item)

        # 轮流从各类别取物品
        category_queue = list(category_items.keys())
        current_pos = {cat: 0 for cat in category_queue}
        continuous_count = {cat: 0 for cat in category_queue}

        while len(reranked) < top_k and category_queue:
            # 找到可以添加物品的类别
            added = False
            for _ in range(len(category_queue)):
                category = category_queue[0]
                category_queue = category_queue[1:]

                if current_pos[category] < len(category_items[category]):
                    # 检查连续数量
                    if continuous_count[category] < self.max_same_category:
                        # 添加该类别的下一个物品
                        item = category_items[category][current_pos[category]]
                        reranked.append(item)
                        current_pos[category] += 1
                        continuous_count[category] += 1
                        # 重置其他类别的连续计数
                        for cat in continuous_count:
                            if cat != category:
                                continuous_count[cat] = 0
                        category_queue.append(category)
                        added = True
                        break

            # 如果没有可以添加的物品，退出
            if not added:
                break

        return reranked

# 使用示例
def diversity_rerank_example():
    # 创建示例数据
    num_items = 100
    ranked_items = list(range(num_items))

    # 随机分配类别
    categories = ['electronics', 'books', 'clothing', 'food', 'sports']
    category_dict = {item: np.random.choice(categories) for item in ranked_items}

    # 多样性重排
    reranker = DiversityReRanker(category_dict, max_same_category=2)
    reranked_items = reranker.rerank(ranked_items, top_k=50)

    # 查看重排后的类别分布
    reranked_categories = [category_dict[item] for item in reranked_items[:20]]
    print(f"重排后前20个物品的类别: {reranked_categories}")

    return reranked_items
```

---

## 三、评估指标

### 3.1 离线评估指标

**准确率指标**：

1. **AUC（Area Under ROC Curve）**
```python
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt

def calculate_auc(y_true, y_pred):
    """
    计算AUC

    Args:
        y_true: 真实标签 (0或1)
        y_pred: 预测概率

    Returns:
        auc: AUC值
    """
    auc = roc_auc_score(y_true, y_pred)
    return auc

def plot_roc_curve(y_true, y_pred):
    """绘制ROC曲线"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

# 示例使用
def auc_example():
    # 创建示例数据
    np.random.seed(42)
    n_samples = 1000

    # 模拟预测概率
    y_pred = np.random.rand(n_samples)

    # 模拟真实标签（与预测概率相关）
    y_true = (y_pred + np.random.normal(0, 0.3, n_samples) > 0.5).astype(int)

    # 计算AUC
    auc = calculate_auc(y_true, y_pred)
    print(f"AUC: {auc:.4f}")

    # 绘制ROC曲线
    plot_roc_curve(y_true, y_pred)

    return auc
```

2. **NDCG（Normalized Discounted Cumulative Gain）**
```python
def dcg_at_k(relevances, k):
    """
    计算DCG@k

    Args:
        relevances: 相关性列表
        k: top-k

    Returns:
        dcg: DCG值
    """
    relevances = np.array(relevances)[:k]
    if relevances.size:
        # 使用log2(i+2)因为索引从0开始
        return np.sum(relevances / np.log2(np.arange(2, relevances.size + 2)))
    return 0.0

def ndcg_at_k(relevances, k):
    """
    计算NDCG@k

    Args:
        relevances: 相关性列表
        k: top-k

    Returns:
        ndcg: NDCG值
    """
    dcg = dcg_at_k(relevances, k)

    # 理想DCG：排序后的相关性
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)

    if idcg > 0:
        return dcg / idcg
    return 0.0

def evaluate_ndcg(y_true, y_pred, k=10):
    """
    评估NDCG@k

    Args:
        y_true: 真实相关性列表
        y_pred: 预测分数列表
        k: top-k

    Returns:
        ndcg: NDCG@k值
    """
    # 按预测分数排序
    sorted_indices = np.argsort(y_pred)[::-1]
    sorted_relevances = [y_true[i] for i in sorted_indices]

    # 计算NDCG
    ndcg = ndcg_at_k(sorted_relevances, k)
    return ndcg

# 示例使用
def ndcg_example():
    # 创建示例数据
    np.random.seed(42)

    # 真实相关性（0-5分）
    y_true = np.random.randint(0, 6, size=100)

    # 预测分数
    y_pred = np.random.rand(100)

    # 计算不同k值的NDCG
    for k in [5, 10, 20]:
        ndcg = evaluate_ndcg(y_true, y_pred, k)
        print(f"NDCG@{k}: {ndcg:.4f}")

    return ndcg
```

3. **MAP（Mean Average Precision）**
```python
def average_precision(y_true, y_pred, k=None):
    """
    计算Average Precision

    Args:
        y_true: 真实标签列表
        y_pred: 预测分数列表
        k: top-k（None表示全部）

    Returns:
        ap: Average Precision
    """
    # 按预测分数排序
    sorted_indices = np.argsort(y_pred)[::-1]
    sorted_labels = [y_true[i] for i in sorted_indices]

    if k is not None:
        sorted_labels = sorted_labels[:k]

    # 计算AP
    precision_at_k = []
    num_relevant = 0

    for i, label in enumerate(sorted_labels):
        if label == 1:
            num_relevant += 1
            precision_at_k.append(num_relevant / (i + 1))

    if precision_at_k:
        return np.mean(precision_at_k)
    return 0.0

def mean_average_precision(y_true_list, y_pred_list, k=None):
    """
    计算Mean Average Precision

    Args:
        y_true_list: 多个查询的真实标签列表
        y_pred_list: 多个查询的预测分数列表
        k: top-k

    Returns:
        map: Mean Average Precision
    """
    aps = []
    for y_true, y_pred in zip(y_true_list, y_pred_list):
        ap = average_precision(y_true, y_pred, k)
        aps.append(ap)

    return np.mean(aps)

# 示例使用
def map_example():
    # 创建示例数据
    np.random.seed(42)

    # 模拟10个查询
    num_queries = 10
    num_items_per_query = 100

    y_true_list = []
    y_pred_list = []

    for _ in range(num_queries):
        # 真实标签（0或1）
        y_true = np.random.randint(0, 2, size=num_items_per_query)
        # 预测分数
        y_pred = np.random.rand(num_items_per_query)

        y_true_list.append(y_true)
        y_pred_list.append(y_pred)

    # 计算MAP
    for k in [5, 10, 20, None]:
        map_score = mean_average_precision(y_true_list, y_pred_list, k)
        if k is None:
            print(f"MAP: {map_score:.4f}")
        else:
            print(f"MAP@{k}: {map_score:.4f}")

    return map_score
```

4. **召回率（Recall）和准确率（Precision）**
```python
def precision_at_k(y_true, y_pred, k):
    """
    计算Precision@k

    Args:
        y_true: 真实标签列表
        y_pred: 预测分数列表
        k: top-k

    Returns:
        precision: Precision@k
    """
    # 获取top-k预测
    top_k_indices = np.argsort(y_pred)[::-1][:k]
    top_k_labels = [y_true[i] for i in top_k_indices]

    # 计算准确率
    precision = np.sum(top_k_labels) / k
    return precision

def recall_at_k(y_true, y_pred, k):
    """
    计算Recall@k

    Args:
        y_true: 真实标签列表
        y_pred: 预测分数列表
        k: top-k

    Returns:
        recall: Recall@k
    """
    # 获取top-k预测
    top_k_indices = np.argsort(y_pred)[::-1][:k]
    top_k_labels = [y_true[i] for i in top_k_indices]

    # 计算召回率
    num_relevant = np.sum(y_true)
    if num_relevant > 0:
        recall = np.sum(top_k_labels) / num_relevant
    else:
        recall = 0.0
    return recall

# 示例使用
def precision_recall_example():
    # 创建示例数据
    np.random.seed(42)

    # 真实标签（0或1）
    y_true = np.random.randint(0, 2, size=1000)
    # 预测分数
    y_pred = np.random.rand(1000)

    # 计算不同k值的Precision和Recall
    print("k\tPrecision\tRecall")
    for k in [5, 10, 20, 50, 100]:
        precision = precision_at_k(y_true, y_pred, k)
        recall = recall_at_k(y_true, y_pred, k)
        print(f"{k}\t{precision:.4f}\t\t{recall:.4f}")

    return precision, recall
```

### 3.2 在线评估指标

**业务指标**：
1. **CTR（Click-Through Rate）**：点击率
2. **CVR（Conversion Rate）**：转化率
3. **GMV（Gross Merchandise Volume）**：成交总额
4. **停留时长**：用户使用时长
5. **DAU/MAU**：日活/月活

```python
def calculate_ctr(clicks, impressions):
    """计算点击率"""
    if impressions == 0:
        return 0.0
    return clicks / impressions

def calculate_cvr(conversions, clicks):
    """计算转化率"""
    if clicks == 0:
        return 0.0
    return conversions / clicks

def calculate_ctcvr(clicks, conversions, impressions):
    """计算点击转化率（pCTR * pCVR）"""
    if impressions == 0:
        return 0.0
    return conversions / impressions

# 示例使用
def business_metrics_example():
    # 模拟数据
    impressions = 10000  # 曝光次数
    clicks = 500  # 点击次数
    conversions = 50  # 转化次数

    # 计算指标
    ctr = calculate_ctr(clicks, impressions)
    cvr = calculate_cvr(conversions, clicks)
    ctcvr = calculate_ctcvr(clicks, conversions, impressions)

    print(f"曝光次数: {impressions}")
    print(f"点击次数: {clicks}")
    print(f"转化次数: {conversions}")
    print(f"CTR: {ctr:.4f} ({ctr*100:.2f}%)")
    print(f"CVR: {cvr:.4f} ({cvr*100:.2f}%)")
    print(f"CTCVR: {ctcvr:.4f} ({ctcvr*100:.2f}%)")

    return ctr, cvr, ctcvr
```

### 3.3 A/B测试

**A/B测试流程**：
1. 确定评估指标
2. 设计实验方案（分组策略、样本量）
3. 随机分流用户
4. 收集数据
5. 统计显著性检验
6. 得出结论

```python
import numpy as np
from scipy import stats

class ABTest:
    """A/B测试分析"""
    def __init__(self, alpha=0.05, power=0.8):
        """
        Args:
            alpha: 显著性水平（通常为0.05）
            power: 统计功效（通常为0.8）
        """
        self.alpha = alpha
        self.power = power

    def calculate_sample_size(self, baseline_rate, mde=0.05):
        """
        计算所需样本量

        Args:
            baseline_rate: 基线转化率
            mde: 最小可检测效应（Minimum Detectable Effect）

        Returns:
            sample_size: 每组所需样本量
        """
        # 使用双样本比例检验的样本量公式
        z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        z_beta = stats.norm.ppf(self.power)

        p1 = baseline_rate
        p2 = baseline_rate * (1 + mde)
        p_bar = (p1 + p2) / 2

        sample_size = (z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) +
                      z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2)))**2 / (p1 - p2)**2

        return int(np.ceil(sample_size))

    def t_test(self, control_metrics, treatment_metrics):
        """
        t检验：比较两组均值

        Args:
            control_metrics: 对照组指标
            treatment_metrics: 实验组指标

        Returns:
            t_statistic: t统计量
            p_value: p值
            significant: 是否显著
        """
        t_statistic, p_value = stats.ttest_ind(treatment_metrics, control_metrics)
        significant = p_value < self.alpha

        return t_statistic, p_value, significant

    def proportion_test(self, control_success, control_total,
                        treatment_success, treatment_total):
        """
        比例检验：比较两组比例

        Args:
            control_success: 对照组成功数
            control_total: 对照组总数
            treatment_success: 实验组成功数
            treatment_total: 实验组总数

        Returns:
            z_statistic: z统计量
            p_value: p值
            significant: 是否显著
        """
        # 计算比例
        p1 = control_success / control_total
        p2 = treatment_success / treatment_total

        # 合并比例
        p_pooled = (control_success + treatment_success) / (control_total + treatment_total)

        # 标准误差
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/control_total + 1/treatment_total))

        # z统计量
        z_statistic = (p2 - p1) / se

        # 双尾检验p值
        p_value = 2 * (1 - stats.norm.cdf(abs(z_statistic)))
        significant = p_value < self.alpha

        return z_statistic, p_value, significant

    def confidence_interval(self, metrics, confidence=0.95):
        """
        计算置信区间

        Args:
            metrics: 指标数组
            confidence: 置信水平

        Returns:
            mean: 均值
            lower: 下界
            upper: 上界
        """
        mean = np.mean(metrics)
        std = np.std(metrics, ddof=1)
        n = len(metrics)

        # t分布临界值
        t_critical = stats.t.ppf((1 + confidence) / 2, n - 1)

        # 误差范围
        margin_error = t_critical * std / np.sqrt(n)

        lower = mean - margin_error
        upper = mean + margin_error

        return mean, lower, upper

# 示例使用
def ab_test_example():
    # 创建A/B测试实例
    ab_test = ABTest(alpha=0.05, power=0.8)

    # 1. 计算样本量
    baseline_ctr = 0.05  # 基线CTR 5%
    mde = 0.10  # 期望检测到10%的相对提升
    sample_size = ab_test.calculate_sample_size(baseline_ctr, mde)
    print(f"每组所需样本量: {sample_size}")

    # 2. 模拟A/B测试数据
    np.random.seed(42)
    n = sample_size

    # 对照组（CTR = 5%）
    control_clicks = np.random.binomial(1, 0.05, n)
    control_ctr = np.mean(control_clicks)

    # 实验组（CTR = 5.5%，提升10%）
    treatment_clicks = np.random.binomial(1, 0.055, n)
    treatment_ctr = np.mean(treatment_clicks)

    print(f"对照组CTR: {control_ctr:.4f}")
    print(f"实验组CTR: {treatment_ctr:.4f}")
    print(f"相对提升: {(treatment_ctr - control_ctr) / control_ctr * 100:.2f}%")

    # 3. 比例检验
    control_success = np.sum(control_clicks)
    treatment_success = np.sum(treatment_clicks)

    z_stat, p_value, significant = ab_test.proportion_test(
        control_success, n,
        treatment_success, n
    )

    print(f"\n比例检验结果:")
    print(f"z统计量: {z_stat:.4f}")
    print(f"p值: {p_value:.4f}")
    print(f"是否显著: {'是' if significant else '否'}")

    # 4. 计算置信区间
    mean, lower, upper = ab_test.confidence_interval(treatment_clicks)
    print(f"\n实验组CTR 95%置信区间: [{lower:.4f}, {upper:.4f}]")

    return ab_test, z_stat, p_value
```

---

## 四、推荐系统挑战与解决方案

### 4.1 冷启动问题

**问题**：新用户/新物品缺乏历史数据，难以准确推荐。

**解决方案**：
1. **基于内容的推荐**：利用用户注册信息、物品内容特征
2. **热门推荐**：推荐全局热门物品
3. **注册引导**：让用户选择感兴趣的话题
4. **多臂老虎机**：探索-利用平衡

```python
class ColdStartRecommender:
    """冷启动推荐器"""
    def __init__(self):
        self.hot_items = []  # 热门物品
        self.content_features = {}  # 物品内容特征

    def recommend_for_new_user(self, user_profile, n_items=10):
        """为新用户推荐"""
        # 1. 基于用户画像推荐
        content_based_items = self._content_based_recommend(user_profile, n_items // 2)

        # 2. 混合热门物品
        hot_items = self.hot_items[:n_items - len(content_based_items)]

        # 3. 合并
        recommendations = content_based_items + hot_items
        return recommendations

    def recommend_new_items(self, user_history, all_new_items, n_items=10):
        """推荐新物品"""
        # 基于用户历史行为，推荐相似的新物品
        recommendations = []
        for item in all_new_items:
            similarity = self._calculate_similarity(user_history, item)
            recommendations.append((item, similarity))

        # 排序
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return [item for item, score in recommendations[:n_items]]

    def _content_based_recommend(self, user_profile, n_items):
        """基于内容的推荐"""
        # 实现基于用户画像的推荐
        pass

    def _calculate_similarity(self, user_history, item):
        """计算相似度"""
        # 实现相似度计算
        pass
```

### 4.2 数据稀疏性

**问题**：用户-物品交互矩阵极度稀疏（通常<1%填充）。

**解决方案**：
1. **矩阵分解**：学习潜在因子
2. **迁移学习**：从其他领域迁移知识
3. **数据增强**：利用辅助信息
4. **深度学习**：自动特征学习

### 4.3 可扩展性

**问题**：处理海量用户和物品（百万到亿级）。

**解决方案**：
1. **分布式计算**：Spark、Flink
2. **近似最近邻搜索**：Faiss、HNSW
3. **在线学习**：增量更新模型
4. **模型压缩**：量化、剪枝

### 4.4 实时性要求

**问题**：需要实时响应用户行为变化。

**解决方案**：
1. **流式计算**：Flink实时特征工程
2. **在线学习**：FTRL增量更新
3. **缓存策略**：Redis缓存热门结果
4. **模型轻量化**：减少计算复杂度

---

## 五、完整推荐系统示例

### 5.1 端到端推荐流程

```python
class RecommenderSystem:
    """完整的推荐系统"""
    def __init__(self):
        self.recall_model = None
        self.ranking_model = None
        self.reranker = None

    def train(self, train_data):
        """训练模型"""
        # 1. 训练召回模型
        self.recall_model = self._train_recall_model(train_data)

        # 2. 训练排序模型
        self.ranking_model = self._train_ranking_model(train_data)

        # 3. 初始化重排器
        self.reranker = MMRReRanker(lambda_param=0.7)

    def recommend(self, user_id, context, top_k=50):
        """
        生成推荐

        Args:
            user_id: 用户ID
            context: 上下文信息
            top_k: 返回top-k

        Returns:
            recommendations: 推荐列表 [(item_id, score), ...]
        """
        # 1. 召回：从百万级物品中筛选千级候选
        recall_items, recall_scores = self._recall(user_id, context, n_items=1000)

        # 2. 排序：对千级候选进行精准排序
        ranked_items, ranking_scores = self._rank(
            user_id, recall_items, context
        )

        # 3. 重排：考虑多样性和业务规则
        final_items = self._rerank(ranked_items, ranking_scores, context, top_k)

        return final_items

    def _recall(self, user_id, context, n_items=1000):
        """召回阶段"""
        # 多路召回
        items_list = []
        scores_list = []

        # 协同过滤召回
        cf_items, cf_scores = self.recall_model['cf'].recall(user_id, n_items // 4)
        items_list.append(cf_items)
        scores_list.append(cf_scores)

        # 向量召回
        vec_items, vec_scores = self.recall_model['vector'].recall(user_id, n_items // 4)
        items_list.append(vec_items)
        scores_list.append(vec_scores)

        # 热门召回
        hot_items, hot_scores = self.recall_model['hot'].recall(user_id, n_items // 4)
        items_list.append(hot_items)
        scores_list.append(hot_scores)

        # 融合多路召回结果
        fused_items, fused_scores = self._fuse_recalls(items_list, scores_list)

        return fused_items[:n_items], fused_scores[:n_items]

    def _rank(self, user_id, items, context):
        """排序阶段"""
        # 批量预测
        features = self._construct_ranking_features(user_id, items, context)
        scores = self.ranking_model.predict(features)

        # 排序
        sorted_indices = np.argsort(scores)[::-1]
        ranked_items = [items[i] for i in sorted_indices]
        ranking_scores = [scores[i] for i in sorted_indices]

        return ranked_items, ranking_scores

    def _rerank(self, items, scores, context, top_k):
        """重排阶段"""
        # MMR重排
        reranked_items = self.reranker.rerank(
            items, scores,
            item_features=None,  # 实际应用需要传入
            top_k=top_k
        )
        return reranked_items

    def _fuse_recalls(self, items_list, scores_list):
        """融合多路召回结果"""
        # 实现多路召回融合
        pass

    def _construct_ranking_features(self, user_id, items, context):
        """构造排序特征"""
        # 实现特征工程
        pass

    def _train_recall_model(self, train_data):
        """训练召回模型"""
        # 实现召回模型训练
        pass

    def _train_ranking_model(self, train_data):
        """训练排序模型"""
        # 实现排序模型训练
        pass

# 使用示例
def recommender_system_example():
    # 创建推荐系统
    recommender = RecommenderSystem()

    # 训练模型（实际使用时传入真实数据）
    # recommender.train(train_data)

    # 为用户推荐
    user_id = 12345
    context = {'time': '2024-01-01 12:00:00', 'device': 'mobile'}
    recommendations = recommender.recommend(user_id, context, top_k=50)

    print(f"为用户{user_id}推荐了{len(recommendations)}个物品")
    print(f"前10个推荐: {recommendations[:10]}")

    return recommendations
```

### 5.2 评估与优化

```python
class RecommenderEvaluator:
    """推荐系统评估器"""
    def __init__(self):
        self.metrics = []

    def evaluate_offline(self, model, test_data):
        """离线评估"""
        results = {}

        # 1. AUC
        y_true = test_data['labels']
        y_pred = model.predict(test_data['features'])
        results['auc'] = roc_auc_score(y_true, y_pred)

        # 2. NDCG@k
        for k in [5, 10, 20]:
            results[f'ndcg_{k}'] = self._calculate_ndcg(y_true, y_pred, k)

        # 3. MAP@k
        for k in [5, 10, 20]:
            results[f'map_{k}'] = self._calculate_map(y_true, y_pred, k)

        # 4. Precision@k和Recall@k
        for k in [5, 10, 20]:
            results[f'precision_{k}'] = precision_at_k(y_true, y_pred, k)
            results[f'recall_{k}'] = recall_at_k(y_true, y_pred, k)

        return results

    def evaluate_online(self, ab_test_config):
        """在线A/B测试"""
        # 实现在线评估
        pass

    def _calculate_ndcg(self, y_true, y_pred, k):
        """计算NDCG@k"""
        return ndcg_at_k(y_true, y_pred, k)

    def _calculate_map(self, y_true, y_pred, k):
        """计算MAP@k"""
        return average_precision(y_true, y_pred, k)

    def print_results(self, results):
        """打印评估结果"""
        print("=" * 50)
        print("离线评估结果")
        print("=" * 50)

        for metric, value in results.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")

        print("=" * 50)

# 示例使用
def evaluation_example():
    # 创建评估器
    evaluator = RecommenderEvaluator()

    # 模拟测试数据
    np.random.seed(42)
    n_samples = 1000

    test_data = {
        'features': np.random.rand(n_samples, 50),
        'labels': np.random.randint(0, 2, n_samples)
    }

    # 模拟模型预测
    model = None  # 实际使用时传入真实模型
    y_pred = np.random.rand(n_samples)

    # 离线评估
    results = evaluator.evaluate_offline(model, test_data)
    evaluator.print_results(results)

    return results
```

---

## 六、总结与展望

### 6.1 本讲总结

本讲深入介绍了推荐系统的核心概念、架构设计和评估方法：

**核心要点**：
1. **推荐系统价值**：个性化、自动化、规模化、实时性
2. **三层架构**：召回（百万→千级）→ 排序（千级→百级）→ 重排（百级→几十）
3. **评估指标**：
   - 离线：AUC、NDCG、MAP、Precision/Recall
   - 在线：CTR、CVR、GMV、停留时长
4. **A/B测试**：科学的在线评估方法

**技术挑战**：
- 冷启动：新用户/新物品缺乏数据
- 数据稀疏：交互矩阵<1%填充
- 可扩展性：处理海量用户和物品
- 实时性：快速响应用户行为变化

### 6.2 未来趋势

1. **深度学习推荐**：更复杂的模型结构（Transformer、图神经网络）
2. **多模态推荐**：融合文本、图像、视频等多模态信息
3. **实时推荐**：毫秒级实时响应
4. **可解释性**：理解推荐决策过程
5. **公平性**：避免推荐系统偏见
6. **大模型+推荐**：利用LLM的语义理解能力

### 6.3 学习建议

**实践建议**：
1. 从简单方法开始（协同过滤）
2. 在公开数据集上实验（MovieLens、Amazon）
3. 关注工业级实践（Netflix、YouTube、阿里巴巴）
4. 持续学习最新研究论文

**推荐资源**：
- **书籍**：《Recommender Systems Handbook》
- **论文**：RecSys会议、KDD杯
- **开源项目**：Surprise、LightFM、DeepFM
- **数据集**：MovieLens、Amazon Review、Kaggle

---

## 参考文献

1. **经典论文**：
   - Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. Computer, 42(8), 30-37.
   - Cheng, H. T., et al. (2016). Wide & Deep Learning for Recommender Systems. arXiv:1606.07792.

2. **评估指标**：
   - Wang, Y., et al. (2013). A survey of precision-recall curves for information retrieval. Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery.

3. **工业实践**：
   - Davidson, J., et al. (2010). The YouTube video recommendation system. RecSys.
   - Covington, P., Adams, J., & Sargin, E. (2016). Deep neural networks for YouTube recommendations. RecSys.

---

**下一讲预告**：第52讲《协同过滤算法深入解析》

我们将深入探讨协同过滤算法，包括：
- User-based和Item-based协同过滤
- 矩阵分解（SVD、SVD++、ALS）
- 隐语义模型
- Netflix Prize案例研究
- 协同过滤的改进与优化

敬请期待！
