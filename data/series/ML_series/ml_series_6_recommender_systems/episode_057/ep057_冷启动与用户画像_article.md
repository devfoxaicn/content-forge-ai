# 第57讲：冷启动与用户画像
## 推荐系统的"第一公里"问题

---

## 课程概览

冷启动是推荐系统最棘手的挑战：新用户没有行为历史，新物品没有交互数据。本讲将深入探讨冷启动策略、用户画像构建、兴趣建模，以及在新人引导、注册转化中的工程实践。

**核心要点**：
- 冷启动问题的分类与挑战
- 新用户冷启动策略
- 新物品冷启动策略
- 用户标签体系与画像构建
- 兴趣建模与生命周期
- 在新人引导中的应用

---

## 一、冷启动问题分类

### 1.1 三种冷启动

```python
class ColdStartType:
    USER_COLD_START = "user"      # 新用户
    ITEM_COLD_START = "item"      # 新物品
    SYSTEM_COLD_START = "system"  # 新系统

# 冷启动的困难
# 1. 数据稀疏性：无法计算协同过滤
# 2. 兴趣未知：无法个性化
# 3. 评估困难：缺乏历史数据对比
```

### 1.2 冷启动评估指标

```python
def cold_start_metrics(recommendations, user_actions, time_window=7):
    """
    冷启动专用指标
    
    Args:
        recommendations: 推荐列表
        user_actions: 用户行为
        time_window: 观察时间窗口（天）
    """
    metrics = {}
    
    # 1. 首次互动时间
    first_action_time = min([action['time'] for action in user_actions])
    metrics['time_to_first_action'] = first_action_time
    
    # 2. 早期转化率
    early_actions = [a for a in user_actions if a['time'] <= time_window * 86400]
    metrics['early_conversion_rate'] = len(early_actions) / len(recommendations)
    
    # 3. 留存率
    metrics['retention_rate'] = 1.0 if early_actions else 0.0
    
    return metrics
```

---

## 二、新用户冷启动策略

### 2.1 基于注册信息的推荐

```python
import numpy as np

class DemographicBasedRecommender:
    """基于人口统计学的推荐器"""
    def __init__(self):
        self.user_profiles = {}  # {demographic: popular_items}
    
    def fit(self, users, interactions):
        """
        按人口统计学分组统计热门物品
        
        Args:
            users: [{user_id, age, gender, location, ...}]
            interactions: [{user_id, item_id, rating}]
        """
        from collections import defaultdict
        
        # 按属性分组
        age_groups = defaultdict(list)
        gender_groups = defaultdict(list)
        location_groups = defaultdict(list)
        
        for user in users:
            age_groups[user['age'] // 10 * 10].append(user['user_id'])  # 年龄段
            gender_groups[user['gender']].append(user['user_id'])
            location_groups[user['location']].append(user['user_id'])
        
        # 统计每组的热门物品
        self.age_items = self._compute_popular_items(age_groups, interactions)
        self.gender_items = self._compute_popular_items(gender_groups, interactions)
        self.location_items = self._compute_popular_items(location_groups, interactions)
    
    def _compute_popular_items(self, groups, interactions):
        """计算每组的热门物品"""
        result = {}
        for group_key, user_ids in groups.items():
            # 筛选该组的交互
            group_interactions = [
                i for i in interactions 
                if i['user_id'] in user_ids
            ]
            
            # 统计物品热度
            from collections import Counter
            item_counts = Counter([i['item_id'] for i in group_interactions])
            
            # 取top-k
            result[group_key] = [item for item, _ in item_counts.most_common(100)]
        
        return result
    
    def recommend(self, user, k=10):
        """为新用户推荐"""
        candidates = []
        
        # 年龄组推荐
        age_group = user['age'] // 10 * 10
        if age_group in self.age_items:
            candidates.extend(self.age_items[age_group][:k//3])
        
        # 性别推荐
        if user['gender'] in self.gender_items:
            candidates.extend(self.gender_items[user['gender']][:k//3])
        
        # 地区推荐
        if user['location'] in self.location_items:
            candidates.extend(self.location_items[user['location']][:k//3])
        
        # 去重并返回
        return list(dict.fromkeys(candidates))[:k]
```

### 2.2 基于社交关系的推荐

```python
class SocialBasedRecommender:
    """基于社交关系的推荐器"""
    def __init__(self):
        self.social_graph = {}
        self.user_items = {}
    
    def fit(self, social_relations, interactions):
        """
        Args:
            social_relations: [{user_id, friend_id}]
            interactions: [{user_id, item_id, rating}]
        """
        # 构建社交图
        from collections import defaultdict
        self.social_graph = defaultdict(set)
        for rel in social_relations:
            self.social_graph[rel['user_id']].add(rel['friend_id'])
        
        # 统计每个用户的物品偏好
        from collections import defaultdict, Counter
        for user_id in self.social_graph.keys():
            user_inter = [i for i in interactions if i['user_id'] == user_id]
            self.user_items[user_id] = [
                item for item, _ in Counter([i['item_id'] for i in user_inter]).most_common(50)
            ]
    
    def recommend(self, user_id, k=10):
        """基于朋友的推荐"""
        # 获取朋友列表
        friends = self.social_graph.get(user_id, set())
        
        if not friends:
            return []  # 无社交关系，无法推荐
        
        # 聚合朋友的物品偏好
        from collections import Counter
        friend_items = []
        for friend_id in friends:
            if friend_id in self.user_items:
                friend_items.extend(self.user_items[friend_id])
        
        # 统计热度
        item_counts = Counter(friend_items)
        
        # 返回top-k
        return [item for item, _ in item_counts.most_common(k)]
```

### 2.3 主动学习：引导用户选择兴趣

```python
class ActiveLearningColdStart:
    """主动学习冷启动：通过问卷收集用户兴趣"""
    def __init__(self, item_categories, n_questions=5):
        self.item_categories = item_categories
        self.n_questions = n_questions
        self.user_preferences = {}
    
    def select_questions(self, user_id, asked_questions=None):
        """
        选择最优问题（信息增益最大化）
        
        Args:
            asked_questions: 已问过的问题
        Returns:
            questions_to_ask: 推荐的问题列表
        """
        if asked_questions is None:
            asked_questions = []
        
        # 策略1：选择最流行但区分度高的类别
        category_popularity = {}
        for category, items in self.item_categories.items():
            category_popularity[category] = len(items)
        
        # 策略2：选择互补的类别（覆盖不同领域）
        diverse_questions = self._select_diverse_categories(
            category_popularity, 
            self.n_questions - len(asked_questions)
        )
        
        return diverse_questions
    
    def _select_diverse_categories(self, popularity, k):
        """选择多样化的类别"""
        # 简化实现：选择top-k但确保类别多样性
        sorted_cats = sorted(popularity.items(), key=lambda x: x[1], reverse=True)
        
        selected = []
        covered_domains = set()
        
        for category, _ in sorted_cats:
            domain = category.split('/')[0]  # 假设类别有层级
            if domain not in covered_domains:
                selected.append(category)
                covered_domains.add(domain)
                if len(selected) >= k:
                    break
        
        return selected
    
    def update_preferences(self, user_id, preferences):
        """更新用户偏好"""
        self.user_preferences[user_id] = preferences
    
    def recommend(self, user_id, k=10):
        """基于用户偏好推荐"""
        preferences = self.user_preferences.get(user_id, {})
        
        if not preferences:
            return []
        
        # 从用户喜欢的类别中选择物品
        recommended = []
        for category, score in sorted(preferences.items(), 
                                     key=lambda x: x[1], reverse=True):
            if category in self.item_categories:
                items = self.item_categories[category][:k//2]
                recommended.extend(items)
                if len(recommended) >= k:
                    break
        
        return recommended[:k]
```

---

## 三、新物品冷启动策略

### 3.1 基于内容的推荐

```python
class ContentBasedColdStart:
    """基于内容的新物品推荐"""
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
        self.item_profiles = {}  # {item_id: features}
        self.user_interests = {}  # {user_id: interest_vector}
    
    def add_new_item(self, item_id, content):
        """
        添加新物品
        
        Args:
            content: {title, description, tags, category, ...}
        """
        # 提取内容特征
        features = self.feature_extractor.extract(content)
        self.item_profiles[item_id] = features
    
    def find_target_users(self, item_id, top_k=100):
        """为新物品找到目标用户"""
        if item_id not in self.item_profiles:
            return []
        
        item_features = self.item_profiles[item_id]
        
        # 计算与所有用户兴趣的相似度
        similarities = {}
        for user_id, user_interest in self.user_interests.items():
            sim = self._cosine_similarity(item_features, user_interest)
            similarities[user_id] = sim
        
        # 返回top-k用户
        sorted_users = sorted(similarities.items(), 
                            key=lambda x: x[1], reverse=True)
        return [user_id for user_id, _ in sorted_users[:top_k]]
    
    def _cosine_similarity(self, vec1, vec2):
        import numpy as np
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
```

### 3.2 探索-利用策略

```python
class ExplorationColdStart:
    """探索-利用策略处理新物品"""
    def __init__(self, exploration_rate=0.2):
        self.exploration_rate = exploration_rate
        self.new_items = []  # [(item_id, added_time)]
        self.item_stats = {}  # {item_id: {impressions, clicks}}
    
    def add_new_items(self, item_ids):
        """添加新物品到探索池"""
        import time
        current_time = time.time()
        for item_id in item_ids:
            self.new_items.append((item_id, current_time))
            self.item_stats[item_id] = {'impressions': 0, 'clicks': 0}
    
    def recommend_with_exploration(self, user_id, base_recommendations, k=10):
        """
        混合推荐：基础推荐 + 探索新物品
        
        Args:
            base_recommendations: 基础模型的推荐结果
            k: 最终返回数量
        """
        import random
        
        # 决定探索还是利用
        if random.random() < self.exploration_rate:
            # 探索：推荐新物品
            n_explore = int(k * 0.3)  # 30%用于探索
            
            # 选择需要曝光的新物品
            candidates = []
            for item_id, added_time in self.new_items:
                stats = self.item_stats[item_id]
                # 优先选择曝光次数少的新物品
                if stats['impressions'] < 100:
                    candidates.append((item_id, stats['impressions']))
            
            # 按曝光次数排序（少曝光的优先）
            candidates.sort(key=lambda x: x[1])
            explore_items = [item_id for item_id, _ in candidates[:n_explore]]
            
            # 基础推荐填充剩余
            base_items = [item for item, _ in base_recommendations[:k - n_explore]]
            
            return explore_items + base_items
        else:
            # 利用：使用基础推荐
            return [item for item, _ in base_recommendations[:k]]
    
    def record_feedback(self, item_id, clicked):
        """记录反馈"""
        if item_id in self.item_stats:
            self.item_stats[item_id]['impressions'] += 1
            if clicked:
                self.item_stats[item_id]['clicks'] += 1
    
    def graduate_items(self, click_threshold=0.05):
        """毕业物品：达到阈值后移出探索池"""
        graduated = []
        remaining = []
        
        for item_id, added_time in self.new_items:
            stats = self.item_stats[item_id]
            if stats['impressions'] >= 100:
                ctr = stats['clicks'] / stats['impressions']
                if ctr >= click_threshold:
                    graduated.append(item_id)
                    continue
            remaining.append((item_id, added_time))
        
        self.new_items = remaining
        return graduated
```

### 3.3 Bandit算法处理新物品

```python
class NewItemBandit:
    """使用Bandit算法探索新物品"""
    def __init__(self, n_arms=100):
        from collections import defaultdict
        self.counts = defaultdict(int)
        self.rewards = defaultdict(float)
        self.new_items = set()
    
    def add_new_item(self, item_id):
        self.new_items.add(item_id)
    
    def select_item(self, user_id, strategy='ucb'):
        """选择物品展示"""
        if not self.new_items:
            return None
        
        # 优先探索新物品
        if strategy == 'ucb':
            # UCB算法
            best_item = None
            best_ucb = -float('inf')
            
            for item_id in self.new_items:
                if self.counts[item_id] == 0:
                    return item_id  # 未尝试过，优先
                
                # UCB公式
                avg_reward = self.rewards[item_id] / self.counts[item_id]
                ucb = avg_reward + (2 * np.log(sum(self.counts.values())) / 
                                    self.counts[item_id]) ** 0.5
                
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_item = item_id
            
            return best_item
        
        elif strategy == 'thompson':
            # Thompson Sampling
            best_item = None
            best_sample = -float('inf')
            
            for item_id in self.new_items:
                alpha = 1 + self.rewards[item_id]
                beta = 1 + self.counts[item_id] - self.rewards[item_id]
                sample = np.random.beta(alpha, beta)
                
                if sample > best_sample:
                    best_sample = sample
                    best_item = item_id
            
            return best_item
    
    def update(self, item_id, reward):
        self.counts[item_id] += 1
        self.rewards[item_id] += reward
    
    def graduate(self, min_counts=100):
        """毕业机制"""
        graduated = []
        for item_id in list(self.new_items):
            if self.counts[item_id] >= min_counts:
                graduated.append(item_id)
                self.new_items.remove(item_id)
        return graduated
```

---

## 四、用户画像构建

### 4.1 标签体系

```python
class UserTaggingSystem:
    """用户标签系统"""
    def __init__(self):
        self.tag_hierarchy = {}  # 标签层级
        self.user_tags = {}      # {user_id: {tag: weight}}
    
    def define_tag_hierarchy(self, tags):
        """
        定义标签层级
        
        Example:
            tags = {
                'interest': {
                    'tech': ['AI', 'programming', 'gadgets'],
                    'sports': ['basketball', 'football', 'tennis']
                },
                'demographic': {
                    'age': ['18-24', '25-34', '35-44'],
                    'gender': ['male', 'female']
                }
            }
        """
        self.tag_hierarchy = tags
    
    def extract_tags_from_behavior(self, user_behaviors):
        """
        从行为中提取标签
        
        Args:
            user_behaviors: [{user_id, item_id, action_type, timestamp}]
        """
        from collections import defaultdict
        
        # 按用户聚合
        user_actions = defaultdict(list)
        for behavior in user_behaviors:
            user_actions[behavior['user_id']].append(behavior)
        
        # 提取标签
        for user_id, actions in user_actions.items():
            tags = defaultdict(float)
            
            for action in actions:
                # 根据物品类型打标签
                item_tags = self._get_item_tags(action['item_id'])
                
                # 根据行为类型加权
                weight = self._get_action_weight(action['action_type'])
                
                for tag in item_tags:
                    tags[tag] += weight
            
            # 归一化
            total = sum(tags.values())
            if total > 0:
                tags = {tag: weight/total for tag, weight in tags.items()}
            
            self.user_tags[user_id] = dict(tags)
    
    def _get_item_tags(self, item_id):
        # 简化：从物品特征获取标签
        return ['tech', 'AI']  # 示例
    
    def _get_action_weight(self, action_type):
        weights = {
            'view': 1.0,
            'click': 2.0,
            'like': 5.0,
            'share': 10.0,
            'purchase': 20.0
        }
        return weights.get(action_type, 1.0)
```

### 4.2 兴趣建模

```python
import numpy as np

class UserInterestModel:
    """用户兴趣建模"""
    def __init__(self, n_topics=100):
        self.n_topics = n_topics
        self.user_interests = {}  # {user_id: topic_vector}
        self.topic_embeddings = np.random.randn(n_topics, 128)
    
    def train(self, interactions, item_features):
        """
        训练用户兴趣模型
        
        Args:
            interactions: [{user_id, item_id, rating}]
            item_features: {item_id: feature_vector}
        """
        from collections import defaultdict
        
        # 按用户聚合
        user_items = defaultdict(list)
        for inter in interactions:
            user_items[inter['user_id']].append(inter['item_id'])
        
        # 学习用户兴趣向量
        for user_id, item_ids in user_items.items():
            # 获取用户交互物品的特征
            features = []
            for item_id in item_ids[:50]:  # 限制数量
                if item_id in item_features:
                    features.append(item_features[item_id])
            
            if features:
                # 平均池化
                user_vec = np.mean(features, axis=0)
                
                # 映射到主题空间
                topic_sim = cosine_similarity(
                    user_vec.reshape(1, -1),
                    self.topic_embeddings
                ).flatten()
                
                self.user_interests[user_id] = topic_sim
    
    def get_interest_profile(self, user_id):
        """获取用户兴趣画像"""
        if user_id not in self.user_interests:
            return None
        
        interest_vector = self.user_interests[user_id]
        
        # 获取top-k兴趣主题
        top_topics = np.argsort(interest_vector)[-10:][::-1]
        
        profile = {
            'topics': top_topics.tolist(),
            'scores': interest_vector[top_topics].tolist()
        }
        
        return profile
    
    def update_interest(self, user_id, item_features, feedback):
        """增量更新用户兴趣"""
        if user_id not in self.user_interests:
            # 新用户，初始化
            self.user_interests[user_id] = np.zeros(self.n_topics)
        
        # 根据反馈更新
        learning_rate = 0.1
        for item_feature in item_features:
            # 计算主题相似度
            topic_sim = cosine_similarity(
                item_feature.reshape(1, -1),
                self.topic_embeddings
            ).flatten()
            
            # 更新
            self.user_interests[user_id] += learning_rate * feedback * topic_sim
        
        # 归一化
        self.user_interests[user_id] /= np.linalg.norm(self.user_interests[user_id])
```

---

## 五、用户生命周期管理

### 5.1 生命周期阶段

```python
class UserLifecycleManager:
    """用户生命周期管理"""
    def __init__(self):
        self.stages = {
            'new': {'duration': 7, 'strategy': 'cold_start'},
            'active': {'duration': 30, 'strategy': 'engagement'},
            'churned': {'duration': float('inf'), 'strategy': 'retention'}
        }
        self.user_stages = {}
        self.user_activities = {}
    
    def classify_user(self, user_id, current_time):
        """分类用户阶段"""
        if user_id not in self.user_activities:
            return 'new'
        
        activities = self.user_activities[user_id]
        last_activity = max([a['time'] for a in activities])
        
        days_inactive = (current_time - last_activity) / 86400
        
        if days_inactive > 30:
            return 'churned'
        elif days_inactive < 7:
            return 'active'
        else:
            return 'dormant'
    
    def get_strategy(self, user_id, current_time):
        """获取推荐策略"""
        stage = self.classify_user(user_id, current_time)
        
        if stage == 'new':
            return {
                'focus': 'exploration',
                'diversity': 'high',
                'popular_weight': 0.5
            }
        elif stage == 'active':
            return {
                'focus': 'personalization',
                'diversity': 'medium',
                'popular_weight': 0.2
            }
        elif stage == 'churned':
            return {
                'focus': 'reactivation',
                'diversity': 'high',
                'popular_weight': 0.7
            }
        else:
            return {
                'focus': 'engagement',
                'diversity': 'medium',
                'popular_weight': 0.3
            }
    
    def record_activity(self, user_id, activity_type, time):
        """记录用户活动"""
        if user_id not in self.user_activities:
            self.user_activities[user_id] = []
        
        self.user_activities[user_id].append({
            'type': activity_type,
            'time': time
        })
```

### 5.2 新人引导优化

```python
class OnboardingOptimizer:
    """新人引导优化"""
    def __init__(self):
        self.onboarding_flows = {}  # {flow_id: steps}
        self.flow_performance = {}  # {flow_id: metrics}
    
    def design_onboarding_flow(self, user_segments):
        """
        为不同用户段设计引导流程
        
        Args:
            user_segments: {segment: [user_ids]}
        """
        flows = {}
        
        for segment, user_ids in user_segments.items():
            if segment == 'general':
                # 通用用户：简单引导
                flows['general'] = [
                    {'step': 'welcome', 'type': 'message'},
                    {'step': 'select_interests', 'type': 'selection', 'n_choices': 5},
                    {'step': 'first_recommendation', 'type': 'recommendation', 'n_items': 10}
                ]
            
            elif segment == 'expert':
                # 专家用户：快速上手
                flows['expert'] = [
                    {'step': 'welcome', 'type': 'message'},
                    {'step': 'advanced_setup', 'type': 'form'},
                    {'step': 'first_recommendation', 'type': 'recommendation', 'n_items': 20}
                ]
            
            elif segment == 'casual':
                # 休闲用户：游戏化引导
                flows['casual'] = [
                    {'step': 'gamified_welcome', 'type': 'game'},
                    {'step': 'quick_preferences', 'type': 'quiz', 'n_questions': 3},
                    {'step': 'first_recommendation', 'type': 'recommendation', 'n_items': 5}
                ]
        
        return flows
    
    def optimize_flow(self, flow_id, conversion_rates):
        """优化引导流程"""
        # A/B测试分析
        # 简化：返回优化建议
        if conversion_rates < 0.3:
            return 'simplify_flow'
        elif conversion_rates > 0.7:
            return 'add_more_steps'
        else:
            return 'keep_current'
```

---

## 总结

本讲深入探讨了冷启动与用户画像：

**核心要点回顾**：
1. **三种冷启动**：用户、物品、系统
2. **新用户策略**：人口统计、社交关系、主动学习
3. **新物品策略**：内容推荐、探索-利用、Bandit算法
4. **用户画像**：标签体系、兴趣建模
5. **生命周期**：新用户→活跃→流失，不同阶段不同策略
6. **新人引导**：分segment设计引导流程

**实践建议**：
- 冷启动需要多种策略组合
- 用户标签要定期更新
- 兴趣模型要考虑时间衰减
- 引导流程要持续A/B测试优化

---

## 参考资料

1. **论文**：
   - Rashid et al. "Getting to Know You: Learning New User Preferences in Recommender Systems" (2002)
   - Krohn-Grimberghe et al. "Cold-Start Recommendations for New Users" (2012)

2. **工业实践**：
   - Netflix冷启动实践
   - Spotify Discover Weekly设计

---

**第57讲完**。下讲将深入探讨实时推荐与在线学习。
