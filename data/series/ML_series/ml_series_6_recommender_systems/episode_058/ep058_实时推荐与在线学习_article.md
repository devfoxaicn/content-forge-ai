# 第58讲：实时推荐与在线学习
## 毫秒级响应的推荐系统

---

## 课程概览

传统推荐系统使用离线训练、定期更新的方式，但用户兴趣是实时变化的。本讲将深入探讨实时推荐架构、流计算框架、在线学习算法，以及Flink实战、实时反馈闭环构建等前沿技术。

**核心要点**：
- 实时推荐的挑战与价值
- 流计算架构：Flink、Spark Streaming
- 实时特征工程
- 在线学习算法
- 实时反馈闭环
- 曝光去偏技术

---

## 一、实时推荐的挑战

### 1.1 为什么需要实时推荐

```python
# 场景1：用户刚刚点击了iPhone 15
# 离线系统：明天才更新推荐，继续推荐Android手机
# 实时系统：立刻调整策略，推荐iOS生态产品

# 场景2：直播正在发生
# 离线系统：推荐昨天热门的内容
# 实时系统：推荐当前直播间

# 实时推荐的价值
# 1. 捕捉短期兴趣
# 2. 提升用户体验
# 3. 增加即时转化
# 4. 减少推荐滞后性
```

### 1.2 实时性分级

```python
class RealtimeLevel:
    """实时性分级"""
    BATCH = 3600        # 批处理（小时级）
    NEAR_REALTIME = 60  # 近实时（分钟级）
    REALTIME = 1        # 实时（秒级）
    ULTRA_REALTIME = 0.1  # 超实时（100ms级）

# 不同场景的实时性需求
scenarios = {
    '电商推荐': RealtimeLevel.NEAR_REALTIME,
    '新闻推荐': RealtimeLevel.REALTIME,
    '直播推荐': RealtimeLevel.ULTRA_REALTIME,
    '短视频推荐': RealtimeLevel.REALTIME
}
```

---

## 二、流计算架构

### 2.1 Lambda架构

```python
class LambdaArchitecture:
    """Lambda架构：批处理+流处理"""
    def __init__(self):
        self.batch_layer = None      # 批处理层（离线）
        self.speed_layer = None      # 流处理层（实时）
        self.serving_layer = None    # 服务层（合并结果）
    
    def process(self, data_stream):
        """
        数据流处理
        """
        # 1. 批处理层：完整数据，高延迟
        batch_view = self.batch_layer.process(data_stream)
        
        # 2. 流处理层：增量数据，低延迟
        stream_view = self.speed_layer.process(data_stream)
        
        # 3. 服务层：合并视图
        final_view = self.serving_layer.merge(
            batch_view, 
            stream_view
        )
        
        return final_view

# Kappa架构：纯流处理
class KappaArchitecture:
    """Kappa架构：一切皆流"""
    def __init__(self):
        self.stream_processor = None
        self.message_log = None  # 消息日志（可重放）
    
    def process(self, data_stream):
        # 所有计算都通过流处理完成
        return self.stream_processor.process(data_stream)
```

### 2.2 Flink实时推荐

```python
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.common.types import Row
from pyflink.datastream.functions import MapFunction, KeyedProcessFunction

class RealtimeRecommendationFlink:
    """基于Flink的实时推荐"""
    def __init__(self):
        self.env = StreamExecutionEnvironment.get_execution_environment()
    
    def build_pipeline(self):
        """构建实时推荐pipeline"""
        # 1. 数据源：用户行为流
        behavior_stream = self.env.add_source(self.behavior_source())
        
        # 2. 实时特征提取
        feature_stream = behavior_stream.map(
            BehaviorFeatureExtractor()
        )
        
        # 3. 用户状态更新（Keyed Stream）
        user_state_stream = feature_stream.key_by(
            lambda x: x.user_id
        ).process(
            UserStateUpdater()
        )
        
        # 4. 实时推荐
        recommendation_stream = user_state_stream.map(
            RealtimePredictor()
        )
        
        # 5. 输出
        recommendation_stream.add_sink(self.recommendation_sink())
        
        # 执行
        self.env.execute("Realtime Recommendation")
    
    def behavior_source(self):
        """用户行为数据源"""
        # Kafka、Kinesis等
        pass
    
    def recommendation_sink(self):
        """推荐结果输出"""
        # Redis、HBase等
        pass

class BehaviorFeatureExtractor(MapFunction):
    """行为特征提取器"""
    def map(self, behavior):
        # 提取实时特征
        features = {
            'user_id': behavior.user_id,
            'item_id': behavior.item_id,
            'action_type': behavior.action_type,
            'timestamp': behavior.timestamp,
            # 实时统计特征
            'user_action_count_1h': self._count_recent_actions(behavior.user_id, 3600),
            'item_action_count_1h': self._count_recent_item_actions(behavior.item_id, 3600)
        }
        return Row(**features)

class UserStateUpdater(KeyedProcessFunction):
    """用户状态更新器"""
    def open(self, runtime_context):
        # 初始化状态
        self.user_state = runtime_context.get_keyed_state(
            "user_interest_state"
        )
    
    def process_element(self, feature, context):
        # 更新用户兴趣状态
        current_state = self.user_state.get() or {}
        updated_state = self._update_interest(current_state, feature)
        self.user_state.update(updated_state)
        
        yield updated_state
    
    def _update_interest(self, state, feature):
        # 增量更新用户兴趣
        # 使用滑动窗口、指数衰减等
        return state

class RealtimePredictor(MapFunction):
    """实时预测器"""
    def __init__(self):
        self.model = None  # 加载预训练模型
    
    def map(self, user_state):
        # 实时预测
        recommendations = self.model.predict(user_state)
        return recommendations
```

### 2.3 Spark Streaming实现

```python
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession

class RealtimeRecommendationSpark:
    """基于Spark Streaming的实时推荐"""
    def __init__(self, batch_interval=5):
        self.spark = SparkSession.builder \
            .appName("RealtimeRec") \
            .getOrCreate()
        
        self.ssc = StreamingContext(
            self.spark.sparkContext, 
            batch_interval
        )
    
    def process_stream(self, kafka_params):
        """处理数据流"""
        from pyspark.streaming.kafka import KafkaUtils
        
        # 1. 创建Kafka流
        behavior_stream = KafkaUtils.createDirectStream(
            self.ssc,
            ['user-behaviors'],
            kafka_params
        )
        
        # 2. 解析JSON
        parsed_stream = behavior_stream.map(
            lambda x: json.loads(x[1])
        )
        
        # 3. 窗口化操作
        windowed_stream = parsed_stream.window(
            300,  # 5分钟窗口
            60    # 滑动间隔1分钟
        )
        
        # 4. 聚合特征
        feature_stream = windowed_stream.map(
            self._extract_features
        ).reduceByKey(
            self._merge_features
        )
        
        # 5. 实时推荐
        recommendation_stream = feature_stream.map(
            self._predict
        )
        
        # 6. 输出结果
        recommendation_stream.foreachRDD(
            self._save_recommendations
        )
        
        # 启动
        self.ssc.start()
        self.ssc.awaitTermination()
    
    def _extract_features(self, behavior):
        # 提取特征
        return (behavior['user_id'], {
            'item_id': behavior['item_id'],
            'action': behavior['action_type'],
            'timestamp': behavior['timestamp']
        })
    
    def _merge_features(self, feat1, feat2):
        # 合并特征
        return {**feat1, **feat2}
    
    def _predict(self, user_id_features):
        user_id, features = user_id_features
        # 调用模型预测
        recommendations = self.model.predict(features)
        return (user_id, recommendations)
    
    def _save_recommendations(self, rdd):
        def save_partition(iter):
            # 批量写入存储
            for user_id, recs in iter:
                # 写入Redis/HBase
                pass
        
        rdd.foreachPartition(save_partition)
```

---

## 三、实时特征工程

### 3.1 滑动窗口特征

```python
from collections import deque
import time

class SlidingWindowFeatures:
    """滑动窗口特征计算"""
    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.user_windows = {}  # {user_id: deque}
        self.item_windows = {}  # {item_id: deque}
    
    def add_event(self, user_id, item_id, action, timestamp):
        """添加事件"""
        # 添加到用户窗口
        if user_id not in self.user_windows:
            self.user_windows[user_id] = deque(maxlen=self.window_size)
        self.user_windows[user_id].append({
            'item_id': item_id,
            'action': action,
            'timestamp': timestamp
        })
        
        # 添加到物品窗口
        if item_id not in self.item_windows:
            self.item_windows[item_id] = deque(maxlen=self.window_size)
        self.item_windows[item_id].append({
            'user_id': user_id,
            'action': action,
            'timestamp': timestamp
        })
    
    def get_user_features(self, user_id, current_time):
        """获取用户实时特征"""
        if user_id not in self.user_windows:
            return {}
        
        window = self.user_windows[user_id]
        
        features = {}
        
        # 1. 时间窗口统计
        time_windows = [300, 3600, 86400]  # 5分钟、1小时、1天
        
        for tw in time_windows:
            tw_events = [
                e for e in window 
                if current_time - e['timestamp'] <= tw
            ]
            
            features[f'action_count_{tw}s'] = len(tw_events)
            features[f'unique_items_{tw}s'] = len(set(e['item_id'] for e in tw_events))
            
            # 行为分布
            action_dist = {}
            for e in tw_events:
                action_dist[e['action']] = action_dist.get(e['action'], 0) + 1
            
            for action, count in action_dist.items():
                features[f'{action}_count_{tw}s'] = count
        
        return features
    
    def get_item_features(self, item_id, current_time):
        """获取物品实时特征"""
        if item_id not in self.item_windows:
            return {}
        
        window = self.item_windows[item_id]
        
        features = {}
        
        # 1. 时间热度
        time_windows = [300, 3600, 86400]
        
        for tw in time_windows:
            tw_events = [
                e for e in window 
                if current_time - e['timestamp'] <= tw
            ]
            
            features[f'view_count_{tw}s'] = len(tw_events)
            features[f'unique_users_{tw}s'] = len(set(e['user_id'] for e in tw_events))
        
        return features
```

### 3.2 指数衰减特征

```python
class ExponentialDecayFeatures:
    """指数衰减特征：给予最近事件更高权重"""
    def __init__(self, decay_rate=0.1):
        self.decay_rate = decay_rate
        self.user_features = {}  # {user_id: feature_vector}
        self.item_features = {}  # {item_id: feature_vector}
    
    def update_user(self, user_id, item_features, action_weight, timestamp):
        """更新用户特征"""
        if user_id not in self.user_features:
            self.user_features[user_id] = np.zeros_like(item_features)
        
        # 指数衰减
        current_features = self.user_features[user_id]
        decayed_features = current_features * np.exp(-self.decay_rate)
        
        # 增量更新
        updated_features = decayed_features + action_weight * item_features
        self.user_features[user_id] = updated_features
    
    def get_user_similarity(self, user_id, item_features):
        """计算用户与物品的相似度"""
        if user_id not in self.user_features:
            return 0.0
        
        user_features = self.user_features[user_id]
        similarity = np.dot(user_features, item_features)
        return similarity
```

---

## 四、在线学习

### 4.1 在线梯度下降

```python
import numpy as np

class OnlineLogisticRegression:
    """在线逻辑回归"""
    def __init__(self, n_features, learning_rate=0.01):
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.weights = np.zeros(n_features)
        self.bias = 0.0
    
    def predict_proba(self, features):
        """预测概率"""
        logits = np.dot(features, self.weights) + self.bias
        proba = 1.0 / (1.0 + np.exp(-logits))
        return proba
    
    def update(self, features, label):
        """在线更新"""
        # 预测
        proba = self.predict_proba(features)
        
        # 计算梯度
        error = label - proba
        gradient = error * features
        
        # 更新权重
        self.weights += self.learning_rate * gradient
        self.bias += self.learning_rate * error
    
    def batch_update(self, features_list, labels_list):
        """批量更新"""
        for features, label in zip(features_list, labels_list):
            self.update(features, label)

# 使用示例
model = OnlineLogisticRegression(n_features=100, learning_rate=0.01)

# 在线学习循环
for event in event_stream:
    features = extract_features(event)
    label = event['label']  # 点击=1，未点击=0
    
    # 立即更新模型
    model.update(features, label)
    
    # 使用最新模型预测
    next_prediction = model.predict_proba(next_features)
```

### 4.2 FTRL-Proximal算法

```python
class FTRLProximal:
    """Follow-The-Regularized-Leader + Proximal Gradient"""
    def __init__(self, n_features, alpha=0.1, beta=1.0, lambda1=0.1, lambda2=1.0):
        self.n_features = n_features
        self.alpha = alpha  # 学习率参数
        self.beta = beta    # 学习率参数
        self.lambda1 = lambda1  # L1正则化
        self.lambda2 = lambda2  # L2正则化
        
        # 模型参数
        self.weights = np.zeros(n_features)
        
        # 累积变量
        self.z = np.zeros(n_features)  # 累积梯度
        self.n = np.zeros(n_features)  # 累积平方梯度
    
    def predict_proba(self, features):
        """预测概率"""
        logits = np.dot(features, self.weights)
        proba = 1.0 / (1.0 + np.exp(-logits))
        return proba
    
    def update(self, features, label):
        """FTRL更新"""
        # 预测
        proba = self.predict_proba(features)
        
        # 计算梯度
        gradient = proba - label
        
        # 更新累积变量
        self.z += gradient
        self.n += gradient ** 2
        
        # 更新权重（Proximal操作）
        for i in range(self.n_features):
            # 自适应学习率
            learning_rate = self.alpha / (self.beta + np.sqrt(self.n[i]))
            
            # Proximal梯度更新（L1正则化）
            if abs(self.z[i]) <= self.lambda1:
                self.weights[i] = 0
            else:
                self.weights[i] = -(
                    np.sign(self.z[i]) * self.lambda1 - self.z[i]
                ) / (
                    (self.beta + np.sqrt(self.n[i])) / self.alpha + self.lambda2
                )
    
    def get_sparsity(self):
        """获取模型稀疏度"""
        nonzero = np.count_nonzero(self.weights)
        sparsity = 1.0 - nonzero / self.n_features
        return sparsity
```

### 4.3 因子分解机的在线学习

```python
class OnlineFM:
    """在线因子分解机"""
    def __init__(self, n_features, n_factors=10, learning_rate=0.01):
        self.n_features = n_features
        self.n_factors = n_factors
        
        # 模型参数
        self.w0 = 0.0  # 偏置
        self.w = np.zeros(n_features)  # 一阶权重
        self.V = np.random.randn(n_features, n_factors) * 0.01  # 二阶交互
        
        self.learning_rate = learning_rate
    
    def predict(self, features):
        """预测"""
        # 一阶项
        linear_term = np.dot(features, self.w)
        
        # 二阶项
        interaction_term = 0.0
        for f in range(self.n_factors):
            v_f = self.V[:, f]
            interaction_term += 0.5 * (
                np.dot(features * v_f, features * v_f) - 
                np.dot(features ** 2, v_f ** 2)
            )
        
        return self.w0 + linear_term + interaction_term
    
    def update(self, features, label):
        """在线更新"""
        # 预测
        prediction = self.predict(features)
        
        # 计算误差
        error = label - prediction
        
        # 更新w0
        self.w0 -= self.learning_rate * error
        
        # 更新w
        self.w -= self.learning_rate * error * features
        
        # 更新V
        for f in range(self.n_factors):
            v_f = self.V[:, f]
            
            # 计算梯度
            grad_v = features * (np.dot(v_f, features) - features ** 2 * v_f)
            
            # 更新
            self.V[:, f] += self.learning_rate * error * grad_v
```

---

## 五、实时反馈闭环

### 5.1 反馈闭环设计

```python
class RealtimeFeedbackLoop:
    """实时反馈闭环"""
    def __init__(self, model, feature_store, model_store):
        self.model = model
        self.feature_store = feature_store
        self.model_store = model_store
    
    def on_impression(self, user_id, item_ids, context):
        """曝光事件"""
        # 1. 记录曝光
        self.feature_store.record_impression(user_id, item_ids, context)
        
        # 2. 更新用户状态（预期反馈）
        self.feature_store.update_user_pending_feedback(user_id, item_ids)
    
    def on_click(self, user_id, item_id, context):
        """点击事件"""
        # 1. 获取特征
        features = self.feature_store.get_features(user_id, item_id, context)
        
        # 2. 立即更新模型
        self.model.update(features, label=1)
        
        # 3. 清除pending反馈
        self.feature_store.clear_pending_feedback(user_id, item_id)
        
        # 4. 记录到特征库
        self.feature_store.record_feedback(user_id, item_id, 'click', context)
    
    def on_no_click(self, user_id, exposed_items, context):
        """未点击事件（隐式负反馈）"""
        # 对所有曝光但未点击的物品记录负反馈
        for item_id in exposed_items:
            features = self.feature_store.get_features(user_id, item_id, context)
            
            # 使用较小的学习率（因为是隐式反馈）
            self.model.update(features, label=0, lr_scale=0.1)
            
            self.feature_store.record_feedback(user_id, item_id, 'no_click', context)
    
    def periodic_model_sync(self):
        """定期同步模型到存储"""
        # 将在线学习的模型参数保存
        self.model_store.save(self.model.get_weights())
```

### 5.2 曝光去偏

```python
class PositionBiasCorrection:
    """位置偏差校正"""
    def __init__(self, n_positions=10):
        self.n_positions = n_positions
        self.position_clicks = np.zeros(n_positions)
        self.position_impressions = np.zeros(n_positions)
    
    def update(self, position, clicked):
        """更新位置统计"""
        if position >= self.n_positions:
            return
        
        self.position_impressions[position] += 1
        if clicked:
            self.position_clicks[position] += 1
    
    def get_position_bias(self, position):
        """获取位置偏差"""
        if self.position_impressions[position] == 0:
            return 1.0
        
        click_rate = self.position_clicks[position] / self.position_impressions[position]
        
        # 归一化（以位置0为基准）
        if self.position_impressions[0] > 0:
            base_rate = self.position_clicks[0] / self.position_impressions[0]
            bias = click_rate / base_rate
        else:
            bias = 1.0
        
        return bias
    
    def correct_prediction(self, raw_score, position):
        """校正预测分数"""
        bias = self.get_position_bias(position)
        corrected_score = raw_score / bias
        return corrected_score

class PropensityScoreCorrection:
    """倾向分数校正"""
    def __init__(self):
        self.propensity_model = None
    
    def train_propensity_model(self, logged_data):
        """
        训练倾向分数模型
        
        Args:
            logged_data: [{features, position, item_properties}]
        """
        from sklearn.linear_model import LogisticRegression
        
        X = []
        y = []
        
        for data in logged_data:
            # 特征：位置、物品属性等
            features = [
                data['position'],
                data['item_category'],
                data['item_popularity']
            ]
            
            X.append(features)
            y.append(1)  # 所有曝光的数据
        
        # 训练模型预测曝光概率
        self.propensity_model = LogisticRegression()
        self.propensity_model.fit(X, y)
    
    def get_propensity_score(self, features):
        """获取倾向分数"""
        if self.propensity_model is None:
            return 1.0
        
        propensity = self.propensity_model.predict_proba([features])[0][1]
        return propensity
    
    def inverse_propensity_weighting(self, reward, propensity):
        """逆倾向分数加权"""
        ipw = reward / propensity
        return ipw
```

---

## 总结

本讲深入探讨了实时推荐与在线学习：

**核心要点回顾**：
1. **实时性分级**：批处理→近实时→实时→超实时
2. **流计算架构**：Lambda、Kappa架构
3. **Flink/Spark**：实时推荐pipeline
4. **实时特征**：滑动窗口、指数衰减
5. **在线学习**：OGD、FTRL、Online FM
6. **反馈闭环**：曝光→点击→模型更新
7. **去偏技术**：位置偏差、倾向分数校正

**实践建议**：
- 实时推荐需要流计算框架支持
- 特征要设计为可增量更新
- 在线学习要注意模型稳定性
- 曝光偏差要持续校正

---

## 参考资料

1. **论文**：
   - McMahan et al. "Ad Click Prediction: A View from the Trenches" (KDD '13)
   - Graepel et al. "Web-Scale Bayesian Click-Through Rate Prediction for Sponsored Search Advertising in Microsoft's Bing Search Engine" (ICML '10)

2. **开源工具**：
   - Apache Flink: https://flink.apache.org/
   - Apache Spark Streaming: https://spark.apache.org/streaming/

---

**第58讲完**。下讲将深入探讨AB测试与推荐系统评估。
