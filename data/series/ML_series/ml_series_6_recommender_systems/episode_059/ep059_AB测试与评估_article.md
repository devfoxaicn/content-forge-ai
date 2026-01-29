# 第59讲：AB测试与推荐系统评估
## 科学评估推荐系统的效果

---

## 课程概览

推荐系统需要科学的评估方法来验证改进效果。本讲将深入探讨AB测试的设计与实施、统计显著性检验、常见评估陷阱（Simpson悖论、选择偏差、数据泄漏），以及CUPED等高级评估技术。

**核心要点**：
- 离线评估vs在线评估
- AB测试实验设计
- 统计显著性检验
- Simpson悖论与混淆变量
- CUPED方差缩减
- 增量评估最佳实践

---

## 一、评估指标体系

### 1.1 离线评估指标

```python
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

# 分类指标
def classification_metrics(y_true, y_pred, y_score):
    """分类评估指标"""
    metrics = {}
    
    # AUC
    metrics['auc'] = roc_auc_score(y_true, y_score)
    
    # Precision/Recall/F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary'
    )
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    
    # LogLoss
    eps = 1e-15
    y_score_clipped = np.clip(y_score, eps, 1 - eps)
    logloss = -np.mean(
        y_true * np.log(y_score_clipped) + 
        (1 - y_true) * np.log(1 - y_score_clipped)
    )
    metrics['logloss'] = logloss
    
    return metrics

# 排序指标
def ranking_metrics(relevances, k=10):
    """
    排序评估指标
    
    Args:
        relevances: [rel1, rel2, ...] 相关性列表
        k: top-k位置
    """
    metrics = {}
    
    # DCG@K
    relevances = np.array(relevances)[:k]
    discounts = np.log2(np.arange(len(relevances)) + 2)
    dcg = np.sum(relevances / discounts)
    metrics[f'dcg@{k}'] = dcg
    
    # NDCG@K
    ideal_relevances = sorted(relevances, reverse=True)
    ideal_dcg = np.sum(ideal_relevances / discounts)
    metrics[f'ndcg@{k}'] = dcg / ideal_dcg if ideal_dcg > 0 else 0
    
    # MAP@K
    if relevances.sum() > 0:
        precisions = []
        num_relevant = 0
        for i, rel in enumerate(relevances):
            if rel == 1:
                num_relevant += 1
                precisions.append(num_relevant / (i + 1))
        metrics[f'map@{k}'] = np.mean(precisions) if precisions else 0
    else:
        metrics[f'map@{k}'] = 0
    
    # MRR
    for i, rel in enumerate(relevances):
        if rel == 1:
            metrics['mrr'] = 1.0 / (i + 1)
            break
    else:
        metrics['mrr'] = 0
    
    return metrics
```

### 1.2 在线评估指标

```python
# 业务指标
def business_metrics(user_activities):
    """
    业务评估指标
    
    Args:
        user_activities: [{user_id, actions, revenue, ...}]
    """
    metrics = {}
    
    # CTR (Click-Through Rate)
    total_impressions = sum(len(a['impressions']) for a in user_activities)
    total_clicks = sum(a['clicks'] for a in user_activities)
    metrics['ctr'] = total_clicks / total_impressions if total_impressions > 0 else 0
    
    # CVR (Conversion Rate)
    total_conversions = sum(a['conversions'] for a in user_activities)
    metrics['cvr'] = total_conversions / total_clicks if total_clicks > 0 else 0
    
    # GMV (Gross Merchandise Value)
    total_revenue = sum(a['revenue'] for a in user_activities)
    metrics['gmv'] = total_revenue
    
    # ARPU (Average Revenue Per User)
    metrics['arpu'] = total_revenue / len(user_activities) if user_activities else 0
    
    # 留存率
    returning_users = sum(1 for a in user_activities if a['is_returning'])
    metrics['retention_rate'] = returning_users / len(user_activities)
    
    return metrics

# 用户参与度指标
def engagement_metrics(user_sessions):
    """用户参与度指标"""
    metrics = {}
    
    # 平均会话时长
    session_durations = [s['duration'] for s in user_sessions]
    metrics['avg_session_duration'] = np.mean(session_durations)
    
    # 日活跃用户
    metrics['dau'] = len(set(s['user_id'] for s in user_sessions))
    
    # 页面浏览深度
    page_depths = [len(s['page_views']) for s in user_sessions]
    metrics['avg_page_depth'] = np.mean(page_depths)
    
    # 跳出率
    bounce_count = sum(1 for s in user_sessions if len(s['page_views']) == 1)
    metrics['bounce_rate'] = bounce_count / len(user_sessions)
    
    return metrics
```

---

## 二、AB测试实验设计

### 2.1 分流策略

```python
import hashlib

class ABTestSplitter:
    """AB测试分流器"""
    def __init__(self, test_name, traffic_split=0.5):
        self.test_name = test_name
        self.traffic_split = traffic_split  # 实验组流量比例
    
    def assign_group(self, user_id):
        """
        为用户分配实验组
        
        Args:
            user_id: 用户ID
        Returns:
            group: 'control' or 'treatment'
        """
        # 使用一致性哈希确保同一用户始终分到同一组
        hash_input = f"{self.test_name}:{user_id}".encode('utf-8')
        hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
        
        # 归一化到[0, 1]
        normalized = hash_value / (2 ** 128 - 1)
        
        # 分组
        if normalized < self.traffic_split:
            return 'treatment'
        else:
            return 'control'
    
    def batch_assign(self, user_ids):
        """批量分配用户"""
        assignments = {}
        for user_id in user_ids:
            assignments[user_id] = self.assign_group(user_id)
        return assignments

# 渐进式流量分配
class ProgressiveRollout:
    """渐进式流量分配"""
    def __init__(self, test_name, stages=[0.01, 0.05, 0.1, 0.25, 0.5]):
        self.test_name = test_name
        self.stages = stages
        self.current_stage = 0
    
    def get_traffic_split(self):
        """获取当前阶段的流量分配"""
        return self.stages[self.current_stage]
    
    def advance_stage(self):
        """推进到下一阶段"""
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            return True
        return False
    
    def should_check_safety(self):
        """是否需要安全检查"""
        # 前3个阶段需要检查
        return self.current_stage < 3
```

### 2.2 样本量计算

```python
import scipy.stats as stats

def calculate_sample_size(baseline_rate, mde, alpha=0.05, power=0.8, two_tailed=True):
    """
    计算AB测试所需样本量
    
    Args:
        baseline_rate: 基线转化率（如CTR=0.05）
        mde: Minimum Detectable Effect（最小可检测效应）
        alpha: 显著性水平（通常0.05）
        power: 统计功效（通常0.8）
        two_tailed: 是否双尾检验
    Returns:
        sample_size: 每组所需样本量
    """
    # Z分数
    if two_tailed:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)
    
    z_beta = stats.norm.ppf(power)
    
    # 转化率
    p1 = baseline_rate
    p2 = baseline_rate * (1 + mde)
    
    # 合并标准差
    p_pooled = (p1 + p2) / 2
    se = np.sqrt(p_pooled * (1 - p_pooled))
    
    # 效应量
    effect_size = abs(p2 - p1)
    
    # 样本量计算
    sample_size = 2 * (se ** 2) * ((z_alpha + z_beta) ** 2) / (effect_size ** 2)
    
    return int(np.ceil(sample_size))

# 示例：计算CTR实验的样本量
baseline_ctr = 0.05  # 5% CTR
mde = 0.02  # 期望检测2%的相对提升（即5.1% vs 5%）
sample_size = calculate_sample_size(baseline_ctr, mde)

print(f"每组需要样本量: {sample_size:,}")
print(f"总样本量: {sample_size * 2:,}")
```

---

## 三、统计显著性检验

### 3.1 假设检验

```python
class HypothesisTest:
    """假设检验"""
    
    @staticmethod
    def z_test(control_metric, treatment_metric, control_size, treatment_size):
        """
        Z检验（适用于大样本）
        
        Args:
            control_metric: 对照组指标值（如CTR）
            treatment_metric: 实验组指标值
            control_size: 对照组样本量
            treatment_size: 实验组样本量
        Returns:
            z_score: Z分数
            p_value: p值
            significant: 是否显著
        """
        # 合并标准误差
        se = np.sqrt(
            control_metric * (1 - control_metric) / control_size +
            treatment_metric * (1 - treatment_metric) / treatment_size
        )
        
        # Z分数
        z_score = (treatment_metric - control_metric) / se
        
        # 双尾p值
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        # 显著性判断（alpha=0.05）
        significant = p_value < 0.05
        
        return {
            'z_score': z_score,
            'p_value': p_value,
            'significant': significant
        }
    
    @staticmethod
    def t_test(control_values, treatment_values):
        """
        t检验（适用于小样本）
        
        Args:
            control_values: 对照组观测值列表
            treatment_values: 实验组观测值列表
        """
        from scipy import stats
        
        t_stat, p_value = stats.ttest_ind(treatment_values, control_values)
        
        return {
            't_stat': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def chi_square_test(control_conversions, control_total, 
                        treatment_conversions, treatment_total):
        """
        卡方检验（适用于分类数据）
        
        Args:
            control_conversions: 对照组转化数
            control_total: 对照组总数
            treatment_conversions: 实验组转化数
            treatment_total: 实验组总数
        """
        # 构建列联表
        control_non_conversions = control_total - control_conversions
        treatment_non_conversions = treatment_total - treatment_conversions
        
        observed = np.array([
            [control_conversions, control_non_conversions],
            [treatment_conversions, treatment_non_conversions]
        ])
        
        # 卡方检验
        chi2, p_value, dof, expected = stats.chi2_contingency(observed)
        
        return {
            'chi2': chi2,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
```

### 3.2 置信区间

```python
def confidence_interval(metric, sample_size, confidence=0.95):
    """
    计算置信区间（比例数据）
    
    Args:
        metric: 指标值（如CTR）
        sample_size: 样本量
        confidence: 置信水平
    Returns:
        (lower_bound, upper_bound): 置信区间
    """
    alpha = 1 - confidence
    z = stats.norm.ppf(1 - alpha / 2)
    
    # 标准误差
    se = np.sqrt(metric * (1 - metric) / sample_size)
    
    # 置信区间
    lower_bound = metric - z * se
    upper_bound = metric + z * se
    
    return lower_bound, upper_bound

# 相对提升的置信区间
def relative_lift_ci(control_metric, treatment_metric, 
                    control_size, treatment_size, confidence=0.95):
    """
    计算相对提升的置信区间
    """
    # 计算两组的置信区间
    control_lower, control_upper = confidence_interval(
        control_metric, control_size, confidence
    )
    treatment_lower, treatment_upper = confidence_interval(
        treatment_metric, treatment_size, confidence
    )
    
    # 相对提升
    lift_point = (treatment_metric - control_metric) / control_metric
    
    # 保守估计（使用置信区间的边界）
    lift_lower = (treatment_lower - control_upper) / control_upper
    lift_upper = (treatment_upper - control_lower) / control_lower
    
    return lift_point, lift_lower, lift_upper
```

---

## 四、常见评估陷阱

### 4.1 Simpson悖论

```python
def detect_simpson_paradox(data):
    """
    检测Simpson悖论
    
    Args:
        data: {segment: {group: {conversions, total}}}
    Returns:
        has_paradox: 是否存在悖论
        explanation: 解释
    """
    # 计算整体转化率
    control_total = sum(data[s]['control']['total'] for s in data)
    control_conv = sum(data[s]['control']['conversions'] for s in data)
    control_rate = control_conv / control_total
    
    treatment_total = sum(data[s]['treatment']['total'] for s in data)
    treatment_conv = sum(data[s]['treatment']['conversions'] for s in data)
    treatment_rate = treatment_conv / treatment_total
    
    # 检查每个分段的趋势
    segment_trends = {}
    for segment, segment_data in data.items():
        seg_control_rate = segment_data['control']['conversions'] / \
                           segment_data['control']['total']
        seg_treatment_rate = segment_data['treatment']['conversions'] / \
                              segment_data['treatment']['total']
        
        segment_trends[segment] = seg_treatment_rate > seg_control_rate
    
    # 整体趋势
    overall_trend = treatment_rate > control_rate
    
    # 检测悖论：所有分段趋势一致，但整体趋势相反
    has_paradox = (
        all(segment_trends.values()) and not overall_trend
    ) or (
        not any(segment_trends.values()) and overall_trend
    )
    
    explanation = {
        'overall_rates': {
            'control': control_rate,
            'treatment': treatment_rate,
            'treatment_better': overall_trend
        },
        'segment_trends': segment_trends,
        'has_paradox': has_paradox
    }
    
    return has_paradox, explanation

# 示例：经典的Simpson悖论
example_data = {
    'new_users': {
        'control': {'conversions': 40, 'total': 100},
        'treatment': {'conversions': 80, 'total': 200}
    },
    'returning_users': {
        'control': {'conversions': 200, 'total': 400},
        'treatment': {'conversions': 60, 'total': 100}
    }
}

has_paradox, explanation = detect_simpson_paradox(example_data)
print(f"Simpson Paradox Detected: {has_paradox}")
print(explanation)
```

### 4.2 选择偏差

```python
class SelectionBiasDetector:
    """选择偏差检测"""
    
    @staticmethod
    def check_sample_ratio_ratio(control_size, treatment_size, 
                                expected_ratio=0.5, threshold=0.05):
        """
        检查样本比例是否异常（SRM: Sample Ratio Mismatch）
        
        Args:
            control_size: 对照组样本量
            treatment_size: 实验组样本量
            expected_ratio: 期望的实验组比例
            threshold: 允许的偏差阈值
        """
        total = control_size + treatment_size
        actual_ratio = treatment_size / total
        
        # 卡方检验
        expected_control = total * (1 - expected_ratio)
        expected_treatment = total * expected_ratio
        
        observed = [control_size, treatment_size]
        expected = [expected_control, expected_treatment]
        
        chi2, p_value = stats.chisquare(observed, f_exp=expected)
        
        has_srm = p_value < 0.05
        
        return {
            'has_srm': has_srm,
            'p_value': p_value,
            'actual_ratio': actual_ratio,
            'expected_ratio': expected_ratio
        }
    
    @staticmethod
    def check_pre_experiment_balance(control_metrics, treatment_metrics):
        """
        检查实验前用户特征是否平衡
        
        Args:
            control_metrics: {user_id: {metric: value}}
            treatment_metrics: {user_id: {metric: value}}
        """
        imbalances = {}
        
        # 对每个特征进行t检验
        all_metrics = set()
        for user_metrics in control_metrics.values():
            all_metrics.update(user_metrics.keys())
        
        for metric in all_metrics:
            control_values = [
                user_metrics.get(metric, 0) 
                for user_metrics in control_metrics.values()
            ]
            treatment_values = [
                user_metrics.get(metric, 0) 
                for user_metrics in treatment_metrics.values()
            ]
            
            t_stat, p_value = stats.ttest_ind(control_values, treatment_values)
            imbalances[metric] = {
                'p_value': p_value,
                'balanced': p_value >= 0.05
            }
        
        return imbalances
```

---

## 五、高级评估技术

### 5.1 CUPED方差缩减

```python
class CUPED:
    """
    Controlled-Experiment Using Pre-Experiment Data
    
    使用实验前数据减少方差，提升统计功效
    """
    
    def __init__(self):
        self.theta = None  # 协方差系数
    
    def fit(self, pre_experiment_metrics, experiment_metrics):
        """
        计算CUPED系数
        
        Args:
            pre_experiment_metrics: 实验前指标 {user_id: metric}
            experiment_metrics: 实验期间指标 {user_id: metric}
        """
        import pandas as pd
        
        # 构建DataFrame
        df = pd.DataFrame({
            'pre': [pre_experiment_metrics.get(uid, 0) for uid in experiment_metrics.keys()],
            'exp': list(experiment_metrics.values())
        }).dropna()
        
        # 计算协方差
        cov_pre_exp = df[['pre', 'exp']].cov().iloc[0, 1]
        var_pre = df['pre'].var()
        
        # CUPED系数
        self.theta = cov_pre_exp / var_pre
        
        return self.theta
    
    def adjust(self, pre_metric, experiment_metric):
        """
        调整实验指标
        
        Args:
            pre_metric: 用户实验前指标
            experiment_metric: 用户实验期间指标
        Returns:
            adjusted_metric: 调整后的指标
        """
        if self.theta is None:
            raise ValueError("CUPED not fitted. Call fit() first.")
        
        adjusted = experiment_metric - self.theta * (pre_metric - np.mean(pre_metric))
        return adjusted
    
    def evaluate(self, control_data, treatment_data):
        """
        使用CUPED评估实验效果
        
        Args:
            control_data: {user_id: {'pre': metric, 'exp': metric}}
            treatment_data: {user_id: {'pre': metric, 'exp': metric}}
        Returns:
            results: 评估结果
        """
        # 合并数据计算theta
        all_users = set(control_data.keys()) | set(treatment_data.keys())
        
        pre_metrics = {uid: control_data.get(uid, treatment_data.get(uid))['pre'] 
                      for uid in all_users}
        exp_metrics = {}
        for uid in all_users:
            if uid in control_data:
                exp_metrics[uid] = control_data[uid]['exp']
            elif uid in treatment_data:
                exp_metrics[uid] = treatment_data[uid]['exp']
        
        # 计算theta
        self.fit(pre_metrics, exp_metrics)
        
        # 调整两组的指标
        control_adjusted = [
            self.adjust(data['pre'], data['exp'])
            for data in control_data.values()
        ]
        treatment_adjusted = [
            self.adjust(data['pre'], data['exp'])
            for data in treatment_data.values()
        ]
        
        # 计算调整后的差异
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(treatment_adjusted, control_adjusted)
        
        # 方差缩减比例
        original_var = np.var(
            [data['exp'] for data in list(control_data.values()) + list(treatment_data.values())]
        )
        adjusted_var = np.var(control_adjusted + treatment_adjusted)
        variance_reduction = 1 - adjusted_var / original_var
        
        return {
            't_stat': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'variance_reduction': variance_reduction,
            'theta': self.theta
        }
```

### 5.2 增量评估

```python
class IncrementalityEvaluator:
    """增量评估：衡量真实的增量效果"""
    
    def __init__(self):
        self.holdout_group = None  # 不接受推广的对照组
    
    def design_holdout_experiment(self, user_ids, holdout_ratio=0.1):
        """
        设计holdout实验
        
        Args:
            user_ids: 用户ID列表
            holdout_ratio: holdout组比例
        Returns:
            assignments: {user_id: group}
        """
        import random
        
        n_holdout = int(len(user_ids) * holdout_ratio)
        holdout_users = random.sample(user_ids, n_holdout)
        
        assignments = {}
        for user_id in user_ids:
            if user_id in holdout_users:
                assignments[user_id] = 'holdout'
            else:
                assignments[user_id] = 'treatment'
        
        return assignments
    
    def calculate_lift(self, treatment_metrics, holdout_metrics):
        """
        计算增量提升
        
        Args:
            treatment_metrics: 推广组指标
            holdout_metrics: holdout组指标
        Returns:
            lift_metrics: 增量指标
        """
        treatment_mean = np.mean(treatment_metrics)
        holdout_mean = np.mean(holdout_metrics)
        
        # 绝对增量
        absolute_lift = treatment_mean - holdout_mean
        
        # 相对增量
        relative_lift = absolute_lift / holdout_mean if holdout_mean > 0 else 0
        
        # iROI (incremental ROI)
        # 假设成本已知
        cost_per_user = 1.0  # 示例
        revenue_per_user = treatment_mean
        incremental_revenue = absolute_lift
        
        iroi = incremental_revenue / cost_per_user
        
        return {
            'absolute_lift': absolute_lift,
            'relative_lift': relative_lift,
            'iroi': iroi,
            'treatment_mean': treatment_mean,
            'holdout_mean': holdout_mean
        }
```

---

## 六、最佳实践

### 6.1 实验监控

```python
class ExperimentMonitor:
    """实验监控"""
    
    def __init__(self, test_name, metrics_to_track):
        self.test_name = test_name
        self.metrics_to_track = metrics_to_track
        self.alerts = []
    
    def check_sample_size(self, current_samples, required_samples):
        """检查样本量"""
        ratio = current_samples / required_samples
        if ratio < 0.5:
            self.alerts.append({
                'type': 'sample_size',
                'severity': 'warning',
                'message': f"样本量不足: {ratio:.1%} of required"
            })
    
    def check_metric_stability(self, metric_history, window=7):
        """检查指标稳定性（避免辛普森悖论）"""
        if len(metric_history) < window * 2:
            return
        
        recent = metric_history[-window:]
        earlier = metric_history[-(window*2):-window]
        
        # 检查趋势是否一致
        recent_trend = np.polyfit(range(window), recent, 1)[0]
        earlier_trend = np.polyfit(range(window), earlier, 1)[0]
        
        if np.sign(recent_trend) != np.sign(earlier_trend):
            self.alerts.append({
                'type': 'metric_stability',
                'severity': 'critical',
                'message': '指标趋势不稳定，可能存在Simpson悖论'
            })
    
    def check_health_metrics(self, control_health, treatment_health):
        """检查健康指标（如加载时间、错误率）"""
        for metric, control_val in control_health.items():
            treatment_val = treatment_health.get(metric, 0)
            
            # 如果实验组健康指标恶化超过10%
            if treatment_val > control_val * 1.1:
                self.alerts.append({
                    'type': 'health_metric',
                    'severity': 'critical',
                    'message': f"{metric}恶化: {treatment_val/control_val:.1%}"
                })
```

---

## 总结

本讲深入探讨了AB测试与推荐系统评估：

**核心要点回顾**：
1. **评估指标**：离线指标（AUC/NDCG）vs 在线指标（CTR/GMV）
2. **实验设计**：分流策略、样本量计算、渐进式 rollout
3. **统计检验**：Z检验、t检验、卡方检验、置信区间
4. **评估陷阱**：Simpson悖论、选择偏差、数据泄漏
5. **高级技术**：CUPED方差缩减、增量评估
6. **最佳实践**：持续监控、健康指标检查

**实践建议**：
- 实验前计算所需样本量
- 检查SRM（样本比例异常）
- 监控分段趋势避免Simpson悖论
- 使用CUPED提升统计功效
- 始终监控健康指标

---

## 参考资料

1. **论文**：
   - Kohavi et al. "Controlled Experiments on the Web: Survey and Practical Guide" (Data Mining and Knowledge Discovery 2009)
   - Deng et al. "All One-Hoss Bosses: Bidding in Configurable Automated Scoring Systems" (KDD '13)

2. **在线资源**：
   - Netflix Experimentation Platform: https://netflixtechblog.com/
   - Google Analytics AB Testing: https://support.google.com/analytics/

---

**第59讲完**。Series 6最后一讲将探讨大模型与推荐系统的融合。
