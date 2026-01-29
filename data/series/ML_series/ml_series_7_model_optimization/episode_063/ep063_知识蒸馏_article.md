# 第63讲：知识蒸馏Knowledge Distillation

## 课程概览
知识蒸馏通过让小模型学习大模型的知识，实现模型压缩。

## 一、蒸馏基础

### 1.1 核心思想
教师模型(大) → 学生模型(小)

### 1.2 蒸馏损失
```python
def distillation_loss(student_logits, teacher_logits, labels, temperature=3, alpha=0.5):
    """知识蒸馏损失"""
    
    # 软标签（教师概率分布）
    soft_targets = F.softmax(teacher_logits / temperature, dim=1)
    
    # 学生软预测
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    
    # 软损失
    soft_loss = F.kl_div(soft_student, soft_targets, reduction='batchmean')
    soft_loss *= (temperature ** 2)
    
    # 硬损失
    hard_loss = F.cross_entropy(student_logits, labels)
    
    # 组合
    loss = alpha * soft_loss + (1 - alpha) * hard_loss
    
    return loss
```

## 二、Logits蒸馏

```python
class LogitsDistillation:
    def __init__(self, teacher, student, temperature=3):
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
    
    def train_step(self, batch):
        x, y = batch
        
        # 教师预测
        with torch.no_grad():
            teacher_logits = self.teacher(x)
        
        # 学生预测
        student_logits = self.student(x)
        
        # 蒸馏损失
        loss = distillation_loss(student_logits, teacher_logits, y, self.temperature)
        
        # 更新学生
        loss.backward()
        optimizer.step()
        
        return loss.item()
```

## 三、特征蒸馏

```python
class FeatureDistillation(nn.Module):
    """中间层特征蒸馏"""
    
    def __init__(self, student, teacher):
        super().__init__()
        self.student = student
        self.teacher = teacher
        # 匹配层
        self.adapter = nn.Linear(teacher.feature_dim, student.feature_dim)
    
    def forward(self, x):
        # 教师特征
        with torch.no_grad():
            teacher_feat = self.teacher.extract_features(x)
        
        # 学生特征
        student_feat = self.student.extract_features(x)
        
        # 特征对齐
        aligned_teacher = self.adapter(teacher_feat)
        
        # 特征蒸馏损失
        loss = F.mse_loss(student_feat, aligned_teacher)
        
        return loss
```

## 四、关系蒸馏

```python
def relation_distillation(teacher_features, student_features):
    """学习特征间的关系"""
    
    # 计算教师特征关系矩阵
    teacher_relations = compute_relations(teacher_features)
    
    # 计算学生特征关系矩阵
    student_relations = compute_relations(student_features)
    
    # 关系蒸馏损失
    loss = F.mse_loss(student_relations, teacher_relations)
    
    return loss

def compute_relations(features):
    """计算特征间关系"""
    # 归一化
    normalized = F.normalize(features, p=2, dim=1)
    # 相似度矩阵
    relations = torch.mm(normalized, normalized.t())
    return relations
```

## 五、自蒸馏

```python
class SelfDistillation(nn.Module):
    """自蒸馏：学生也是自己的老师"""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.stop_gradient = False
    
    def forward(self, x):
        # 主分支
        main_output = self.model(x)
        
        # 辅助分支（历史输出）
        if not self.stop_gradient:
            self.history_output = main_output.detach()
        
        # 自蒸馏损失
        if hasattr(self, 'history_output'):
            loss = F.mse_loss(main_output, self.history_output)
        
        return main_output
```

## 六、在线蒸馏

```python
class OnlineDistillation(nn.Module):
    """互学习：多个模型互相学习"""
    
    def __init__(self, model_a, model_b):
        super().__init__()
        self.model_a = model_a
        self.model_b = model_b
    
    def forward(self, x, labels):
        # 模型A预测
        logits_a = self.model_a(x)
        
        # 模型B预测
        logits_b = self.model_b(x)
        
        # 交叉蒸馏
        loss_a = distillation_loss(logits_a, logits_b.detach(), labels)
        loss_b = distillation_loss(logits_b, logits_a.detach(), labels)
        
        # 总损失
        loss = loss_a + loss_b
        
        return loss
```

## 七、实践案例

### DistilBERT
```python
class DistilBERT:
    """DistilBERT实现"""
    
    def __init__(self, teacher_bert):
        self.teacher = teacher_bert
        # 初始化学生（6层）
        self.student = self.create_student()
    
    def distill(self, dataset):
        for batch in dataset:
            # 教师输出
            with torch.no_grad():
                teacher_out = self.teacher(batch)
            
            # 学生输出
            student_out = self.student(batch)
            
            # 多任务蒸馏
            loss = (
                distillation_loss(student_out.logits, teacher_out.logits) +
                0.5 * cosine_loss(student_out.hidden, teacher_out.hidden) +
                0.1 * vocabulary_loss(student_out, teacher_out)
            )
            
            loss.backward()
```

## 八、高级技巧

### 数据增强蒸馏
```python
def augmented_distillation(teacher, student, x):
    """使用增强数据进行蒸馏"""
    
    # 原始数据
    teacher_out_orig = teacher(x)
    student_out_orig = student(x)
    
    # 增强数据
    x_aug = augment(x)
    teacher_out_aug = teacher(x_aug)
    student_out_aug = student(x_aug)
    
    # 组合损失
    loss = (
        distillation_loss(student_out_orig, teacher_out_orig) +
        distillation_loss(student_out_aug, teacher_out_aug)
    )
    
    return loss
```

## 总结
知识蒸馏是模型压缩的有效方法，关键在于设计合适的蒸馏目标。

**下一讲**：神经架构搜索NAS
