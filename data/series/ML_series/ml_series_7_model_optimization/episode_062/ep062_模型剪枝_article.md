# 第62讲：模型剪枝与网络瘦身

## 课程概览
模型剪枝通过移除冗余参数来压缩模型，是提升推理效率的重要手段。

## 一、剪枝基础

### 1.1 剪枝类型
- 非结构化剪枝：移除单个权重
- 结构化剪枝：移除整个通道/层

### 1.2 剪枝流程
```python
def prune_model(model, pruning_ratio=0.3):
    # 1. 计算重要性
    importance = compute_importance(model)
    
    # 2. 确定剪枝阈值
    threshold = np.percentile(importance, pruning_ratio * 100)
    
    # 3. 应用剪枝
    apply_pruning(model, threshold)
    
    # 4. 微调恢复
    fine_tune(model)
    
    return model
```

## 二、非结构化剪枝

### 2.1 L1正则化剪枝
```python
def l1_regularized_pruning(model, lambda_reg=0.01):
    """L1正则化促使稀疏化"""
    
    # 训练时加入L1正则
    for param in model.parameters():
        l1_penalty = lambda_reg * torch.sum(torch.abs(param))
        loss += l1_penalty
    
    # 训练后剪枝小权重
    with torch.no_grad():
        for param in model.parameters():
            mask = torch.abs(param) > 0.01
            param.mul_(mask.float())
```

### 2.2 迭代剪枝
```python
def iterative_pruning(model, iterations=5):
    """迭代式剪枝"""
    
    for i in range(iterations):
        # 训练
        train(model, epochs=10)
        
        # 剪枝
        prune_ratio = 0.2 + i * 0.1
        prune_model(model, prune_ratio)
        
        # 微调
        fine_tune(model, epochs=5)
    
    return model
```

## 三、结构化剪枝

### 3.1 通道剪枝
```python
def channel_pruning(conv_layer, pruning_ratio):
    """通道级剪枝"""
    
    # 计算每个通道的重要性（如L1范数）
    importance = torch.norm(conv_layer.weight, p=1, dim=(1, 2, 3))
    
    # 确定要剪枝的通道
    num_channels = int(pruning_ratio * conv_layer.out_channels)
    _, indices_to_prune = torch.topk(importance, num_channels, largest=False)
    
    # 创建mask
    mask = torch.ones(conv_layer.out_channels)
    mask[indices_to_prune] = 0
    
    return mask
```

### 3.2 层剪枝
```python
def layer_pruning(model, threshold):
    """移除不重要的层"""
    
    layers_to_prune = []
    for i, layer in enumerate(model.layers):
        # 评估层重要性
        importance = evaluate_layer_importance(model, layer)
        
        if importance < threshold:
            layers_to_prune.append(i)
    
    # 移除层
    model.remove_layers(layers_to_prune)
    
    return model
```

## 四、Lottery Ticket Hypothesis

```python
def lottery_ticket_pruning(model, dataset):
    """彩票假设剪枝"""
    
    # 1. 训练初始模型
    train(model, dataset, epochs=100)
    
    # 2. 剪枝
    mask = prune_model(model, ratio=0.9)
    
    # 3. 重置权重到初始值
    reset_weights(model, init_weights)
    
    # 4. 在mask下重新训练
    train_with_mask(model, dataset, mask, epochs=100)
    
    return model, mask
```

## 五、动态剪枝

```python
class DynamicPruning(nn.Module):
    """动态剪枝网络"""
    
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.gate_params = nn.ParameterList([
            nn.Parameter(torch.ones(1)) for _ in base_model.layers
        ])
    
    def forward(self, x):
        for layer, gate in zip(self.base_model.layers, self.gate_params):
            # 动态gate决定是否执行
            if torch.sigmoid(gate) > 0.5:
                x = layer(x)
        return x
```

## 六、剪枝实践

### 6.1 Torch剪枝
```python
import torch.nn.utils.prune as prune

# 结构化L1剪枝
prune.l1_structured(module, name='weight', amount=0.3)

# 移除剪枝重参数化
prune.remove(module, 'weight')
```

### 6.2 剪枝+量化组合
```python
def prune_and_quantize(model):
    """先剪枝后量化"""
    
    # 1. 剪枝
    pruned_model = prune_model(model, ratio=0.5)
    
    # 2. 微调
    fine_tune(pruned_model)
    
    # 3. 量化
    quantized_model = quantize(pruned_model)
    
    return quantized_model
```

## 总结
剪枝能显著减少模型参数和计算量，但需要平衡精度和效率。

**下一讲**：知识蒸馏
