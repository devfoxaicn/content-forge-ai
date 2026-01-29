# 第60讲：大模型与推荐系统的融合
## 下一代推荐系统的范式变革

---

## 课程概览

大语言模型（LLM）的兴起为推荐系统带来了新的可能性。本讲将深入探讨LLM4Rec、生成式推荐、对话式推荐、多模态推荐，以及Prompt Engineering在推荐中的应用和RLHF对齐技术。

**核心要点**：
- LLM在推荐中的优势与挑战
- 生成式推荐vs判别式推荐
- 对话式推荐系统
- 多模态推荐（文本+图像+视频）
- Prompt Engineering技巧
- RLHF与推荐对齐

---

## 一、LLM4Rec：大模型做推荐

### 1.1 为什么用LLM做推荐

```python
# 传统推荐系统 vs LLM推荐

# 传统：判别式模型
# input: user_id, item_id → output: score
# 优点：高效、可解释
# 缺点：冷启动困难、泛化性差

# LLM：生成式模型
# input: natural language → output: recommendations + explanation
# 优点：零样本/少样本、可对话、多模态
# 缺点：计算成本高、幻觉问题

# LLM在推荐中的优势：
# 1. 零样本能力：无需训练即可推荐
# 2. 可解释性：能生成推荐理由
# 3. 对话能力：可以交互式推荐
# 4. 知识融合：利用世界知识
# 5. 多模态：理解文本、图像、视频
```

### 1.2 LLM4Rec架构

```python
import torch
import torch.nn as nn

class LLM4Rec(nn.Module):
    """基于LLM的推荐模型"""
    def __init__(self, llm_model, item_embeddings, adapter_dim=64):
        super(LLM4Rec, self).__init__()
        
        # 预训练LLM（如LLaMA、GPT）
        self.llm = llm_model
        
        # 物品embedding层
        self.item_embeddings = nn.Embedding(
            len(item_embeddings), 
            len(item_embeddings[0])
        )
        self.item_embeddings.weight.data = torch.tensor(item_embeddings)
        
        # Adapter层：将物品embedding映射到LLM空间
        self.adapter = nn.Sequential(
            nn.Linear(len(item_embeddings[0]), adapter_dim),
            nn.ReLU(),
            nn.Linear(adapter_dim, llm_model.config.hidden_size)
        )
        
        # LoRA微调参数
        self.lora_A = nn.Parameter(torch.randn(llm_model.config.hidden_size, 8))
        self.lora_B = nn.Parameter(torch.randn(8, llm_model.config.hidden_size))
    
    def forward(self, input_ids, item_ids, attention_mask=None):
        """
        Args:
            input_ids: LLM输入token IDs
            item_ids: 候选物品ID
            attention_mask: 注意力mask
        Returns:
            logits: 推荐logits
        """
        # 1. 获取物品embedding
        item_emb = self.item_embeddings(item_ids)  # [B, D]
        item_emb = self.adapter(item_emb)  # [B, hidden_size]
        
        # 2. LLM前向传播
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # 3. 获取最后一层hidden state
        last_hidden = outputs.last_hidden_state  # [B, seq_len, hidden_size]
        
        # 4. 注入物品信息（LoRA方式）
        lora_output = torch.matmul(
            torch.matmul(last_hidden, self.lora_A),
            self.lora_B
        )
        enhanced_hidden = last_hidden + lora_output
        
        # 5. 计算与物品的相似度
        # 使用CLS token或平均池化
        sequence_repr = enhanced_hidden.mean(dim=1)  # [B, hidden_size]
        
        # 内积得到推荐分数
        scores = (sequence_repr * item_emb).sum(dim=1)  # [B]
        
        return scores
```

---

## 二、生成式推荐

### 2.1 推荐即生成

```python
class GenerativeRecommender:
    """生成式推荐：直接生成推荐物品"""
    def __init__(self, llm, item_catalog):
        self.llm = llm
        self.item_catalog = item_catalog  # 物品目录
        self.item_to_id = {item['name']: item['id'] for item in item_catalog}
    
    def recommend(self, user_profile, context, top_k=10):
        """
        生成式推荐
        
        Args:
            user_profile: {history, preferences, demographics}
            context: {time, location, device}
        Returns:
            recommendations: 推荐物品列表
        """
        # 构建prompt
        prompt = self._build_recommendation_prompt(user_profile, context)
        
        # 生成推荐
        generated_text = self.llm.generate(
            prompt,
            max_length=200,
            temperature=0.7,
            top_p=0.9
        )
        
        # 解析生成的物品
        recommendations = self._parse_recommendations(generated_text)
        
        return recommendations[:top_k]
    
    def _build_recommendation_prompt(self, user_profile, context):
        """构建推荐prompt"""
        prompt = f"""
你是一个智能推荐系统。根据用户信息和上下文，推荐合适的物品。

用户信息：
- 历史行为: {', '.join(user_profile['history'][-10:])}
- 偏好标签: {', '.join(user_profile['preferences'])}
- 年龄段: {user_profile['demographics']['age_group']}

上下文：
- 时间: {context['time']}
- 地点: {context['location']}

请推荐5个最相关的物品，简要说明推荐理由。
格式：物品名 - 推荐理由
"""
        return prompt
    
    def _parse_recommendations(self, generated_text):
        """解析生成的推荐文本"""
        recommendations = []
        lines = generated_text.strip().split('\n')
        
        for line in lines:
            if '-' in line:
                parts = line.split('-', 1)
                if len(parts) == 2:
                    item_name = parts[0].strip()
                    reason = parts[1].strip()
                    
                    if item_name in self.item_to_id:
                        recommendations.append({
                            'item_id': self.item_to_id[item_name],
                            'item_name': item_name,
                            'reason': reason
                        })
        
        return recommendations
```

### 2.2 概率生成模型

```python
class ProbabilisticGenerativeRec:
    """概率生成式推荐"""
    def __init__(self, item_vocab_size, hidden_dim=256):
        self.item_vocab_size = item_vocab_size
        
        # 生成器：GPT-like结构
        self.generator = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8),
            num_layers=6
        )
        
        # 物品embedding
        self.item_embedding = nn.Embedding(item_vocab_size, hidden_dim)
    
    def sample(self, user_context, num_samples=10, temperature=1.0):
        """
        从生成模型采样推荐
        
        Args:
            user_context: 用户上下文embedding
            num_samples: 采样数量
            temperature: 采样温度
        Returns:
            sampled_items: 采样的物品ID
        """
        # 初始化
        current_tokens = torch.zeros(1, 1, dtype=torch.long)  # SOS token
        hidden_states = user_context.unsqueeze(0)
        
        sampled_items = []
        
        for _ in range(num_samples):
            # 生成下一个token
            logits = self.generator(
                self.item_embedding(current_tokens),
                hidden_states
            )
            
            # 温度缩放
            logits = logits / temperature
            
            # 采样
            probs = torch.softmax(logits[-1, -1, :], dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            # 更新
            current_tokens = torch.cat([current_tokens, next_token.unsqueeze(0)], dim=1)
            
            # 记录物品
            if next_token.item() < self.item_vocab_size:
                sampled_items.append(next_token.item())
        
        return sampled_items
```

---

## 三、对话式推荐

### 3.1 对话式推荐架构

```python
class ConversationalRecommender:
    """对话式推荐系统"""
    def __init__(self, llm, knowledge_base):
        self.llm = llm
        self.knowledge_base = knowledge_base
        
        # 对话历史
        self.conversation_history = []
        
        # 用户偏好模型
        self.user_preferences = {}
    
    def chat(self, user_message, user_id):
        """对话式推荐"""
        # 1. 更新对话历史
        self.conversation_history.append({
            'role': 'user',
            'content': user_message
        })
        
        # 2. 理解用户意图
        intent = self._understand_intent(user_message)
        
        # 3. 更新用户偏好
        self._update_preferences(user_id, user_message, intent)
        
        # 4. 生成回复
        if intent == 'request_recommendation':
            response = self._generate_recommendation(user_id)
        elif intent == 'provide_feedback':
            response = self._handle_feedback(user_id, user_message)
        elif intent == 'ask_question':
            response = self._answer_question(user_id, user_message)
        else:
            response = self._clarify_intent(user_message)
        
        # 5. 更新对话历史
        self.conversation_history.append({
            'role': 'assistant',
            'content': response
        })
        
        return response
    
    def _understand_intent(self, message):
        """理解用户意图"""
        # 简化实现：使用关键词匹配
        if any(word in message for word in ['推荐', '建议', '想看']):
            return 'request_recommendation'
        elif any(word in message for word in ['喜欢', '不喜欢', '好的', '不行']):
            return 'provide_feedback'
        elif '?' in message or '什么' in message or '怎么' in message:
            return 'ask_question'
        else:
            return 'unknown'
    
    def _generate_recommendation(self, user_id):
        """生成推荐"""
        # 获取用户偏好
        preferences = self.user_preferences.get(user_id, {})
        
        # 检索知识库
        candidates = self.knowledge_base.search(
            category=preferences.get('category'),
            price_range=preferences.get('price_range')
        )
        
        # LLM生成推荐回复
        prompt = f"""
基于用户偏好：{preferences}

候选物品：{candidates[:5]}

请生成个性化的推荐回复，包含推荐理由。
"""
        
        response = self.llm.generate(prompt)
        return response
    
    def _handle_feedback(self, user_id, feedback):
        """处理用户反馈"""
        # 提取情感（正面/负面）
        sentiment = self._analyze_sentiment(feedback)
        
        # 更新偏好
        if sentiment == 'positive':
            # 强化当前偏好
            pass
        else:
            # 调整偏好
            pass
        
        return "收到您的反馈，我会调整推荐策略。"
    
    def _answer_question(self, user_id, question):
        """回答问题"""
        # 从知识库检索相关信息
        relevant_info = self.knowledge_base.search(question)
        
        # LLM生成回答
        prompt = f"""
用户问题：{question}

相关信息：{relevant_info}

请提供准确、有帮助的回答。
"""
        
        answer = self.llm.generate(prompt)
        return answer
```

### 3.2 多轮对话策略

```python
class MultiTurnDialogue:
    """多轮对话策略"""
    
    def __init__(self):
        self.dialogue_state = {}
        self.candidate_pool = {}
    
    def initialize_dialogue(self, user_id):
        """初始化对话"""
        self.dialogue_state[user_id] = {
            'turn': 0,
            'constraints': {},
            'preferences': {},
            'rejected_items': set()
        }
        
        # 初始候选池
        self.candidate_pool[user_id] = self._get_initial_candidates(user_id)
    
    def next_turn(self, user_id, user_input):
        """下一轮对话"""
        state = self.dialogue_state[user_id]
        state['turn'] += 1
        
        # 理解用户输入
        updated = self._update_state_from_input(user_id, user_input)
        
        # 对话策略
        if state['turn'] == 1:
            # 第一轮：开放式提问
            response = self._ask_open_question()
        elif state['turn'] <= 3:
            # 第2-3轮：澄清偏好
            response = self._clarify_preferences()
        elif state['turn'] <= 5:
            # 第4-5轮：展示推荐并获取反馈
            response = self._show_and_get_feedback()
        else:
            # 第6轮+：最终推荐
            response = self._final_recommendation()
        
        return response
    
    def _ask_open_question(self):
        """开放式提问"""
        return "您好！请问您今天想找什么类型的商品？可以说说您的具体需求。"
    
    def _clarify_preferences(self):
        """澄清偏好"""
        return "明白了。请问您对价格有什么要求吗？还有其他偏好吗？"
    
    def _show_and_get_feedback(self):
        """展示推荐并获取反馈"""
        # 从候选池选择
        candidates = list(self.candidate_pool[user_id])[:3]
        
        response = "根据您的需求，我为您推荐以下几个商品：\n"
        for i, item in enumerate(candidates, 1):
            response += f"{i}. {item['name']} - {item.get('description', '')}\n"
        
        response += "您觉得怎么样？需要调整推荐吗？"
        
        return response
```

---

## 四、多模态推荐

### 4.1 文本+图像推荐

```python
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

class MultiModalRecommender:
    """多模态推荐：文本+图像"""
    def __init__(self):
        # 使用CLIP作为视觉-语言编码器
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # 推荐头
        self.recommendation_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
    
    def encode_multimodal(self, text, image):
        """
        编码多模态输入
        
        Args:
            text: 文本描述
            image: PIL Image
        Returns:
            embedding: 融合的embedding
        """
        # CLIP编码
        inputs = self.clip_processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        outputs = self.clip_model(**inputs)
        
        # 文本特征
        text_features = outputs.text_embeds  # [1, 512]
        
        # 图像特征
        image_features = outputs.image_embeds  # [1, 512]
        
        # 融合
        fused = (text_features + image_features) / 2
        
        return fused
    
    def score_item(self, user_query, item_text, item_image):
        """为物品打分"""
        # 编码用户查询
        query_emb = self.encode_multimodal(
            user_query['text'],
            user_query.get('image')
        )
        
        # 编码物品
        item_emb = self.encode_multimodal(item_text, item_image)
        
        # 计算相似度
        similarity = torch.cosine_similarity(query_emb, item_emb)
        
        # 额外的打分头
        combined = torch.cat([query_emb, item_emb], dim=1)
        score = self.recommendation_head(combined)
        
        # 融合
        final_score = 0.7 * similarity + 0.3 * score.squeeze()
        
        return final_score.item()
    
    def recommend(self, user_query, item_catalog, top_k=10):
        """多模态推荐"""
        scores = []
        
        for item in item_catalog:
            score = self.score_item(
                user_query,
                item['text_description'],
                item['image']
            )
            scores.append((item, score))
        
        # 排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return [item for item, _ in scores[:top_k]]
```

### 4.2 视频推荐

```python
class VideoRecommender:
    """视频推荐：理解视频内容"""
    def __init__(self):
        # 视频编码器（使用预训练模型）
        self.video_encoder = self._load_video_encoder()
        
        # 文本编码器
        from transformers import AutoModel
        self.text_encoder = AutoModel.from_pretrained("bert-base-uncased")
    
    def _load_video_encoder(self):
        """加载视频编码器"""
        # 可以使用VideoMAE、TimeSformer等
        # 简化示例
        return nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2)),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(64, 512)
        )
    
    def encode_video(self, video_frames):
        """
        编码视频
        
        Args:
            video_frames: [T, H, W, C] 视频帧
        Returns:
            embedding: 视频embedding
        """
        # 简化：实际中需要更复杂的处理
        import torch
        video_tensor = torch.from_numpy(video_frames).permute(3, 0, 1, 2).unsqueeze(0)
        video_tensor = video_tensor.float() / 255.0
        
        embedding = self.video_encoder(video_tensor)
        return embedding
```

---

## 五、Prompt Engineering技巧

### 5.1 推荐专用Prompt模板

```python
class RecommendationPrompts:
    """推荐系统Prompt模板"""
    
    @staticmethod
    def zero_shot_prompt(user_profile, context):
        """零样本推荐"""
        prompt = f"""
你是一个专业的推荐系统。请根据以下信息为用户推荐物品。

用户画像：
- 年龄: {user_profile['age']}
- 兴趣标签: {', '.join(user_profile['interests'])}
- 最近浏览: {', '.join(user_profile['recent_views'])}

当前场景：
- 时间: {context['time']}
- 平台: {context['platform']}

请推荐5个最相关的物品，包括物品名称和简短理由。
"""
        return prompt
    
    @staticmethod
    def few_shot_prompt(user_profile, context, examples):
        """少样本推荐"""
        prompt = "你是一个专业的推荐系统。以下是几个推荐示例：\n\n"
        
        for i, example in enumerate(examples, 1):
            prompt += f"""示例 {i}:
用户: {example['user_query']}
推荐: {', '.join(example['recommendations'])}
理由: {example['reasoning']}

"""
        
        prompt += f"""现在请为以下用户推荐：
用户画像: {user_profile}
场景: {context}

推荐: 
"""
        return prompt
    
    @staticmethod
    def chain_of_thought_prompt(user_query):
        """思维链推荐"""
        prompt = f"""
用户查询: {user_query}

请一步步分析并给出推荐：

步骤1: 理解用户需求
- 用户想要什么类型的产品？
- 用户的预算范围是多少？
- 有什么特殊要求？

步骤2: 分析候选产品
- 哪些产品符合用户需求？
- 这些产品的优缺点是什么？

步骤3: 比较和排序
- 根据相关性排序
- 考虑用户可能感兴趣的因素

步骤4: 最终推荐
请列出top-5推荐及理由。
"""
        return prompt
```

### 5.2 Prompt优化策略

```python
class PromptOptimizer:
    """Prompt优化器"""
    
    def __init__(self, validation_set):
        self.validation_set = validation_set
    
    def optimize(self, base_prompt, n_iterations=10):
        """优化prompt"""
        best_prompt = base_prompt
        best_score = self._evaluate_prompt(base_prompt)
        
        for iteration in range(n_iterations):
            # 生成变体
            variants = self._generate_variants(best_prompt)
            
            # 评估变体
            for variant in variants:
                score = self._evaluate_prompt(variant)
                if score > best_score:
                    best_prompt = variant
                    best_score = score
            
            print(f"Iteration {iteration + 1}: Best score = {best_score:.3f}")
        
        return best_prompt
    
    def _generate_variants(self, prompt):
        """生成prompt变体"""
        variants = []
        
        # 策略1: 添加示例
        variant1 = prompt + "\n\n示例：\n用户：喜欢iPhone的用户\n推荐：MacBook, AirPods, Apple Watch"
        variants.append(variant1)
        
        # 策略2: 调整语气
        variant2 = prompt.replace("请推荐", "我建议你考虑")
        variants.append(variant2)
        
        # 策略3: 添加约束
        variant3 = prompt + "\n\n注意：推荐的产品应该价格合理、评价良好。"
        variants.append(variant3)
        
        return variants
    
    def _evaluate_prompt(self, prompt):
        """评估prompt质量"""
        total_score = 0
        
        for example in self.validation_set:
            # 使用prompt生成推荐
            recommendations = self.llm.generate(prompt.format(**example))
            
            # 计算与真实推荐的匹配度
            score = self._calculate_match(
                recommendations,
                example['ground_truth']
            )
            total_score += score
        
        return total_score / len(self.validation_set)
```

---

## 六、RLHF与推荐对齐

### 6.1 RLHF在推荐中的应用

```python
class RecommenderRLHF:
    """使用RLHF对齐推荐系统"""
    
    def __init__(self, policy_model, reward_model):
        self.policy_model = policy_model  # 推荐模型
        self.reward_model = reward_model  # 奖励模型
    
    def collect_feedback(self, user_id, recommendations):
        """收集用户反馈"""
        feedback = []
        
        for item in recommendations:
            # 展示给用户
            user_action = self._present_to_user(user_id, item)
            
            # 记录反馈
            feedback.append({
                'item': item,
                'clicked': user_action == 'click',
                'satisfied': user_action == 'satisfied',
                'time_spent': user_action.get('time', 0)
            })
        
        return feedback
    
    def train_reward_model(self, feedback_data):
        """训练奖励模型"""
        # 使用用户反馈训练奖励模型
        # 目标：预测人类偏好
        
        features = []
        rewards = []
        
        for feedback in feedback_data:
            # 特征：推荐物品的属性
            item_features = self._extract_item_features(feedback['item'])
            features.append(item_features)
            
            # 奖励：综合多个反馈信号
            reward = 0
            if feedback['clicked']:
                reward += 1
            if feedback['satisfied']:
                reward += 2
            reward += feedback['time_spent'] / 60  # 每分钟1分
            
            rewards.append(reward)
        
        # 训练奖励模型
        self.reward_model.fit(features, rewards)
    
    def rlhf_finetuning(self, user_data, n_episodes=100):
        """RLHF微调"""
        import torch.optim as optim
        
        optimizer = optim.Adam(self.policy_model.parameters(), lr=1e-5)
        
        for episode in range(n_episodes):
            episode_rewards = []
            
            for user_id in user_data:
                # 生成推荐
                recommendations = self.policy_model.recommend(user_id, top_k=10)
                
                # 收集反馈
                feedback = self.collect_feedback(user_id, recommendations)
                
                # 计算奖励
                features = [self._extract_item_features(f['item']) for f in feedback]
                predicted_rewards = self.reward_model.predict(features)
                total_reward = sum(predicted_rewards)
                
                episode_rewards.append(total_reward)
                
                # 策略梯度更新
                loss = self._compute_policy_loss(recommendations, predicted_rewards)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            avg_reward = sum(episode_rewards) / len(episode_rewards)
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}")
```

---

## 总结

本讲深入探讨了大模型与推荐系统的融合：

**核心要点回顾**：
1. **LLM4Rec**：利用LLM的零样本、生成、对话能力
2. **生成式推荐**：直接生成推荐物品而非判别分数
3. **对话式推荐**：多轮交互、澄清偏好、个性化
4. **多模态推荐**：文本+图像+视频的联合理解
5. **Prompt Engineering**：零样本、少样本、思维链
6. **RLHF对齐**：用人类反馈优化推荐策略

**实践建议**：
- LLM适合冷启动和解释性推荐
- 对话式推荐需要良好的对话管理
- 多模态推荐需要融合不同模态的特征
- Prompt要设计为推荐任务专用
- RLHF需要大量高质量的人类反馈

---

## 参考资料

1. **论文**：
   - Sun et al. "Is ChatGPT a Good Recommender? A Preliminary Study" (2023)
   - Hua et al. "Large Language Models as Zero-Shot Conversational Recommenders" (2023)

2. **开源项目**：
   - ChatGPT-Recommendation: https://github.com/microsoft/recommenders

---

**第60讲完**，Series 6（推荐系统系列）全部完成！下一系列将探讨模型优化技术。
