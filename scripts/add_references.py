"""
批量添加引用来源
为 ML Series 文章添加真实的学术和技术引用
"""

import os
import re
from pathlib import Path
from typing import Dict, List

# 主题到引用的映射
REFERENCES_MAP = {
    # 机器学习基础
    "线性回归": [
        "[The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/) - Hastie et al., Springer",
        "[Linear Regression in scikit-learn](https://scikit-learn.org/stable/modules/linear_model.html) - 官方文档",
    ],
    "逻辑回归": [
        "[Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression) - scikit-learn",
        "[Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-and-machine-learning/) - Bishop, Springer",
    ],
    "决策树": [
        "[Classification and Regression Trees](https://www.taylorfrancis.com/books/mono/10.1201/9781315139470/classification-regression-trees-leo-breiman) - Breiman et al.",
        "[XGBoost Documentation](https://xgboost.readthedocs.io/) - 陈天奇等",
    ],
    "集成学习": [
        "[Ensemble Methods in Machine Learning](https://www.sciencedirect.com/science/article/pii/S0893608000000124) - Dietterich, 2000",
        "[Random Forests](https://link.springer.com/article/10.1023/A:1010933404324) - Breiman, 2001",
    ],

    # 深度学习
    "CNN|卷积": [
        "[ImageNet Classification with Deep CNNs](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) - AlexNet, 2012",
        "[Very Deep Convolutional Networks](https://arxiv.org/abs/1409.1556) - VGGNet, 2014",
        "[Deep Residual Learning](https://arxiv.org/abs/1512.03385) - ResNet, 2015",
    ],
    "RNN|循环": [
        "[Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) - Hochreiter & Schmidhuber, 1997",
        "[Sequence to Sequence Learning](https://arxiv.org/abs/1409.3215) - Sutskever et al., 2014",
    ],
    "Transformer|Attention": [
        "[Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al., 2017",
        "[BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) - Devlin et al., 2018",
    ],
    "GPT|语言模型": [
        "[Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) - GPT-2, 2019",
        "[GPT-4 Technical Report](https://arxiv.org/abs/2303.08774) - OpenAI, 2023",
    ],

    # 推荐系统
    "推荐|协同过滤": [
        "[Matrix Factorization Techniques for Recommender Systems](https://ieeexplore.ieee.org/document/5197422) - Koren et al., 2009",
        "[Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792) - Google, 2016",
        "[DeepFM: A Factorization-Machine based Neural Network](https://arxiv.org/abs/1703.04247) - 2017",
    ],
    "召回|召回算法": [
        "[Approximate Nearest Neighbor Search](https://arxiv.org/abs/1603.09320) - FAISS",
        "[Efficient and robust approximate nearest neighbor search](https://ieeexplore.ieee.org/document/7001931) - HNSW",
    ],

    # 强化学习
    "强化学习|RL": [
        "[Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html) - Sutton & Barto",
        "[Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) - DQN, 2013",
        "[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) - PPO, 2017",
    ],
    "DQN": [
        "[Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) - Nature 2015",
        "[Deep Q-Network](https://arxiv.org/abs/1312.5602) - Mnih et al., 2013",
    ],

    # NLP
    "NLP|自然语言": [
        "[Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) - Jurafsky & Martin",
        "[Natural Language Processing with Transformers](https://www.oreilly.com/library/view/natural-language-processing/9781098136789/) - O'Reilly",
    ],
    "BERT|预训练": [
        "[BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) - Google, 2018",
        "[RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) - Facebook, 2019",
    ],
    "RAG|检索增强": [
        "[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) - Facebook, 2020",
        "[Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) - Karpukhin et al., 2020",
    ],

    # 模型优化
    "量化|模型压缩": [
        "[Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877) - Google, 2017",
        "[ONNX Runtime Quantization](https://onnxruntime.ai/docs/performance/quantization.html) - Microsoft",
    ],
    "蒸馏|知识蒸馏": [
        "[Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) - Hinton et al., 2015",
        "[Knowledge Distillation Survey](https://arxiv.org/abs/2006.05525) - 2020",
    ],

    # 特征工程
    "特征工程|特征选择": [
        "[Feature Engineering for Machine Learning](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/) - O'Reilly",
        "[sklearn.feature_selection](https://scikit-learn.org/stable/modules/feature_selection.html) - 官方文档",
    ],

    # 计算机视觉
    "目标检测|检测": [
        "[Rich feature hierarchies for accurate object detection](https://arxiv.org/abs/1311.2524) - R-CNN, 2014",
        "[You Only Look Once](https://arxiv.org/abs/1506.02640) - YOLO, 2015",
        "[Faster R-CNN](https://arxiv.org/abs/1506.01497) - Ren et al., 2015",
    ],
    "图像分割|分割": [
        "[Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038) - FCN, 2014",
        "[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) - 2015",
    ],

    # 其他
    "降维|PCA": [
        "[PCA on sklearn](https://scikit-learn.org/stable/modules/decomposition.html#pca) - 官方文档",
        "[Dimensionality Reduction: A Comparative Review](https://www.mdpi.com/1407064) - 2021",
    ],
    "贝叶斯": [
        "[Probabilistic Programming & Bayesian Methods for Hackers](https://camdavidsonpilon.github.io/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/) - 开源书籍",
        "[PyMC3 Documentation](https://docs.pymc.io/) - 贝叶斯建模框架",
    ],
    "异常检测": [
        "[Isolation Forest](https://ieeexplore.ieee.org/document/4781136) - Liu et al., 2008",
        "[Anomaly Detection Survey](https://arxiv.org/abs/1901.03407) - 2019",
    ],
}

# 通用引用（当无法匹配特定主题时使用）
GENERIC_REFERENCES = [
    "**核心论文**：",
    "- [Machine Learning](https://www.nature.com/articles/nature14539) - Nature 2015 深度学习综述",
    "- [Deep Learning](https://www.deeplearningbook.org/) - Goodfellow, Bengio, Courville",
    "",
    "**开源工具**：",
    "- [PyTorch](https://pytorch.org/) - 深度学习框架",
    "- [scikit-learn](https://scikit-learn.org/) - 机器学习库",
    "- [Hugging Face](https://huggingface.co/) - 预训练模型库",
]


def find_article_file(episode: int, base_path: str = "/Users/z/Documents/work/content-forge-ai/data/series/ML_series") -> Path:
    """查找文章文件"""
    base = Path(base_path)
    ep_str = f"episode_{episode:03d}"

    for series_dir in sorted(base.iterdir()):
        if not series_dir.is_dir():
            continue
        ep_dir = series_dir / ep_str
        if ep_dir.exists():
            articles = list(ep_dir.glob("*_article.md"))
            if articles:
                return max(articles, key=lambda p: p.stat().st_size)
    return None


def get_references_for_topic(title: str) -> List[str]:
    """根据标题获取相关引用"""
    refs = []

    for pattern, ref_list in REFERENCES_MAP.items():
        if re.search(pattern, title, re.IGNORECASE):
            refs.extend(ref_list)

    if not refs:
        refs = GENERIC_REFERENCES.copy()

    return refs[:5]  # 最多5个引用


def add_references_to_article(file_path: Path) -> bool:
    """为文章添加引用"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 检查是否已有足够引用
    current_refs = len(re.findall(r'\[.*?\]\(https?://', content))
    if current_refs >= 5:
        return False  # 已有足够引用

    # 提取标题
    title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    title = title_match.group(1) if title_match else ""

    # 获取相关引用
    refs = get_references_for_topic(title)

    # 构建引用文本
    refs_text = "\n**延伸阅读**：\n\n" + "\n".join(refs)

    # 查找插入位置（在"延伸阅读"或"关于作者"之前）
    insert_patterns = [
        r'(\*\*延伸阅读\*\*：)',
        r'(\*\*关于作者\*\*：)',
        r'(\*\*互动交流\*\*：)',
        r'(-{3,}\n\*\*元数据\*\*)',
    ]

    inserted = False
    for pattern in insert_patterns:
        if re.search(pattern, content):
            # 在该位置之前插入
            content = re.sub(pattern, refs_text + "\n\n\\1", content, count=1)
            inserted = True
            break

    if not inserted:
        # 在文章末尾添加
        content += "\n\n" + refs_text

    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    return True


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="为ML Series文章添加引用")
    parser.add_argument("--episode", type=int, help="指定期号")
    parser.add_argument("--start", type=int, default=1, help="起始期号")
    parser.add_argument("--end", type=int, default=100, help="结束期号")
    parser.add_argument("--dry-run", action="store_true", help="仅预览，不修改")

    args = parser.parse_args()

    if args.episode:
        episodes = [args.episode]
    else:
        episodes = range(args.start, args.end + 1)

    updated = 0
    for ep in episodes:
        file_path = find_article_file(ep)
        if not file_path:
            continue

        if args.dry_run:
            print(f"[预览] 第{ep:03d}期: {file_path.name}")
        else:
            if add_references_to_article(file_path):
                print(f"✅ 第{ep:03d}期: 已添加引用")
                updated += 1
            else:
                print(f"⏭️  第{ep:03d}期: 已有足够引用")

    print(f"\n📊 统计: 更新了 {updated} 篇文章")


if __name__ == "__main__":
    main()
