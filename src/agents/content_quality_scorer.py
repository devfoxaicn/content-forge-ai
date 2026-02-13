"""
内容质量自动评分系统
基于多维度的文章质量评估

评分维度：
1. 内容深度 (25%)
2. 代码质量 (20%)
3. 结构完整性 (15%)
4. 可读性 (15%)
5. 技术准确性 (15%)
6. SEO优化 (10%)
"""

from typing import Dict, Any, List, Tuple
import re
from dataclasses import dataclass
from src.agents.base import BaseAgent


@dataclass
class QualityDimension:
    """质量维度"""
    name: str
    score: float
    max_score: float
    weight: float
    details: List[str]


class ContentQualityScorer(BaseAgent):
    """内容质量自动评分系统"""

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        super().__init__(config, prompts)

        # 评分权重
        self.weights = {
            "content_depth": 0.25,
            "code_quality": 0.20,
            "structure": 0.15,
            "readability": 0.15,
            "accuracy": 0.15,
            "seo": 0.10
        }

        # 质量标准
        self.standards = {
            "min_word_count": 8000,
            "ideal_word_count": 12000,
            "min_code_blocks": 2,
            "ideal_code_blocks": 5,
            "min_headers": 5,
            "ideal_headers": 10,
            "max_paragraph_length": 200,
            "min_examples": 3
        }

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行质量评分

        Args:
            state: 包含 longform_article 的状态

        Returns:
            更新后的状态，包含质量评分报告
        """
        self.log("开始内容质量评分...")

        article = state.get("longform_article", "")
        if not article:
            self.log("警告：未找到文章内容", "WARNING")
            return state

        # 执行各维度评分
        dimensions = {
            "content_depth": self._score_content_depth(article),
            "code_quality": self._score_code_quality(article),
            "structure": self._score_structure(article),
            "readability": self._score_readability(article),
            "accuracy": self._score_accuracy(article, state),
            "seo": self._score_seo(state.get("seo_metadata", {}))
        }

        # 计算加权总分
        total_score = sum(
            dim.score * self.weights[name]
            for name, dim in dimensions.items()
        )

        # 生成报告
        quality_report = {
            "total_score": round(total_score, 1),
            "grade": self._get_grade(total_score),
            "dimensions": {
                name: {
                    "score": dim.score,
                    "max_score": dim.max_score,
                    "weight": dim.weight,
                    "details": dim.details
                }
                for name, dim in dimensions.items()
            },
            "recommendations": self._generate_recommendations(dimensions),
            "passed": total_score >= 70
        }

        self.log(f"质量评分完成：{total_score:.1f}/100 ({quality_report['grade']})")

        return {
            **state,
            "quality_report": quality_report
        }

    def _score_content_depth(self, article: str) -> QualityDimension:
        """
        评估内容深度

        考虑因素：
        - 文章长度
        - 技术术语密度
        - 是否包含深入分析
        """
        details = []
        score = 0
        max_score = 100

        # 文章长度评分 (40分)
        word_count = len(article)
        if word_count >= self.standards["ideal_word_count"]:
            length_score = 40
            details.append(f"✅ 文章长度充足 ({word_count:,} 字)")
        elif word_count >= self.standards["min_word_count"]:
            length_score = 30
            details.append(f"⚠️ 文章长度达标 ({word_count:,} 字)")
        else:
            length_score = int(word_count / self.standards["min_word_count"] * 30)
            details.append(f"❌ 文章长度不足 ({word_count:,} 字，建议 {self.standards['min_word_count']:,}+)")
        score += length_score

        # 技术术语密度 (30分)
        tech_terms = self._count_technical_terms(article)
        term_density = tech_terms / (word_count / 1000) if word_count > 0 else 0
        if term_density >= 10:
            term_score = 30
            details.append(f"✅ 技术术语丰富 ({tech_terms} 个)")
        elif term_density >= 5:
            term_score = 20
            details.append(f"⚠️ 技术术语适中 ({tech_terms} 个)")
        else:
            term_score = 10
            details.append(f"❌ 技术术语较少 ({tech_terms} 个)")
        score += term_score

        # 深度分析标记 (30分)
        depth_indicators = [
            (r'##\s+.*深入', '深入分析章节'),
            (r'##\s+.*原理', '原理解析章节'),
            (r'##\s+.*实战', '实战应用章节'),
            (r'##\s+.*优化', '优化策略章节'),
            (r'##\s+.*对比', '对比分析章节'),
        ]

        depth_score = 0
        found_indicators = []
        for pattern, name in depth_indicators:
            if re.search(pattern, article):
                depth_score += 6
                found_indicators.append(name)

        if depth_score >= 24:
            details.append(f"✅ 内容深度良好 ({', '.join(found_indicators)})")
        elif depth_score >= 12:
            details.append(f"⚠️ 内容深度适中 ({', '.join(found_indicators)})")
        else:
            details.append("❌ 缺少深度分析章节")
        score += min(30, depth_score)

        return QualityDimension(
            name="内容深度",
            score=score,
            max_score=max_score,
            weight=self.weights["content_depth"],
            details=details
        )

    def _score_code_quality(self, article: str) -> QualityDimension:
        """
        评估代码质量

        考虑因素：
        - 代码块数量
        - 代码注释完整性
        - 代码语言多样性
        """
        details = []
        score = 0
        max_score = 100

        # 提取代码块
        code_blocks = re.findall(r'```(\w+)?\n(.*?)```', article, re.DOTALL)
        code_count = len(code_blocks)

        # 代码块数量评分 (40分)
        if code_count >= self.standards["ideal_code_blocks"]:
            count_score = 40
            details.append(f"✅ 代码示例充足 ({code_count} 个)")
        elif code_count >= self.standards["min_code_blocks"]:
            count_score = 25
            details.append(f"⚠️ 代码示例达标 ({code_count} 个)")
        elif code_count > 0:
            count_score = 15
            details.append(f"❌ 代码示例较少 ({code_count} 个)")
        else:
            count_score = 0
            details.append("❌ 缺少代码示例")
        score += count_score

        # 代码注释评分 (30分)
        commented_blocks = 0
        for lang, code in code_blocks:
            # 检查是否有注释
            has_comments = bool(re.search(r'#.*|//.*|/\*.*\*/|""".*"""', code, re.DOTALL))
            if has_comments:
                commented_blocks += 1

        if code_count > 0:
            comment_ratio = commented_blocks / code_count
            if comment_ratio >= 0.8:
                comment_score = 30
                details.append(f"✅ 代码注释完善 ({commented_blocks}/{code_count})")
            elif comment_ratio >= 0.5:
                comment_score = 20
                details.append(f"⚠️ 代码注释适中 ({commented_blocks}/{code_count})")
            else:
                comment_score = 10
                details.append(f"❌ 代码注释不足 ({commented_blocks}/{code_count})")
        else:
            comment_score = 0
        score += comment_score

        # 语言多样性评分 (30分)
        languages = set(lang for lang, _ in code_blocks if lang)
        if len(languages) >= 3:
            diversity_score = 30
            details.append(f"✅ 代码语言多样 ({', '.join(languages)})")
        elif len(languages) >= 2:
            diversity_score = 20
            details.append(f"⚠️ 代码语言适中 ({', '.join(languages)})")
        elif len(languages) == 1:
            diversity_score = 10
            details.append(f"单一语言 ({list(languages)[0]})")
        else:
            diversity_score = 0
        score += diversity_score

        return QualityDimension(
            name="代码质量",
            score=score,
            max_score=max_score,
            weight=self.weights["code_quality"],
            details=details
        )

    def _score_structure(self, article: str) -> QualityDimension:
        """
        评估结构完整性

        考虑因素：
        - 章节数量
        - 标题层级
        - 是否包含必要章节
        """
        details = []
        score = 0
        max_score = 100

        # 提取标题
        headers = re.findall(r'^(#{1,3})\s+(.+)$', article, re.MULTILINE)
        header_count = len(headers)

        # 章节数量评分 (40分)
        if header_count >= self.standards["ideal_headers"]:
            structure_score = 40
            details.append(f"✅ 章节结构完善 ({header_count} 个章节)")
        elif header_count >= self.standards["min_headers"]:
            structure_score = 25
            details.append(f"⚠️ 章节结构达标 ({header_count} 个章节)")
        else:
            structure_score = int(header_count / self.standards["min_headers"] * 25)
            details.append(f"❌ 章节结构不足 ({header_count} 个章节)")
        score += structure_score

        # 必要章节检查 (30分)
        required_sections = [
            (r'#+\s*引言|简介|背景', '引言'),
            (r'#+\s*原理|核心|架构', '核心原理'),
            (r'#+\s*实现|代码|实践', '实践应用'),
            (r'#+\s*总结|结论|展望', '总结展望'),
        ]

        found_sections = []
        for pattern, name in required_sections:
            if re.search(pattern, article):
                found_sections.append(name)

        section_score = len(found_sections) * 7.5
        if len(found_sections) == len(required_sections):
            details.append(f"✅ 必要章节完整 ({', '.join(found_sections)})")
        else:
            missing = [name for _, name in required_sections if name not in found_sections]
            details.append(f"⚠️ 缺少章节: {', '.join(missing)}")
        score += section_score

        # 层级结构评分 (30分)
        h1_count = len([h for h in headers if len(h[0]) == 1])
        h2_count = len([h for h in headers if len(h[0]) == 2])
        h3_count = len([h for h in headers if len(h[0]) == 3])

        if h1_count >= 1 and h2_count >= 3 and h3_count >= 2:
            hierarchy_score = 30
            details.append("✅ 标题层级合理")
        elif h2_count >= 3:
            hierarchy_score = 20
            details.append("⚠️ 标题层级适中")
        else:
            hierarchy_score = 10
            details.append("❌ 标题层级简单")
        score += hierarchy_score

        return QualityDimension(
            name="结构完整性",
            score=score,
            max_score=max_score,
            weight=self.weights["structure"],
            details=details
        )

    def _score_readability(self, article: str) -> QualityDimension:
        """
        评估可读性

        考虑因素：
        - 段落长度
        - 句子复杂度
        - 术语解释
        """
        details = []
        score = 0
        max_score = 100

        # 段落分析 (40分)
        paragraphs = [p for p in article.split('\n\n') if p.strip() and not p.startswith('#')]
        if paragraphs:
            avg_length = sum(len(p) for p in paragraphs) / len(paragraphs)
            long_paragraphs = sum(1 for p in paragraphs if len(p) > self.standards["max_paragraph_length"])

            if avg_length <= 150 and long_paragraphs == 0:
                para_score = 40
                details.append(f"✅ 段落长度适中 (平均 {avg_length:.0f} 字)")
            elif avg_length <= 200:
                para_score = 25
                details.append(f"⚠️ 部分段落较长 ({long_paragraphs} 个)")
            else:
                para_score = 15
                details.append(f"❌ 段落过长 (平均 {avg_length:.0f} 字)")
        else:
            para_score = 0
        score += para_score

        # 列表和表格使用 (30分)
        lists = len(re.findall(r'^\s*[-*+]\s+', article, re.MULTILINE))
        tables = len(re.findall(r'\|.+\|', article))

        if lists >= 5 and tables >= 1:
            format_score = 30
            details.append(f"✅ 格式丰富 (列表 {lists} 个，表格 {tables} 个)")
        elif lists >= 3:
            format_score = 20
            details.append(f"⚠️ 格式适中 (列表 {lists} 个)")
        else:
            format_score = 10
            details.append("❌ 格式单一")
        score += format_score

        # 术语解释 (30分)
        # 检查是否有术语解释模式
        term_explanations = len(re.findall(r'[（(][^)）]+[)）]|：[^。\n]+', article))
        if term_explanations >= 10:
            explain_score = 30
            details.append(f"✅ 术语解释充分 ({term_explanations} 处)")
        elif term_explanations >= 5:
            explain_score = 20
            details.append(f"⚠️ 术语解释适中 ({term_explanations} 处)")
        else:
            explain_score = 10
            details.append("❌ 缺少术语解释")
        score += explain_score

        return QualityDimension(
            name="可读性",
            score=score,
            max_score=max_score,
            weight=self.weights["readability"],
            details=details
        )

    def _score_accuracy(self, article: str, state: Dict[str, Any]) -> QualityDimension:
        """
        评估技术准确性

        考虑因素：
        - 事实核查结果
        - 代码审查结果
        - 引用来源
        """
        details = []
        score = 0
        max_score = 100

        # 检查事实核查结果 (40分)
        fact_check = state.get("fact_check_result", {})
        if fact_check.get("passed", True):
            fact_score = 40
            details.append("✅ 事实核查通过")
        else:
            issues = fact_check.get("issues", [])
            fact_score = max(20, 40 - len(issues) * 10)
            details.append(f"⚠️ 事实核查发现 {len(issues)} 个问题")
        score += fact_score

        # 检查代码审查结果 (30分)
        code_review = state.get("code_review_result", {})
        if code_review.get("passed", True):
            code_score = 30
            details.append("✅ 代码审查通过")
        else:
            issues = code_review.get("issues", [])
            code_score = max(15, 30 - len(issues) * 5)
            details.append(f"⚠️ 代码审查发现 {len(issues)} 个问题")
        score += code_score

        # 检查引用来源 (30分)
        references = len(re.findall(r'\[.*?\]\(https?://', article))
        if references >= 5:
            ref_score = 30
            details.append(f"✅ 引用来源充足 ({references} 个)")
        elif references >= 2:
            ref_score = 20
            details.append(f"⚠️ 引用来源适中 ({references} 个)")
        else:
            ref_score = 10
            details.append("❌ 缺少引用来源")
        score += ref_score

        return QualityDimension(
            name="技术准确性",
            score=score,
            max_score=max_score,
            weight=self.weights["accuracy"],
            details=details
        )

    def _score_seo(self, seo_metadata: Dict[str, Any]) -> QualityDimension:
        """
        评估 SEO 优化
        """
        details = []
        score = 0
        max_score = 100

        if not seo_metadata:
            return QualityDimension(
                name="SEO优化",
                score=50,
                max_score=max_score,
                weight=self.weights["seo"],
                details=["⚠️ 未进行 SEO 优化"]
            )

        # SEO 评分 (50分)
        seo_score = seo_metadata.get("seo_score", 0)
        if seo_score >= 80:
            details.append(f"✅ SEO 评分优秀 ({seo_score}/100)")
            score += 50
        elif seo_score >= 60:
            details.append(f"⚠️ SEO 评分达标 ({seo_score}/100)")
            score += 35
        else:
            details.append(f"❌ SEO 评分不足 ({seo_score}/100)")
            score += 20

        # 结构化数据 (30分)
        if seo_metadata.get("structured_data"):
            details.append("✅ 包含结构化数据")
            score += 30
        else:
            details.append("❌ 缺少结构化数据")

        # 内链建议 (20分)
        internal_links = seo_metadata.get("internal_links_suggestions", [])
        if internal_links:
            details.append(f"✅ 内链建议 ({len(internal_links)} 个)")
            score += 20
        else:
            details.append("⚠️ 无内链建议")

        return QualityDimension(
            name="SEO优化",
            score=score,
            max_score=max_score,
            weight=self.weights["seo"],
            details=details
        )

    def _count_technical_terms(self, article: str) -> int:
        """统计技术术语数量"""
        # 常见技术术语模式
        tech_patterns = [
            r'\b(?:API|SDK|REST|GraphQL|gRPC|HTTP|HTTPS|TCP|UDP|JSON|XML|YAML)\b',
            r'\b(?:TensorFlow|PyTorch|Keras|scikit-learn|NumPy|Pandas)\b',
            r'\b(?:Transformer|CNN|RNN|LSTM|GRU|GAN|VAE|Diffusion)\b',
            r'\b(?:fine-tuning|pre-training|RLHF|SFT|DPO)\b',
            r'\b(?:embedding|tokenization|attention|layer|batch|epoch)\b',
        ]

        count = 0
        for pattern in tech_patterns:
            count += len(re.findall(pattern, article, re.IGNORECASE))

        return count

    def _get_grade(self, score: float) -> str:
        """获取等级"""
        if score >= 90:
            return "A+ (优秀)"
        elif score >= 85:
            return "A (良好)"
        elif score >= 80:
            return "B+ (较好)"
        elif score >= 75:
            return "B (达标)"
        elif score >= 70:
            return "C+ (及格)"
        elif score >= 60:
            return "C (勉强)"
        else:
            return "D (不及格)"

    def _generate_recommendations(self, dimensions: Dict[str, QualityDimension]) -> List[str]:
        """生成优化建议"""
        recommendations = []

        for name, dim in dimensions.items():
            if dim.score < 60:
                # 找出问题详情
                issues = [d for d in dim.details if d.startswith('❌') or d.startswith('⚠️')]
                for issue in issues:
                    recommendations.append(f"[{dim.name}] {issue[2:]}")

        return recommendations[:10]  # 最多10条建议
