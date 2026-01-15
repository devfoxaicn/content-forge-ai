"""
质量评估Agent - 基于世界级10大标准进行多维度评分
"""

import re
from typing import Dict, Any, List, Optional
from datetime import datetime
from src.agents.base import BaseAgent


class QualityEvaluatorAgent(BaseAgent):
    """质量评估Agent - 多维度评分并生成改进建议"""

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        super().__init__(config, prompts)
        self.min_score = config.get("min_score", 7.0)
        self.generate_improvement = config.get("generate_improvement", True)
        self.full_evaluation = config.get("full_evaluation", True)

        # 质量维度权重配置
        self.dimension_weights = {
            "structure": 0.15,      # 清晰的层次结构
            "depth": 0.25,          # 深度与广度平衡
            "accuracy": 0.20,       # 技术准确性
            "readability": 0.15,    # 可读性
            "visual": 0.10,         # 可视化呈现
            "timeliness": 0.15      # 与时俱进
        }

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行质量评估

        Args:
            state: 当前工作流状态，包含longform_article, code_review_result, fact_check_result

        Returns:
            更新后的状态，包含quality_report
        """
        self.log("开始质量评估...")

        try:
            # 获取文章和相关审查结果
            article = state.get("longform_article", {})
            content = article.get("full_content", "")
            title = article.get("title", "")

            code_review = state.get("code_review_result", {})
            fact_check = state.get("fact_check_result", {})

            if not content:
                self.log("未找到文章内容，跳过质量评估", "WARNING")
                return {**state, "quality_report": None}

            # 多维度评分
            dimension_scores = self._evaluate_all_dimensions(
                title, content, code_review, fact_check
            )

            # 计算加权总分
            overall_score = self._calculate_weighted_score(dimension_scores)

            # 生成改进建议
            improvements = self._generate_improvements(dimension_scores, overall_score)

            # 如果分数低于min_score，生成优化版本
            improved_article = None
            if overall_score < self.min_score and self.generate_improvement:
                self.log(f"质量分数 {overall_score:.1f} 低于阈值 {self.min_score}，生成改进建议...")
                improved_article = self._generate_improvement_suggestions(article, improvements)

            # 构建质量报告
            report = {
                "overall_score": overall_score,
                "dimension_scores": dimension_scores,
                "improvements": improvements,
                "improved_article": improved_article,
                "meets_threshold": overall_score >= self.min_score,
                "metadata": {
                    "evaluated_at": datetime.now().isoformat(),
                    "article_length": len(content),
                    "word_count": len(content),
                    "code_blocks": code_review.get("total_blocks", 0) if code_review else 0,
                    "fact_claims": fact_check.get("total_claims", 0) if fact_check else 0
                }
            }

            self.log(f"质量评估完成，总分: {overall_score:.1f}/10")

            return {**state, "quality_report": report}

        except Exception as e:
            self.log(f"质量评估失败: {e}", "ERROR")
            return {
                **state,
                "quality_report": {
                    "overall_score": 6.0,
                    "error": str(e)
                }
            }

    def _evaluate_all_dimensions(self, title: str, content: str,
                                  code_review: Dict[str, Any],
                                  fact_check: Dict[str, Any]) -> Dict[str, Any]:
        """执行所有维度的评估"""
        scores = {}

        # 1. 结构评分
        scores["structure"] = self._evaluate_structure(content)

        # 2. 深度评分
        scores["depth"] = self._evaluate_depth(content)

        # 3. 准确性评分（基于code_review和fact_check）
        scores["accuracy"] = self._evaluate_accuracy(content, code_review, fact_check)

        # 4. 可读性评分
        scores["readability"] = self._evaluate_readability(content)

        # 5. 可视化评分
        scores["visual"] = self._evaluate_visual(content)

        # 6. 时效性评分
        scores["timeliness"] = self._evaluate_timeliness(content)

        return scores

    def _evaluate_structure(self, content: str) -> Dict[str, Any]:
        """评估文章结构"""
        score = 0.0
        feedback = []

        # 检查标题层级
        headings = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILELINE)
        if len(headings) >= 5:
            score += 2.0
        elif len(headings) >= 3:
            score += 1.0
        else:
            feedback.append("建议增加更多章节标题，提升结构清晰度")

        # 检查是否有引言
        if re.search(r'##\s*(引言|概述|介绍|背景)', content):
            score += 1.5
        else:
            feedback.append("建议添加引言章节")

        # 检查是否有总结
        if re.search(r'##\s*(总结|结语|展望|未来)', content):
            score += 1.5
        else:
            feedback.append("建议添加总结章节")

        # 检查逻辑连贯性（过渡词）
        transitions = ['因此', '所以', '然而', '但是', '此外', '另外', '首先', '其次', '最后']
        transition_count = sum(1 for t in transitions if t in content)
        if transition_count >= 5:
            score += 2.0
        else:
            feedback.append("建议增加过渡词，提升逻辑连贯性")

        # 检查章节平衡
        sections = re.split(r'##\s+', content)
        section_lengths = [len(s) for s in sections if len(s) > 100]
        if section_lengths:
            avg_length = sum(section_lengths) / len(section_lengths)
            max_length = max(section_lengths)
            if max_length / avg_length < 3.0:  # 最大章节不超过平均的3倍
                score += 2.0
            else:
                feedback.append("部分章节篇幅过长，建议适当平衡")
        else:
            score += 1.0

        # 检查列表使用
        if re.search(r'^\s*[-*+]\s+', content, re.MULTILINE):
            score += 1.0

        return {
            "score": min(score, 10.0),
            "weight": self.dimension_weights["structure"],
            "feedback": feedback
        }

    def _evaluate_depth(self, content: str) -> Dict[str, Any]:
        """评估内容深度"""
        score = 0.0
        feedback = []

        # 检查字数
        word_count = len(content)
        if word_count >= 10000:
            score += 2.0
        elif word_count >= 5000:
            score += 1.5
        elif word_count >= 3000:
            score += 1.0
        else:
            feedback.append("文章篇幅较短，建议增加内容深度")

        # 检查代码示例
        code_blocks = re.findall(r'```\w*\n.*?```', content, re.DOTALL)
        if len(code_blocks) >= 3:
            score += 2.0
        elif len(code_blocks) >= 1:
            score += 1.0
        else:
            feedback.append("建议添加代码示例，提升技术深度")

        # 检查技术细节
        technical_keywords = ['实现', '原理', '算法', '架构', '机制', '优化', '性能', '部署']
        keyword_count = sum(1 for kw in technical_keywords if kw in content)
        if keyword_count >= 5:
            score += 2.0
        elif keyword_count >= 3:
            score += 1.0
        else:
            feedback.append("建议增加技术细节的讲解")

        # 检查数据支撑
        if re.search(r'\d+%', content) or re.search(r'\d+倍', content):
            score += 1.5
        else:
            feedback.append("建议添加数据支撑，增强说服力")

        # 检查实例/案例
        if re.search(r'案例|示例|实例|实践|应用', content):
            score += 1.5
        else:
            feedback.append("建议添加实际案例，增强实用性")

        # 检查最佳实践
        if re.search(r'最佳实践|建议|推荐|注意事项', content):
            score += 1.0

        return {
            "score": min(score, 10.0),
            "weight": self.dimension_weights["depth"],
            "feedback": feedback
        }

    def _evaluate_accuracy(self, content: str,
                          code_review: Optional[Dict[str, Any]],
                          fact_check: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """评估技术准确性"""
        score = 5.0  # 基础分
        feedback = []

        # 基于代码审查结果
        if code_review:
            code_score = code_review.get("score", 5.0)
            # 将0-10分映射到0-5分
            score += (code_score / 10.0) * 3.0

            if code_score < 7.0:
                feedback.append("代码示例存在问题，建议检查")

        # 基于事实核查结果
        if fact_check:
            accuracy_rate = fact_check.get("accuracy_rate", 0.8)
            # 将准确率映射到0-2分
            score += accuracy_rate * 2.0

            high_risk_count = len(fact_check.get("high_risk_claims", []))
            if high_risk_count > 0:
                feedback.append(f"发现{high_risk_count}个高风险声明，建议核实")
        else:
            score += 1.5  # 无事实核查时给予部分分数

        # 检查技术术语使用
        if re.search(r'TODO|FIXME|待补充|后续完善', content):
            score -= 1.0
            feedback.append("文章中包含未完成的内容标记")

        return {
            "score": min(max(score, 0), 10.0),
            "weight": self.dimension_weights["accuracy"],
            "feedback": feedback
        }

    def _evaluate_readability(self, content: str) -> Dict[str, Any]:
        """评估可读性"""
        score = 0.0
        feedback = []

        # 移除代码块后计算
        content_no_code = re.sub(r'```.+?```', '', content, flags=re.DOTALL)

        # 检查段落长度（按空行分段）
        paragraphs = [p.strip() for p in content_no_code.split('\n\n') if p.strip()]
        if len(paragraphs) >= 10:
            score += 2.0
        elif len(paragraphs) >= 5:
            score += 1.0

        # 检查平均段落长度
        if paragraphs:
            avg_para_length = sum(len(p) for p in paragraphs) / len(paragraphs)
            if 100 <= avg_para_length <= 500:  # 理想段落长度
                score += 2.0
            elif avg_para_length > 800:
                feedback.append("部分段落过长，建议拆分")
            else:
                score += 1.0

        # 检查句子长度（粗略估算）
        sentences = re.split(r'[。！？]', content_no_code)
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            avg_sentence_length = sum(len(s) for s in sentences) / len(sentences)
            if avg_sentence_length <= 50:  # 理想句子长度
                score += 2.0
            elif avg_sentence_length <= 100:
                score += 1.0
            else:
                feedback.append("部分句子过长，建议简化")

        # 检查术语解释
        if re.search(r'所谓|是指|指的是|即', content):
            score += 1.0

        # 检查格式统一性
        has_bold = '**' in content
        has_italic = '*' in content
        if has_bold or has_italic:
            score += 1.0

        # 检查空行使用
        blank_lines = content.count('\n\n')
        if blank_lines >= 5:
            score += 1.0
        else:
            feedback.append("建议增加空行，提升可读性")

        return {
            "score": min(score, 10.0),
            "weight": self.dimension_weights["readability"],
            "feedback": feedback
        }

    def _evaluate_visual(self, content: str) -> Dict[str, Any]:
        """评估可视化呈现"""
        score = 0.0
        feedback = []

        # 检查图表
        diagram_markers = ['图表', '架构图', '流程图', '示意图', '图解']
        has_diagram_mentions = any(marker in content for marker in diagram_markers)
        if has_diagram_mentions:
            score += 3.0
        else:
            feedback.append("建议添加架构图或流程图")

        # 检查Mermaid代码
        if re.search(r'```mermaid|```graph', content):
            score += 4.0
        else:
            feedback.append("建议使用Mermaid添加图表")

        # 检查表格
        if re.search(r'\|.+\|', content):
            score += 1.5

        # 检查列表
        if re.search(r'^\s*[-*+]\s+', content, re.MULTILINE):
            score += 1.5

        return {
            "score": min(score, 10.0),
            "weight": self.dimension_weights["visual"],
            "feedback": feedback
        }

    def _evaluate_timeliness(self, content: str) -> Dict[str, Any]:
        """评估时效性"""
        score = 0.0
        feedback = []

        # 检查时间相关词汇
        time_keywords = ['最新', '当前', '目前', '近期', '2024', '2025', '2026']
        time_keyword_count = sum(1 for kw in time_keywords if kw in content)
        if time_keyword_count >= 3:
            score += 3.0
        elif time_keyword_count >= 1:
            score += 1.5
        else:
            feedback.append("建议明确指出信息的时效性")

        # 检查前瞻性内容
        forward_keywords = ['未来', '展望', '趋势', '发展', '方向', '规划']
        forward_count = sum(1 for kw in forward_keywords if kw in content)
        if forward_count >= 2:
            score += 3.0
        else:
            feedback.append("建议添加未来发展趋势的分析")

        # 检查版本信息
        if re.search(r'版本|v\d+\.', content):
            score += 2.0

        # 检查实践验证
        if re.search(r'实践|验证|测试|实际', content):
            score += 2.0

        return {
            "score": min(score, 10.0),
            "weight": self.dimension_weights["timeliness"],
            "feedback": feedback
        }

    def _calculate_weighted_score(self, dimension_scores: Dict[str, Any]) -> float:
        """计算加权总分"""
        total_score = 0.0
        total_weight = 0.0

        for dimension, data in dimension_scores.items():
            score = data.get("score", 0)
            weight = data.get("weight", 0)
            total_score += score * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def _generate_improvements(self, dimension_scores: Dict[str, Any],
                              overall_score: float) -> List[str]:
        """生成改进建议"""
        improvements = []

        # 收集所有维度的反馈
        for dimension, data in dimension_scores.items():
            feedback = data.get("feedback", [])
            if feedback:
                dimension_name = {
                    "structure": "结构",
                    "depth": "深度",
                    "accuracy": "准确性",
                    "readability": "可读性",
                    "visual": "可视化",
                    "timeliness": "时效性"
                }.get(dimension, dimension)

                for item in feedback:
                    improvements.append(f"[{dimension_name}] {item}")

        # 添加总体建议
        if overall_score < 5.0:
            improvements.append("文章质量较低，建议全面重构")
        elif overall_score < 7.0:
            improvements.append("文章质量中等，建议重点改进低分维度")
        elif overall_score < 9.0:
            improvements.append("文章质量良好，继续完善可达到优秀")

        return improvements

    def _generate_improvement_suggestions(self, article: Dict[str, Any],
                                         improvements: List[str]) -> Dict[str, Any]:
        """生成改进建议（不直接重写，只提供建议）"""
        title = article.get("title", "")
        content = article.get("full_content", "")

        prompt = f"""基于以下质量评估结果，为文章提供具体的改进建议。

文章标题：{title}

改进建议：
{chr(10).join(f'- {imp}' for imp in improvements[:10])}

请提供：
1. **结构改进建议**：如何优化文章结构
2. **内容补充建议**：哪些内容需要补充或深化
3. **具体修改方案**：针对每个问题的具体修改建议

请以Markdown格式返回，不要重写整篇文章，只提供改进建议。"""

        try:
            response = self._call_llm(prompt)

            return {
                "suggestions": response,
                "generated_at": datetime.now().isoformat()
            }
        except Exception as e:
            self.log(f"生成改进建议失败: {e}", "WARNING")
            return {
                "suggestions": "无法生成详细建议，请参考上述改进意见进行优化。",
                "error": str(e)
            }
