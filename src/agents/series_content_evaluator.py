"""
系列内容质量评估Agent v8.0
评估生成的系列文章是否达到顶级科技媒体标准

融合copy-editing的七遍编辑法
- 清晰度检查
- 声音和语调检查
- "So What"检查
- 证据检查
- 具体性检查
- 情感共鸣检查
- 无障碍检查
"""

from typing import Dict, Any, List
import re
from loguru import logger


class SeriesContentEvaluatorAgent:
    """系列内容质量评估Agent"""

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        self.config = config
        self.prompts = prompts
        self.name = "series_content_evaluator"

    def evaluate(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估文章质量

        Args:
            article: 包含title和full_content的字典

        Returns:
            评估结果字典
        """
        title = article.get("title", "")
        content = article.get("full_content", "")

        logger.info(f"[SeriesContentEvaluator] 开始评估文章: {title}")

        # 执行七个维度的检查
        scores = {}
        feedback = {}

        # 1. 清晰度检查
        clarity_score, clarity_feedback = self._check_clarity(content)
        scores["clarity"] = clarity_score
        feedback["clarity"] = clarity_feedback

        # 2. 声音和语调检查
        voice_score, voice_feedback = self._check_voice_and_tone(content)
        scores["voice"] = voice_score
        feedback["voice"] = voice_feedback

        # 3. "So What"检查
        so_what_score, so_what_feedback = self._check_so_what(content)
        scores["so_what"] = so_what_score
        feedback["so_what"] = so_what_feedback

        # 4. 证据检查
        evidence_score, evidence_feedback = self._check_evidence(content)
        scores["evidence"] = evidence_score
        feedback["evidence"] = evidence_feedback

        # 5. 具体性检查
        specificity_score, specificity_feedback = self._check_specificity(content)
        scores["specificity"] = specificity_score
        feedback["specificity"] = specificity_feedback

        # 6. 情感共鸣检查
        emotion_score, emotion_feedback = self._check_emotional_resonance(content)
        scores["emotion"] = emotion_score
        feedback["emotion"] = emotion_feedback

        # 7. 无障碍检查
        accessibility_score, accessibility_feedback = self._check_accessibility(content)
        scores["accessibility"] = accessibility_score
        feedback["accessibility"] = accessibility_feedback

        # 8. 技术深度检查
        depth_score, depth_feedback = self._check_technical_depth(content)
        scores["depth"] = depth_score
        feedback["depth"] = depth_feedback

        # 9. 实践价值检查
        practicality_score, practicality_feedback = self._check_practical_value(content)
        scores["practicality"] = practicality_score
        feedback["practicality"] = practicality_feedback

        # 计算总分
        weights = {
            "clarity": 0.15,
            "voice": 0.10,
            "so_what": 0.15,
            "evidence": 0.15,
            "specificity": 0.10,
            "emotion": 0.10,
            "accessibility": 0.10,
            "depth": 0.20,  # 技术深度权重最高
            "practicality": 0.15,
        }

        total_score = sum(scores[k] * weights.get(k, 0.1) for k in scores.keys())

        # 确定评级
        if total_score >= 90:
            grade = "S"
            status = "达到顶级科技媒体发布标准"
        elif total_score >= 80:
            grade = "A"
            status = "达到优秀科技媒体发布标准"
        elif total_score >= 70:
            grade = "B"
            status = "达到一般科技媒体发布标准"
        elif total_score >= 60:
            grade = "C"
            status = "需要重大改进才能发布"
        else:
            grade = "D"
            status = "不建议发布，需要全面重写"

        result = {
            "title": title,
            "total_score": round(total_score, 2),
            "grade": grade,
            "status": status,
            "scores": scores,
            "feedback": feedback,
            "word_count": len(content),
            "recommendations": self._generate_recommendations(scores, feedback)
        }

        logger.info(f"[SeriesContentEvaluator] 评估完成: 总分 {total_score:.2f} ({grade}级)")

        return result

    def _check_clarity(self, content: str) -> tuple[float, str]:
        """清晰度检查"""
        issues = []
        score = 100.0

        # 检查句子长度
        sentences = re.split(r'[。！？.!?]', content)
        long_sentences = [s for s in sentences if len(s) > 100]
        if len(long_sentences) > len(sentences) * 0.1:  # 超过10%的句子过长
            issues.append(f"发现{len(long_sentences)}个过长的句子（>100字），建议拆分")
            score -= 10

        # 检查段落长度
        paragraphs = content.split('\n\n')
        long_paragraphs = [p for p in paragraphs if len(p) > 500]
        if len(long_paragraphs) > len(paragraphs) * 0.2:  # 超过20%的段落过长
            issues.append(f"发现{len(long_paragraphs)}个过长的段落（>500字），建议拆分")
            score -= 10

        # 检查术语密度
        tech_terms = ['Transformer', 'GPT', 'LLM', 'RAG', 'Agent', 'Fine-tuning', 'LoRA']
        term_count = sum(content.count(term) for term in tech_terms)
        if term_count == 0:
            issues.append("未发现技术术语，内容可能过于基础")
            score -= 20
        elif term_count < 10:
            issues.append(f"技术术语密度较低（{term_count}个），可以增加更多专业术语")
            score -= 5

        # 检查是否有足够的标题和结构
        headers = re.findall(r'^#+\s', content, re.MULTILINE)
        if len(headers) < 10:
            issues.append(f"标题数量较少（{len(headers)}个），建议增加更多小标题以提升可读性")
            score -= 10

        feedback = "清晰度检查通过" if not issues else "；".join(issues)
        return max(0, score), feedback

    def _check_voice_and_tone(self, content: str) -> tuple[float, str]:
        """声音和语调检查"""
        issues = []
        score = 100.0

        # 检查是否使用第一人称（技术文章应保持客观）
        first_person = content.count("我") + content.count("我们")
        if first_person > len(content) / 100:  # 每100字超过1个
            issues.append(f"第一人称使用过多（{first_person}次），建议保持客观")
            score -= 15

        # 检查是否过于口语化
        colloquialisms = ["咱们", "吧", "呢", "呀", "哦", "嗯"]
        colloquial_count = sum(content.count(word) for word in colloquialisms)
        if colloquial_count > 20:
            issues.append(f"口语化表达过多（{colloquial_count}次），建议使用更正式的表达")
            score -= 15

        # 检查是否有专业的技术表达
        professional_phrases = ["研究表明", "数据显示", "实验证明", "理论上", "实际上"]
        professional_count = sum(content.count(phrase) for phrase in professional_phrases)
        if professional_count < 3:
            issues.append("专业技术表达较少，建议增加更多学术性表达")
            score -= 10

        feedback = "语调检查通过" if not issues else "；".join(issues)
        return max(0, score), feedback

    def _check_so_what(self, content: str) -> tuple[float, str]:
        """"So What"检查 - 每个论点是否回答了"为什么重要\""""
        issues = []
        score = 100.0

        # 检查是否有"因此"、"所以"、"这意味着"等连接词
        so_what_phrases = ["因此", "所以", "这意味着", "这表明", "由此可见"]
        so_what_count = sum(content.count(phrase) for phrase in so_what_phrases)
        if so_what_count < 10:
            issues.append(f"缺少价值连接词（{so_what_count}次），建议增加更多\"因此\"、\"所以\"等表达")
            score -= 20

        # 检查是否有明确的价值陈述
        value_phrases = ["价值", "意义", "重要性", "优势", "好处", "作用"]
        value_count = sum(content.count(phrase) for phrase in value_phrases)
        if value_count < 5:
            issues.append(f"价值陈述较少（{value_count}次），建议更多说明技术价值")
            score -= 15

        # 检查引言部分是否有价值声明
        intro = content[:500]  # 假设前500字是引言
        if "为什么" not in intro and "重要性" not in intro:
            issues.append("引言部分缺少价值声明，建议增加\"为什么这很重要\"的说明")
            score -= 15

        feedback = "So What检查通过" if not issues else "；".join(issues)
        return max(0, score), feedback

    def _check_evidence(self, content: str) -> tuple[float, str]:
        """证据检查 - 每个论断是否有数据支撑"""
        issues = []
        score = 100.0

        # 检查是否有数字和数据
        numbers = re.findall(r'\d+\.?\d*%|\d+\.?\d*[万亿千百]', content)
        if len(numbers) < 10:
            issues.append(f"数据引用较少（{len(numbers)}处），建议增加更多具体数据")
            score -= 20

        # 检查是否有引用
        citations = re.findall(r'\[.*?\]|\(.*?\d{4}.*?\)', content)
        if len(citations) < 5:
            issues.append(f"引用较少（{len(citations)}处），建议增加更多权威引用")
            score -= 15

        # 检查是否有代码示例
        code_blocks = re.findall(r'```.*?```', content, re.DOTALL)
        if len(code_blocks) < 3:
            issues.append(f"代码示例较少（{len(code_blocks)}个），建议增加更多代码示例")
            score -= 15

        # 检查是否有案例研究
        case_keywords = ["案例", "例如", "比如", "实际应用", "实践中"]
        case_count = sum(content.count(keyword) for keyword in case_keywords)
        if case_count < 5:
            issues.append(f"案例引用较少（{case_count}次），建议增加更多实际案例")
            score -= 10

        feedback = "证据检查通过" if not issues else "；".join(issues)
        return max(0, score), feedback

    def _check_specificity(self, content: str) -> tuple[float, str]:
        """具体性检查 - 避免空泛表述"""
        issues = []
        score = 100.0

        # 检查是否有空泛的形容词
        vague_adjectives = ["很好", "非常", "极其", "强大", "优秀", "出色"]
        vague_count = sum(content.count(adj) for adj in vague_adjectives)
        if vague_count > 20:
            issues.append(f"空泛形容词过多（{vague_count}次），建议使用更具体的描述")
            score -= 15

        # 检查是否有具体的版本号、参数等
        specific_info = re.findall(r'\d+\.\d+(\.\d+)?|v\d+\.\d+|参数|配置', content)
        if len(specific_info) < 10:
            issues.append(f"具体技术信息较少（{len(specific_info)}处），建议增加更多版本号、参数等具体信息")
            score -= 20

        # 检查是否有具体的工具、框架名称
        tool_names = re.findall(r'[A-Z][a-zA-Z]*(?:\.js|\.py| AI| API| SDK)', content)
        if len(tool_names) < 5:
            issues.append(f"工具/框架名称较少（{len(tool_names)}个），建议增加更多具体工具名称")
            score -= 10

        feedback = "具体性检查通过" if not issues else "；".join(issues)
        return max(0, score), feedback

    def _check_emotional_resonance(self, content: str) -> tuple[float, str]:
        """情感共鸣检查"""
        issues = []
        score = 100.0

        # 检查Hook（开头100字）
        hook = content[:100]
        # 好的Hook通常包含：数字、故事、问题、对比
        hook_quality = 0
        if re.search(r'\d+', hook):
            hook_quality += 1
        if "问题" in hook or "为什么" in hook or "如何" in hook:
            hook_quality += 1
        if "但是" in hook or "然而" in hook or "不过" in hook:
            hook_quality += 1

        if hook_quality < 2:
            issues.append("开头Hook质量不够高，建议使用数字、问题或对比吸引读者")
            score -= 20

        # 检查是否有故事性元素
        story_indicators = ["故事", "经历", "历程", "发展", "演进"]
        story_count = sum(content.count(indicator) for indicator in story_indicators)
        if story_count < 3:
            issues.append(f"故事性元素较少（{story_count}次），建议增加技术演进的故事线")
            score -= 15

        # 检查结尾是否有启发性和行动号召
        ending = content[-500:]
        if "建议" not in ending and "推荐" not in ending:
            issues.append("结尾缺少行动号召，建议增加对读者的建议或推荐")
            score -= 10

        feedback = "情感共鸣检查通过" if not issues else "；".join(issues)
        return max(0, score), feedback

    def _check_accessibility(self, content: str) -> tuple[float, str]:
        """无障碍检查 - 是否易于理解"""
        issues = []
        score = 100.0

        # 检查是否有足够的标题层次
        h1 = content.count('# ')
        h2 = content.count('## ')
        h3 = content.count('### ')
        if h1 != 1:
            issues.append(f"主标题数量不正确（应为1个，实际{h1}个）")
            score -= 10
        if h2 < 5:
            issues.append(f"二级标题较少（{h2}个），建议增加更多小标题")
            score -= 10
        if h3 < 10:
            issues.append(f"三级标题较少（{h3}个），建议进一步细分内容")
            score -= 5

        # 检查是否有列表（提升可读性）
        lists = content.count('- ') + content.count('* ')
        if lists < 20:
            issues.append(f"列表项较少（{lists}个），建议使用更多列表来组织信息")
            score -= 10

        # 检查是否有足够的加粗强调
        bold = content.count('**')
        if bold < 20:
            issues.append(f"加粗强调较少（{bold}处），建议对关键词进行加强调")
            score -= 5

        feedback = "无障碍检查通过" if not issues else "；".join(issues)
        return max(0, score), feedback

    def _check_technical_depth(self, content: str) -> tuple[float, str]:
        """技术深度检查 - 这是最重要的维度"""
        issues = []
        score = 100.0

        # 检查是否有架构解析
        architecture_keywords = ["架构", "设计", "模块", "组件", "系统"]
        architecture_count = sum(content.count(kw) for kw in architecture_keywords)
        if architecture_count < 5:
            issues.append(f"架构解析内容较少（{architecture_count}次），建议增加系统架构解析")
            score -= 20

        # 检查是否有算法解析
        algorithm_keywords = ["算法", "复杂度", "优化", "时间复杂度", "空间复杂度"]
        algorithm_count = sum(content.count(kw) for kw in algorithm_keywords)
        if algorithm_count < 3:
            issues.append(f"算法解析较少（{algorithm_count}次），建议增加算法复杂度分析")
            score -= 15

        # 检查是否有对比分析
        comparison_keywords = ["对比", "比较", "区别", "差异", "优缺点", "优势"]
        comparison_count = sum(content.count(kw) for kw in comparison_keywords)
        if comparison_count < 5:
            issues.append(f"对比分析较少（{comparison_count}次），建议增加与同类技术的对比")
            score -= 15

        # 检查是否有性能分析
        performance_keywords = ["性能", "速度", "延迟", "吞吐量", "效率", "优化"]
        performance_count = sum(content.count(kw) for kw in performance_keywords)
        if performance_count < 5:
            issues.append(f"性能分析较少（{performance_count}次），建议增加性能数据和分析")
            score -= 15

        # 检查是否有前沿性内容
        cutting_edge_keywords = ["前沿", "最新", "突破", "创新", "未来", "趋势"]
        cutting_edge_count = sum(content.count(kw) for kw in cutting_edge_keywords)
        if cutting_edge_count < 3:
            issues.append(f"前沿性内容较少（{cutting_edge_count}次），建议增加技术趋势和未来展望")
            score -= 10

        # 检查是否有数学/理论内容
        math_content = re.search(r'\$.*?\$|公式|定理|原理', content)
        if not math_content:
            issues.append("缺少数学或理论内容，建议增加核心算法的数学描述")
            score -= 10

        feedback = "技术深度检查通过" if not issues else "；".join(issues)
        return max(0, score), feedback

    def _check_practical_value(self, content: str) -> tuple[float, str]:
        """实践价值检查"""
        issues = []
        score = 100.0

        # 检查是否有代码示例
        code_blocks = re.findall(r'```(?:python|javascript|bash|yaml)?\n.*?```', content, re.DOTALL)
        if len(code_blocks) < 3:
            issues.append(f"代码示例不足（{len(code_blocks)}个），建议至少提供3个完整代码示例")
            score -= 25

        # 检查是否有部署指南
        deployment_keywords = ["部署", "安装", "配置", "运行", "环境"]
        deployment_count = sum(content.count(kw) for kw in deployment_keywords)
        if deployment_count < 5:
            issues.append(f"部署指导较少（{deployment_count}次），建议增加部署和配置说明")
            score -= 20

        # 检查是否有最佳实践
        best_practice_keywords = ["最佳实践", "建议", "推荐", "注意", "避免"]
        best_practice_count = sum(content.count(kw) for kw in best_practice_keywords)
        if best_practice_count < 10:
            issues.append(f"最佳实践内容较少（{best_practice_count}次），建议增加实践经验分享")
            score -= 20

        # 检查是否有工具推荐
        tool_keywords = ["工具", "框架", "库", "平台", "推荐"]
        tool_count = sum(content.count(kw) for kw in tool_keywords)
        if tool_count < 5:
            issues.append(f"工具推荐较少（{tool_count}次），建议推荐相关工具和框架")
            score -= 15

        feedback = "实践价值检查通过" if not issues else "；".join(issues)
        return max(0, score), feedback

    def _generate_recommendations(self, scores: Dict[str, float], feedback: Dict[str, str]) -> str:
        """生成改进建议"""
        recommendations = []

        # 找出得分最低的3个维度
        sorted_scores = sorted(scores.items(), key=lambda x: x[1])[:3]

        for dimension, score in sorted_scores:
            if score < 80:
                dimension_names = {
                    "clarity": "清晰度",
                    "voice": "语调",
                    "so_what": "价值表达",
                    "evidence": "证据支撑",
                    "specificity": "具体性",
                    "emotion": "吸引力",
                    "accessibility": "可读性",
                    "depth": "技术深度",
                    "practicality": "实践价值"
                }
                recommendations.append(f"- **{dimension_names.get(dimension, dimension)}**（{score:.0f}分）：{feedback.get(dimension, '')}")

        if recommendations:
            return "\n".join(recommendations)
        else:
            return "文章质量优秀，无需改进"
