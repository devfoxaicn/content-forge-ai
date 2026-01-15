"""
一致性检查Agent - 检查文章的术语、引用、数据一致性
纯Python实现，无需额外API成本
"""

import re
import json
from typing import Dict, Any, List, Optional, Set, Tuple
from collections import defaultdict
from src.agents.base import BaseAgent


class ConsistencyCheckerAgent(BaseAgent):
    """一致性检查Agent - 确保术语、引用、数据的一致性"""

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        super().__init__(config, prompts)
        self.use_memory_mcp = config.get("use_memory_mcp", False)
        self.check_terminology = config.get("check_terminology", True)
        self.check_citations = config.get("check_citations", True)
        self.check_data = config.get("check_data", True)

        # 术语一致性规则（可扩展）
        self.terminology_rules = {
            # 技术术语统一翻译
            "Transformer": {"translation": "Transformer", "variants": ["变换器", "转换器"]},
            "Attention": {"translation": "注意力机制", "variants": ["注意机制"]},
            "Token": {"translation": "Token", "variants": ["词元", "标记"]},
            "Embedding": {"translation": "Embedding", "variants": ["嵌入", "向量化"]},
            "Fine-tuning": {"translation": "微调", "variants": ["精调", "训练"]},
            "Inference": {"translation": "推理", "variants": ["推断", "预测"]},
            "Backpropagation": {"translation": "反向传播", "variants": ["BP", "反向传"]},
            "Hyperparameter": {"translation": "超参数", "variants": ["超参", "参数"]},
        }

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行一致性检查

        Args:
            state: 当前工作流状态，包含longform_article

        Returns:
            更新后的状态，包含consistency_check_result
        """
        self.log("开始一致性检查...")

        try:
            # 获取长文本文章
            article = state.get("longform_article", {})
            content = article.get("full_content", "")
            title = article.get("title", "")

            if not content:
                self.log("未找到文章内容，跳过一致性检查", "WARNING")
                return {**state, "consistency_check_result": None}

            issues = []
            suggestions = []
            terminology_dict = {}

            # 1. 术语一致性检查
            if self.check_terminology:
                term_result = self._check_terminology_consistency(content)
                issues.extend(term_result["issues"])
                suggestions.extend(term_result["suggestions"])
                terminology_dict = term_result["terminology_dict"]

            # 2. 引用完整性检查
            if self.check_citations:
                citation_result = self._check_citation_consistency(content)
                issues.extend(citation_result["issues"])
                suggestions.extend(citation_result["suggestions"])

            # 3. 数据一致性检查
            if self.check_data:
                data_result = self._check_data_consistency(content)
                issues.extend(data_result["issues"])
                suggestions.extend(data_result["suggestions"])

            # 4. 格式一致性检查
            format_result = self._check_format_consistency(content)
            issues.extend(format_result["issues"])
            suggestions.extend(format_result["suggestions"])

            # 计算一致性分数
            consistency_score = self._calculate_consistency_score(issues, content)

            # 生成术语表
            glossary = self._generate_glossary(content, terminology_dict)

            # 构建检查结果
            result = {
                "score": consistency_score,
                "total_issues": len(issues),
                "issues": issues,
                "suggestions": suggestions[:20],  # 限制建议数量
                "glossary": glossary,
                "meets_threshold": consistency_score >= 8.0,
                "summary": self._generate_summary(consistency_score, len(issues), len(glossary))
            }

            self.log(f"一致性检查完成，分数: {consistency_score:.1f}/10，问题: {len(issues)}")

            return {**state, "consistency_check_result": result}

        except Exception as e:
            self.log(f"一致性检查失败: {e}", "ERROR")
            return {
                **state,
                "consistency_check_result": {
                    "score": 7.0,
                    "error": str(e)
                }
            }

    def _check_terminology_consistency(self, content: str) -> Dict[str, Any]:
        """检查术语翻译一致性"""
        issues = []
        suggestions = []
        terminology_dict = defaultdict(lambda: {"count": 0, "locations": []})

        # 提取所有可能的技术术语（大写开头的英文单词）
        technical_terms = re.findall(r'\b[A-Z][a-zA-Z]+\b', content)

        # 统计每个术语的出现次数和位置
        for term in technical_terms:
            if term in self.terminology_rules:
                terminology_dict[term]["count"] += 1

        # 检查是否有不一致的翻译
        for term, rule in self.terminology_rules.items():
            expected_translation = rule["translation"]
            variants = rule["variants"]

            for variant in variants:
                if variant in content:
                    issues.append({
                        "type": "术语不一致",
                        "severity": "medium",
                        "description": f"发现术语'{term}'的不一致翻译: '{variant}'",
                        "location": self._find_term_location(content, variant),
                        "suggestion": f"统一使用'{expected_translation}'"
                    })
                    suggestions.append(f"[术语] 将'{variant}'统一为'{expected_translation}'")

        # 检查中英文混用
        for term in terminology_dict:
            count = terminology_dict[term]["count"]
            if count > 0:
                # 检查是否有英文但无中文翻译
                has_translation = self.terminology_rules.get(term, {}).get("translation", "")
                if has_translation and has_translation != term:
                    # 统计中英文使用比例
                    chinese_count = content.count(has_translation)
                    english_count = count

                    if chinese_count == 0 and english_count > 2:
                        suggestions.append(f"[术语] '{term}'首次出现时建议添加中文翻译'{has_translation}'")

        return {
            "issues": issues,
            "suggestions": suggestions,
            "terminology_dict": dict(terminology_dict)
        }

    def _check_citation_consistency(self, content: str) -> Dict[str, Any]:
        """检查引用完整性"""
        issues = []
        suggestions = []

        # 检查章节引用
        chapter_refs = re.findall(r'第[一二三四五六七八九十\d]+章|章节\s+\d+|参见\s+第.+章', content)
        defined_chapters = re.findall(r'^#{1,3}\s+(第.+章|.+章$)', content, re.MULTILINE)

        for ref in chapter_refs:
            # 简化检查：只检查是否在定义的章节中
            found = False
            for defined in defined_chapters:
                if defined[1] and defined[1] in ref:
                    found = True
                    break

            if not found and "章" in ref:
                issues.append({
                    "type": "引用不完整",
                    "severity": "low",
                    "description": f"章节引用'{ref}'可能不存在",
                    "suggestion": "检查章节编号是否正确"
                })

        # 检查文献引用格式
        citation_patterns = [
            r'\[\d+\]',  # [1], [2]
            r'\[.+?\]',  # [Author, 2023]
            r'\([^)]+\d{4}\)'  # (Author, 2023)
        ]

        has_citations = any(re.search(pattern, content) for pattern in citation_patterns)

        if not has_citations:
            suggestions.append("[引用] 建议添加文献引用，增强文章权威性")
        else:
            # 检查是否有参考文献列表
            has_reference_list = bool(re.search(r'##?\s*参考文献|References|引用', content, re.IGNORECASE))
            if not has_reference_list:
                issues.append({
                    "type": "引用不完整",
                    "severity": "medium",
                    "description": "文章中包含引用但没有参考文献列表",
                    "suggestion": "在文末添加参考文献列表"
                })
                suggestions.append("[引用] 建议添加参考文献列表")

        return {
            "issues": issues,
            "suggestions": suggestions
        }

    def _check_data_consistency(self, content: str) -> Dict[str, Any]:
        """检查数据一致性"""
        issues = []
        suggestions = []

        # 提取所有数字和百分比
        numbers = re.findall(r'\d+\.?\d*\s*(?:%|倍|倍速|万|千|亿|GB|MB|KB|ms|s|分钟|小时|天)', content)

        # 检查是否有矛盾的声明（简化版）
        # 例如：既说"提升3倍"又说"降低50%"
        increase_matches = re.findall(r'(?:提升|增加|增长|提高)\s*(\d+\.?\d*)\s*(?:倍|%)', content)
        decrease_matches = re.findall(r'(?:降低|减少|下降|缩短)\s*(\d+\.?\d*)\s*(?:倍|%)', content)

        if len(increase_matches) > 0 and len(decrease_matches) > 0:
            # 检查是否有矛盾的描述（同一指标既增又减）
            # 这里做简化检查，实际需要更复杂的语义分析
            pass

        # 检查版本号一致性
        version_patterns = [
            r'v?\d+\.\d+(?:\.\d+)?',
            r'Python\s*\d+\.\d+',
            r'GLM-\d+(?:\.\d+)?',
            r'GPT-\d+(?:\.\d+)?'
        ]

        versions_found = []
        for pattern in version_patterns:
            matches = re.findall(pattern, content)
            versions_found.extend(matches)

        # 检查是否有版本号格式不一致
        if versions_found:
            # 提取版本格式
            version_formats = set()
            for v in versions_found:
                if '.' in v:
                    parts = v.count('.')
                    version_formats.add(parts)

            if len(version_formats) > 1:
                suggestions.append("[数据] 建议统一版本号格式（如：v1.0.0）")

        # 检查数据单位一致性
        unit_patterns = {
            '内存': ['GB', 'MB', 'KB'],
            '时间': ['ms', 's', '分钟', '小时'],
            '数量': ['万', '千', '百万', '亿']
        }

        for category, units in unit_patterns.items():
            used_units = []
            for unit in units:
                if unit in content:
                    used_units.append(unit)

            if len(used_units) > 1:
                # 检查是否混用不同单位
                suggestions.append(f"[数据] {category}单位混用: {', '.join(used_units)}，建议统一")

        return {
            "issues": issues,
            "suggestions": suggestions
        }

    def _check_format_consistency(self, content: str) -> Dict[str, Any]:
        """检查格式一致性"""
        issues = []
        suggestions = []

        # 检查标题层级一致性
        headings = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)
        if headings:
            # 检查是否跳跃层级
            prev_level = 0
            for level, title in headings:
                current_level = len(level)
                if current_level > prev_level + 1:
                    issues.append({
                        "type": "格式不一致",
                        "severity": "low",
                        "description": f"标题层级跳跃: 从H{prev_level}到H{current_level}",
                        "location": title,
                        "suggestion": "标题层级应逐级递增"
                    })
                prev_level = current_level

        # 检查列表格式一致性
        bullet_styles = []
        if re.search(r'^\s*-\s+', content, re.MULTILINE):
            bullet_styles.append("-")
        if re.search(r'^\s*\*\s+', content, re.MULTILINE):
            bullet_styles.append("*")
        if re.search(r'^\s*\+\s+', content, re.MULTILINE):
            bullet_styles.append("+")

        if len(bullet_styles) > 1:
            suggestions.append(f"[格式] 列表符号混用: {', '.join(bullet_styles)}，建议统一")

        # 检查代码块语言标识
        code_blocks = re.findall(r'```(\w*)', content)
        if code_blocks:
            # 检查是否有未标识语言的代码块
            unnamed_blocks = code_blocks.count("")
            if unnamed_blocks > 0:
                suggestions.append(f"[格式] {unnamed_blocks}个代码块缺少语言标识，建议添加")

        return {
            "issues": issues,
            "suggestions": suggestions
        }

    def _calculate_consistency_score(self, issues: List[Dict[str, Any]], content: str) -> float:
        """计算一致性分数（0-10分）"""
        base_score = 10.0

        # 根据问题严重程度扣分
        for issue in issues:
            severity = issue.get("severity", "low")
            if severity == "high":
                base_score -= 0.5
            elif severity == "medium":
                base_score -= 0.3
            else:
                base_score -= 0.1

        return max(0.0, min(10.0, base_score))

    def _generate_glossary(self, content: str, terminology_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成术语表"""
        glossary = []

        # 从术语规则中提取
        for term, rule in self.terminology_rules.items():
            if term in terminology_dict:
                glossary.append({
                    "term": term,
                    "translation": rule["translation"],
                    "count": terminology_dict[term]["count"]
                })

        # 按出现次数排序
        glossary.sort(key=lambda x: x["count"], reverse=True)

        return glossary[:20]  # 返回前20个术语

    def _find_term_location(self, content: str, term: str) -> str:
        """查找术语在文章中的位置"""
        lines = content.split('\n')
        for idx, line in enumerate(lines):
            if term in line:
                # 返回上下文
                context_start = max(0, idx - 1)
                context_end = min(len(lines), idx + 2)
                context_lines = lines[context_start:context_end]
                return "...\n".join(context_lines)[:100] + "..."
        return "未找到具体位置"

    def _generate_summary(self, score: float, issue_count: int, glossary_count: int) -> str:
        """生成一致性检查总结"""
        parts = [
            f"一致性分数: {score:.1f}/10",
            f"发现问题: {issue_count} 个",
            f"技术术语: {glossary_count} 个"
        ]

        if score >= 9.0:
            parts.append("✅ 一致性优秀")
        elif score >= 7.0:
            parts.append("⚠️ 一致性良好，有改进空间")
        else:
            parts.append("❌ 一致性需要改进")

        return " | ".join(parts)
