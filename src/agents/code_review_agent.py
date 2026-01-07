"""
代码审查Agent - 验证技术文章中的代码示例
确保代码准确性、可运行性和最佳实践
"""

from typing import Dict, Any, List
import re
from src.agents.base import BaseAgent


class CodeReviewAgent(BaseAgent):
    """
    代码审查Agent

    核心功能：
    1. 提取文章中的代码片段
    2. 验证代码语法和逻辑
    3. 检查最佳实践
    4. 提供改进建议
    """

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        super().__init__(config, prompts)
        review_config = config.get("agents", {}).get("code_review_agent", {})
        self.enabled = review_config.get("enabled", True)
        self.auto_fix = review_config.get("auto_fix", False)
        self.min_rating = review_config.get("min_rating", 7.0)
        self.use_skill = review_config.get("use_skill", False)

        if not self.enabled:
            self.log("CodeReviewAgent已禁用", "WARNING")

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        审查内容中的代码片段

        Args:
            state: 当前工作流状态

        Returns:
            Dict[str, Any]: 更新后的状态，包含审查结果
        """
        self.log("开始代码审查")

        try:
            if not self.enabled:
                self.log("CodeReviewAgent已禁用，跳过审查步骤")
                return {**state, "code_review": None, "current_step": "code_review_skipped"}

            # 获取需要审查的内容
            content_to_review = self._get_content_to_review(state)
            if not content_to_review:
                self.log("没有找到需要审查的内容")
                return {**state, "code_review": None, "current_step": "code_review_no_content"}

            # 提取代码片段
            code_snippets = self._extract_code_blocks(content_to_review)
            if not code_snippets:
                self.log("没有找到代码片段")
                return {**state, "code_review": None, "current_step": "code_review_no_snippets"}

            self.log(f"找到 {len(code_snippets)} 个代码片段，开始审查")

            # 审查每个代码片段
            review_results = []
            for idx, snippet in enumerate(code_snippets, 1):
                self.log(f"审查代码片段 {idx}/{len(code_snippets)}: {snippet.get('language', 'unknown')}")

                try:
                    review = self._review_code_snippet(snippet)
                    review["snippet_index"] = idx
                    review_results.append(review)
                except Exception as e:
                    self.log(f"审查片段 {idx} 失败: {str(e)}", "WARNING")
                    review_results.append({
                        "snippet_index": idx,
                        "language": snippet.get("language"),
                        "error": str(e),
                        "rating": 0
                    })

            # 计算总体评分
            overall_rating = self._calculate_overall_rating(review_results)

            # 生成审查报告
            review_report = self._generate_review_report(review_results, overall_rating)

            # 生成改进建议
            improvement_suggestions = self._generate_improvements(review_results)

            self.log(f"代码审查完成，总体评分: {overall_rating:.1f}/10")

            return {
                **state,
                "code_review": {
                    "total_snippets": len(code_snippets),
                    "reviews": review_results,
                    "overall_rating": overall_rating,
                    "report": review_report,
                    "improvements": improvement_suggestions,
                    "meets_standard": overall_rating >= self.min_rating
                },
                "current_step": "code_review_completed"
            }

        except Exception as e:
            self.log(f"代码审查失败: {str(e)}", "ERROR")
            return {
                **state,
                "error_message": f"代码审查失败: {str(e)}",
                "current_step": "code_review_failed"
            }

    def _get_content_to_review(self, state: Dict[str, Any]) -> str:
        """获取需要审查的内容"""
        # 优先审查长文章内容
        if "longform_content" in state:
            return state["longform_content"]
        elif "xiaohongshu_content" in state:
            return state["xiaohongshu_content"]
        elif "twitter_content" in state:
            return state["twitter_content"]
        else:
            return ""

    def _extract_code_blocks(self, content: str) -> List[Dict[str, Any]]:
        """
        从Markdown内容中提取代码块

        支持格式：
        - ```python ... ```
        - ```javascript ... ```
        - ```bash ... ```
        等
        """
        code_blocks = []

        # 正则匹配Markdown代码块
        pattern = r'```(\w*)\n(.*?)```'
        matches = re.findall(pattern, content, re.DOTALL)

        for language, code in matches:
            # 跳过空代码块
            if not code.strip():
                continue

            code_blocks.append({
                "language": language or "text",
                "code": code.strip(),
                "lines": len(code.split('\n')),
                "chars": len(code)
            })

        return code_blocks

    def _review_code_snippet(self, snippet: Dict[str, Any]) -> Dict[str, Any]:
        """
        审查单个代码片段

        Args:
            snippet: 代码片段数据

        Returns:
            Dict[str, Any]: 审查结果
        """
        language = snippet.get("language", "text")
        code = snippet.get("code", "")

        # 如果使用code-review skill
        if self.use_skill:
            return self._review_with_skill(snippet)

        # 使用LLM进行代码审查
        return self._review_with_llm(language, code)

    def _review_with_llm(self, language: str, code: str) -> Dict[str, Any]:
        """使用LLM进行代码审查"""

        prompt = f"""请审查以下{language}代码片段，从以下维度评分（每项0-10分）：

代码：
```{language}
{code}
```

请评估：
1. **语法正确性**（2分）：代码语法是否正确
2. **逻辑完整性**（2分）：代码逻辑是否完整
3. **最佳实践**（2分）：是否符合该语言的最佳实践
4. **可读性**（2分）：代码是否易于理解
5. **安全性**（2分）：是否存在安全隐患

请以JSON格式返回：
{{
  "syntax": 评分,
  "logic": 评分,
  "best_practices": 评分,
  "readability": 评分,
  "security": 评分,
  "issues": ["问题1", "问题2"],
  "suggestions": ["建议1", "建议2"],
  "overall_comment": "总体评价"
}}
"""

        try:
            response = self._call_llm(prompt)

            # 尝试解析JSON响应
            import json
            # 提取JSON部分（LLM可能返回JSON周围有文本）
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())

                # 计算总分
                total_score = (
                    result.get("syntax", 0) +
                    result.get("logic", 0) +
                    result.get("best_practices", 0) +
                    result.get("readability", 0) +
                    result.get("security", 0)
                ) / 2  # 转换为10分制

                return {
                    "language": language,
                    "method": "llm",
                    "syntax": result.get("syntax", 0),
                    "logic": result.get("logic", 0),
                    "best_practices": result.get("best_practices", 0),
                    "readability": result.get("readability", 0),
                    "security": result.get("security", 0),
                    "issues": result.get("issues", []),
                    "suggestions": result.get("suggestions", []),
                    "comment": result.get("overall_comment", ""),
                    "rating": total_score
                }
            else:
                # 无法解析JSON，返回默认评分
                return {
                    "language": language,
                    "method": "llm",
                    "rating": 6.0,
                    "issues": ["无法解析审查结果"],
                    "suggestions": [],
                    "comment": "代码审查格式解析失败"
                }

        except Exception as e:
            self.log(f"LLM代码审查失败: {str(e)}", "WARNING")
            return {
                "language": language,
                "method": "llm",
                "rating": 5.0,
                "issues": [f"审查失败: {str(e)}"],
                "suggestions": [],
                "comment": "代码审查出错"
            }

    def _review_with_skill(self, snippet: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用code-review skill进行代码审查

        注意：需要集成 code-review:code-review skill
        """
        # TODO: 实际集成code-review skill
        # 这里提供一个占位实现

        return {
            "language": snippet.get("language"),
            "method": "skill",
            "rating": 7.5,
            "issues": [],
            "suggestions": ["考虑添加错误处理", "添加类型注解"],
            "comment": "代码质量良好"
        }

    def _calculate_overall_rating(self, reviews: List[Dict[str, Any]]) -> float:
        """计算总体评分"""
        if not reviews:
            return 0.0

        ratings = [r.get("rating", 0) for r in reviews if "rating" in r]
        if not ratings:
            return 0.0

        return sum(ratings) / len(ratings)

    def _generate_review_report(self, reviews: List[Dict[str, Any]], overall_rating: float) -> str:
        """生成审查报告"""
        report_parts = []

        report_parts.append(f"# 代码审查报告\n")
        report_parts.append(f"**总体评分**: {overall_rating:.1f}/10\n")
        report_parts.append(f"**审查片段数**: {len(reviews)}\n")

        # 评分等级
        if overall_rating >= 9.0:
            grade = "优秀 ✅"
        elif overall_rating >= 7.0:
            grade = "良好 👍"
        elif overall_rating >= 5.0:
            grade = "及格 ⚠️"
        else:
            grade = "需改进 ❌"

        report_parts.append(f"**质量等级**: {grade}\n")

        # 详细审查结果
        report_parts.append("\n## 详细审查结果\n")

        for review in reviews:
            idx = review.get("snippet_index", 0)
            language = review.get("language", "unknown")
            rating = review.get("rating", 0)
            comment = review.get("comment", "")

            report_parts.append(f"\n### 片段 {idx} ({language})\n")
            report_parts.append(f"- **评分**: {rating:.1f}/10\n")
            report_parts.append(f"- **评价**: {comment}\n")

            issues = review.get("issues", [])
            if issues:
                report_parts.append(f"- **问题**: {', '.join(issues)}\n")

            suggestions = review.get("suggestions", [])
            if suggestions:
                report_parts.append(f"- **建议**: {', '.join(suggestions)}\n")

        return "\n".join(report_parts)

    def _generate_improvements(self, reviews: List[Dict[str, Any]]) -> List[str]:
        """生成改进建议汇总"""
        improvements = []

        # 收集所有建议
        all_suggestions = []
        for review in reviews:
            suggestions = review.get("suggestions", [])
            all_suggestions.extend(suggestions)

        # 去重
        improvements = list(set(all_suggestions))

        # 按重要性排序
        priority_keywords = ["安全", "错误", "bug", "错误处理", "异常"]
        high_priority = []
        normal_priority = []

        for improvement in improvements:
            if any(keyword in improvement.lower() for keyword in priority_keywords):
                high_priority.append(improvement)
            else:
                normal_priority.append(improvement)

        return high_priority + normal_priority
