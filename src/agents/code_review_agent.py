"""
代码审查Agent - 检查文章中代码示例的正确性和最佳实践
使用纯Python实现（ast模块），无需额外API成本
"""

import re
import ast
from typing import Dict, Any, List, Optional
from src.agents.base import BaseAgent


class CodeReviewAgent(BaseAgent):
    """代码审查Agent - 验证代码示例的语法正确性和最佳实践"""

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        super().__init__(config, prompts)
        self.supported_languages = ["python", "javascript", "typescript", "java", "go", "rust", "cpp", "c"]

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行代码审查

        Args:
            state: 当前工作流状态，包含longform_article

        Returns:
            更新后的状态，包含code_review_result
        """
        self.log("开始代码审查...")

        try:
            # 获取长文本文章
            article = state.get("longform_article", {})
            content = article.get("full_content", "")

            if not content:
                self.log("未找到文章内容，跳过代码审查", "WARNING")
                return {**state, "code_review_result": None}

            # 提取代码块
            code_blocks = self._extract_code_blocks(content)
            self.log(f"提取到 {len(code_blocks)} 个代码块")

            if not code_blocks:
                self.log("文章中无代码块，跳过审查", "INFO")
                return {
                    **state,
                    "code_review_result": {
                        "total_blocks": 0,
                        "valid_blocks": 0,
                        "reviews": [],
                        "score": 10.0,
                        "summary": "无需审查"
                    }
                }

            # 审查每个代码块
            review_results = []
            for idx, block in enumerate(code_blocks):
                review = self._review_code_block(block, idx)
                review_results.append(review)

            # 计算总体质量分数
            quality_score = self._calculate_quality_score(review_results)

            # 生成总结
            summary = self._generate_summary(review_results, quality_score)

            # 构建审查结果
            result = {
                "total_blocks": len(code_blocks),
                "valid_blocks": sum(1 for r in review_results if r.get("syntax_valid", False)),
                "score": quality_score,
                "reviews": review_results,
                "summary": summary,
                "improvements_needed": quality_score < 8.0
            }

            self.log(f"代码审查完成，质量分数: {quality_score:.1f}/10")

            return {**state, "code_review_result": result}

        except Exception as e:
            self.log(f"代码审查失败: {e}", "ERROR")
            # 失败时返回空结果，不影响后续流程
            return {
                **state,
                "code_review_result": {
                    "total_blocks": 0,
                    "error": str(e),
                    "score": 5.0  # 中等分数
                }
            }

    def _extract_code_blocks(self, content: str) -> List[Dict[str, Any]]:
        """从Markdown内容中提取代码块"""
        code_blocks = []

        # 匹配Markdown代码块 ```language ... ```
        pattern = r'```(\w*)\n(.*?)```'
        matches = re.findall(pattern, content, re.DOTALL)

        for language, code in matches:
            language = language.lower().strip()
            if not language or language not in self.supported_languages:
                language = "text"  # 未知语言

            code_blocks.append({
                "language": language,
                "code": code.strip(),
                "line_count": len(code.split('\n'))
            })

        return code_blocks

    def _review_code_block(self, block: Dict[str, Any], index: int) -> Dict[str, Any]:
        """审查单个代码块"""
        language = block.get("language", "")
        code = block.get("code", "")

        review = {
            "index": index,
            "language": language,
            "code_preview": code[:100] + "..." if len(code) > 100 else code,
            "line_count": block.get("line_count", 0),
            "syntax_valid": False,
            "has_errors": False,
            "issues": [],
            "suggestions": [],
            "improved_version": None
        }

        # 根据语言选择审查方法
        if language == "python":
            review.update(self._review_python_code(code))
        elif language in ["javascript", "typescript", "java", "go", "rust", "cpp", "c"]:
            review.update(self._review_generic_code(code, language))
        else:
            # 不支持的语言，跳过语法检查
            review["syntax_valid"] = True
            review["suggestions"].append("代码语言不支持自动审查，建议人工验证")

        # 通用检查
        review.update(self._check_best_practices(code, language))

        return review

    def _review_python_code(self, code: str) -> Dict[str, Any]:
        """审查Python代码"""
        result = {
            "syntax_valid": False,
            "has_errors": False,
            "issues": [],
            "suggestions": []
        }

        try:
            # 尝试解析AST
            tree = ast.parse(code)
            result["syntax_valid"] = True

            # 检查代码质量
            issues = []
            suggestions = []

            # 检查是否有导入
            has_imports = any(isinstance(node, (ast.Import, ast.ImportFrom))
                            for node in ast.walk(tree))
            if not has_imports and len(code.split('\n')) > 5:
                suggestions.append("建议添加必要的import语句")

            # 检查是否有函数定义
            has_functions = any(isinstance(node, ast.FunctionDef)
                              for node in ast.walk(tree))
            if not has_functions and len(code.split('\n')) > 10:
                suggestions.append("建议将代码封装为函数，提高可重用性")

            # 检查是否有注释
            has_comments = '#' in code
            if not has_comments and len(code.split('\n')) > 5:
                suggestions.append("建议添加代码注释，提高可读性")

            # 检查是否有文档字符串
            if has_functions:
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if not ast.get_docstring(node):
                            suggestions.append(f"函数 '{node.name}' 缺少文档字符串")

            result["issues"] = issues
            result["suggestions"] = suggestions

        except SyntaxError as e:
            result["syntax_valid"] = False
            result["has_errors"] = True
            result["issues"].append(f"语法错误: {e.msg} (行 {e.lineno})")

        except Exception as e:
            result["issues"].append(f"分析错误: {str(e)}")

        return result

    def _review_generic_code(self, code: str, language: str) -> Dict[str, Any]:
        """审查其他语言代码（基础检查）"""
        result = {
            "syntax_valid": True,  # 假设语法正确（无法深度解析）
            "has_errors": False,
            "issues": [],
            "suggestions": []
        }

        # 基础检查
        suggestions = []

        # 检查是否有注释
        comment_patterns = {
            "javascript": ["//", "/*"],
            "typescript": ["//", "/*"],
            "java": ["//", "/*"],
            "go": ["//", "/*"],
            "rust": ["//", "/*"],
            "cpp": ["//", "/*"],
            "c": ["//", "/*"]
        }

        comment_markers = comment_patterns.get(language, [])
        has_comments = any(marker in code for marker in comment_markers)

        if not has_comments and len(code.split('\n')) > 5:
            suggestions.append(f"建议添加{language}注释，提高代码可读性")

        # 检查是否有函数/方法定义
        function_patterns = {
            "javascript": [r"function\s+\w+", r"=>\s*{"],
            "typescript": [r"function\s+\w+", r"=>\s*{", r"\w+\s*\([^)]*\)\s*:"],
            "java": [r"(public|private|protected)?\s*(static)?\s*\w+\s+\w+\s*\("],
            "go": [r"func\s+\w+"],
            "rust": [r"fn\s+\w+"],
            "cpp": [r"\w+\s+\w+\s*\(", r"auto\s+\w+"],
            "c": [r"\w+\s+\w+\s*\("]
        }

        patterns = function_patterns.get(language, [])
        has_functions = any(re.search(pattern, code) for pattern in patterns)

        if not has_functions and len(code.split('\n')) > 10:
            suggestions.append("建议将代码封装为函数/方法，提高可重用性")

        result["suggestions"] = suggestions

        return result

    def _check_best_practices(self, code: str, language: str) -> Dict[str, Any]:
        """检查代码最佳实践"""
        issues = []
        suggestions = []

        # 通用最佳实践检查

        # 1. 检查是否有硬编码的敏感信息
        sensitive_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']'
        ]

        for pattern in sensitive_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                issues.append("警告：代码中可能包含硬编码的敏感信息（密码、密钥等）")

        # 2. 检查异常处理
        if language == "python":
            if "try:" not in code and ("open(" in code or "request" in code):
                suggestions.append("建议添加异常处理（try-except）")

        # 3. 检查资源释放
        if language == "python":
            if "open(" in code and "with open(" not in code:
                suggestions.append("建议使用 'with open()' 进行文件操作，自动管理资源")

        # 4. 检查代码复杂度
        line_count = len(code.split('\n'))
        if line_count > 50:
            suggestions.append(f"代码块较长（{line_count}行），建议拆分为多个函数")

        # 5. 检查是否有TODO/FIXME标记
        if "TODO" in code or "FIXME" in code:
            suggestions.append("代码中包含TODO/FIXME标记，建议完善")

        return {
            "best_practice_issues": issues,
            "best_practice_suggestions": suggestions
        }

    def _calculate_quality_score(self, reviews: List[Dict[str, Any]]) -> float:
        """计算代码质量总分（0-10分）"""
        if not reviews:
            return 10.0

        total_score = 0.0

        for review in reviews:
            block_score = 10.0

            # 语法正确性权重：40%
            if review.get("syntax_valid"):
                block_score *= 1.0
            else:
                block_score *= 0.6  # 语法错误扣40%

            # 最佳实践问题权重：30%
            issues_count = len(review.get("issues", [])) + len(review.get("best_practice_issues", []))
            block_score -= min(issues_count * 0.5, 3.0)

            # 改进建议权重：30%
            suggestions_count = len(review.get("suggestions", [])) + len(review.get("best_practice_suggestions", []))
            block_score -= min(suggestions_count * 0.3, 3.0)

            # 确保分数在0-10之间
            block_score = max(0.0, min(10.0, block_score))
            total_score += block_score

        return total_score / len(reviews)

    def _generate_summary(self, reviews: List[Dict[str, Any]], score: float) -> str:
        """生成代码审查总结"""
        if not reviews:
            return "无需审查"

        total_blocks = len(reviews)
        valid_blocks = sum(1 for r in reviews if r.get("syntax_valid", False))
        total_issues = sum(len(r.get("issues", [])) + len(r.get("best_practice_issues", []))
                         for r in reviews)
        total_suggestions = sum(len(r.get("suggestions", [])) + len(r.get("best_practice_suggestions", []))
                              for r in reviews)

        summary_parts = [
            f"共审查 {total_blocks} 个代码块",
            f"语法正确: {valid_blocks}/{total_blocks}",
            f"发现问题: {total_issues} 个",
            f"改进建议: {total_suggestions} 条"
        ]

        if score >= 8.0:
            summary_parts.append("✅ 代码质量优秀")
        elif score >= 6.0:
            summary_parts.append("⚠️ 代码质量良好，有改进空间")
        else:
            summary_parts.append("❌ 代码质量需要改进")

        return " | ".join(summary_parts)
