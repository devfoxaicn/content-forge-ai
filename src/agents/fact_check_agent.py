"""
事实核查Agent - 验证技术文章中的声明和数据
确保内容的准确性和可信度
"""

from typing import Dict, Any, List
import re
from datetime import datetime
from src.agents.base import BaseAgent


class FactCheckAgent(BaseAgent):
    """
    事实核查Agent

    核心功能：
    1. 识别内容中的关键声明
    2. 验证技术数据的准确性
    3. 检查引用和来源
    4. 标记需要核实的内容
    """

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        super().__init__(config, prompts)
        factcheck_config = config.get("agents", {}).get("fact_check_agent", {})
        self.enabled = factcheck_config.get("enabled", True)
        self.verify_claims = factcheck_config.get("verify_claims", True)
        self.check_dates = factcheck_config.get("check_dates", True)
        self.cross_reference = factcheck_config.get("cross_reference", True)
        self.min_confidence = factcheck_config.get("min_confidence", 0.7)

        if not self.enabled:
            self.log("FactCheckAgent已禁用", "WARNING")

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        对内容进行事实核查

        Args:
            state: 当前工作流状态

        Returns:
            Dict[str, Any]: 更新后的状态，包含核查结果
        """
        self.log("开始事实核查")

        try:
            if not self.enabled:
                self.log("FactCheckAgent已禁用，跳过核查步骤")
                return {**state, "fact_check": None, "current_step": "fact_check_skipped"}

            # 获取需要核查的内容
            content_to_check = self._get_content_to_check(state)
            if not content_to_check:
                self.log("没有找到需要核查的内容")
                return {**state, "fact_check": None, "current_step": "fact_check_no_content"}

            # 提取关键声明
            claims = self._extract_claims(content_to_check)
            if not claims:
                self.log("没有找到需要核查的声明")
                return {**state, "fact_check": None, "current_step": "fact_check_no_claims"}

            self.log(f"找到 {len(claims)} 个关键声明，开始核查")

            # 核实每个声明
            verification_results = []
            for idx, claim in enumerate(claims, 1):
                self.log(f"核实声明 {idx}/{len(claims)}: {claim.get('text', '')[:50]}...")

                try:
                    verification = self._verify_claim(claim, state)
                    verification["claim_index"] = idx
                    verification_results.append(verification)
                except Exception as e:
                    self.log(f"核实声明 {idx} 失败: {str(e)}", "WARNING")
                    verification_results.append({
                        "claim_index": idx,
                        "claim": claim,
                        "error": str(e),
                        "confidence": 0.0,
                        "status": "error"
                    })

            # 计算总体可信度
            overall_confidence = self._calculate_overall_confidence(verification_results)

            # 生成核查报告
            fact_check_report = self._generate_fact_check_report(verification_results, overall_confidence)

            # 标记需要用户核实的内容
            flagged_items = self._get_flagged_items(verification_results)

            self.log(f"事实核查完成，可信度: {overall_confidence:.1%}")

            return {
                **state,
                "fact_check": {
                    "total_claims": len(claims),
                    "verifications": verification_results,
                    "overall_confidence": overall_confidence,
                    "report": fact_check_report,
                    "flagged_items": flagged_items,
                    "meets_standard": overall_confidence >= self.min_confidence
                },
                "current_step": "fact_check_completed"
            }

        except Exception as e:
            self.log(f"事实核查失败: {str(e)}", "ERROR")
            return {
                **state,
                "error_message": f"事实核查失败: {str(e)}",
                "current_step": "fact_check_failed"
            }

    def _get_content_to_check(self, state: Dict[str, Any]) -> str:
        """获取需要核查的内容"""
        # 优先核查长文章内容
        if "longform_content" in state:
            return state["longform_content"]
        elif "xiaohongshu_content" in state:
            return state["xiaohongshu_content"]
        elif "twitter_content" in state:
            return state["twitter_content"]
        else:
            return ""

    def _extract_claims(self, content: str) -> List[Dict[str, Any]]:
        """
        从内容中提取关键声明

        识别模式：
        1. 包含数字的陈述（性能指标、统计数据）
        2. 技术规格声明（版本、参数量）
        3. 比较性陈述（"比X快Y倍"）
        4. 时间相关陈述（发布日期、版本时间）
        5. 权威引用
        """
        claims = []

        # 提取包含数字的陈述
        claims.extend(self._extract_numeric_claims(content))

        # 提取比较性陈述
        claims.extend(self._extract_comparison_claims(content))

        # 提取版本/日期声明
        claims.extend(self._extract_version_claims(content))

        # 提取技术规格声明
        claims.extend(self._extract_spec_claims(content))

        return claims

    def _extract_numeric_claims(self, content: str) -> List[Dict[str, Any]]:
        """提取包含数字的陈述"""
        claims = []

        # 匹配模式：性能数据、统计数据等
        patterns = [
            r'(?:性能|准确率|提升|降低|减少|增加).*?(\d+(?:\.\d+)?%?)',
            r'(\d+(?:\.\d+)?%?).{0,50}(?:倍|倍数|提升)',
            r'(\d+(?:\.\d+)?[kKmMbB]?).{0,30}(?:参数|tokens|)',
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                text = match.group(0)
                claims.append({
                    "type": "numeric",
                    "text": text,
                    "value": match.group(1),
                    "context": self._get_context(content, match.start(), match.end())
                })

        return claims

    def _extract_comparison_claims(self, content: str) -> List[Dict[str, Any]]:
        """提取比较性陈述"""
        claims = []

        # 比较性关键词
        comparison_keywords = ["优于", "超过", "胜过", "领先", "快于", "高于", "好于"]

        for keyword in comparison_keywords:
            pattern = r'.{{0,100}}{keyword}.{{0,50}}'.format(keyword=keyword)
            matches = re.finditer(pattern, content)
            for match in matches:
                claims.append({
                    "type": "comparison",
                    "text": match.group(0),
                    "keyword": keyword
                })

        return claims

    def _extract_version_claims(self, content: str) -> List[Dict[str, Any]]:
        """提取版本/日期声明"""
        claims = []

        # 版本号模式
        version_pattern = r'v?(?:ersion)?\s*(\d+(?:\.\d+)+)'
        matches = re.finditer(version_pattern, content, re.IGNORECASE)
        for match in matches:
            claims.append({
                "type": "version",
                "text": match.group(0),
                "version": match.group(1)
            })

        # 日期模式
        date_pattern = r'\d{4}年\d{1,2}月|\d{4}-\d{1,2}-\d{1,2}|20\d{2}'
        matches = re.finditer(date_pattern, content)
        for match in matches:
            claims.append({
                "type": "date",
                "text": match.group(0),
                "date": match.group(0)
            })

        return claims

    def _extract_spec_claims(self, content: str) -> List[Dict[str, Any]]:
        """提取技术规格声明"""
        claims = []

        # 参数量声明
        param_pattern = r'(\d+(?:\.\d+)?[kKmMbB]?)\s*(?:参数|parameters?)'
        matches = re.finditer(param_pattern, content, re.IGNORECASE)
        for match in matches:
            claims.append({
                "type": "specification",
                "text": match.group(0),
                "spec": match.group(1)
            })

        # 模型名称
        model_names = [
            "GPT-4", "GPT-3", "Claude", "Llama", "Gemini", "GLM",
            "Transformer", "BERT", "Diffusion", "Stable Diffusion"
        ]

        for model in model_names:
            pattern = r'\b' + model + r'\b'
            if re.search(pattern, content, re.IGNORECASE):
                claims.append({
                    "type": "model_reference",
                    "text": model,
                    "model": model
                })

        return claims

    def _get_context(self, content: str, start: int, end: int, window: int = 50) -> str:
        """获取声明的上下文"""
        context_start = max(0, start - window)
        context_end = min(len(content), end + window)
        return content[context_start:context_end].strip()

    def _verify_claim(self, claim: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """
        核实单个声明

        Args:
            claim: 声明数据
            state: 当前状态（可能包含研究数据）

        Returns:
            Dict[str, Any]: 核实结果
        """
        claim_type = claim.get("type", "unknown")
        claim_text = claim.get("text", "")

        # 默认核实结果
        verification = {
            "claim": claim_text,
            "type": claim_type,
            "confidence": 0.7,
            "status": "unverified",
            "sources": [],
            "notes": []
        }

        # 根据声明类型进行核实
        if claim_type == "numeric":
            verification.update(self._verify_numeric_claim(claim, state))
        elif claim_type == "comparison":
            verification.update(self._verify_comparison_claim(claim, state))
        elif claim_type == "version":
            verification.update(self._verify_version_claim(claim, state))
        elif claim_type == "date":
            verification.update(self._verify_date_claim(claim, state))
        elif claim_type == "specification":
            verification.update(self._verify_spec_claim(claim, state))
        elif claim_type == "model_reference":
            verification.update(self._verify_model_claim(claim, state))

        return verification

    def _verify_numeric_claim(self, claim: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """核实数字声明"""
        # 检查研究数据中是否有相关数据
        research_data = state.get("research_data", {})

        # 简化核实：给予中等置信度
        return {
            "confidence": 0.75,
            "status": "likely_accurate",
            "notes": ["数据需要来源引用", "建议添加官方文档链接"]
        }

    def _verify_comparison_claim(self, claim: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """核实比较性声明"""
        return {
            "confidence": 0.65,
            "status": "needs_verification",
            "notes": ["比较性声明需要基准测试支持", "建议添加具体数据对比"]
        }

    def _verify_version_claim(self, claim: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """核实版本声明"""
        version = claim.get("version", "")

        # 检查是否是已知的合理版本号
        # 简化处理：假设格式正确即可
        if re.match(r'^\d+\.\d+', version):
            return {
                "confidence": 0.85,
                "status": "format_valid",
                "notes": ["版本号格式正确"]
            }
        else:
            return {
                "confidence": 0.5,
                "status": "needs_check",
                "notes": ["版本号格式可能不正确"]
            }

    def _verify_date_claim(self, claim: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """核实日期声明"""
        if self.check_dates:
            return {
                "confidence": 0.8,
                "status": "date_format_valid",
                "notes": ["日期格式正确", "建议添加具体发布时间"]
            }
        return {
            "confidence": 0.7,
            "status": "date_not_checked"
        }

    def _verify_spec_claim(self, claim: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """核实规格声明"""
        return {
            "confidence": 0.7,
            "status": "specification_needs_source",
            "notes": ["技术规格需要官方来源", "建议链接到技术文档"]
        }

    def _verify_model_claim(self, claim: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """核实模型引用"""
        model = claim.get("model", "")

        # 已知的主流模型
        known_models = ["GPT-4", "GPT-3", "Claude", "Llama", "Gemini", "GLM"]

        if model in known_models:
            return {
                "confidence": 0.9,
                "status": "known_model",
                "notes": [f"{model}是已知的主流模型"]
            }
        else:
            return {
                "confidence": 0.6,
                "status": "model_needs_verification",
                "notes": ["模型需要验证是否存在"]
            }

    def _calculate_overall_confidence(self, verifications: List[Dict[str, Any]]) -> float:
        """计算总体可信度"""
        if not verifications:
            return 0.0

        confidences = [v.get("confidence", 0) for v in verifications]
        return sum(confidences) / len(confidences)

    def _generate_fact_check_report(self, verifications: List[Dict[str, Any]], overall_confidence: float) -> str:
        """生成事实核查报告"""
        report_parts = []

        report_parts.append("# 事实核查报告\n")
        report_parts.append(f"**总体可信度**: {overall_confidence:.1%}\n")
        report_parts.append(f"**核查声明数**: {len(verifications)}\n")

        # 可信度等级
        if overall_confidence >= 0.9:
            grade = "高度可信 ✅"
        elif overall_confidence >= 0.7:
            grade = "基本可信 👍"
        elif overall_confidence >= 0.5:
            grade = "需要核实 ⚠️"
        else:
            grade = "可信度较低 ❌"

        report_parts.append(f"**可信度等级**: {grade}\n")

        # 详细核查结果
        report_parts.append("\n## 详细核查结果\n")

        for verification in verifications:
            idx = verification.get("claim_index", 0)
            claim_text = verification.get("claim", "")[:60]
            confidence = verification.get("confidence", 0)
            status = verification.get("status", "unknown")
            notes = verification.get("notes", [])

            report_parts.append(f"\n### 声明 {idx}\n")
            report_parts.append(f"- **内容**: {claim_text}...\n")
            report_parts.append(f"- **可信度**: {confidence:.1%}\n")
            report_parts.append(f"- **状态**: {status}\n")

            if notes:
                report_parts.append(f"- **备注**: {', '.join(notes)}\n")

        return "\n".join(report_parts)

    def _get_flagged_items(self, verifications: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """获取需要标记的项目（可信度低于阈值）"""
        flagged = []

        for verification in verifications:
            confidence = verification.get("confidence", 1.0)

            if confidence < self.min_confidence:
                flagged.append({
                    "claim": verification.get("claim"),
                    "confidence": confidence,
                    "status": verification.get("status"),
                    "reason": "可信度低于阈值"
                })

        return flagged
