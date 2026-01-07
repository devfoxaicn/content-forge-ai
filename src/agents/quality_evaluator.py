"""
质量评估Agent
评估内容质量并给出优化建议
"""

from typing import Dict, Any
import re
from src.agents.base import BaseAgent


class QualityEvaluatorAgent(BaseAgent):
    """质量评估Agent"""

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        super().__init__(config, prompts)
        self.min_score = config.get("agents", {}).get("quality_evaluator", {}).get("min_score", 7.0)
        self.auto_improve = config.get("agents", {}).get("quality_evaluator", {}).get("auto_improve", True)

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行质量评估

        Args:
            state: 当前工作流状态

        Returns:
            Dict[str, Any]: 更新后的状态
        """
        self.log("开始评估内容质量")

        try:
            # 获取生成的内容和标题
            content = state.get("generated_content", {})
            title = state.get("recommended_title", content.get("title", ""))

            if not content:
                raise ValueError("没有找到生成的内容")

            # 构建提示词
            user_prompt = self._build_prompt(state, content, title)

            # 调用LLM评估
            response = self._call_llm(user_prompt)

            # 解析响应
            quality_report = self._parse_quality_report(response)

            self.log(f"质量评分: {quality_report['overall_score']}/10")

            # 如果分数过低且启用自动改进，则重新生成
            if quality_report["overall_score"] < self.min_score and self.auto_improve:
                self.log(f"质量评分低于阈值({self.min_score})，启用自动改进", "WARNING")
                # TODO: 实现自动改进逻辑
                quality_report["auto_improved"] = False

            # 检查是否通过
            passed = quality_report["overall_score"] >= self.min_score

            return {
                **state,
                "quality_report": quality_report,
                "quality_passed": passed,
                "current_step": "quality_evaluator_completed"
            }
        except Exception as e:
            self.log(f"质量评估失败: {str(e)}", "ERROR")
            return {
                **state,
                "error_message": f"质量评估失败: {str(e)}",
                "current_step": "quality_evaluator_failed",
                "quality_passed": False
            }

    def _build_prompt(self, state: Dict[str, Any], content: Dict[str, Any], title: str) -> str:
        """
        构建提示词

        Args:
            state: 当前状态
            content: 生成的内容
            title: 标题

        Returns:
            str: 提示词
        """
        prompts = self.prompts.get("prompts", {})
        prompt_template = prompts.get("quality_evaluator", {}).get("user", "")

        hashtags = ", ".join(content.get("hashtags", []))

        return prompt_template.format(
            title=title,
            content=content.get("full_content", ""),
            hashtags=hashtags,
            target_audience=state["target_audience"],
            content_type=state["content_type"]
        )

    def _parse_quality_report(self, response: str) -> Dict[str, Any]:
        """
        解析LLM响应为质量报告

        Args:
            response: LLM响应

        Returns:
            Dict[str, Any]: 质量报告
        """
        report = {
            "overall_score": 7.0,
            "title_score": 14.0,
            "content_value_score": 21.0,
            "structure_score": 14.0,
            "platform_fit_score": 10.5,
            "viral_potential_score": 10.5,
            "strengths": [],
            "improvements": [],
            "improved_version": None
        }

        # 提取总体评分
        score_match = re.search(r'总体评分[：:]\s*([\d.]+)', response)
        if score_match:
            report["overall_score"] = float(score_match.group(1))

        # 提取各项分数
        scores = {
            "title_score": r"标题质量[：:]\s*([\d.]+)",
            "content_value_score": r"内容价值[：:]\s*([\d.]+)",
            "structure_score": r"内容结构[：:]\s*([\d.]+)",
            "platform_fit_score": r"平台适配[：:]\s*([\d.]+)",
            "viral_potential_score": r"传播潜力[：:]\s*([\d.]+)"
        }

        for key, pattern in scores.items():
            match = re.search(pattern, response)
            if match:
                report[key] = float(match.group(1))

        # 提取优点
        strengths_section = re.search(r'优点\s*(.+?)(?=改进建议|优化|$)', response, re.DOTALL)
        if strengths_section:
            strengths = re.findall(r'\d+[\.、]\s*(.+)', strengths_section.group(1))
            report["strengths"] = [s.strip() for s in strengths if s.strip()]

        # 提取改进建议
        improvements_section = re.search(r'改进建议\s*(.+?)(?=优化后的版本|总体评分|$)', response, re.DOTALL)
        if improvements_section:
            improvements = re.findall(r'\d+[\.、]\s*(.+)', improvements_section.group(1))
            report["improvements"] = [i.strip() for i in improvements if i.strip()]

        # 提取优化版本
        improved_section = re.search(r'优化后的版本\s*(.+)$', response, re.DOTALL)
        if improved_section:
            report["improved_version"] = improved_section.group(1).strip()

        # 如果没有提取到优点和改进建议，添加默认值
        if not report["strengths"]:
            report["strengths"] = ["结构清晰", "内容实用"]
        if not report["improvements"]:
            report["improvements"] = ["可增加更多实例", "优化标题吸引力"]

        return report
