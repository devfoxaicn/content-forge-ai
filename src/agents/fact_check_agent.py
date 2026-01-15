"""
事实核查Agent - 验证文章中的事实性声明
使用zhipuai为主，Tavily仅用于关键验证（限制使用）
"""

import re
import json
from typing import Dict, Any, List, Optional
from src.agents.base import BaseAgent


class FactCheckAgent(BaseAgent):
    """事实核查Agent - 识别并验证文章中的事实性声明"""

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        super().__init__(config, prompts)
        self.confidence_threshold = config.get("confidence_threshold", 0.8)
        self.max_tavily_calls = config.get("max_tavily_calls", 10)  # 限制Tavily调用次数
        self.use_tavily = config.get("use_tavily", True)

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行事实核查

        Args:
            state: 当前工作流状态，包含longform_article

        Returns:
            更新后的状态，包含fact_check_result
        """
        self.log("开始事实核查...")

        try:
            # 获取长文本文章
            article = state.get("longform_article", {})
            content = article.get("full_content", "")
            title = article.get("title", "")

            if not content:
                self.log("未找到文章内容，跳过事实核查", "WARNING")
                return {**state, "fact_check_result": None}

            # 使用LLM提取事实性声明
            claims = self._extract_claims_via_llm(title, content)
            self.log(f"提取到 {len(claims)} 个事实性声明")

            if not claims:
                self.log("未提取到事实性声明", "INFO")
                return {
                    **state,
                    "fact_check_result": {
                        "total_claims": 0,
                        "verified_claims": [],
                        "accuracy_rate": 1.0,
                        "high_risk_claims": [],
                        "summary": "无需核查"
                    }
                }

            # 验证声明（选择性使用Tavily）
            verified_claims = []
            tavily_call_count = 0

            for claim in claims:
                # 判断是否需要Tavily验证
                needs_verification = self._needs_tavily_verification(claim)

                if needs_verification and self.use_tavily and tavily_call_count < self.max_tavily_calls:
                    # 使用Tavily验证关键声明
                    verified = self._verify_with_tavily(claim)
                    tavily_call_count += 1
                else:
                    # 使用zhipuai的内置知识进行验证
                    verified = self._verify_with_llm_knowledge(claim)

                verified_claims.append(verified)

            self.log(f"事实核查完成，Tavily调用次数: {tavily_call_count}/{self.max_tavily_calls}")

            # 计算准确率
            accuracy_rate = sum(1 for c in verified_claims
                              if c.get("confidence_score", 0) > self.confidence_threshold) / len(verified_claims)

            # 识别高风险声明
            high_risk_claims = [c for c in verified_claims
                              if c.get("confidence_score", 0) < 0.5]

            # 生成总结
            summary = self._generate_summary(verified_claims, accuracy_rate, high_risk_claims)

            # 构建核查结果
            result = {
                "total_claims": len(claims),
                "verified_claims": verified_claims,
                "accuracy_rate": accuracy_rate,
                "high_risk_claims": high_risk_claims,
                "tavily_calls": tavily_call_count,
                "summary": summary,
                "needs_correction": len(high_risk_claims) > 0
            }

            self.log(f"事实核查完成，准确率: {accuracy_rate:.1%}")

            return {**state, "fact_check_result": result}

        except Exception as e:
            self.log(f"事实核查失败: {e}", "ERROR")
            return {
                **state,
                "fact_check_result": {
                    "total_claims": 0,
                    "error": str(e),
                    "accuracy_rate": 0.8  # 默认值
                }
            }

    def _extract_claims_via_llm(self, title: str, content: str) -> List[Dict[str, Any]]:
        """使用LLM从文章中提取事实性声明"""
        prompt = f"""请从以下文章中提取需要验证的事实性声明。

标题：{title}

文章内容：
{content[:3000]}

请提取以下类型的事实性声明：
1. **版本号**：如 v2.5.0, GLM-4.7, Python 3.9
2. **性能数据**：如 "速度提升3倍", "延迟降低50%", "支持100万并发"
3. **技术规格**：如 "支持128K上下文", "内存占用1GB"
4. **日期时间**：如 "2024年1月发布", "最近版本"
5. **比较性陈述**：如 "优于GPT-4", "比TensorFlow快2倍"
6. **统计数据**：如 "已有100万用户", "市场份额30%"

请以JSON格式返回，格式如下：
{{
  "claims": [
    {{
      "statement": "声明内容",
      "type": "版本号|性能数据|技术规格|日期时间|比较性陈述|统计数据",
      "context": "声明的上下文（前后的句子）"
    }}
  ]
}}

只返回JSON，不要其他内容。"""

        try:
            response = self._call_llm(prompt)

            # 解析JSON
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                return data.get("claims", [])

            # 如果解析失败，返回空列表
            self.log("解析声明提取结果失败", "WARNING")
            return []

        except Exception as e:
            self.log(f"提取声明失败: {e}", "ERROR")
            return []

    def _needs_tavily_verification(self, claim: Dict[str, Any]) -> bool:
        """判断声明是否需要Tavily验证"""
        claim_type = claim.get("type", "")
        statement = claim.get("statement", "")

        # 以下类型的声明需要Tavily验证
        high_priority_types = [
            "性能数据",
            "比较性陈述",
            "统计数据"
        ]

        # 包含具体数字的声明需要验证
        has_numbers = bool(re.search(r'\d+', statement))

        # 是高优先级类型且包含数字
        return claim_type in high_priority_types and has_numbers

    def _verify_with_tavily(self, claim: Dict[str, Any]) -> Dict[str, Any]:
        """使用Tavily API验证声明"""
        try:
            from tavily import TavilyClient
            import os

            api_key = os.environ.get("TAVILY_API_KEY")
            if not api_key:
                self.log("未设置TAVILY_API_KEY，降级到LLM验证", "WARNING")
                return self._verify_with_llm_knowledge(claim)

            client = TavilyClient(api_key=api_key)
            statement = claim.get("statement", "")

            # 搜索验证
            search_result = client.search(
                query=statement,
                search_depth="basic",
                max_results=3,
                include_raw_content=False
            )

            # 评估可信度
            confidence_score = self._assess_credibility_from_tavily(search_result, statement)

            # 提取来源
            sources = []
            for result in search_result.get("results", [])[:3]:
                sources.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "score": result.get("score", 0)
                })

            return {
                **claim,
                "verification_method": "tavily",
                "confidence_score": confidence_score,
                "sources": sources,
                "status": "verified" if confidence_score > 0.8 else "uncertain"
            }

        except Exception as e:
            self.log(f"Tavily验证失败: {e}，降级到LLM验证", "WARNING")
            return self._verify_with_llm_knowledge(claim)

    def _verify_with_llm_knowledge(self, claim: Dict[str, Any]) -> Dict[str, Any]:
        """使用zhipuai的内置知识验证声明"""
        statement = claim.get("statement", "")
        claim_type = claim.get("type", "")
        context = claim.get("context", "")

        prompt = f"""请基于你的知识库验证以下事实性声明的准确性。

声明类型：{claim_type}
声明内容：{statement}
上下文：{context}

请评估：
1. 该声明是否准确？
2. 你的置信度是多少（0-1之间）？
3. 如果不准确，正确的信息是什么？

请以JSON格式返回：
{{
  "accurate": true/false,
  "confidence": 0.0-1.0,
  "correction": "如果准确则为空，否则提供正确信息",
  "reasoning": "评估理由"
}}

只返回JSON，不要其他内容。"""

        try:
            response = self._call_llm(prompt)

            # 解析JSON
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())

                confidence = data.get("confidence", 0.7)
                accurate = data.get("accurate", True)

                return {
                    **claim,
                    "verification_method": "llm_knowledge",
                    "confidence_score": confidence if accurate else 1.0 - confidence,
                    "status": "verified" if confidence > 0.7 else "uncertain",
                    "correction": data.get("correction", ""),
                    "reasoning": data.get("reasoning", "")
                }

        except Exception as e:
            self.log(f"LLM知识验证失败: {e}", "WARNING")

        # 默认返回中等置信度
        return {
            **claim,
            "verification_method": "llm_knowledge",
            "confidence_score": 0.7,
            "status": "uncertain",
            "reasoning": "验证过程出现问题，建议人工核实"
        }

    def _assess_credibility_from_tavily(self, search_result: Dict[str, Any], statement: str) -> float:
        """根据Tavily搜索结果评估可信度"""
        results = search_result.get("results", [])

        if not results:
            return 0.3  # 无搜索结果，低置信度

        # 计算置信度分数
        confidence_score = 0.0

        # 结果数量评分（最多3个结果）
        result_count_score = min(len(results) / 3, 1.0) * 0.3
        confidence_score += result_count_score

        # 内容相关性评分
        statement_lower = statement.lower()
        relevant_count = 0

        for result in results:
            title = result.get("title", "").lower()
            content = result.get("content", "").lower()

            # 检查是否包含关键词
            keywords = re.findall(r'\b\w{3,}\b', statement_lower)
            keyword_matches = sum(1 for kw in keywords if kw in title or kw in content)

            if keyword_matches >= 3:
                relevant_count += 1

        relevance_score = (relevant_count / len(results)) * 0.7
        confidence_score += relevance_score

        return min(confidence_score, 1.0)

    def _generate_summary(self, verified_claims: List[Dict[str, Any]],
                         accuracy_rate: float, high_risk_claims: List[Dict[str, Any]]) -> str:
        """生成事实核查总结"""
        if not verified_claims:
            return "无需核查"

        total_claims = len(verified_claims)
        verified_count = sum(1 for c in verified_claims if c.get("status") == "verified")
        tavily_count = sum(1 for c in verified_claims if c.get("verification_method") == "tavily")

        summary_parts = [
            f"共核查 {total_claims} 个声明",
            f"已验证: {verified_count}/{total_claims}",
            f"准确率: {accuracy_rate:.1%}",
            f"Tavily验证: {tavily_count} 个"
        ]

        if high_risk_claims:
            summary_parts.append(f"⚠️ 发现 {len(high_risk_claims)} 个高风险声明")
        else:
            summary_parts.append("✅ 未发现高风险声明")

        return " | ".join(summary_parts)
