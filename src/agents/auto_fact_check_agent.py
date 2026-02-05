"""
轻量级事实核查Agent v11.0 - 使用LLM内置知识快速核查关键声明

特点:
- 仅对Top10重要新闻进行核查（节省成本）
- 使用LLM内置知识（无需Tavily API）
- 核查重点：版本号、性能数据、技术规格
- 输出置信度评分，低于阈值的新闻标记为需要人工审核
"""

from typing import Dict, Any, List
from src.agents.base import BaseAgent
from loguru import logger


class AutoFactCheckAgent(BaseAgent):
    """轻量级事实核查Agent v11.0"""

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        super().__init__(config, prompts)
        self.name = "auto_fact_check"

        # 获取配置
        agent_config = config.get("agents", {}).get("auto_fact_check", {})
        self.check_top_n = agent_config.get("check_top_n", 10)
        self.use_tavily = agent_config.get("use_tavily", False)
        self.confidence_threshold = agent_config.get("confidence_threshold", 0.7)

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """执行轻量级事实核查"""
        self.log(f"开始事实核查（Top {self.check_top_n}条）...")

        # 获取评分后的新闻
        scored_trends = state.get("scored_trends", {})

        if not scored_trends:
            self.log("没有评分数据，跳过事实核查", "WARNING")
            return {**state, "current_step": "fact_check_skipped"}

        # 收集所有新闻并按重要性排序
        all_trends = []
        for category, trends in scored_trends.items():
            for trend in trends:
                all_trends.append(trend)

        # 按评分排序
        all_trends.sort(key=lambda x: x.get("final_score", 0), reverse=True)

        # 取Top N进行核查
        top_trends = all_trends[:self.check_top_n]

        if not top_trends:
            self.log("没有新闻需要核查", "WARNING")
            return {**state, "current_step": "fact_check_skipped"}

        self.log(f"开始核查Top {len(top_trends)}条新闻...")

        # 对每条新闻进行事实核查
        verified_trends = []
        high_confidence_count = 0
        low_confidence_count = 0

        for trend in top_trends:
            # 提取关键声明
            claims = self._extract_key_claims(trend)

            if not claims:
                # 没有提取到关键声明，给予中等置信度
                trend["fact_check"] = {
                    "confidence": 0.6,
                    "needs_review": False,
                    "claims_checked": [],
                    "notes": "未提取到关键声明"
                }
                verified_trends.append(trend)
                continue

            # 使用LLM知识验证
            verified = self._verify_with_llm(claims, trend)

            # 添加核查结果
            trend["fact_check"] = verified

            if verified["confidence"] >= self.confidence_threshold:
                high_confidence_count += 1
            else:
                low_confidence_count += 1

            verified_trends.append(trend)

        # 重新排序（高置信度优先）
        verified_trends.sort(key=lambda x: x.get("fact_check", {}).get("confidence", 0), reverse=True)

        self.log(f"事实核查完成: {len(verified_trends)}条")
        self.log(f"  - 高置信度(>={self.confidence_threshold}): {high_confidence_count}条")
        self.log(f"  - 低置信度(<{self.confidence_threshold}): {low_confidence_count}条")

        # 更新state
        # 注意：我们需要将核查后的新闻放回原始分类结构中
        updated_scored_trends = self._merge_verified_trends(scored_trends, verified_trends)

        return {
            **state,
            "scored_trends": updated_scored_trends,
            "fact_check_stats": {
                "total": len(verified_trends),
                "high_confidence": high_confidence_count,
                "low_confidence": low_confidence_count
            },
            "current_step": "fact_check_completed"
        }

    def _extract_key_claims(self, trend: Dict[str, Any]) -> List[str]:
        """提取关键声明"""
        claims = []

        title = trend.get("title", "")
        description = trend.get("description", "")

        # 提取版本号
        import re
        version_patterns = [
            r'GPT-\d+(\.\d+)?',
            r'GPT-\d+[a-z]',
            r'Llma \d+(\.[A-Za-z])?',
            r'Claude \d+(\.[A-Za-z])?',
            r'Gemini \d+(\.[A-Za-z])?',
            r'v\d+(\.\d+)+',
            r'Version \d+(\.\d+)+',
            r'\d+\.\d+(\.\d+)?',
        ]

        for pattern in version_patterns:
            matches = re.findall(pattern, title, re.IGNORECASE)
            claims.extend([f"版本: {m}" for m in matches])

        # 提取性能数据
        performance_patterns = [
            r'\d+%',
            r'\d+倍',
            r'\d+x',
            r'\d+倍性能',
            r'\d+亿参数',
            r'\d+B参数',
            r'\d+M参数',
        ]

        for pattern in performance_patterns:
            matches = re.findall(pattern, title + description)
            claims.extend([f"性能数据: {m}" for m in matches])

        # 提取公司/产品名称
        entities = self._extract_entities(title + description)
        claims.extend([f"实体: {e}" for e in entities])

        return list(set(claims))  # 去重

    def _extract_entities(self, text: str) -> List[str]:
        """提取实体名称（简单版）"""
        entities = []

        # 常见AI实体列表
        ai_entities = [
            "OpenAI", "Anthropic", "Google", "Microsoft", "Meta",
            "ChatGPT", "GPT-4", "Claude", "Gemini", "Llama",
            "Hugging Face", "NVIDIA", "AMD", "Intel",
            "Transformer", "BERT", "ResNet", "YOLO"
        ]

        for entity in ai_entities:
            if entity.lower() in text.lower():
                entities.append(entity)

        return list(set(entities))

    def _verify_with_llm(self, claims: List[str], trend: Dict[str, Any]) -> Dict[str, Any]:
        """使用LLM内置知识验证声明"""
        # 构建验证提示词
        prompt = f"""请验证以下新闻中的关键声明是否准确：

标题: {trend.get('title', '')}
描述: {trend.get('description', '')}

关键声明:
{chr(10).join(f'- {claim}' for claim in claims)}

请以JSON格式返回验证结果：
{{
    "confidence": 0.0-1.0之间的置信度分数,
    "claims_checked": ["验证通过的声明列表"],
    "issues_found": ["发现的问题列表"],
    "notes": "其他备注"
}}

注意：
- 只使用你的内置知识，不要进行网络搜索
- 对于版本号、性能数据等技术声明要严格验证
- 如果不确定，给予较低的置信度
"""

        try:
            response = self._call_llm(prompt)

            # 尝试解析JSON响应
            import json
            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                # JSON解析失败，返回默认值
                result = {
                    "confidence": 0.6,
                    "claims_checked": [],
                    "issues_found": [],
                    "notes": "LLM响应解析失败"
                }

            # 确保包含必要字段
            if "confidence" not in result:
                result["confidence"] = 0.6
            if "needs_review" not in result:
                result["needs_review"] = result["confidence"] < self.confidence_threshold
            if "claims_checked" not in result:
                result["claims_checked"] = claims
            if "issues_found" not in result:
                result["issues_found"] = []

            return result

        except Exception as e:
            self.log(f"LLM验证失败: {e}", "WARNING")
            return {
                "confidence": 0.5,
                "needs_review": True,
                "claims_checked": [],
                "issues_found": ["验证过程出错"],
                "notes": str(e)
            }

    def _merge_verified_trends(self, original_trends: Dict[str, List[Dict]],
                                verified_trends: List[Dict]) -> Dict[str, List[Dict]]:
        """将核查后的新闻合并回原始分类结构"""
        # 创建verified_trends的索引
        verified_index = {t.get("url", ""): t for t in verified_trends}

        # 更新原始trends
        for category, trends in original_trends.items():
            for i, trend in enumerate(trends):
                url = trend.get("url", "")
                if url in verified_index:
                    # 更新fact_check信息
                    trends[i]["fact_check"] = verified_index[url].get("fact_check", {})

        return original_trends
