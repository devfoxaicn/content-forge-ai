"""
内容增强Agent v11.0 - 为重要新闻补充背景和影响分析

特点:
- 使用trafilatura提取完整文章内容
- 为重要性>=70的新闻生成背景分析
- 使用textstat评估可读性
- 自动生成行业影响分析
"""

from typing import Dict, Any, List
import trafilatura
import textstat
from src.agents.base import BaseAgent
from loguru import logger


class ContentEnhancerAgent(BaseAgent):
    """内容增强Agent v11.0"""

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        super().__init__(config, prompts)
        self.name = "content_enhancer"

        # 获取配置
        agent_config = config.get("agents", {}).get("content_enhancer", {})
        self.extract_full_content = agent_config.get("extract_full_content", True)
        self.min_importance_score = agent_config.get("min_importance_score", 70)
        self.enable_background = agent_config.get("enable_background", True)
        self.enable_impact = agent_config.get("enable_impact", True)

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """执行内容增强"""
        self.log(f"开始内容增强（最低重要性: {self.min_importance_score}）...")

        # 获取评分后的新闻
        scored_trends = state.get("scored_trends", {})

        if not scored_trends:
            self.log("没有评分数据，跳过内容增强", "WARNING")
            return {**state, "current_step": "content_enhancer_skipped"}

        enhanced_count = 0
        total_count = 0

        # 对每个分类的新闻进行增强
        for category, trends in scored_trends.items():
            for i, trend in enumerate(trends):
                total_count += 1

                # 只对高重要性新闻进行增强
                importance = trend.get("final_score", trend.get("importance_score", 0))

                if importance >= self.min_importance_score:
                    # 提取完整内容
                    if self.extract_full_content:
                        full_content = self._extract_full_content(trend.get("url", ""))
                        if full_content:
                            trend["full_content"] = full_content
                            trend["content_length"] = len(full_content)

                            # 计算可读性
                            try:
                                readability = textstat.flesch_reading_ease(full_content)
                                trend["readability_score"] = readability
                            except:
                                trend["readability_score"] = None

                    # 生成背景分析
                    if self.enable_background:
                        background = self._generate_background(trend)
                        if background:
                            trend["background_analysis"] = background

                    # 生成影响分析
                    if self.enable_impact:
                        impact = self._generate_impact(trend)
                        if impact:
                            trend["impact_analysis"] = impact

                    enhanced_count += 1

            # 更新该分类的trends
            scored_trends[category] = trends

        self.log(f"内容增强完成: {enhanced_count}/{total_count}条新闻已增强")

        return {
            **state,
            "scored_trends": scored_trends,
            "content_enhancer_stats": {
                "total": total_count,
                "enhanced": enhanced_count
            },
            "current_step": "content_enhancer_completed"
        }

    def _extract_full_content(self, url: str) -> str:
        """使用trafilatura提取完整文章内容"""
        if not url:
            return None

        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                content = trafilatura.extract(downloaded)
                return content
        except Exception as e:
            self.log(f"提取内容失败: {url}, {e}", "WARNING")

        return None

    def _generate_background(self, trend: Dict[str, Any]) -> str:
        """生成背景分析"""
        prompt = f"""为以下新闻生成背景分析（120-150字）：

标题: {trend.get('title', '')}
描述: {trend.get('description', '')}

请提供：
1. 技术发展脉络
2. 相关历史事件
3. 公司/机构背景

要求：
- 具体性：使用数据和例子
- 简洁性：3-5句话
- 准确性：基于已知事实"""

        try:
            response = self._call_llm(prompt)
            return response.strip()
        except Exception as e:
            self.log(f"生成背景分析失败: {e}", "WARNING")
            return None

    def _generate_impact(self, trend: Dict[str, Any]) -> str:
        """生成影响分析"""
        prompt = f"""为以下新闻生成行业影响分析（120-150字）：

标题: {trend.get('title', '')}
描述: {trend.get('description', '')}

请分析：
1. 对不同群体的影响（开发者、企业、用户）
2. 市场格局变化
3. 技术发展方向

要求：
- 利益导向：说明"这意味着什么"
- 可操作：提供具体建议
- 前瞻性：预测未来趋势"""

        try:
            response = self._call_llm(prompt)
            return response.strip()
        except Exception as e:
            self.log(f"生成影响分析失败: {e}", "WARNING")
            return None
