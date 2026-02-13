"""
Agent模块
导出所有Agent类

组织结构：
- Core: 基础类和工具
- Auto Mode: 每日新闻简报工作流
- Series Mode: 系列文章生成工作流
- Quality: 质量评估和优化
- Utility: 辅助工具
"""

# ============================================================================
# Core - 基础类
# ============================================================================
from src.agents.base import (
    BaseAgent,
    AgentStatus,
    AgentMetrics,
    with_retry
)

# ============================================================================
# Auto Mode - 每日新闻简报工作流 (v11.0)
# ============================================================================
# 数据获取
from src.agents.ai_trend_analyzer_real import RealAITrendAnalyzerAgent
from src.agents.concurrent_fetch_agent import ConcurrentFetchAgent

# 内容处理
from src.agents.time_weight_agent import TimeWeightAgent
from src.agents.auto_fact_check_agent import AutoFactCheckAgent
from src.agents.content_enhancer_agent import ContentEnhancerAgent
from src.agents.translation_refiner_agent import TranslationRefinerAgent
from src.agents.trend_categorizer_agent import TrendCategorizerAgent
from src.agents.news_scoring_agent import NewsScoringAgent

# 内容生成
from src.agents.trends_digest_agent import TrendsDigestAgent
from src.agents.world_class_digest_agent_v8 import WorldClassDigestAgentV9
from src.agents.news_digest_agent_v3 import NewsDigestAgent  # v3 implementation

# ============================================================================
# Series Mode - 系列文章生成工作流
# ============================================================================
# 内容生成
from src.agents.longform_generator import LongFormGeneratorAgent
from src.agents.research_agent import ResearchAgent

# 质量保证
from src.agents.code_review_agent import CodeReviewAgent
from src.agents.fact_check_agent import FactCheckAgent
from src.agents.quality_evaluator_agent import QualityEvaluatorAgent
from src.agents.consistency_checker_agent import ConsistencyCheckerAgent
from src.agents.visualization_generator_agent import VisualizationGeneratorAgent
from src.agents.citation_formatter_agent import CitationFormatterAgent
from src.agents.series_content_evaluator import SeriesContentEvaluatorAgent

# ============================================================================
# Quality - 质量评估和优化
# ============================================================================
from src.agents.seo_optimizer_agent import SEOOptimizerAgent
from src.agents.content_quality_scorer import ContentQualityScorer

# ============================================================================
# Utility - 辅助工具
# ============================================================================
from src.agents.title_optimizer import TitleOptimizerAgent
from src.agents.image_generator import ImageGeneratorAgent


# ============================================================================
# Exports
# ============================================================================
__all__ = [
    # Core
    "BaseAgent",
    "AgentStatus",
    "AgentMetrics",
    "with_retry",

    # Auto Mode - Data Fetching
    "RealAITrendAnalyzerAgent",
    "ConcurrentFetchAgent",

    # Auto Mode - Processing
    "TimeWeightAgent",
    "AutoFactCheckAgent",
    "ContentEnhancerAgent",
    "TranslationRefinerAgent",
    "TrendCategorizerAgent",
    "NewsScoringAgent",

    # Auto Mode - Generation
    "TrendsDigestAgent",
    "WorldClassDigestAgentV9",
    "NewsDigestAgent",  # v3 implementation

    # Series Mode - Generation
    "LongFormGeneratorAgent",
    "ResearchAgent",

    # Series Mode - Quality Assurance
    "CodeReviewAgent",
    "FactCheckAgent",
    "QualityEvaluatorAgent",
    "ConsistencyCheckerAgent",
    "VisualizationGeneratorAgent",
    "CitationFormatterAgent",
    "SeriesContentEvaluatorAgent",

    # Quality
    "SEOOptimizerAgent",
    "ContentQualityScorer",

    # Utility
    "TitleOptimizerAgent",
    "ImageGeneratorAgent",
]


# ============================================================================
# Agent Registry - 便捷访问
# ============================================================================
AGENT_REGISTRY = {
    # Auto Mode
    "concurrent_fetch": ConcurrentFetchAgent,
    "time_weight": TimeWeightAgent,
    "auto_fact_check": AutoFactCheckAgent,
    "content_enhancer": ContentEnhancerAgent,
    "translation_refiner": TranslationRefinerAgent,
    "trend_categorizer": TrendCategorizerAgent,
    "news_scoring": NewsScoringAgent,
    "world_class_digest": WorldClassDigestAgentV9,

    # Series Mode
    "research": ResearchAgent,
    "longform_generator": LongFormGeneratorAgent,
    "code_review": CodeReviewAgent,
    "fact_check": FactCheckAgent,
    "quality_evaluator": QualityEvaluatorAgent,
    "consistency_checker": ConsistencyCheckerAgent,
    "visualization_generator": VisualizationGeneratorAgent,
    "citation_formatter": CitationFormatterAgent,

    # Quality
    "seo_optimizer": SEOOptimizerAgent,
    "content_quality_scorer": ContentQualityScorer,
}


def get_agent_class(name: str):
    """
    通过名称获取 Agent 类

    Args:
        name: Agent 名称（如 "research", "longform_generator"）

    Returns:
        Agent 类

    Raises:
        KeyError: 如果 Agent 不存在
    """
    if name not in AGENT_REGISTRY:
        available = ", ".join(AGENT_REGISTRY.keys())
        raise KeyError(f"Agent '{name}' not found. Available: {available}")
    return AGENT_REGISTRY[name]
