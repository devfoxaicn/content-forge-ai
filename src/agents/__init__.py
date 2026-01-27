"""
Agent模块
导出所有Agent类（内容工厂模式：精简版）
"""

from src.agents.base import BaseAgent
from src.agents.ai_trend_analyzer_real import RealAITrendAnalyzerAgent
from src.agents.trends_digest_agent import TrendsDigestAgent
from src.agents.longform_generator import LongFormGeneratorAgent
# xiaohongshu_refiner and twitter_generator removed (Refine/Custom mode cleanup)
from src.agents.title_optimizer import TitleOptimizerAgent
from src.agents.image_generator import ImageGeneratorAgent

__all__ = [
    "BaseAgent",
    "RealAITrendAnalyzerAgent",
    "TrendsDigestAgent",
    "LongFormGeneratorAgent",
    # "XiaohongshuRefinerAgent",  # Removed (Refine/Custom mode cleanup)
    # "TwitterGeneratorAgent",  # Removed (Refine/Custom mode cleanup)
    "TitleOptimizerAgent",
    "ImageGeneratorAgent"
]
