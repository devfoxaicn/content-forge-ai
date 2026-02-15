"""
工作流状态管理
定义多Agent协同系统的状态结构
"""

from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime


class TrendingTopic(TypedDict):
    """热门话题"""
    topic: str
    views: int
    engagement_rate: float
    difficulty: str  # easy, medium, hard
    description: str


class ContentIdea(TypedDict):
    """内容创意"""
    title: str
    angle: str
    value_proposition: str
    content_format: str  # 图文/视频/混合
    expected_virality: str


class GeneratedContent(TypedDict):
    """生成的内容"""
    title: str
    intro: str
    body: str
    ending: str
    hashtags: List[str]
    word_count: int


class ImageSuggestion(TypedDict):
    """图像建议"""
    style: str
    scene_description: str
    color_scheme: str
    sd_prompt: Optional[str]
    sd_negative_prompt: Optional[str]
    shooting_suggestion: Optional[str]


class QualityReport(TypedDict):
    """质量评估报告"""
    overall_score: float
    title_score: float
    content_value_score: float
    structure_score: float
    strengths: List[str]
    improvements: List[str]
    improved_version: Optional[str]


class WorkflowState(TypedDict):
    """
    工作流状态
    包含所有Agent之间的共享数据
    """
    # 输入参数
    user_request: Dict[str, Any]
    topic: str
    target_audience: str
    content_type: str
    keywords: List[str]

    # 中间状态
    current_step: str
    error_message: Optional[str]
    retry_count: int
    max_retries: int

    # Agent输出
    trending_topics: Optional[List[TrendingTopic]]
    selected_topic: Optional[str]
    content_ideas: Optional[List[ContentIdea]]
    selected_idea: Optional[ContentIdea]
    generated_content: Optional[GeneratedContent]
    optimized_titles: Optional[List[str]]
    recommended_title: Optional[str]
    image_suggestions: Optional[List[ImageSuggestion]]
    quality_report: Optional[QualityReport]

    # Topic模式专用字段
    content_style: Optional[str]  # 内容风格: xiaohongshu/weixin/tech_blog/video_script
    image_style: Optional[str]  # 配图风格: tech_abstract/dark_minimal/soft_gradient/illustration
    platforms: Optional[List[str]]  # 目标平台列表
    enable_fact_check: Optional[bool]  # 是否启用事实核查
    enable_image: Optional[bool]  # 是否启用配图生成
    research_data: Optional[Dict[str, Any]]  # 调研数据
    research_report: Optional[str]  # 调研报告
    article: Optional[str]  # 长文本文章
    fact_check_result: Optional[Dict[str, Any]]  # 事实核查结果
    platform_outputs: Optional[Dict[str, str]]  # 多平台适配输出
    image_outputs: Optional[Dict[str, Any]]  # 配图输出

    # 最终输出
    final_output: Optional[Dict[str, Any]]

    # 元数据
    start_time: str
    end_time: Optional[str]
    execution_time: Optional[float]
    agent_execution_order: List[str]
    tokens_used: Dict[str, int]

    # 配置
    config: Dict[str, Any]


def create_initial_state(
    topic: Optional[str] = None,
    target_audience: str = "大众用户",
    content_type: str = "干货分享",
    keywords: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
    # Topic模式参数
    content_style: Optional[str] = None,
    image_style: Optional[str] = None,
    platforms: Optional[List[str]] = None,
    enable_fact_check: bool = True,
    enable_image: bool = True
) -> WorkflowState:
    """
    创建初始工作流状态

    Args:
        topic: 内容主题标识（可选，留空则基于实时热点自动生成）
        target_audience: 目标受众
        content_type: 内容类型
        keywords: 关键词列表
        config: 配置字典
        content_style: 内容风格 (topic模式专用)
        image_style: 配图风格 (topic模式专用)
        platforms: 目标平台列表 (topic模式专用)
        enable_fact_check: 是否启用事实核查 (topic模式专用)
        enable_image: 是否启用配图生成 (topic模式专用)

    Returns:
        WorkflowState: 初始状态
    """
    # 如果没有提供topic，使用auto作为标识
    if topic is None:
        topic = "auto"

    return WorkflowState(
        # 输入参数
        user_request={
            "topic": topic,
            "target_audience": target_audience,
            "content_type": content_type,
            "keywords": keywords or []
        },
        topic=topic,
        target_audience=target_audience,
        content_type=content_type,
        keywords=keywords or [],

        # 中间状态
        current_step="init",
        error_message=None,
        retry_count=0,
        max_retries=3,

        # Agent输出（初始化为None）
        trending_topics=None,
        selected_topic=None,
        content_ideas=None,
        selected_idea=None,
        generated_content=None,
        optimized_titles=None,
        recommended_title=None,
        image_suggestions=None,
        quality_report=None,

        # Topic模式专用字段
        content_style=content_style,
        image_style=image_style,
        platforms=platforms,
        enable_fact_check=enable_fact_check,
        enable_image=enable_image,
        research_data=None,
        research_report=None,
        article=None,
        fact_check_result=None,
        platform_outputs=None,
        image_outputs=None,

        # 最终输出
        final_output=None,

        # 元数据
        start_time=datetime.now().isoformat(),
        end_time=None,
        execution_time=None,
        agent_execution_order=[],
        tokens_used={},

        # 配置
        config=config or {}
    )


def update_state(state: WorkflowState, updates: Dict[str, Any]) -> WorkflowState:
    """
    更新工作流状态

    Args:
        state: 当前状态
        updates: 要更新的字段

    Returns:
        WorkflowState: 更新后的状态
    """
    return {**state, **updates}


def add_agent_to_order(state: WorkflowState, agent_name: str) -> WorkflowState:
    """
    记录Agent执行顺序

    Args:
        state: 当前状态
        agent_name: Agent名称

    Returns:
        WorkflowState: 更新后的状态
    """
    execution_order = state["agent_execution_order"].copy()
    execution_order.append(agent_name)
    return {**state, "agent_execution_order": execution_order}


def calculate_execution_time(state: WorkflowState) -> WorkflowState:
    """
    计算执行时间

    Args:
        state: 当前状态

    Returns:
        WorkflowState: 更新后的状态
    """
    start = datetime.fromisoformat(state["start_time"])
    end = datetime.now()
    execution_time = (end - start).total_seconds()

    return {
        **state,
        "end_time": end.isoformat(),
        "execution_time": execution_time
    }
