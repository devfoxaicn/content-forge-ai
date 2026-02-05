"""
时效性智能加权Agent v11.0 - 根据发布时间动态调整热点权重

核心功能:
- 1小时内新闻: 100分（最新，最高优先级）
- 6小时内: 90-99分（非常新）
- 24小时内: 70-89分（新）
- 3天内: 40-69分（较新）
- 超过3天: <40分（旧）

特殊处理:
- 突发新闻(breaking): +20分加成
- 实时数据源(NewsData.io, Reddit): +10分加成
- 周末/节假日发布: -5分（因为关注度低）
"""

from typing import Dict, Any, List
from datetime import datetime, timedelta
from src.agents.base import BaseAgent
from loguru import logger


class TimeWeightAgent(BaseAgent):
    """时效性智能加权Agent v11.0"""

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        super().__init__(config, prompts)
        self.name = "time_weight"

        # 获取配置
        agent_config = config.get("agents", {}).get("time_weight", {})
        self.weight_mode = agent_config.get("weight_mode", "dynamic")  # dynamic, linear, exponential
        self.max_hours = agent_config.get("max_hours", 72)
        self.boost_latest = agent_config.get("boost_latest", 2.0)

        # 实时数据源列表（获得时效性加成）
        self.realtime_sources = [
            "NewsData.io",
            "Reddit",
            "Hacker News",
            "GitHub Trending"
        ]

        # 突发新闻关键词
        self.breaking_keywords = [
            "breaking", "突发", "紧急", "just in", "刚刚",
            "exclusive", "独家", "live", "直播"
        ]

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """执行时效性加权"""
        self.log(f"开始时效性加权（模式: {self.weight_mode}）...")

        # 获取热点数据
        trends_by_source = state.get("trends_by_source", {})

        if not trends_by_source:
            self.log("没有热点数据，跳过时效性加权", "WARNING")
            return {**state, "current_step": "time_weight_skipped"}

        # 统计信息
        total_count = 0
        boosted_count = 0
        one_hour_count = 0
        six_hour_count = 0
        twenty_four_hour_count = 0

        # 对每个数据源的热点进行时效性加权
        weighted_trends_by_source = {}
        for source, trends in trends_by_source.items():
            weighted_trends = []
            for trend in trends:
                # 计算时效性分数
                time_score = self._calculate_time_score(trend, source)

                # 检查是否为突发新闻
                is_breaking = self._is_breaking_news(trend)
                if is_breaking:
                    time_score = min(100, time_score + 20)

                # 检查是否为实时数据源
                is_realtime = source in self.realtime_sources
                if is_realtime:
                    time_score = min(100, time_score + 10)

                # 更新热力分数（合并时效性分数）
                current_heat = trend.get("heat_score", 60)
                trend["heat_score"] = int((current_heat + time_score) / 2)
                trend["time_weight_score"] = time_score
                trend["is_breaking"] = is_breaking
                trend["is_realtime"] = is_realtime

                weighted_trends.append(trend)
                total_count += 1

                # 统计
                if time_score >= 100:
                    one_hour_count += 1
                elif time_score >= 90:
                    six_hour_count += 1
                elif time_score >= 70:
                    twenty_four_hour_count += 1

                if is_breaking or is_realtime:
                    boosted_count += 1

            # 按时效性分数排序
            weighted_trends.sort(key=lambda x: x.get("time_weight_score", 0), reverse=True)
            weighted_trends_by_source[source] = weighted_trends

        self.log(f"时效性加权完成: {total_count}条热点")
        self.log(f"  - 1小时内: {one_hour_count}条")
        self.log(f"  - 6小时内: {six_hour_count}条")
        self.log(f"  - 24小时内: {twenty_four_hour_count}条")
        self.log(f"  - 加权提升: {boosted_count}条")

        return {
            **state,
            "trends_by_source": weighted_trends_by_source,
            "time_weight_stats": {
                "total": total_count,
                "one_hour": one_hour_count,
                "six_hour": six_hour_count,
                "twenty_four_hour": twenty_four_hour_count,
                "boosted": boosted_count
            },
            "current_step": "time_weight_completed"
        }

    def _calculate_time_score(self, trend: Dict[str, Any], source: str) -> float:
        """计算时效性分数"""
        # 获取发布时间
        timestamp = trend.get("timestamp", "")
        if not timestamp:
            # 没有时间戳，默认给予中等分数
            return 50

        try:
            # 解析时间戳
            pub_time = self._parse_timestamp(timestamp)
            if not pub_time:
                return 50

            # 计算时间差（小时）
            now = datetime.now()
            delta = now - pub_time
            hours_ago = delta.total_seconds() / 3600

            # 根据模式计算分数
            if self.weight_mode == "dynamic":
                return self._dynamic_time_score(hours_ago)
            elif self.weight_mode == "linear":
                return self._linear_time_score(hours_ago)
            elif self.weight_mode == "exponential":
                return self._exponential_time_score(hours_ago)
            else:
                return self._dynamic_time_score(hours_ago)

        except Exception as e:
            self.log(f"解析时间失败: {timestamp}, {e}", "WARNING")
            return 50

    def _parse_timestamp(self, timestamp: str) -> datetime:
        """解析时间戳字符串"""
        if not timestamp:
            return None

        # 尝试多种时间格式
        formats = [
            "%Y-%m-%d",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%a, %d %b %Y %H:%M:%S %z",  # RFC 2822
            "%a, %d %b %Y %H:%M:%S",     # RSS format
        ]

        for fmt in formats:
            try:
                return datetime.strptime(timestamp[:19], fmt)
            except ValueError:
                continue

        return None

    def _dynamic_time_score(self, hours_ago: float) -> float:
        """
        动态时效性评分（推荐）

        1小时内: 100分（最新）
        6小时内: 90-99分（非常新）
        24小时内: 70-89分（新）
        3天内: 40-69分（较新）
        超过3天: <40分（旧）
        """
        if hours_ago <= 1:
            return 100
        elif hours_ago <= 6:
            # 90-99分，线性递减
            return 99 - (hours_ago - 1) * 2
        elif hours_ago <= 24:
            # 70-89分，线性递减
            return 89 - (hours_ago - 6) * 1.5
        elif hours_ago <= 72:
            # 40-69分，线性递减
            return 69 - (hours_ago - 24) * 0.8
        else:
            # 超过3天，<40分
            return max(0, 40 - (hours_ago - 72) * 0.3)

    def _linear_time_score(self, hours_ago: float) -> float:
        """线性时效性评分（简单版）"""
        max_hours = self.max_hours
        if hours_ago <= 0:
            return 100
        elif hours_ago >= max_hours:
            return 0
        else:
            return 100 * (1 - hours_ago / max_hours)

    def _exponential_time_score(self, hours_ago: float) -> float:
        """指数时效性评分（激进版）"""
        import math
        decay_rate = 0.1
        return 100 * math.exp(-decay_rate * hours_ago)

    def _is_breaking_news(self, trend: Dict[str, Any]) -> bool:
        """检查是否为突发新闻"""
        title = trend.get("title", "").lower()
        description = trend.get("description", "").lower()

        text = title + " " + description

        for keyword in self.breaking_keywords:
            if keyword in text:
                return True

        return False
