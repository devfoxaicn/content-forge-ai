"""
时间过滤工具类 - 用于24小时内内容过滤
"""
from datetime import datetime, timedelta
from typing import Optional, Union
from loguru import logger


class TimeFilter:
    """24小时时间过滤器"""

    def __init__(self, hours: int = 24):
        """
        初始化时间过滤器

        Args:
            hours: 时间窗口（小时），默认24小时
        """
        self.hours = hours
        self.cutoff_time = datetime.now() - timedelta(hours=hours)

    def is_within_time_window(self, timestamp: Union[str, datetime, int, float]) -> bool:
        """
        检查时间戳是否在时间窗口内

        Args:
            timestamp: 时间戳，可以是字符串、datetime对象、Unix时间戳

        Returns:
            bool: 是否在时间窗口内
        """
        try:
            dt = self._parse_timestamp(timestamp)
            if dt is None:
                return False
            return dt >= self.cutoff_time
        except Exception as e:
            logger.warning(f"时间解析失败: {timestamp}, 错误: {e}")
            return False

    def _parse_timestamp(self, timestamp: Union[str, datetime, int, float]) -> Optional[datetime]:
        """
        解析各种格式的时间戳 (v9.1: 增强版，支持更多RSS/HTTP格式)

        Args:
            timestamp: 时间戳

        Returns:
            datetime对象或None
        """
        if timestamp is None:
            return None

        # 已经是datetime对象
        if isinstance(timestamp, datetime):
            return timestamp

        # Unix时间戳（秒或毫秒）
        if isinstance(timestamp, (int, float)):
            # 判断是秒还是毫秒
            if timestamp > 1e12:  # 毫秒
                return datetime.fromtimestamp(timestamp / 1000)
            else:  # 秒
                return datetime.fromtimestamp(timestamp)

        # 字符串格式
        if isinstance(timestamp, str):
            if not timestamp.strip():
                return None

            timestamp_clean = timestamp.strip()

            # ========== v9.1: 增强的时间格式支持 ==========

            # 格式1: ISO 8601标准格式
            iso_formats = [
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
            ]
            for fmt in iso_formats:
                try:
                    return datetime.strptime(timestamp_clean, fmt)
                except ValueError:
                    continue

            # 格式2: RSS/Atom格式 (RFC 2822) - "Thu, 29 Jan 2026 15:30:00 GMT"
            # 需要处理时区缩写
            rss_formats = [
                "%a, %d %b %Y %H:%M:%S %Z",   # Thu, 29 Jan 2026 15:30:00 GMT
                "%a, %d %b %Y %H:%M:%S %z",   # Thu, 29 Jan 2026 15:30:00 +0000
                "%a, %d %b %Y %H:%M:%S",     # Thu, 29 Jan 2026 15:30:00
                "%a, %d %b %Y",              # Thu, 29 Jan 2026
                "%d %b %Y %H:%M:%S %Z",      # 29 Jan 2026 15:30:00 GMT
                "%d %b %Y",                  # 29 Jan 2026
            ]

            # 时区缩写映射（RFC 2822）
            timezone_map = {
                "GMT": "+0000", "UTC": "+0000",
                "EST": "-0500", "EDT": "-0400",
                "CST": "-0600", "CDT": "-0500",
                "MST": "-0700", "MDT": "-0600",
                "PST": "-0800", "PDT": "-0700",
            }

            for fmt in rss_formats:
                try:
                    # 替换时区缩写为数字偏移
                    temp_timestamp = timestamp_clean
                    for tz_abbr, tz_offset in timezone_map.items():
                        if tz_abbr in temp_timestamp:
                            temp_timestamp = temp_timestamp.replace(tz_abbr, tz_offset)
                            break

                    return datetime.strptime(temp_timestamp, fmt)
                except ValueError:
                    continue

            # 格式3: HTTP Date格式 - "Thursday, 29-Jan-26 15:30:00 GMT"
            try:
                # 替换时区
                temp_timestamp = timestamp_clean
                for tz_abbr, tz_offset in timezone_map.items():
                    if tz_abbr in temp_timestamp:
                        temp_timestamp = temp_timestamp.replace(tz_abbr, tz_offset)
                        break

                # 尝试解析带连字符的日期
                for fmt in ["%a, %d-%b-%y %H:%M:%S %z", "%a, %d-%b-%y %H:%M:%S"]:
                    try:
                        return datetime.strptime(temp_timestamp, fmt)
                    except ValueError:
                        continue
            except:
                pass

            # 格式4: 使用dateutil自动解析（支持大多数格式）
            try:
                from dateutil import parser as dateutil_parser
                return dateutil_parser.parse(timestamp_clean)
            except Exception as e:
                logger.debug(f"dateutil解析失败: {timestamp_clean}, 错误: {e}")

        return None

    def filter_items(self, items: list, timestamp_key: str = "published_at") -> list:
        """
        过滤列表中的项目

        Args:
            items: 项目列表
            timestamp_key: 时间戳字段名

        Returns:
            过滤后的列表
        """
        filtered = []
        for item in items:
            timestamp = item.get(timestamp_key)
            if timestamp and self.is_within_time_window(timestamp):
                filtered.append(item)

        logger.info(f"时间过滤: {len(items)} -> {len(filtered)} (保留率: {len(filtered)/len(items)*100:.1f}%)" if items else "无数据")
        return filtered

    def get_hours_until_cutoff(self, timestamp: Union[str, datetime, int, float]) -> Optional[int]:
        """
        获取距离截止时间的小时数

        Args:
            timestamp: 时间戳

        Returns:
            距离截止时间的小时数，None表示无法解析
        """
        dt = self._parse_timestamp(timestamp)
        if dt is None:
            return None

        delta = dt - self.cutoff_time
        return int(delta.total_seconds() / 3600)


def create_time_filter(hours: int = 24) -> TimeFilter:
    """
    创建时间过滤器的工厂函数

    Args:
        hours: 时间窗口（小时）

    Returns:
        TimeFilter实例
    """
    return TimeFilter(hours=hours)


# 便捷函数
def is_within_24h(timestamp: Union[str, datetime, int, float]) -> bool:
    """检查时间戳是否在24小时内"""
    filter_24h = create_time_filter(hours=24)
    return filter_24h.is_within_time_window(timestamp)


def filter_last_24h(items: list, timestamp_key: str = "published_at") -> list:
    """过滤最近24小时的项目"""
    filter_24h = create_time_filter(hours=24)
    return filter_24h.filter_items(items, timestamp_key=timestamp_key)
