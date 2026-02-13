"""
基础Agent类
所有Agent都继承自这个基类

Enhanced Features:
- 自动重试机制
- 执行指标收集
- 统一错误处理
- 超时控制
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, TypeVar
from dataclasses import dataclass, field
from datetime import datetime
import os
import time
import functools
from enum import Enum

# Python 3.9 compatibility for ParamSpec
try:
    from typing import ParamSpec
except ImportError:
    from typing_extensions import ParamSpec

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import yaml

# 导入API配置管理器
try:
    from src.utils.api_config import get_api_config
except ImportError:
    def get_api_config():
        return None

# 类型提示
P = ParamSpec('P')
T = TypeVar('T')


class AgentStatus(Enum):
    """Agent 执行状态"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RETRY = "retry"


@dataclass
class AgentMetrics:
    """Agent 执行指标"""
    agent_name: str
    start_time: datetime = None
    end_time: datetime = None
    duration_seconds: float = 0.0
    status: AgentStatus = AgentStatus.PENDING
    retry_count: int = 0
    error_message: str = ""
    llm_calls: int = 0
    total_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "status": self.status.value,
            "retry_count": self.retry_count,
            "error_message": self.error_message,
            "llm_calls": self.llm_calls,
            "total_tokens": self.total_tokens
        }


def with_retry(max_retries: int = 3, delay: float = 2.0, backoff: float = 2.0):
    """
    重试装饰器

    Args:
        max_retries: 最大重试次数
        delay: 初始延迟（秒）
        backoff: 延迟增长因子
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(self, *args: P.args, **kwargs: P.kwargs) -> T:
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries + 1):
                try:
                    return func(self, *args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        self.log(f"尝试 {attempt + 1}/{max_retries + 1} 失败: {e}, {current_delay:.1f}秒后重试", "WARNING")
                        self.metrics.retry_count += 1
                        self.metrics.status = AgentStatus.RETRY
                        time.sleep(current_delay)
                        current_delay *= backoff

            # 所有重试都失败
            self.log(f"所有 {max_retries + 1} 次尝试均失败", "ERROR")
            raise last_exception
        return wrapper
    return decorator


class BaseAgent(ABC):
    """
    Agent基类（增强版）

    Features:
    - 自动重试机制
    - 执行指标收集
    - 统一错误处理
    - 日志记录
    """

    # 类级别指标收集
    _global_metrics: Dict[str, AgentMetrics] = {}

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        """
        初始化Agent

        Args:
            config: Agent配置
            prompts: 提示词配置
        """
        self.config = config
        self.prompts = prompts
        self.name = self.__class__.__name__

        # 初始化指标
        self.metrics = AgentMetrics(agent_name=self.name)

        # 初始化LLM
        self.llm = self._init_llm()

        # 检查是否启用了 thinking 模式
        llm_config = self.config.get("llm", {})
        if llm_config.get("provider") == "zhipuai":
            thinking_config = llm_config.get("zhipuai", {}).get("thinking", {})
            if thinking_config.get("enabled", False):
                thinking_type = thinking_config.get("type", "enabled")
                self.log(f"GLM-4.7 Thinking 模式已启用: {thinking_type}", "INFO")

        # 加载系统提示词
        self.system_prompt = self._load_system_prompt()

    def _init_llm(self) -> ChatOpenAI:
        """
        初始化LLM

        Returns:
            ChatOpenAI: LLM实例
        """
        llm_config = self.config.get("llm", {})
        provider = llm_config.get("provider", "zhipuai")

        if provider == "zhipuai":
            zhipuai_config = llm_config.get("zhipuai", {})

            # 使用APIConfigManager获取密钥和端点
            api_config = get_api_config()
            if api_config:
                try:
                    api_key = api_config.get_api_key("zhipuai")
                    base_url = api_config.get_endpoint("llm.zhipuai.base_url")
                except ValueError:
                    # Fallback到环境变量
                    api_key = os.getenv("ZHIPUAI_API_KEY")
                    base_url = "https://open.bigmodel.cn/api/coding/paas/v4/"  # 编码专用端点
                    if not api_key:
                        raise ValueError("请设置环境变量 ZHIPUAI_API_KEY")
            else:
                # Fallback到原有逻辑
                api_key = os.getenv("ZHIPUAI_API_KEY")
                base_url = zhipuai_config.get("base_url", "https://open.bigmodel.cn/api/coding/paas/v4/")  # 编码专用端点
                if not api_key:
                    raise ValueError("请设置环境变量 ZHIPUAI_API_KEY")

            # 基础参数
            llm_kwargs = {
                "openai_api_base": base_url,
                "openai_api_key": api_key,
                "model": zhipuai_config.get("model", "glm-4.7"),
                "temperature": zhipuai_config.get("temperature", 0.7),
                "max_tokens": zhipuai_config.get("max_tokens", 8000),
                "timeout": zhipuai_config.get("timeout", 600)  # 默认10分钟超时
            }

            # Thinking 深度思考模式（GLM-4.7 专属）
            thinking_config = zhipuai_config.get("thinking", {})
            if thinking_config.get("enabled", False):
                thinking_type = thinking_config.get("type", "enabled")
                # 通过 model_kwargs 传递 thinking 参数
                llm_kwargs["model_kwargs"] = {
                    "thinking": {"type": thinking_type}
                }

            return ChatOpenAI(**llm_kwargs)
        elif provider == "openai":
            openai_config = llm_config.get("openai", {})

            # 使用APIConfigManager获取密钥
            api_config = get_api_config()
            if api_config:
                try:
                    api_key = api_config.get_api_key("openai")
                except ValueError:
                    api_key = os.getenv("OPENAI_API_KEY")
                    if not api_key:
                        raise ValueError("请设置环境变量 OPENAI_API_KEY")
            else:
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("请设置环境变量 OPENAI_API_KEY")

            return ChatOpenAI(
                openai_api_key=api_key,
                model=openai_config.get("model", "gpt-4o"),
                temperature=self.config.get("temperature", 0.7),
                max_tokens=self.config.get("max_tokens", 2000)
            )
        else:
            raise ValueError(f"不支持的LLM提供商: {provider}")

    def _load_system_prompt(self) -> str:
        """
        加载系统提示词

        Returns:
            str: 系统提示词
        """
        agent_name = self.name.replace("Agent", "").lower()
        prompts = self.prompts.get("prompts", {})

        # 检查是否有自定义系统提示词
        custom_prompts = self.config.get("prompts", {}).get("custom_system_prompts", {})
        if custom_prompts and custom_prompts.get(agent_name):
            return custom_prompts[agent_name]

        # 从prompts.yaml加载
        agent_prompts = prompts.get(agent_name, {})
        return agent_prompts.get("system", "你是一个专业的小红书内容创作助手。")

    @abstractmethod
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行Agent逻辑

        Args:
            state: 当前工作流状态

        Returns:
            Dict[str, Any]: 更新后的状态
        """
        pass

    def execute_with_metrics(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        带指标收集的执行方法（包装器）

        Args:
            state: 当前工作流状态

        Returns:
            Dict[str, Any]: 更新后的状态
        """
        self.metrics.start_time = datetime.now()
        self.metrics.status = AgentStatus.RUNNING

        try:
            result = self.execute(state)
            self.metrics.status = AgentStatus.SUCCESS
            return result
        except Exception as e:
            self.metrics.status = AgentStatus.FAILED
            self.metrics.error_message = str(e)
            self.log(f"执行失败: {e}", "ERROR")
            raise
        finally:
            self.metrics.end_time = datetime.now()
            self.metrics.duration_seconds = (
                self.metrics.end_time - self.metrics.start_time
            ).total_seconds()

            # 记录到全局指标
            BaseAgent._global_metrics[self.name] = self.metrics
            self.log(f"执行完成，耗时 {self.metrics.duration_seconds:.2f}秒", "INFO")

    def _call_llm(self, user_prompt: str) -> str:
        """
        调用LLM（带指标收集）

        Args:
            user_prompt: 用户提示词

        Returns:
            str: LLM响应
        """
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt)
        ]

        response = self.llm.invoke(messages)
        self.metrics.llm_calls += 1

        # 尝试获取 token 使用量
        if hasattr(response, 'response_metadata'):
            token_usage = response.response_metadata.get('token_usage', {})
            self.metrics.total_tokens += token_usage.get('total_tokens', 0)

        return response.content

    def _call_llm_with_retry(
        self,
        user_prompt: str,
        max_retries: int = 3,
        delay: float = 2.0
    ) -> str:
        """
        带重试的LLM调用

        Args:
            user_prompt: 用户提示词
            max_retries: 最大重试次数
            delay: 重试延迟

        Returns:
            str: LLM响应
        """
        last_exception = None
        current_delay = delay

        for attempt in range(max_retries + 1):
            try:
                return self._call_llm(user_prompt)
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    self.log(f"LLM调用失败 {attempt + 1}/{max_retries + 1}: {e}", "WARNING")
                    self.metrics.retry_count += 1
                    time.sleep(current_delay)
                    current_delay *= 2

        self.log(f"LLM调用全部失败", "ERROR")
        raise last_exception

    def log(self, message: str, level: str = "INFO"):
        """
        日志输出

        Args:
            message: 日志消息
            level: 日志级别
        """
        from loguru import logger
        logger.log(level, f"[{self.name}] {message}")

    def safe_execute(
        self,
        state: Dict[str, Any],
        default_return: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        安全执行（捕获异常，返回默认值）

        Args:
            state: 当前工作流状态
            default_return: 失败时的默认返回值

        Returns:
            Dict[str, Any]: 更新后的状态或默认值
        """
        try:
            return self.execute_with_metrics(state)
        except Exception as e:
            self.log(f"安全执行捕获异常: {e}", "ERROR")
            if default_return is not None:
                return default_return
            return {
                **state,
                "error_message": str(e),
                "failed_agent": self.name
            }

    @classmethod
    def get_global_metrics(cls) -> Dict[str, Dict[str, Any]]:
        """
        获取所有Agent的全局指标

        Returns:
            Dict[str, Dict[str, Any]]: 指标字典
        """
        return {
            name: metrics.to_dict()
            for name, metrics in cls._global_metrics.items()
        }

    @classmethod
    def reset_global_metrics(cls):
        """重置全局指标"""
        cls._global_metrics = {}

    def get_metrics(self) -> AgentMetrics:
        """获取当前Agent的指标"""
        return self.metrics
