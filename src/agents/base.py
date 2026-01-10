"""
基础Agent类
所有Agent都继承自这个基类
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import yaml

# 导入API配置管理器
try:
    from src.utils.api_config import get_api_config
except ImportError:
    # 如果导入失败，提供一个fallback
    def get_api_config():
        return None


class BaseAgent(ABC):
    """Agent基类"""

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

    def _call_llm(self, user_prompt: str) -> str:
        """
        调用LLM

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
        return response.content

    def log(self, message: str, level: str = "INFO"):
        """
        日志输出

        Args:
            message: 日志消息
            level: 日志级别
        """
        from loguru import logger
        logger.log(level, f"[{self.name}] {message}")
