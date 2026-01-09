"""
API配置管理工具

统一管理所有API端点和密钥，避免硬编码和重复代码。
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class APIConfigManager:
    """
    API配置管理器
    
    功能：
    1. 从YAML文件加载API端点配置
    2. 统一管理API密钥的获取和验证
    3. 提供URL模板替换功能
    """
    
    def __init__(self, config_path: str = "config/api_endpoints.yaml"):
        """
        初始化配置管理器
        
        Args:
            config_path: API端点配置文件路径
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        加载配置文件
        
        Returns:
            Dict[str, Any]: 配置字典
        """
        if not self.config_path.exists():
            return {}
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    
    def get_endpoint(self, key_path: str, **kwargs) -> str:
        """
        获取API端点URL
        
        Args:
            key_path: 点分隔的配置路径 (如 "llm.zhipuai.base_url")
            **kwargs: URL模板变量
            
        Returns:
            str: 完整URL
            
        Example:
            api_config = APIConfigManager()
            url = api_config.get_endpoint("data_sources.hackernews.base_url")
            url = api_config.get_endpoint("data_sources.hackernews.item", id=12345)
        """
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            value = value.get(key, {})
        if isinstance(value, str):
            return value.format(**kwargs)
        return value
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """
        获取API密钥（统一管理密钥检查）
        
        Args:
            provider: 提供商名称 (zhipuai, openai, gemini, tavily, newsapi, reddit)
            
        Returns:
            Optional[str]: API密钥
            
        Raises:
            ValueError: 如果环境变量未设置
        """
        key_mapping = {
            "zhipuai": "ZHIPUAI_API_KEY",
            "openai": "OPENAI_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "stability": "STABILITY_API_KEY",
            "tavily": "TAVILY_API_KEY",
            "newsapi": "NEWSAPI_KEY",
            "reddit_client_id": "REDDIT_CLIENT_ID",
            "reddit_client_secret": "REDDIT_CLIENT_SECRET",
            "reddit_user_agent": "REDDIT_USER_AGENT",
        }
        
        env_var = key_mapping.get(provider)
        if not env_var:
            raise ValueError(f"未知的提供商: {provider}")
        
        api_key = os.getenv(env_var)
        if not api_key:
            raise ValueError(f"请设置环境变量 {env_var}")
        return api_key
    
    def has_api_key(self, provider: str) -> bool:
        """
        检查API密钥是否存在
        
        Args:
            provider: 提供商名称
            
        Returns:
            bool: 是否存在密钥
        """
        try:
            self.get_api_key(provider)
            return True
        except ValueError:
            return False


# 全局单例实例
_api_config_instance = None

def get_api_config() -> APIConfigManager:
    """
    获取全局API配置管理器实例
    
    Returns:
        APIConfigManager: 配置管理器实例
    """
    global _api_config_instance
    if _api_config_instance is None:
        _api_config_instance = APIConfigManager()
    return _api_config_instance
