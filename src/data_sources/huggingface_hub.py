"""
Hugging Face Hub API - 模型和数据集数据源
API文档: https://huggingface.co/docs/hub/en/api
"""
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any
from loguru import logger
from src.utils.time_filter import TimeFilter


class HuggingFaceAPI:
    """Hugging Face Hub API客户端"""

    def __init__(self, api_token: str = None):
        """
        初始化Hugging Face API客户端

        Args:
            api_token: API token（可选，某些操作需要认证）
        """
        self.base_url = "https://huggingface.co/api"
        self.headers = {}
        if api_token:
            self.headers["Authorization"] = f"Bearer {api_token}"

    def get_recent_models(
        self,
        limit: int = 50,
        days_ago: int = 1
    ) -> List[Dict[str, Any]]:
        """
        获取最近的模型

        Args:
            limit: 返回数量限制
            days_ago: 搜索最近几天的模型

        Returns:
            模型列表
        """
        params = {
            "limit": limit,
            "sort": "createdAt",
            "direction": "desc"
        }

        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=self.headers,
                params=params,
                timeout=30
            )
            response.raise_for_status()

            models = response.json()

            # 过滤最近N天的模型
            time_filter = TimeFilter(hours=days_ago * 24)
            filtered_models = []

            for model in models:
                created_at = model.get("createdAt", "")
                if created_at and time_filter.is_within_time_window(created_at):
                    filtered_models.append(self._normalize_model(model))

            logger.info(f"Hugging Face: 获取到 {len(models)} 个模型，24小时内 {len(filtered_models)} 个")
            return filtered_models

        except Exception as e:
            logger.error(f"Hugging Face API调用失败: {e}")
            return []

    def get_models_by_task(
        self,
        task: str = "text-generation",
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        按任务获取模型

        Args:
            task: 任务类型
            limit: 返回数量限制

        Returns:
            模型列表
        """
        params = {
            "limit": limit,
            "task": task,
            "sort": "downloads",
            "direction": "desc"
        }

        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=self.headers,
                params=params,
                timeout=30
            )
            response.raise_for_status()

            models = response.json()

            # 只返回最近7天的
            time_filter = TimeFilter(hours=7 * 24)
            filtered_models = []

            for model in models:
                created_at = model.get("createdAt", "")
                if created_at and time_filter.is_within_time_window(created_at):
                    filtered_models.append(self._normalize_model(model))

            return filtered_models

        except Exception as e:
            logger.error(f"Hugging Face按任务获取模型失败: {e}")
            return []

    def get_recent_datasets(
        self,
        limit: int = 30,
        days_ago: int = 1
    ) -> List[Dict[str, Any]]:
        """
        获取最近的数据集

        Args:
            limit: 返回数量限制
            days_ago: 搜索最近几天的数据集

        Returns:
            数据集列表
        """
        params = {
            "limit": limit,
            "sort": "createdAt",
            "direction": "desc"
        }

        try:
            response = requests.get(
                f"{self.base_url}/datasets",
                headers=self.headers,
                params=params,
                timeout=30
            )
            response.raise_for_status()

            datasets = response.json()

            # 过滤最近N天的数据集
            time_filter = TimeFilter(hours=days_ago * 24)
            filtered_datasets = []

            for dataset in datasets:
                created_at = dataset.get("createdAt", "")
                if created_at and time_filter.is_within_time_window(created_at):
                    filtered_datasets.append(self._normalize_dataset(dataset))

            logger.info(f"Hugging Face: 获取到 {len(datasets)} 个数据集，24小时内 {len(filtered_datasets)} 个")
            return filtered_datasets

        except Exception as e:
            logger.error(f"Hugging Face数据集API调用失败: {e}")
            return []

    def _normalize_model(self, model: Dict[str, Any]) -> Dict[str, Any]:
        """标准化模型数据格式"""
        model_id = model.get("modelId", "")
        author = model.get("author", "")
        likes = model.get("likes", 0)
        downloads = model.get("downloads", 0)

        # 提取任务标签
        tags = model.get("tags", [])
        task_tags = [tag for tag in tags if tag in [
            "text-generation", "text-classification", "question-answering",
            "translation", "summarization", "image-classification",
            "object-detection", "audio-classification", "automatic-speech-recognition"
        ]]

        return {
            "id": f"hf-{model_id}",
            "title": f"{author}/{model_id}",
            "description": model.get("cardData", {}).get("text", "")[:500] or f"Model by {author}",
            "url": f"https://huggingface.co/{model_id}",
            "published_at": model.get("createdAt", ""),
            "source": "Hugging Face",
            "category": "dev_tools",
            "metadata": {
                "model_id": model_id,
                "author": author,
                "likes": likes,
                "downloads": downloads,
                "tasks": task_tags,
                "pipeline_tag": model.get("pipeline_tag", ""),
                "library_name": model.get("cardData", {}).get("library_name", ""),
                "type": "model"
            }
        }

    def _normalize_dataset(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """标准化数据集数据格式"""
        dataset_id = dataset.get("id", "")
        author = dataset.get("author", "")
        downloads = dataset.get("downloads", 0)
        likes = dataset.get("likes", 0)

        return {
            "id": f"hf-dataset-{dataset_id}",
            "title": f"{author}/{dataset_id}",
            "description": dataset.get("cardData", {}).get("text", "")[:500] or f"Dataset by {author}",
            "url": f"https://huggingface.co/datasets/{dataset_id}",
            "published_at": dataset.get("createdAt", ""),
            "source": "Hugging Face",
            "category": "dev_tools",
            "metadata": {
                "dataset_id": dataset_id,
                "author": author,
                "downloads": downloads,
                "likes": likes,
                "type": "dataset"
            }
        }


def create_huggingface_client(api_token: str = None) -> HuggingFaceAPI:
    """
    创建Hugging Face API客户端的工厂函数

    Args:
        api_token: API token（可选）

    Returns:
        HuggingFaceAPI实例
    """
    return HuggingFaceAPI(api_token=api_token)
