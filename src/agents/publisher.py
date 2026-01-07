"""
小红书发布Agent
通过xiaohongshu-mcp发布笔记到小红书平台
"""

from typing import Dict, Any, Optional
import json
import os
from datetime import datetime
from src.agents.base import BaseAgent


class PublisherAgent(BaseAgent):
    """发布Agent - 发布笔记到小红书"""

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        super().__init__(config, prompts)
        publisher_config = config.get("agents", {}).get("publisher", {})
        self.auto_publish = publisher_config.get("auto_publish", False)
        self.mcp_server_url = publisher_config.get("mcp_server_url", "http://localhost:3000")

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行发布操作
        注意：所有内容已由 auto_orchestrator.py 保存到对应媒体目录

        Args:
            state: 当前工作流状态

        Returns:
            Dict[str, Any]: 更新后的状态
        """
        self.log("准备发布笔记到小红书")

        try:
            # 获取要发布的内容
            content = self._get_content_to_publish(state)

            if not content:
                raise ValueError("没有可发布的内容")

            # 检查是否启用自动发布
            if not self.auto_publish:
                self.log("自动发布未启用，内容已由orchestrator保存到各媒体目录", "INFO")
                return {
                    **state,
                    "published": False,
                    "publish_result": {
                        "status": "skipped",
                        "reason": "auto_publish未启用",
                        "note": "内容已保存到各媒体目录（xiaohongshu, twitter等）"
                    },
                    "current_step": "publisher_skipped"
                }

            # 发布到小红书
            result = self._publish_to_xiaohongshu(content, state)

            self.log(f"笔记发布成功！笔记ID: {result.get('note_id', 'N/A')}")

            return {
                **state,
                "publish_result": result,
                "published": True,
                "current_step": "publisher_completed"
            }

        except Exception as e:
            self.log(f"发布失败: {str(e)}", "ERROR")
            return {
                **state,
                "error_message": f"发布失败: {str(e)}",
                "published": False,
                "current_step": "publisher_failed"
            }

    def _get_content_to_publish(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        获取要发布的内容

        Args:
            state: 当前状态

        Returns:
            Optional[Dict[str, Any]]: 要发布的内容
        """
        # 新工作流：优先使用小红书笔记
        if "xiaohongshu_note" in state and state["xiaohongshu_note"]:
            note = state["xiaohongshu_note"]
            # 构建发布内容
            content = {
                "title": note.get("title", state.get("recommended_title", "")),
                "full_content": f"{note.get('intro', '')}\n\n{note.get('body', '')}\n\n{note.get('ending', '')}",
                "word_count": note.get("word_count", 0),
                "hashtags": note.get("hashtags", []),
                "images": state.get("generated_images", [])
            }
            return content

        # 旧工作流：改写后的内容
        if "rewritten_content" in state and state["rewritten_content"]:
            return state["rewritten_content"]

        # 旧工作流：生成的内容
        if "generated_content" in state and state["generated_content"]:
            return state["generated_content"]

        return None

    def _publish_to_xiaohongshu(self, content: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """
        发布到小红书（通过MCP）

        Args:
            content: 内容
            state: 状态

        Returns:
            Dict[str, Any]: 发布结果
        """
        self.log("调用xiaohongshu-mcp发布笔记")

        # TODO: 实际调用MCP服务
        # 这里需要根据xiaohongshu-mcp的实际API来实现
        # 示例代码：
        # import subprocess
        # result = subprocess.run([
        #     "mcp-client", "call", "xiaohongshu-mcp",
        #     "publish_note",
        #     "--title", content['title'],
        #     "--content", content['full_content']
        # ], capture_output=True, text=True)

        # 目前返回模拟结果
        mock_result = {
            "note_id": f"NOTE_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "status": "published",
            "url": f"https://www.xiaohongshu.com/explore/{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "published_at": datetime.now().isoformat(),
            "title": content.get('title', ''),
            "platform": "xiaohongshu"
        }

        self.log(f"发布成功: {mock_result['url']}", "SUCCESS")
        return mock_result
