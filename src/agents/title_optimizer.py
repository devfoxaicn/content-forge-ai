"""
标题优化Agent
生成爆款标题
"""

from typing import Dict, Any, List
import re
from src.agents.base import BaseAgent


class TitleOptimizerAgent(BaseAgent):
    """标题优化Agent"""

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        super().__init__(config, prompts)
        title_config = config.get("agents", {}).get("title_optimizer", {})
        self.num_titles = title_config.get("num_titles", 10)
        self.use_emojis = title_config.get("use_emojis", True)

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行标题优化

        Args:
            state: 当前工作流状态

        Returns:
            Dict[str, Any]: 更新后的状态
        """
        self.log("开始生成优化标题")

        try:
            # 获取长文本文章或小红书笔记（适配新工作流）
            longform = state.get("longform_article", {})
            xiaohongshu = state.get("xiaohongshu_note", {})

            # 优先使用小红书笔记，其次长文本文章
            if xiaohongshu and xiaohongshu.get("title"):
                content = xiaohongshu
                content_type = "小红书笔记"
            elif longform and longform.get("title"):
                content = longform
                content_type = "长文本文章"
            else:
                raise ValueError("没有找到可用的内容（需要longform_article或xiaohongshu_note）")

            self.log(f"基于{content_type}生成优化标题: {content.get('title', '')}")

            # 构建提示词
            user_prompt = self._build_prompt(state, content)

            # 调用LLM生成标题
            response = self._call_llm(user_prompt)

            # 解析响应
            titles = self._parse_titles(response)

            # 选择最佳标题（选择第一个作为推荐）
            recommended_title = titles[0] if titles else content.get("title", "未命名标题")

            self.log(f"成功生成 {len(titles)} 个标题，推荐: {recommended_title}")

            return {
                **state,
                "optimized_titles": titles,
                "recommended_title": recommended_title,
                "current_step": "title_optimizer_completed"
            }
        except Exception as e:
            self.log(f"标题优化失败: {str(e)}", "ERROR")
            # 失败时返回原内容标题
            longform = state.get("longform_article", {})
            xiaohongshu = state.get("xiaohongshu_note", {})
            fallback_title = xiaohongshu.get("title") or longform.get("title") or "未命名标题"

            return {
                **state,
                "optimized_titles": [fallback_title],
                "recommended_title": fallback_title,
                "error_message": f"标题优化失败: {str(e)}",
                "current_step": "title_optimizer_failed"
            }

    def _build_prompt(self, state: Dict[str, Any], content: Dict[str, Any]) -> str:
        """
        构建提示词

        Args:
            state: 当前状态
            content: 生成的内容

        Returns:
            str: 提示词
        """
        prompts = self.prompts.get("prompts", {})
        prompt_template = prompts.get("title_optimizer", {}).get("user", "")

        # 提取核心价值点
        selected_idea = state.get("selected_idea", {})
        value_proposition = selected_idea.get("value_proposition", "实用的AI工具分享")

        return prompt_template.format(
            topic=state["topic"],
            target_audience=state["target_audience"],
            value_proposition=value_proposition,
            num_titles=self.num_titles
        )

    def _parse_titles(self, response: str) -> List[str]:
        """
        解析LLM响应为标题列表

        Args:
            response: LLM响应

        Returns:
            List[str]: 标题列表
        """
        titles = []

        # 尝试按行分割
        lines = response.split("\n")

        for line in lines:
            line = line.strip()

            # 跳过空行和标题行
            if not line or line.startswith(("标题", "==")):
                continue

            # 移除序号
            line = re.sub(r'^\d+[\.\、]\s*', '', line)
            line = re.sub(r'^[\-\●]\s*', '', line)

            # 移除括号中的说明（如 "（点击率：高）"）
            line = re.sub(r'\([^)]*\)', '', line)
            line = re.sub(r'（[^）]*）', '', line)

            # 清理并添加
            line = line.strip()
            if line and len(line) > 5 and len(line) < 50:  # 标题长度合理
                titles.append(line)

            if len(titles) >= self.num_titles:
                break

        # 如果没有找到足够的标题，返回原内容标题
        if not titles:
            titles = ["AI工具实战：让效率翻倍的秘密武器"]

        return titles
