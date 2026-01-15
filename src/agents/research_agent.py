"""
研究Agent - 网络搜索获取技术资料，支持多种搜索方式
1. zhipuAI联网搜索（推荐，使用年包）
2. Open-WebSearch MCP（免费，多引擎）
3. Tavily API（付费，需要API Key）
4. Mock模式（离线模拟）
"""

from typing import Dict, Any
from src.agents.base import BaseAgent


class ResearchAgent(BaseAgent):
    """研究Agent - 网络搜索获取技术资料"""

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        super().__init__(config, prompts)
        research_config = config.get("agents", {}).get("research_agent", {})
        self.max_results = research_config.get("max_results", 10)
        self.search_depth = research_config.get("search_depth", "advanced")
        self.mock_mode = research_config.get("mock_mode", False)
        self.search_provider = research_config.get("search_provider", "zhipuai")  # zhipuai, tavily, mock

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行网络搜索研究

        Args:
            state: 当前工作流状态

        Returns:
            Dict[str, Any]: 更新后的状态，包含research_data
        """
        self.log("开始执行网络搜索研究...")

        try:
            # 获取主题
            selected_topic = state.get("selected_ai_topic")
            if not selected_topic:
                raise ValueError("没有找到选中的AI话题")

            topic_title = selected_topic.get("title", "")
            topic_description = selected_topic.get("description", "")

            self.log(f"研究主题: {topic_title}")
            self.log(f"搜索提供商: {self.search_provider}")

            # 根据配置选择搜索方式
            if self.mock_mode:
                self.log("使用Mock模式生成研究数据")
                research_data = self._generate_mock_research(topic_title, topic_description)
            elif self.search_provider == "zhipuai":
                # 使用zhipuAI联网搜索（推荐，包含在年包中）
                research_data = self._search_with_zhipuai(topic_title, topic_description)
            elif self.search_provider == "tavily" and self._has_tavily_key():
                # 使用Tavily API（需要付费）
                research_data = self._search_with_tavily(topic_title, topic_description)
            else:
                # 降级到Mock模式
                self.log("未配置搜索API，使用Mock模式")
                research_data = self._generate_mock_research(topic_title, topic_description)

            self.log(f"研究完成，获取到 {len(research_data.get('sources', []))} 个资料来源")

            return {
                **state,
                "research_data": research_data,
                "research_summary": self._generate_summary(research_data),
                "current_step": "research_completed"
            }

        except Exception as e:
            self.log(f"研究失败: {str(e)}", "ERROR")
            # 失败时返回模拟数据
            selected_topic = state.get("selected_ai_topic", {"title": "AI技术", "description": ""})
            research_data = self._generate_mock_research(
                selected_topic.get("title", "AI技术"),
                selected_topic.get("description", "")
            )
            return {
                **state,
                "research_data": research_data,
                "research_summary": self._generate_summary(research_data),
                "current_step": "research_completed"
            }

    def _has_tavily_key(self) -> bool:
        """检查是否有Tavily API key"""
        import os
        tavily_key = os.environ.get("TAVILY_API_KEY")
        return bool(tavily_key and tavily_key != "your_tavily_api_key_here")

    def _search_with_zhipuai(self, topic: str, description: str) -> Dict[str, Any]:
        """
        使用zhipuAI联网搜索（推荐，包含在年包中）
        通过让LLM直接搜索并整理信息
        """
        try:
            # 构建搜索查询
            query = f"{topic} {description}".strip()
            self.log(f"zhipuAI联网搜索查询: {query}")

            # 使用LLM进行联网搜索（zhipuAI内置web_search工具）
            prompt = f"""请联网搜索关于"{query}"的技术资料，并提供以下信息：

1. **技术背景**：该技术的发展历史和现状
2. **核心特性**：主要功能和技术特点
3. **应用场景**：实际应用领域和案例
4. **发展趋势**：未来发展方向和挑战

请搜索以下类型的资源：
- 官方文档和GitHub仓库
- 技术博客和教程
- Stack Overflow和开发者社区
- 学术论文和案例分析

请以JSON格式返回搜索结果：
{{
  "query": "{query}",
  "sources": [
    {{
      "title": "资源标题",
      "url": "链接地址",
      "content": "内容摘要（200字以内）"
    }}
  ],
  "detailed_info": {{
    "background": "技术背景...",
    "core_features": "核心特性...",
    "use_cases": "应用场景...",
    "trends": "发展趋势..."
  }}
}}
"""

            # 调用LLM（zhipuAI会自动使用web_search工具）
            response = self._call_llm(prompt)

            # 解析JSON响应
            import json
            try:
                # 尝试直接解析
                research_data = json.loads(response)
            except json.JSONDecodeError:
                # 如果解析失败，尝试提取JSON部分
                import re
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    research_data = json.loads(json_match.group())
                else:
                    # 完全失败，返回Mock数据
                    self.log("zhipuAI搜索结果解析失败，使用Mock模式", "WARNING")
                    return self._generate_mock_research(topic, description)

            # 验证数据结构
            if "sources" not in research_data:
                research_data["sources"] = []
            if "detailed_info" not in research_data:
                research_data["detailed_info"] = {}

            self.log(f"zhipuAI联网搜索成功，获取到 {len(research_data.get('sources', []))} 个资料来源")
            return research_data

        except Exception as e:
            self.log(f"zhipuAI联网搜索失败: {e}", "ERROR")
            # 降级到Mock模式
            return self._generate_mock_research(topic, description)

    def _search_with_tavily(self, topic: str, description: str) -> Dict[str, Any]:
        """使用Tavily API进行搜索"""
        try:
            from tavily import TavilyClient

            api_key = self._get_api_key("tavily")
            client = TavilyClient(api_key=api_key)

            # 构建搜索查询
            query = f"{topic} {description}".strip()
            self.log(f"Tavily搜索查询: {query}")

            # 执行搜索
            search_result = client.search(
                query=query,
                search_depth=self.search_depth,
                max_results=self.max_results,
                include_raw_content=True,
                include_domains=["github.com", "stackoverflow.com", "medium.com", "dev.to", "arxiv.org"]
            )

            # 提取关键信息
            research_data = {
                "query": query,
                "sources": [],
                "detailed_info": {
                    "background": "",
                    "core_features": "",
                    "use_cases": "",
                    "trends": ""
                }
            }

            # 处理搜索结果
            if search_result.get("results"):
                for result in search_result["results"][:self.max_results]:
                    research_data["sources"].append({
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "content": result.get("content", "")[:1000]  # 限制内容长度
                    })

                # 使用LLM总结搜索结果
                research_data["detailed_info"] = self._summarize_search_results(research_data)

            return research_data

        except Exception as e:
            self.log(f"Tavily搜索失败: {e}", "ERROR")
            # 降级到Mock模式
            return self._generate_mock_research(topic, description)

    def _summarize_search_results(self, research_data: Dict[str, Any]) -> Dict[str, Any]:
        """使用LLM总结搜索结果"""
        try:
            # 构建搜索结果摘要
            sources_text = "\n".join([
                f"- {s['title']}: {s['content'][:200]}..."
                for s in research_data.get("sources", [])[:5]
            ])

            prompt = f"""基于以下搜索结果，提取技术背景信息：

**搜索来源**：
{sources_text}

请提取并分类以下信息：
1. **技术背景**：该技术的发展历史和现状
2. **核心特性**：主要功能和技术特点
3. **应用场景**：实际应用领域和案例
4. **发展趋势**：未来发展方向和挑战

请以JSON格式返回：
{{
  "background": "技术背景...",
  "core_features": "核心特性...",
  "use_cases": "应用场景...",
  "trends": "发展趋势..."
}}
"""

            response = self._call_llm(prompt)

            # 尝试解析JSON
            import json
            try:
                parsed = json.loads(response)
                return parsed
            except:
                # JSON解析失败，返回默认结构
                return {
                    "background": response[:500] if len(response) > 500 else response,
                    "core_features": response[500:1000] if len(response) > 1000 else "",
                    "use_cases": response[1000:1500] if len(response) > 1500 else "",
                    "trends": response[1500:2000] if len(response) > 2000 else ""
                }

        except Exception as e:
            self.log(f"总结搜索结果失败: {e}", "ERROR")
            return {
                "background": "技术背景信息",
                "core_features": "核心功能特性",
                "use_cases": "应用场景说明",
                "trends": "发展趋势分析"
            }

    def _generate_mock_research(self, topic: str, description: str) -> Dict[str, Any]:
        """生成模拟研究数据"""
        # 基于主题生成模拟的研究数据
        return {
            "query": f"{topic} {description}",
            "sources": [
                {
                    "title": f"{topic} - 官方文档",
                    "url": "https://example.com/docs",
                    "content": f"{topic}是一个重要的技术领域，具有广泛的应用前景。"
                },
                {
                    "title": f"{topic} - 技术深度解析",
                    "url": "https://example.com/deep-dive",
                    "content": f"{description} - 该技术在行业内得到了广泛应用，具有很高的研究价值。"
                },
                {
                    "title": f"{topic} - 实践指南",
                    "url": "https://example.com/guide",
                    "content": f"{topic}的实践应用需要考虑多个因素，包括性能、安全性和可扩展性。"
                }
            ],
            "detailed_info": {
                "background": f"{topic}是当前AI领域的重要技术方向。该技术经历了快速的发展，从早期的理论研究到现在的实际应用，已经在多个行业产生了深远的影响。",
                "core_features": f"{topic}的核心特性包括高效的处理能力、灵活的架构设计、强大的扩展性，以及与现有系统的良好兼容性。这些特性使其成为解决复杂问题的理想选择。",
                "use_cases": f"{topic}广泛应用于代码开发、系统架构、数据分析、自动化测试等多个领域。在实际应用中，它能够显著提升开发效率和系统性能。",
                "trends": f"未来，{topic}将继续向智能化、自动化方向发展。随着技术的不断进步，我们可以预期更多的创新应用和更高的性能表现。"
            }
        }

    def _generate_summary(self, research_data: Dict[str, Any]) -> str:
        """生成研究摘要"""
        sources = research_data.get("sources", [])
        details = research_data.get("detailed_info", {})

        summary = f"**研究主题**: {research_data.get('query', '')}\n\n"
        summary += f"**资料来源**: 共获取{len(sources)}个技术资料\n\n"
        summary += f"**核心发现**: \n"
        summary += f"- 技术背景: {details.get('background', '')[:100]}...\n"
        summary += f"- 核心特性: {details.get('core_features', '')[:100]}...\n"
        summary += f"- 应用场景: {details.get('use_cases', '')[:100]}...\n"

        return summary
