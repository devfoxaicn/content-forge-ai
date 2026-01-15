"""
可视化生成Agent - 识别内容并生成Mermaid图表
纯Python实现，无需额外API成本
"""

import re
from typing import Dict, Any, List, Optional
from src.agents.base import BaseAgent


class VisualizationGeneratorAgent(BaseAgent):
    """可视化生成Agent - 为文章生成Mermaid图表"""

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        super().__init__(config, prompts)
        self.generate_mermaid = config.get("generate_mermaid", True)
        self.min_diagrams = config.get("min_diagrams", 3)
        self.diagram_types = ["flowchart", "sequence", "graph", "class", "pie"]

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行可视化生成

        Args:
            state: 当前工作流状态，包含longform_article

        Returns:
            更新后的状态，包含visualization_result
        """
        self.log("开始生成可视化内容...")

        try:
            # 获取长文本文章
            article = state.get("longform_article", {})
            content = article.get("full_content", "")
            title = article.get("title", "")

            if not content:
                self.log("未找到文章内容，跳过可视化生成", "WARNING")
                return {**state, "visualization_result": None}

            # 识别可视化机会
            opportunities = self._identify_visualization_opportunities(content)

            if not opportunities:
                self.log("未发现适合可视化的内容", "INFO")
                return {
                    **state,
                    "visualization_result": {
                        "total_diagrams": 0,
                        "diagrams": [],
                        "summary": "无需可视化"
                    }
                }

            self.log(f"识别到 {len(opportunities)} 个可视化机会")

            # 生成Mermaid图表
            diagrams = []
            for opp in opportunities[:self.min_diagrams]:
                diagram = self._generate_diagram(opp, content)
                if diagram:
                    diagrams.append(diagram)

            # 生成增强版文章（包含图表）
            enhanced_content = self._insert_diagrams_into_content(content, diagrams)

            # 构建结果
            result = {
                "total_diagrams": len(diagrams),
                "diagrams": diagrams,
                "enhanced_content": enhanced_content,
                "summary": f"生成 {len(diagrams)} 个图表"
            }

            self.log(f"可视化生成完成，生成 {len(diagrams)} 个图表")

            return {**state, "visualization_result": result}

        except Exception as e:
            self.log(f"可视化生成失败: {e}", "ERROR")
            return {
                **state,
                "visualization_result": {
                    "total_diagrams": 0,
                    "error": str(e)
                }
            }

    def _identify_visualization_opportunities(self, content: str) -> List[Dict[str, Any]]:
        """识别适合可视化的内容"""
        opportunities = []

        # 1. 识别架构图机会
        arch_keywords = ['架构', 'architecture', '组件', 'module', '服务', 'service', '系统', 'system']
        arch_sections = self._find_sections_with_keywords(content, arch_keywords)
        for section in arch_sections:
            opportunities.append({
                "type": "architecture",
                "diagram_type": "graph",
                "section": section,
                "priority": "high"
            })

        # 2. 识别流程图机会
        flow_keywords = ['流程', 'flow', '步骤', 'step', '过程', 'process', '工作流', 'workflow']
        flow_sections = self._find_sections_with_keywords(content, flow_keywords)
        for section in flow_sections:
            opportunities.append({
                "type": "flowchart",
                "diagram_type": "flowchart",
                "section": section,
                "priority": "high"
            })

        # 3. 识别序列图机会
        seq_keywords = ['交互', 'interaction', '调用', 'call', '请求', 'request', '响应', 'response']
        seq_sections = self._find_sections_with_keywords(content, seq_keywords)
        for section in seq_sections:
            opportunities.append({
                "type": "sequence",
                "diagram_type": "sequence",
                "section": section,
                "priority": "medium"
            })

        # 4. 识别对比图机会
        comparison_keywords = ['对比', 'comparison', '比较', 'versus', 'vs', '差异', 'difference']
        comp_sections = self._find_sections_with_keywords(content, comparison_keywords)
        for section in comp_sections:
            opportunities.append({
                "type": "comparison",
                "diagram_type": "graph",
                "section": section,
                "priority": "medium"
            })

        # 5. 识别类图机会
        class_keywords = ['类', 'class', '对象', 'object', '接口', 'interface', '继承', 'inherit']
        class_sections = self._find_sections_with_keywords(content, class_keywords)
        for section in class_sections:
            opportunities.append({
                "type": "class",
                "diagram_type": "class",
                "section": section,
                "priority": "low"
            })

        # 按优先级排序
        priority_order = {"high": 0, "medium": 1, "low": 2}
        opportunities.sort(key=lambda x: priority_order.get(x["priority"], 3))

        return opportunities

    def _find_sections_with_keywords(self, content: str, keywords: List[str]) -> List[str]:
        """查找包含特定关键词的章节"""
        sections = []
        lines = content.split('\n')
        current_section = []
        current_section_title = ""

        for line in lines:
            # 检查是否是标题
            if line.startswith('#'):
                # 保存上一节
                if current_section_title and current_section:
                    section_text = '\n'.join(current_section)
                    # 检查是否包含关键词
                    if any(kw in section_text.lower() for kw in keywords):
                        sections.append(f"{current_section_title}\n{section_text[:500]}")

                # 开始新节
                current_section_title = line
                current_section = []
            else:
                current_section.append(line)

        # 检查最后一节
        if current_section_title and current_section:
            section_text = '\n'.join(current_section)
            if any(kw in section_text.lower() for kw in keywords):
                sections.append(f"{current_section_title}\n{section_text[:500]}")

        return sections

    def _generate_diagram(self, opportunity: Dict[str, Any], content: str) -> Optional[Dict[str, Any]]:
        """生成单个图表"""
        diagram_type = opportunity["diagram_type"]
        section = opportunity["section"]

        if diagram_type == "graph":
            return self._generate_graph_diagram(section)
        elif diagram_type == "flowchart":
            return self._generate_flowchart_diagram(section)
        elif diagram_type == "sequence":
            return self._generate_sequence_diagram(section)
        elif diagram_type == "class":
            return self._generate_class_diagram(section)
        else:
            return None

    def _generate_graph_diagram(self, section: str) -> Optional[Dict[str, Any]]:
        """生成架构图（graph LR/TD）"""
        # 提取关键组件
        lines = section.split('\n')
        nodes = []
        edges = []

        # 简化的组件提取逻辑
        for line in lines:
            # 查找可能的组件名称（中文或英文）
            if '组件' in line or '服务' in line or '模块' in line:
                # 提取关键词作为节点
                words = re.findall(r'[\u4e00-\u9fa5]+|[A-Za-z]+', line)
                if words:
                    node_name = words[0] if len(words) > 0 else "Node"
                    if len(node_name) >= 2 and node_name not in nodes:
                        nodes.append(node_name)

            # 查找关系
            if '→' in line or '->' in line or '调用' in line:
                words = re.findall(r'[\u4e00-\u9fa5]+|[A-Za-z]+', line)
                if len(words) >= 2:
                    edges.append((words[0], words[1]))

        # 如果没有找到足够信息，使用通用模板
        if len(nodes) < 2:
            return {
                "type": "architecture",
                "mermaid_code": self._get_default_architecture_diagram(),
                "title": "系统架构图",
                "description": "系统架构概览"
            }

        # 生成Mermaid代码
        mermaid_code = "graph TB\n"
        for i, node in enumerate(nodes[:5]):  # 限制节点数量
            node_id = f"N{i}"
            mermaid_code += f"    {node_id}[{node}]\n"

        # 添加边
        for i, (src, dst) in enumerate(edges[:5]):
            mermaid_code += f"    N{i} --> N{i+1}\n"

        return {
            "type": "architecture",
            "mermaid_code": mermaid_code,
            "title": "架构图",
            "description": "系统组件关系"
        }

    def _generate_flowchart_diagram(self, section: str) -> Optional[Dict[str, Any]]:
        """生成流程图"""
        # 提取步骤
        step_keywords = ['第一步', '第二步', '首先', '然后', '接下来', '最后', '第一步是', '第二步是']
        lines = section.split('\n')
        steps = []

        for line in lines:
            for kw in step_keywords:
                if kw in line:
                    # 提取步骤描述
                    step_desc = re.sub(kw, '', line).strip()
                    step_desc = re.sub(r'[:：]', '', step_desc)
                    if step_desc:
                        steps.append(step_desc)
                    break

        # 如果没有找到步骤，使用通用流程
        if len(steps) < 2:
            return {
                "type": "flowchart",
                "mermaid_code": self._get_default_flowchart_diagram(),
                "title": "流程图",
                "description": "处理流程"
            }

        # 生成Mermaid流程图
        mermaid_code = "flowchart TD\n"
        for i, step in enumerate(steps[:5]):
            step_id = f"S{i}"
            mermaid_code += f"    {step_id}[{step[:20]}...]\n"
            if i < len(steps) - 1:
                mermaid_code += f"    {step_id} --> S{i+1}\n"

        return {
            "type": "flowchart",
            "mermaid_code": mermaid_code,
            "title": "流程图",
            "description": "操作流程"
        }

    def _generate_sequence_diagram(self, section: str) -> Optional[Dict[str, Any]]:
        """生成序列图"""
        # 提取参与者
        participant_keywords = ['用户', '系统', '服务', '客户端', '服务器', 'API', '数据库']
        lines = section.split('\n')
        participants = set()

        for line in lines:
            for kw in participant_keywords:
                if kw in line:
                    participants.add(kw)

        # 如果参与者太少，使用默认
        if len(participants) < 2:
            participants = ['用户', '系统', '服务', '数据库']

        # 生成Mermaid序列图
        mermaid_code = "sequenceDiagram\n"
        for i, participant in enumerate(list(participants)[:4]):
            mermaid_code += f"    participant P{i} as {participant}\n"

        # 添加交互（简化版）
        if len(participants) >= 2:
            parts = list(participants)
            mermaid_code += f"    {parts[0]}->>{parts[1]}: 请求\n"
            mermaid_code += f"    {parts[1]}-->>{parts[0]}: 响应\n"

        return {
            "type": "sequence",
            "mermaid_code": mermaid_code,
            "title": "时序图",
            "description": "交互流程"
        }

    def _generate_class_diagram(self, section: str) -> Optional[Dict[str, Any]]:
        """生成类图"""
        # 简化版类图
        return {
            "type": "class",
            "mermaid_code": self._get_default_class_diagram(),
            "title": "类图",
            "description": "类结构"
        }

    def _get_default_architecture_diagram(self) -> str:
        """获取默认架构图"""
        return """graph TB
    A[用户] --> B[API网关]
    B --> C[服务层]
    C --> D[业务逻辑]
    C --> E[数据处理]
    D --> F[数据库]
    E --> F
    F --> G[缓存层]
    G --> A"""

    def _get_default_flowchart_diagram(self) -> str:
        """获取默认流程图"""
        return """flowchart TD
    A[开始] --> B[数据处理]
    B --> C{验证}
    C -->|通过| D[保存]
    C -->|失败| E[错误处理]
    D --> F[结束]
    E --> F"""

    def _get_default_class_diagram(self) -> str:
        """获取默认类图"""
        return """classDiagram
    class BaseModel {
        +train()
        +predict()
        +evaluate()
    }
    class DataLoader {
        +load()
        +transform()
    }
    class Trainer {
        +fit()
        +validate()
    }
    BaseModel --> DataLoader
    BaseModel --> Trainer"""

    def _insert_diagrams_into_content(self, content: str, diagrams: List[Dict[str, Any]]) -> str:
        """将图表插入到文章内容中"""
        if not diagrams:
            return content

        # 在相关章节后插入图表
        enhanced_content = content
        lines = content.split('\n')

        for diagram in diagrams:
            diagram_type = diagram["type"]
            diagram_title = diagram["title"]
            mermaid_code = diagram["mermaid_code"]

            # 找到合适的插入位置
            insert_index = -1
            for i, line in enumerate(lines):
                # 在包含相关关键词的段落后插入
                if diagram_type == "architecture" and ('架构' in line or '组件' in line):
                    # 找到该段落的结尾
                    for j in range(i, min(i + 10, len(lines))):
                        if lines[j].strip() == "" or j == len(lines) - 1:
                            insert_index = j + 1
                            break
                    break
                elif diagram_type == "flowchart" and ('流程' in line or '步骤' in line):
                    for j in range(i, min(i + 10, len(lines))):
                        if lines[j].strip() == "" or j == len(lines) - 1:
                            insert_index = j + 1
                            break
                    break

            # 如果找到合适位置，插入图表
            if insert_index > 0 and insert_index < len(lines):
                diagram_md = f"\n\n**{diagram_title}**\n\n```mermaid\n{mermaid_code}\n```\n\n"
                lines.insert(insert_index, diagram_md)
                break  # 只插入第一个图表，避免重复

        return '\n'.join(lines)
