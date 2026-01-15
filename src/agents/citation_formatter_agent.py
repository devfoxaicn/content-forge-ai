"""
引用格式化Agent - 识别引用并统一格式（GB/T 7714, APA, IEEE）
纯Python实现，无需额外API成本
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from src.agents.base import BaseAgent


class CitationFormatterAgent(BaseAgent):
    """引用格式化Agent - 统一文章引用格式"""

    # 支持的引用格式
    SUPPORTED_STYLES = {
        "gb7714": "GB/T 7714-2015（中国国家标准）",
        "apa": "APA（美国心理学会）",
        "ieee": "IEEE（电气电子工程师学会）"
    }

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        super().__init__(config, prompts)
        self.citation_style = config.get("citation_style", "gb7714")  # 默认使用GB/T 7714
        self.auto_detect = config.get("auto_detect", True)
        self.generate_reference_list = config.get("generate_reference_list", True)

        # 引用标记模式
        self.citation_patterns = [
            r'\[(\d+)\]',  # [1], [2]
            r'\[([A-Za-z]+\s+et?\s*al\.?,?\s*\d{4})\]',  # [Author et al., 2023]
            r'\(([A-Za-z]+\s+et?\s*al\.?,?\s*\d{4})\)',  # (Author et al., 2023)
            r'\[([A-Za-z]+\s+\d{4})\]',  # [Author 2023]
        ]

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行引用格式化

        Args:
            state: 当前工作流状态，包含longform_article

        Returns:
            更新后的状态，包含citation_formatter_result
        """
        self.log("开始引用格式化...")

        try:
            # 获取长文本文章
            article = state.get("longform_article", {})
            content = article.get("full_content", "")
            title = article.get("title", "")

            if not content:
                self.log("未找到文章内容，跳过引用格式化", "WARNING")
                return {**state, "citation_formatter_result": None}

            # 1. 识别文内引用
            inline_citations = self._extract_inline_citations(content)

            # 2. 识别参考文献信息
            references = self._extract_reference_section(content)

            # 3. 标准化参考文献格式
            formatted_references = []
            if references:
                formatted_references = self._format_references(references, self.citation_style)

            # 4. 验证引用完整性
            validation_result = self._validate_citations(inline_citations, formatted_references)

            # 5. 生成增强版文章（更新引用格式）
            enhanced_content = self._enhance_citation_format(content, inline_citations, formatted_references)

            # 6. 生成参考文献列表（如果文章中没有）
            reference_list_section = ""
            if self.generate_reference_list and not self._has_reference_section(content):
                reference_list_section = self._generate_reference_list_section(formatted_references)

            # 构建结果
            result = {
                "citation_style": self.citation_style,
                "style_name": self.SUPPORTED_STYLES.get(self.citation_style, self.citation_style),
                "inline_citation_count": len(inline_citations),
                "reference_count": len(formatted_references),
                "inline_citations": inline_citations,
                "references": formatted_references,
                "validation": validation_result,
                "enhanced_content": enhanced_content,
                "reference_list_section": reference_list_section,
                "meets_threshold": validation_result.get("all_citations_valid", False),
                "summary": self._generate_summary(len(inline_citations), len(formatted_references), validation_result)
            }

            self.log(f"引用格式化完成，识别 {len(inline_citations)} 个文内引用，{len(formatted_references)} 条参考文献")

            return {**state, "citation_formatter_result": result}

        except Exception as e:
            self.log(f"引用格式化失败: {e}", "ERROR")
            return {
                **state,
                "citation_formatter_result": {
                    "error": str(e),
                    "inline_citation_count": 0,
                    "reference_count": 0
                }
            }

    def _extract_inline_citations(self, content: str) -> List[Dict[str, Any]]:
        """提取文内引用"""
        citations = []

        for pattern in self.citation_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                citation_text = match.group(0)
                citation_id = match.group(1) if match.groups() else ""

                # 获取上下文
                start = max(0, match.start() - 50)
                end = min(len(content), match.end() + 50)
                context = content[start:end]

                citations.append({
                    "text": citation_text,
                    "id": citation_id,
                    "position": match.start(),
                    "context": context.strip(),
                    "type": self._classify_citation_type(citation_text)
                })

        # 去重（按位置）
        seen_positions = set()
        unique_citations = []
        for citation in citations:
            if citation["position"] not in seen_positions:
                seen_positions.add(citation["position"])
                unique_citations.append(citation)

        return unique_citations

    def _classify_citation_type(self, citation: str) -> str:
        """分类引用类型"""
        if re.match(r'^\[\d+\]$', citation):
            return "numeric"
        elif "et al" in citation.lower():
            return "author_year_et al"
        elif re.search(r'[A-Za-z]+\s+\d{4}', citation):
            return "author_year"
        else:
            return "unknown"

    def _extract_reference_section(self, content: str) -> List[str]:
        """提取参考文献部分"""
        references = []

        # 查找参考文献章节
        ref_section_patterns = [
            r'##+\s*参考文献\s*\n+(.*?)(?=##+|\Z)',
            r'##+\s*References\s*\n+(.*?)(?=##+|\Z)',
            r'##+\s*引用\s*\n+(.*?)(?=##+|\Z)',
        ]

        for pattern in ref_section_patterns:
            match = re.search(pattern, content, re.DOTALL | re.MULTILINE)
            if match:
                ref_text = match.group(1).strip()
                # 分割为单独的引用条目
                ref_items = re.split(r'^\d+\.?\s*|^\[?\d+\]?\s*|^-\s*', ref_text, flags=re.MULTILINE)
                references = [ref.strip() for ref in ref_items if ref.strip() and len(ref.strip()) > 10]
                break

        return references

    def _has_reference_section(self, content: str) -> bool:
        """检查是否已有参考文献章节"""
        return bool(re.search(r'##+\s*(参考文献|References|引用)', content, re.IGNORECASE))

    def _format_references(self, references: List[str], style: str) -> List[Dict[str, Any]]:
        """格式化参考文献"""
        formatted_refs = []

        for idx, ref in enumerate(references, 1):
            try:
                # 解析引用信息
                parsed = self._parse_reference(ref)

                # 根据格式标准重新格式化
                if style == "gb7714":
                    formatted = self._format_gb7714(parsed)
                elif style == "apa":
                    formatted = self._format_apa(parsed)
                elif style == "ieee":
                    formatted = self._format_ieee(parsed)
                else:
                    formatted = ref  # 不支持的格式，保持原样

                formatted_refs.append({
                    "index": idx,
                    "original": ref,
                    "formatted": formatted,
                    "parsed": parsed,
                    "valid": bool(parsed.get("authors") or parsed.get("title"))
                })
            except Exception as e:
                # 解析失败，保留原样
                formatted_refs.append({
                    "index": idx,
                    "original": ref,
                    "formatted": ref,
                    "parsed": {},
                    "valid": False,
                    "error": str(e)
                })

        return formatted_refs

    def _parse_reference(self, ref: str) -> Dict[str, Any]:
        """解析参考文献信息"""
        parsed = {
            "authors": [],
            "year": "",
            "title": "",
            "source": "",
            "url": "",
            "type": "unknown"
        }

        # 提取年份
        year_match = re.search(r'\((\d{4})\)|(\d{4})[.,]', ref)
        if year_match:
            parsed["year"] = year_match.group(1) or year_match.group(2)

        # 提取URL
        url_match = re.search(r'https?://[^\s\)]+', ref)
        if url_match:
            parsed["url"] = url_match.group(0)

        # 提取标题（通常在年份后面或URL前面）
        title_patterns = [
            r'(?:\(\d{4}\)\.?\s*|\d{4}[.,]\s*)(.+?)(?:\.|,|http)',
            r'[:\[]\s*(.+?)(?:\.|,|http|\[)',
        ]
        for pattern in title_patterns:
            title_match = re.search(pattern, ref)
            if title_match:
                parsed["title"] = title_match.group(1).strip()
                break

        # 提取作者（在年份或标题之前）
        author_patterns = [
            r'^([A-Z][a-z]+\s+et\s+al\.?)',
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+et\s+al\.?)?)',
            r'^([A-Za-z\u4e00-\u9fa5]+(?:\s+[A-Za-z\u4e00-\u9fa5]+)*(?:\s+等)?)',
        ]
        for pattern in author_patterns:
            author_match = re.match(pattern, ref)
            if author_match:
                parsed["authors"] = [author_match.group(1)]
                break

        # 识别文献类型
        if "arxiv" in ref.lower():
            parsed["type"] = "preprint"
        elif "github" in ref.lower():
            parsed["type"] = "software"
        elif parsed["url"] and not parsed["authors"]:
            parsed["type"] = "webpage"
        elif parsed["authors"] and parsed["year"]:
            parsed["type"] = "article"

        return parsed

    def _format_gb7714(self, parsed: Dict[str, Any]) -> str:
        """格式化为GB/T 7714-2015格式"""
        authors = parsed.get("authors", [""])
        year = parsed.get("year", "")
        title = parsed.get("title", "")
        url = parsed.get("url", "")

        author_str = authors[0] if authors else ""
        if "et al" in author_str.lower() or "等" in author_str:
            author_str = author_str.replace(" et al.", "，等").replace(" 等", "，等")

        parts = []
        if author_str:
            parts.append(f"{author_str}. ")
        if year:
            parts.append(f"{year}. ")
        if title:
            parts.append(f"{title}. ")
        if url:
            parts.append(f"URL: {url}")

        return "".join(parts) if parts else "引用格式无法解析"

    def _format_apa(self, parsed: Dict[str, Any]) -> str:
        """格式化为APA格式"""
        authors = parsed.get("authors", [""])
        year = parsed.get("year", "n.d.")
        title = parsed.get("title", "")
        url = parsed.get("url", "")

        author_str = authors[0] if authors else "Anonymous"
        if "et al" in author_str.lower() or "等" in author_str:
            author_str = author_str.replace(" et al.", " et al.")

        return f"{author_str} ({year}). {title}. Retrieved from {url}" if url else f"{author_str} ({year}). {title}."

    def _format_ieee(self, parsed: Dict[str, Any]) -> str:
        """格式化为IEEE格式"""
        authors = parsed.get("authors", [""])
        title = parsed.get("title", "")
        url = parsed.get("url", "")

        author_str = authors[0] if authors else ""
        if "et al" in author_str.lower():
            author_str = author_str.replace(" et al.", " et al.")

        parts = []
        if author_str:
            parts.append(author_str)
        if title:
            parts.append(f'"{title}"')
        if url:
            parts.append(f"[Online]. Available: {url}")

        return ", ".join(parts) if parts else "引用格式无法解析"

    def _validate_citations(
        self,
        inline_citations: List[Dict[str, Any]],
        references: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """验证引用完整性"""
        issues = []
        missing_refs = []

        # 检查文内引用是否都有对应的参考文献
        ref_numbers = set()
        for ref in references:
            ref_numbers.add(ref["index"])

        cited_numbers = set()
        for citation in inline_citations:
            citation_id = citation.get("id", "")
            if citation_id.isdigit():
                cited_numbers.add(int(citation_id))

        # 查找缺失的参考文献
        missing_refs = cited_numbers - ref_numbers
        if missing_refs:
            issues.append({
                "type": "缺失参考文献",
                "severity": "high",
                "description": f"文内引用的参考文献缺失: {', '.join(map(str, sorted(missing_refs)))}",
                "suggestion": "添加缺失的参考文献条目"
            })

        # 查找未引用的参考文献
        unused_refs = ref_numbers - cited_numbers
        if unused_refs:
            issues.append({
                "type": "未引用参考文献",
                "severity": "low",
                "description": f"以下参考文献未被引用: {', '.join(map(str, sorted(unused_refs)))}",
                "suggestion": "删除未使用的参考文献或在正文中添加引用"
            })

        return {
            "all_citations_valid": len(missing_refs) == 0,
            "missing_references": sorted(missing_refs),
            "unused_references": sorted(unused_refs),
            "issues": issues
        }

    def _enhance_citation_format(
        self,
        content: str,
        inline_citations: List[Dict[str, Any]],
        references: List[Dict[str, Any]]
    ) -> str:
        """增强引用格式"""
        enhanced = content

        # 如果有格式化的参考文献，替换原文中的参考文献部分
        if references:
            # 构建新的参考文献部分
            ref_section_title = "## 参考文献"
            new_ref_section = f"\n\n{ref_section_title}\n\n"

            for ref in references:
                if ref["valid"]:
                    new_ref_section += f"{ref['index']}. {ref['formatted']}\n"
                else:
                    new_ref_section += f"{ref['index']}. {ref['original']}\n"

            # 替换或添加参考文献部分
            if self._has_reference_section(content):
                # 替换现有参考文献部分
                ref_patterns = [
                    r'##+\s*参考文献\s*\n+.*?(?=##+|\Z)',
                    r'##+\s*References\s*\n+.*?(?=##+|\Z)',
                ]
                for pattern in ref_patterns:
                    enhanced = re.sub(
                        pattern,
                        new_ref_section.strip() + "\n\n",
                        enhanced,
                        flags=re.DOTALL | re.MULTILINE,
                        count=1
                    )
                    if "参考文献" in enhanced or "References" in enhanced:
                        break

        return enhanced

    def _generate_reference_list_section(self, references: List[Dict[str, Any]]) -> str:
        """生成参考文献列表章节"""
        if not references:
            return ""

        section = "\n\n## 参考文献\n\n"
        for ref in references:
            if ref["valid"]:
                section += f"{ref['index']}. {ref['formatted']}\n"
            else:
                section += f"{ref['index']}. {ref['original']}\n"

        return section

    def _generate_summary(
        self,
        inline_count: int,
        ref_count: int,
        validation: Dict[str, Any]
    ) -> str:
        """生成引用格式化总结"""
        parts = [
            f"引用格式: {self.SUPPORTED_STYLES.get(self.citation_style, self.citation_style)}",
            f"文内引用: {inline_count} 个",
            f"参考文献: {ref_count} 条"
        ]

        if validation.get("all_citations_valid"):
            parts.append("✅ 引用完整")
        else:
            missing = len(validation.get("missing_references", []))
            if missing > 0:
                parts.append(f"⚠️ 缺失 {missing} 条参考文献")

        return " | ".join(parts)
