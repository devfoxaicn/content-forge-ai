"""
SEO优化Agent
为生成的文章添加SEO优化元素，提升搜索引擎排名

Features:
- 自动生成 meta description
- 优化标题结构
- 添加结构化数据
- 关键词密度分析
- 内链建议
"""

from typing import Dict, Any, List, Optional
import re
import json
from src.agents.base import BaseAgent


class SEOOptimizerAgent(BaseAgent):
    """SEO优化Agent"""

    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        super().__init__(config, prompts)
        self.min_title_length = 50
        self.max_title_length = 60
        self.meta_description_length = (120, 160)
        self.keyword_density_target = 0.02  # 2%

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行SEO优化

        Args:
            state: 包含 longform_article 的状态

        Returns:
            更新后的状态，包含 SEO 元数据
        """
        self.log("开始SEO优化...")

        article = state.get("longform_article", "")
        if not article:
            self.log("警告：未找到文章内容", "WARNING")
            return state

        # 提取或生成标题
        title = self._extract_title(article)
        optimized_title = self._optimize_title(title)

        # 生成 meta description
        meta_description = self._generate_meta_description(article)

        # 关键词分析
        keywords = self._extract_keywords(article, state.get("current_topic", {}).get("keywords", []))
        keyword_analysis = self._analyze_keyword_density(article, keywords)

        # 生成结构化数据
        structured_data = self._generate_structured_data(
            title=optimized_title,
            description=meta_description,
            keywords=keywords,
            article=article
        )

        # 内链建议
        internal_links = self._suggest_internal_links(state.get("episode", 0), keywords)

        # SEO 评分
        seo_score = self._calculate_seo_score(
            title=optimized_title,
            description=meta_description,
            keyword_analysis=keyword_analysis,
            article_length=len(article)
        )

        seo_metadata = {
            "optimized_title": optimized_title,
            "meta_description": meta_description,
            "keywords": keywords,
            "keyword_density": keyword_analysis,
            "structured_data": structured_data,
            "internal_links_suggestions": internal_links,
            "seo_score": seo_score,
            "recommendations": self._generate_recommendations(seo_score, keyword_analysis)
        }

        self.log(f"SEO优化完成，评分: {seo_score}/100")

        return {
            **state,
            "seo_metadata": seo_metadata
        }

    def _extract_title(self, article: str) -> str:
        """从文章中提取标题"""
        lines = article.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('# ') and not line.startswith('## '):
                return line[2:].strip()
        return "未命名文章"

    def _optimize_title(self, title: str) -> str:
        """
        优化标题

        规则:
        - 长度在 50-60 字符之间
        - 包含关键信息
        - 吸引点击
        """
        # 如果标题太短，添加副标题
        if len(title) < self.min_title_length:
            # 使用 LLM 生成优化标题
            prompt = f"""请优化以下技术文章标题，使其更具吸引力和SEO友好：

原标题：{title}

要求：
1. 长度控制在 50-60 字符
2. 包含核心技术关键词
3. 使用数字或疑问句增加点击率
4. 保持专业性和准确性

请直接输出优化后的标题："""
            optimized = self._call_llm(prompt).strip()
            return optimized[:self.max_title_length]

        # 如果标题太长，截断
        if len(title) > self.max_title_length:
            return title[:self.max_title_length-3] + "..."

        return title

    def _generate_meta_description(self, article: str) -> str:
        """
        生成 meta description

        要求:
        - 长度 120-160 字符
        - 包含核心关键词
        - 吸引用户点击
        """
        # 提取文章摘要（前500字符）
        summary = article[:500].replace('\n', ' ').strip()

        prompt = f"""请为以下技术文章生成一个SEO友好的 meta description：

文章摘要：
{summary}

要求：
1. 长度严格控制在 120-160 字符之间
2. 包含文章的核心关键词
3. 使用吸引人的语言，鼓励点击
4. 不要使用引号或特殊字符
5. 直接输出描述文本，不要任何前缀

Meta Description："""

        description = self._call_llm(prompt).strip()

        # 确保长度符合要求
        if len(description) < self.meta_description_length[0]:
            description += " 了解更多技术细节和实践经验。"
        elif len(description) > self.meta_description_length[1]:
            description = description[:self.meta_description_length[1]-3] + "..."

        return description

    def _extract_keywords(self, article: str, existing_keywords: List[str]) -> List[str]:
        """
        提取和优化关键词
        """
        # 合并现有关键词和从文章中提取的关键词
        all_keywords = set(existing_keywords)

        # 使用 LLM 提取额外关键词
        prompt = f"""请从以下技术文章中提取 5-8 个核心关键词：

文章内容（前1000字符）：
{article[:1000]}

要求：
1. 选择最具代表性和搜索价值的关键词
2. 优先选择技术术语和专业名词
3. 避免过于宽泛的词汇

请直接输出关键词，用逗号分隔："""

        extracted = self._call_llm(prompt).strip()
        new_keywords = [k.strip() for k in extracted.split(',') if k.strip()]

        all_keywords.update(new_keywords)

        return list(all_keywords)[:10]  # 最多返回10个关键词

    def _analyze_keyword_density(self, article: str, keywords: List[str]) -> Dict[str, float]:
        """
        分析关键词密度
        """
        total_words = len(article.split())
        density = {}

        for keyword in keywords:
            # 计算关键词出现次数
            count = len(re.findall(re.escape(keyword), article, re.IGNORECASE))
            density[keyword] = {
                "count": count,
                "density": count / total_words if total_words > 0 else 0
            }

        return density

    def _generate_structured_data(
        self,
        title: str,
        description: str,
        keywords: List[str],
        article: str
    ) -> Dict[str, Any]:
        """
        生成 Schema.org 结构化数据
        """
        # 估算阅读时间（假设每分钟阅读 500 字）
        word_count = len(article)
        reading_time = max(1, word_count // 500)

        structured_data = {
            "@context": "https://schema.org",
            "@type": "TechArticle",
            "headline": title,
            "description": description,
            "keywords": ", ".join(keywords),
            "wordCount": word_count,
            "timeRequired": f"PT{reading_time}M",  # ISO 8601 duration
            "inLanguage": "zh-CN",
            "author": {
                "@type": "Organization",
                "name": "ContentForge AI"
            },
            "publisher": {
                "@type": "Organization",
                "name": "ContentForge AI",
                "logo": {
                    "@type": "ImageObject",
                    "url": "https://example.com/logo.png"
                }
            }
        }

        return structured_data

    def _suggest_internal_links(self, current_episode: int, keywords: List[str]) -> List[Dict[str, str]]:
        """
        生成内链建议
        """
        suggestions = []

        # 基于关键词推荐相关章节
        keyword_to_episode = {
            "transformer": [1, 14],
            "attention": [1, 14],
            "bert": [2, 3],
            "gpt": [3, 4],
            "diffusion": [96],
            "gan": [20],
            "rlhf": [4],
            "rag": [11, 12, 13, 14, 15, 16, 17, 18],
            "agent": [19, 20, 21, 22, 23, 24, 25, 26],
            "cnn": [12, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
            "nlp": [31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
            "推荐": [51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
        }

        for keyword in keywords:
            keyword_lower = keyword.lower()
            for key, episodes in keyword_to_episode.items():
                if key in keyword_lower:
                    for ep in episodes:
                        if ep != current_episode:
                            suggestions.append({
                                "keyword": keyword,
                                "target_episode": ep,
                                "anchor_text": f"第{ep}期相关内容"
                            })

        return suggestions[:5]  # 最多5个建议

    def _calculate_seo_score(
        self,
        title: str,
        description: str,
        keyword_analysis: Dict[str, Dict],
        article_length: int
    ) -> int:
        """
        计算 SEO 评分 (0-100)
        """
        score = 0

        # 标题评分 (30分)
        if self.min_title_length <= len(title) <= self.max_title_length:
            score += 30
        elif len(title) < self.min_title_length:
            score += 15
        else:
            score += 20

        # 描述评分 (25分)
        min_desc, max_desc = self.meta_description_length
        if min_desc <= len(description) <= max_desc:
            score += 25
        elif len(description) < min_desc:
            score += 10
        else:
            score += 15

        # 关键词密度评分 (25分)
        good_density_count = 0
        for kw, data in keyword_analysis.items():
            density = data["density"]
            if 0.01 <= density <= 0.03:  # 1-3% 是理想密度
                good_density_count += 1

        if good_density_count >= 3:
            score += 25
        elif good_density_count >= 1:
            score += 15
        else:
            score += 5

        # 文章长度评分 (20分)
        if article_length >= 8000:
            score += 20
        elif article_length >= 5000:
            score += 15
        elif article_length >= 3000:
            score += 10
        else:
            score += 5

        return min(100, score)

    def _generate_recommendations(
        self,
        seo_score: int,
        keyword_analysis: Dict[str, Dict]
    ) -> List[str]:
        """
        生成 SEO 优化建议
        """
        recommendations = []

        if seo_score < 70:
            recommendations.append("建议增加文章长度至8000字以上，提升内容深度")

        low_density_keywords = [
            kw for kw, data in keyword_analysis.items()
            if data["density"] < 0.01
        ]
        if low_density_keywords:
            recommendations.append(f"关键词密度过低: {', '.join(low_density_keywords[:3])}，建议在文章中适当增加出现频率")

        high_density_keywords = [
            kw for kw, data in keyword_analysis.items()
            if data["density"] > 0.05
        ]
        if high_density_keywords:
            recommendations.append(f"关键词密度过高: {', '.join(high_density_keywords[:3])}，可能被搜索引擎判定为关键词堆砌")

        if seo_score >= 80:
            recommendations.append("SEO优化良好，文章结构完整，关键词分布合理")

        return recommendations
