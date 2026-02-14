"""
内容质量检查器
逐期检查 ML Series 文章质量，生成质量报告
"""

import os
import re
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class QualityReport:
    """单篇文章质量报告"""
    episode: int
    series_id: str
    title: str
    file_path: str
    word_count: int
    code_blocks: int
    headers: int
    references: int
    has_introduction: bool
    has_core_principle: bool
    has_practice: bool
    has_summary: bool
    score: float
    grade: str
    issues: List[str]
    recommendations: List[str]
    needs_improvement: bool


class ContentQualityChecker:
    """内容质量检查器"""

    # 质量标准
    STANDARDS = {
        "min_word_count": 8000,
        "ideal_word_count": 12000,
        "min_code_blocks": 3,
        "ideal_code_blocks": 5,
        "min_headers": 8,
        "ideal_headers": 12,
        "min_references": 3,
        "ideal_references": 5,
        "passing_score": 80  # 用户设定的优秀线
    }

    # 权重配置
    WEIGHTS = {
        "content_depth": 0.25,
        "code_quality": 0.20,
        "structure": 0.20,
        "completeness": 0.20,
        "references": 0.15
    }

    def __init__(self, base_path: str = "/Users/z/Documents/work/content-forge-ai/data/series/ML_series"):
        self.base_path = Path(base_path)

    def check_episode(self, episode: int, series_id: Optional[str] = None) -> Optional[QualityReport]:
        """
        检查指定期的文章质量

        Args:
            episode: 期号 (1-100)
            series_id: 系列ID（可选，自动检测）

        Returns:
            QualityReport 或 None（如果文章不存在）
        """
        # 查找文章文件
        article_path = self._find_article(episode, series_id)
        if not article_path:
            return None

        # 读取文章内容
        with open(article_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 提取元数据
        title = self._extract_title(content)
        detected_series = self._detect_series(article_path)

        # 计算各项指标
        word_count = len(content)
        code_blocks = len(re.findall(r'```[\s\S]*?```', content))
        headers = len(re.findall(r'^#{1,3}\s+.+$', content, re.MULTILINE))
        references = len(re.findall(r'\[.*?\]\(https?://', content))

        # 检查必要章节（更灵活的匹配）
        has_introduction = bool(re.search(r'#+\s*(\d+\.?\s*)?(第.*章.*引言|引言|简介|背景|概述|导读)', content))
        has_core_principle = bool(re.search(r'#+\s*(\d+\.?\s*)?(第.*章.*原理|第.*章.*核心|原理|核心|架构|基础|概念|技术架构|技术原理)', content))
        has_practice = bool(re.search(r'#+\s*(\d+\.?\s*)?(第.*章.*实践|第.*章.*实现|第.*章.*代码|实践|实现|代码|应用|示例|实战|代码示例)', content))
        has_summary = bool(re.search(r'#+\s*(\d+\.?\s*)?(第.*章.*总结|第.*章.*结论|总结|结论|展望|小结|结语)', content))

        # 计算分数
        score, issues = self._calculate_score(
            word_count, code_blocks, headers, references,
            has_introduction, has_core_principle, has_practice, has_summary
        )

        # 生成等级
        grade = self._get_grade(score)

        # 生成改进建议
        recommendations = self._generate_recommendations(issues)

        return QualityReport(
            episode=episode,
            series_id=detected_series,
            title=title,
            file_path=str(article_path),
            word_count=word_count,
            code_blocks=code_blocks,
            headers=headers,
            references=references,
            has_introduction=has_introduction,
            has_core_principle=has_core_principle,
            has_practice=has_practice,
            has_summary=has_summary,
            score=score,
            grade=grade,
            issues=issues,
            recommendations=recommendations,
            needs_improvement=score < self.STANDARDS["passing_score"]
        )

    def _find_article(self, episode: int, series_id: Optional[str]) -> Optional[Path]:
        """查找文章文件"""
        ep_str = f"episode_{episode:03d}"

        # 如果指定了series_id，直接查找
        if series_id:
            series_path = self.base_path / series_id / ep_str
            if series_path.exists():
                articles = list(series_path.glob("*_article.md"))
                if articles:
                    return max(articles, key=lambda p: p.stat().st_size)

        # 否则搜索所有系列
        for series_dir in sorted(self.base_path.iterdir()):
            if not series_dir.is_dir():
                continue
            ep_dir = series_dir / ep_str
            if ep_dir.exists():
                articles = list(ep_dir.glob("*_article.md"))
                if articles:
                    return max(articles, key=lambda p: p.stat().st_size)

        return None

    def _detect_series(self, article_path: Path) -> str:
        """从路径检测系列ID"""
        parts = article_path.parts
        for part in parts:
            if part.startswith("ml_series_"):
                return part
        return "unknown"

    def _extract_title(self, content: str) -> str:
        """提取文章标题"""
        match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        return match.group(1) if match else "未命名文章"

    def _calculate_score(
        self,
        word_count: int,
        code_blocks: int,
        headers: int,
        references: int,
        has_intro: bool,
        has_core: bool,
        has_practice: bool,
        has_summary: bool
    ) -> tuple:
        """计算质量分数"""
        issues = []
        score = 0

        # 1. 内容深度 (25%)
        if word_count >= self.STANDARDS["ideal_word_count"]:
            depth_score = 25
        elif word_count >= self.STANDARDS["min_word_count"]:
            depth_score = 20
        else:
            depth_score = int(word_count / self.STANDARDS["min_word_count"] * 20)
            issues.append(f"❌ 字数不足: {word_count:,}字 (建议{self.STANDARDS['min_word_count']:,}+)")
        score += depth_score

        # 2. 代码质量 (20%)
        if code_blocks >= self.STANDARDS["ideal_code_blocks"]:
            code_score = 20
        elif code_blocks >= self.STANDARDS["min_code_blocks"]:
            code_score = 15
        elif code_blocks > 0:
            code_score = 10
            issues.append(f"⚠️ 代码示例偏少: {code_blocks}个 (建议{self.STANDARDS['ideal_code_blocks']}+)")
        else:
            code_score = 0
            issues.append(f"❌ 缺少代码示例")
        score += code_score

        # 3. 结构完整性 (20%)
        if headers >= self.STANDARDS["ideal_headers"]:
            struct_score = 20
        elif headers >= self.STANDARDS["min_headers"]:
            struct_score = 15
        else:
            struct_score = int(headers / self.STANDARDS["min_headers"] * 15)
            issues.append(f"❌ 章节结构不足: {headers}个 (建议{self.STANDARDS['ideal_headers']}+)")
        score += struct_score

        # 4. 完整性 (20%)
        completeness = sum([has_intro, has_core, has_practice, has_summary])
        complete_score = completeness * 5

        if not has_intro:
            issues.append("❌ 缺少引言章节")
        if not has_core:
            issues.append("❌ 缺少核心原理章节")
        if not has_practice:
            issues.append("❌ 缺少实践应用章节")
        if not has_summary:
            issues.append("❌ 缺少总结章节")

        score += complete_score

        # 5. 引用来源 (15%)
        if references >= self.STANDARDS["ideal_references"]:
            ref_score = 15
        elif references >= self.STANDARDS["min_references"]:
            ref_score = 10
        elif references > 0:
            ref_score = 5
            issues.append(f"⚠️ 引用来源偏少: {references}个 (建议{self.STANDARDS['ideal_references']}+)")
        else:
            ref_score = 0
            issues.append(f"❌ 缺少引用来源")
        score += ref_score

        return score, issues

    def _get_grade(self, score: float) -> str:
        """获取等级"""
        if score >= 90:
            return "A+ (优秀)"
        elif score >= 85:
            return "A (良好)"
        elif score >= 80:
            return "B+ (较好)"
        elif score >= 75:
            return "B (达标)"
        elif score >= 70:
            return "C+ (及格)"
        elif score >= 60:
            return "C (勉强)"
        else:
            return "D (不及格)"

    def _generate_recommendations(self, issues: List[str]) -> List[str]:
        """生成改进建议"""
        recommendations = []

        for issue in issues:
            if "字数不足" in issue:
                recommendations.append("📝 增加内容深度：补充理论解释、案例分析或扩展讨论")
            elif "代码示例" in issue:
                recommendations.append("💻 添加代码示例：增加可运行的代码片段和注释说明")
            elif "章节结构" in issue:
                recommendations.append("📚 完善章节结构：增加子章节，细化内容组织")
            elif "引言" in issue:
                recommendations.append("📖 添加引言：说明背景、学习目标和前置知识")
            elif "核心原理" in issue:
                recommendations.append("🔬 补充核心原理：详细解释算法/模型的工作机制")
            elif "实践应用" in issue:
                recommendations.append("🛠️ 添加实践应用：包含代码示例、使用场景和最佳实践")
            elif "总结" in issue:
                recommendations.append("📋 添加总结：回顾要点、展望未来方向")
            elif "引用来源" in issue:
                recommendations.append("🔗 添加引用：补充论文、官方文档等技术参考链接")

        return recommendations

    def print_report(self, report: QualityReport):
        """打印质量报告"""
        print(f"\n{'='*60}")
        print(f"📊 第 {report.episode:03d} 期质量报告")
        print(f"{'='*60}")
        print(f"标题: {report.title}")
        print(f"系列: {report.series_id}")
        print(f"文件: {report.file_path}")
        print(f"\n📈 基础指标:")
        print(f"  • 字数: {report.word_count:,}")
        print(f"  • 代码块: {report.code_blocks}")
        print(f"  • 章节数: {report.headers}")
        print(f"  • 引用数: {report.references}")

        print(f"\n📋 章节完整性:")
        print(f"  • 引言: {'✅' if report.has_introduction else '❌'}")
        print(f"  • 核心原理: {'✅' if report.has_core_principle else '❌'}")
        print(f"  • 实践应用: {'✅' if report.has_practice else '❌'}")
        print(f"  • 总结: {'✅' if report.has_summary else '❌'}")

        print(f"\n{'='*60}")
        print(f"🎯 总分: {report.score:.1f}/100 ({report.grade})")
        print(f"{'='*60}")

        if report.needs_improvement:
            print(f"\n⚠️  需要改进 (低于{self.STANDARDS['passing_score']}分)")
            print(f"\n❌ 发现的问题:")
            for issue in report.issues:
                print(f"  {issue}")
            print(f"\n💡 改进建议:")
            for rec in report.recommendations:
                print(f"  {rec}")
        else:
            print(f"\n✅ 质量达标！")

        print(f"\n")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="ML Series 内容质量检查")
    parser.add_argument("--episode", type=int, help="检查指定期号")
    parser.add_argument("--start", type=int, default=1, help="起始期号")
    parser.add_argument("--end", type=int, default=100, help="结束期号")
    parser.add_argument("--report-only", action="store_true", help="仅生成报告，不提示改进")
    parser.add_argument("--json", type=str, help="输出JSON报告到文件")

    args = parser.parse_args()

    checker = ContentQualityChecker()

    if args.episode:
        # 检查单期
        report = checker.check_episode(args.episode)
        if report:
            checker.print_report(report)
        else:
            print(f"❌ 未找到第 {args.episode} 期的文章")
    else:
        # 检查范围
        reports = []
        needs_improvement = []

        for ep in range(args.start, args.end + 1):
            report = checker.check_episode(ep)
            if report:
                reports.append(report)
                if report.needs_improvement:
                    needs_improvement.append(report)

        # 打印摘要
        print(f"\n{'='*60}")
        print(f"📊 ML Series 质量检查摘要")
        print(f"{'='*60}")
        print(f"检查范围: 第{args.start}-{args.end}期")
        print(f"总文章数: {len(reports)}")
        print(f"达标文章: {len(reports) - len(needs_improvement)}")
        print(f"待改进: {len(needs_improvement)}")

        if needs_improvement:
            print(f"\n⚠️  以下文章需要改进:")
            for r in needs_improvement:
                print(f"  • 第{r.episode:03d}期: {r.score:.1f}分 - {r.title[:30]}...")

        # 输出JSON报告
        if args.json:
            with open(args.json, 'w', encoding='utf-8') as f:
                json.dump([asdict(r) for r in reports], f, ensure_ascii=False, indent=2)
            print(f"\n📄 JSON报告已保存到: {args.json}")


if __name__ == "__main__":
    main()
