"""
测试v9.0的6分类系统
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

from datetime import datetime
from src.agents.trend_categorizer_agent import TrendCategorizerAgent
from src.agents.news_scoring_agent import NewsScoringAgent

# 模拟配置
config = {
    "agents": {
        "trend_categorizer": {
            "max_per_category": 5
        },
        "news_scoring": {
            "max_items": 30,
            "min_per_category": 2,
            "max_per_category": 5,
            "scoring_weights": {
                "source_authority": 30,
                "engagement": 20,
                "freshness": 15,
                "category_balance": 15,
                "content_quality": 10,
                "diversity": 10
            }
        }
    }
}

prompts = {}

print("=" * 60)
print("🧪 测试v9.0的6分类系统")
print("=" * 60)

# 模拟各分类的热点数据
test_trends = {
    "📚 学术前沿": [
        {
            "title": "New Transformer Architecture Achieves SOTA on ImageNet",
            "description": "Researchers from MIT propose a novel transformer architecture that achieves state-of-the-art results on ImageNet benchmark.",
            "url": "https://arxiv.org/abs/2401.12345",
            "source": "arXiv",
            "heat_score": 85,
            "timestamp": datetime.now().isoformat()
        },
        {
            "title": "Semantic Scholar: AI Paper Citations Analysis",
            "description": "Analysis of 10 million AI papers shows rapid growth in multimodal research.",
            "url": "https://semanticscholar.org",
            "source": "Semantic Scholar",
            "heat_score": 78,
            "timestamp": datetime.now().isoformat()
        },
        {
            "title": "OpenAlex: Open Citation Database Launched",
            "description": "The world's largest open citation database now includes 250M papers.",
            "url": "https://openalex.org",
            "source": "OpenAlex",
            "heat_score": 72,
            "timestamp": datetime.now().isoformat()
        },
        {
            "title": "Papers with Code: New GPT-4 Implementation Released",
            "description": "Community releases open-source implementation of GPT-4 architecture.",
            "url": "https://paperswithcode.com",
            "source": "Papers with Code",
            "heat_score": 88,
            "timestamp": datetime.now().isoformat()
        },
        {
            "title": "OpenReview: ICLR 2025 Acceptance Statistics",
            "description": "ICLR 2025 conference accepts 25% of submitted papers, lowest rate ever.",
            "url": "https://openreview.net",
            "source": "OpenReview",
            "heat_score": 65,
            "timestamp": datetime.now().isoformat()
        },
        {
            "title": "DBLP: Computer Science Research Trends 2024",
            "description": "Annual report shows AI/ML papers grew 40% year-over-year.",
            "url": "https://dblp.org",
            "source": "DBLP",
            "heat_score": 60,
            "timestamp": datetime.now().isoformat()
        }
    ],

    "🛠️ 开发工具": [
        {
            "title": "Hugging Face: New LLaMA 3 Model Released",
            "description": "Meta releases LLaMA 3 with 70B parameters, available on Hugging Face Hub.",
            "url": "https://huggingface.co",
            "source": "Hugging Face",
            "heat_score": 95,
            "timestamp": datetime.now().isoformat()
        },
        {
            "title": "PyPI: New LangChain Version 2.0 Released",
            "description": "LangChain 2.0 brings major improvements to agent workflows and memory management.",
            "url": "https://pypi.org",
            "source": "PyPI",
            "heat_score": 82,
            "timestamp": datetime.now().isoformat()
        },
        {
            "title": "npm: Vercel AI SDK v4.0 Published",
            "description": "Vercel releases AI SDK v4 with improved streaming and tool use support.",
            "url": "https://npmjs.com",
            "source": "npm",
            "heat_score": 75,
            "timestamp": datetime.now().isoformat()
        },
        {
            "title": "PyTorch 2.4 Release Notes",
            "description": "PyTorch 2.4 brings 40% speed improvement for transformer models.",
            "url": "https://github.com/pytorch/pytorch",
            "source": "PyTorch",
            "heat_score": 90,
            "timestamp": datetime.now().isoformat()
        },
        {
            "title": "TensorFlow 2.17 Released",
            "description": "Google releases TensorFlow 2.17 with Keras 3.0 integration.",
            "url": "https://github.com/tensorflow",
            "source": "TensorFlow",
            "heat_score": 70,
            "timestamp": datetime.now().isoformat()
        }
    ],

    "🦾 AI Agent": [
        {
            "title": "AutoGPT v2.0: Autonomous Agent Framework Update",
            "description": "AutoGPT v2.0 introduces multi-agent collaboration and self-improvement capabilities.",
            "url": "https://github.com/autogpt",
            "source": "GitHub Trending",
            "heat_score": 88,
            "timestamp": datetime.now().isoformat()
        },
        {
            "title": "Product Hunt: AgentGPT - Build AI Agents in Browser",
            "description": "New tool lets anyone build autonomous AI agents without coding.",
            "url": "https://producthunt.com",
            "source": "Product Hunt",
            "heat_score": 80,
            "timestamp": datetime.now().isoformat()
        },
        {
            "title": "Reddit: LangChain Agent Best Practices Discussion",
            "description": "r/LocalLLaMA discusses best practices for building reliable LangChain agents.",
            "url": "https://reddit.com",
            "source": "Reddit",
            "heat_score": 65,
            "timestamp": datetime.now().isoformat()
        },
        {
            "title": "Hacker News: OpenAI Function Calling Guide",
            "description": "Comprehensive guide on implementing function calling with OpenAI API.",
            "url": "https://news.ycombinator.com",
            "source": "Hacker News",
            "heat_score": 92,
            "timestamp": datetime.now().isoformat()
        },
        {
            "title": "Awesome AI Agents: Curated List Updated",
            "description": "New additions to the awesome list include 50+ agent frameworks.",
            "url": "https://github.com/awesome-ai-agents",
            "source": "Awesome AI Agents",
            "heat_score": 55,
            "timestamp": datetime.now().isoformat()
        }
    ],

    "💼 企业应用": [
        {
            "title": "TechCrunch: Enterprise AI Startup Raises $100M",
            "description": "AI startup focused on enterprise automation raises Series B funding.",
            "url": "https://techcrunch.com",
            "source": "TechCrunch AI",
            "heat_score": 78,
            "timestamp": datetime.now().isoformat()
        },
        {
            "title": "VentureBeat: Microsoft Copilot Enterprise Adoption",
            "description": "Report shows 60% of Fortune 500 companies now using Microsoft Copilot.",
            "url": "https://venturebeat.com",
            "source": "VentureBeat AI",
            "heat_score": 82,
            "timestamp": datetime.now().isoformat()
        },
        {
            "title": "AI Business: Healthcare AI Solutions Market Report",
            "description": "Healthcare AI market projected to reach $50B by 2030.",
            "url": "https://artificialintelligence-news.com",
            "source": "AI Business",
            "heat_score": 68,
            "timestamp": datetime.now().isoformat()
        },
        {
            "title": "InfoQ: Enterprise RAG Implementation Patterns",
            "description": "Article discusses production patterns for RAG in enterprise environments.",
            "url": "https://infoq.cn",
            "source": "InfoQ AI",
            "heat_score": 72,
            "timestamp": datetime.now().isoformat()
        }
    ],

    "🌐 消费产品": [
        {
            "title": "Product Hunt: AI Video Generator Launches",
            "description": "New app generates marketing videos from text prompts in seconds.",
            "url": "https://producthunt.com",
            "source": "Product Hunt",
            "heat_score": 85,
            "timestamp": datetime.now().isoformat()
        },
        {
            "title": "a16z Top 100: Best AI Apps of 2024",
            "description": "Andreessen Horowitz releases annual list of top consumer AI applications.",
            "url": "https://a16z.com",
            "source": "a16z",
            "heat_score": 90,
            "timestamp": datetime.now().isoformat()
        },
        {
            "title": "Hacker News Show HN: AI Writing Assistant",
            "description": "Founder launches AI writing assistant that adapts to your style.",
            "url": "https://news.ycombinator.com",
            "source": "Hacker News",
            "heat_score": 75,
            "timestamp": datetime.now().isoformat()
        },
        {
            "title": "App Store: New AI Photo Editor Hits #1",
            "description": "AI-powered photo editor tops App Store charts in 50 countries.",
            "url": "https://apps.apple.com",
            "source": "App Store",
            "heat_score": 80,
            "timestamp": datetime.now().isoformat()
        }
    ],

    "📰 行业资讯": [
        {
            "title": "NewsAPI: Global AI Regulation Roundup",
            "description": "Summary of new AI regulations across US, EU, and Asia.",
            "url": "https://newsapi.org",
            "source": "NewsAPI",
            "heat_score": 70,
            "timestamp": datetime.now().isoformat()
        },
        {
            "title": "MIT Tech Review: The Future of AI Safety",
            "description": "In-depth analysis of emerging AI safety research and challenges.",
            "url": "https://technologyreview.com",
            "source": "MIT Tech Review",
            "heat_score": 88,
            "timestamp": datetime.now().isoformat()
        },
        {
            "title": "The Gradient: Multimodal Learning Survey",
            "description": "Comprehensive survey of multimodal learning techniques and applications.",
            "url": "https://thegradient.pub",
            "source": "The Gradient",
            "heat_score": 65,
            "timestamp": datetime.now().isoformat()
        },
        {
            "title": "MarkTechPost: AI Ethics Framework Proposed",
            "description": "Researchers propose new framework for ethical AI development.",
            "url": "https://marktechpost.com",
            "source": "MarkTechPost",
            "heat_score": 62,
            "timestamp": datetime.now().isoformat()
        },
        {
            "title": "Stanford HAI: AI Index Report 2024",
            "description": "Annual report shows AI adoption accelerating across all industries.",
            "url": "https://hai.stanford.edu",
            "source": "Stanford HAI",
            "heat_score": 92,
            "timestamp": datetime.now().isoformat()
        },
        {
            "title": "Accenture: Technology Vision 2024",
            "description": "Consulting firm identifies key technology trends for 2024.",
            "url": "https://accenture.com",
            "source": "Accenture",
            "heat_score": 68,
            "timestamp": datetime.now().isoformat()
        }
    ]
}

# 将模拟数据转换为trends_by_source格式
trends_by_source = {}
for category, items in test_trends.items():
    for item in items:
        source = item["source"]
        if source not in trends_by_source:
            trends_by_source[source] = []
        trends_by_source[source].append(item)

print(f"\n📊 输入数据: {len(trends_by_source)} 个数据源")
for source, items in trends_by_source.items():
    print(f"  - {source}: {len(items)} 条")

# 创建分类Agent
categorizer = TrendCategorizerAgent(config, prompts)

# 执行分类
print("\n🔄 执行分类 (v9.0: 6分类 + Top5截取)...")
state = {"trends_by_source": trends_by_source}
result = categorizer.execute(state)

categorized_trends = result.get("categorized_trends", {})

print("\n" + "=" * 60)
print("📋 v9.0 分类结果 (Top5截取)")
print("=" * 60)

total_count = 0
for cat_name, cat_data in categorized_trends.items():
    count = cat_data["count"]
    total_count += count
    icon = cat_data["icon"]
    print(f"\n{icon} {cat_name} ({count}条)")
    for i, item in enumerate(cat_data["items"][:5], 1):
        title = item["title"]
        source = item["source"]
        score = item["heat_score"]
        print(f"  {i}. [{source}] {title} (热度: {score})")

print("\n" + "=" * 60)
print(f"✅ 总计: {total_count} 条 (6分类 × Top5 = 最多30条)")
print("=" * 60)

# 测试评分系统
print("\n🔢 测试评分系统 (30数据源权威度评分)...")
scorer = NewsScoringAgent(config, prompts)

# 显示数据源权威度评分
from src.agents.news_scoring_agent import SOURCE_AUTHORITY_SCORES

print("\n📊 数据源权威度评分 (部分):")
print("-" * 60)
for source, score in sorted(SOURCE_AUTHORITY_SCORES.items(), key=lambda x: x[1], reverse=True)[:15]:
    bar = "█" * (score // 5)
    print(f"{source:30s} {score:3d} {bar}")

print("\n✅ v9.0测试完成!")
