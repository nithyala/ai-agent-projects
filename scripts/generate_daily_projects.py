"""
Daily AI Agent Project Generator
Runs via GitHub Actions at 6 PM UTC daily.
Uses Claude API to generate 2 unique, production-quality AI agent projects.
"""

import os
import re
import json
import anthropic
from datetime import datetime
from pathlib import Path

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

AGENT_THEMES = [
    "resume screener that ranks and scores candidates automatically",
    "autonomous code reviewer using static analysis",
    "multi-agent debate system where two AIs argue opposing sides",
    "real-time news summarizer with source credibility scoring",
    "natural language to SQL query generator agent",
    "email triage agent that categorizes and drafts replies",
    "financial report analyzer with risk scoring",
    "customer support escalation agent with sentiment routing",
    "research paper summarizer with citation extraction",
    "meeting transcript to action items converter",
    "social media content strategist agent",
    "competitive analysis agent for SaaS products",
    "job description generator from role requirements",
    "legal document reviewer for contract risks",
    "supply chain disruption alert and mitigation agent",
    "AI changelog generator from git commit history",
    "podcast episode summarizer with key timestamps",
    "multi-agent travel itinerary planner",
    "earnings call sentiment and risk extraction agent",
    "product review aggregator and insight extractor",
]


def get_day_number() -> int:
    projects_dir = Path("projects")
    if not projects_dir.exists():
        return 2
    existing = [d for d in projects_dir.iterdir() if d.is_dir() and d.name.startswith("day")]
    return len(existing) + 1


def pick_themes(day: int) -> list:
    idx1 = (day * 2 - 2) % len(AGENT_THEMES)
    idx2 = (day * 2 - 1) % len(AGENT_THEMES)
    return [AGENT_THEMES[idx1], AGENT_THEMES[idx2]]


def generate_project(theme: str, day: int, proj_num: int) -> dict:
    print(f"\n🤖 Generating Project {proj_num}: {theme}")

    prompt = f"""You are an expert AI/ML engineer. Generate a complete, production-quality AI agent project.

Theme: "{theme}"

Return a JSON object with exactly these keys:
{{
  "project_name": "snake_case_name_max_5_words",
  "title": "Human readable title",
  "description": "One clear sentence describing what it does",
  "agent_code": "Complete Python source code, 200+ lines. Use LangGraph StateGraph with 3+ agent nodes. Each agent has a clear role and specialized system prompt. Use OpenAI GPT-4o. Include TypedDict state, error handling, comments, rich terminal output with ANSI colors, and save a .md output file. Runnable with: python src/agent.py",
  "readme": "Full markdown README: what it does, ASCII architecture diagram, step-by-step explanation, how to run section with commands, sample output, key concepts table",
  "requirements": "pip requirements one per line: langchain>=0.2.0, langchain-openai>=0.1.0, langchain-core>=0.2.0, langgraph>=0.1.0, openai>=1.0.0, python-dotenv>=1.0.0"
}}

Return ONLY valid JSON. No markdown backticks. No explanation outside the JSON."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()
    raw = re.sub(r'^```json\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    return json.loads(raw)


def save_project(project: dict, day: int, proj_num: int) -> str:
    suffix = "b" if proj_num == 2 else "a"
    folder_name = f"day{day:02d}{suffix}_{project['project_name']}"
    base = Path(f"projects/{folder_name}")
    (base / "src").mkdir(parents=True, exist_ok=True)
    (base / "output").mkdir(exist_ok=True)

    (base / "src" / "agent.py").write_text(project["agent_code"])
    (base / "README.md").write_text(project["readme"])
    (base / "requirements.txt").write_text(project["requirements"])

    print(f"✅ Saved: projects/{folder_name}/")
    return folder_name


def update_readme(day: int, projects: list):
    date = datetime.now().strftime("%Y-%m-%d")
    row1 = f"| Day {day:02d}a | [{projects[0]['title']}](projects/{projects[0]['folder']}) | {projects[0]['description']} | {date} |"
    row2 = f"| Day {day:02d}b | [{projects[1]['title']}](projects/{projects[1]['folder']}) | {projects[1]['description']} | {date} |"
    entry = f"{row1}\n{row2}"

    readme = Path("README.md")
    content = readme.read_text() if readme.exists() else ""

    if "<!-- PROJECTS_TABLE -->" in content:
        content = content.replace("<!-- PROJECTS_TABLE -->", f"{entry}\n<!-- PROJECTS_TABLE -->")
    else:
        content += f"\n{entry}"

    readme.write_text(content)
    print("✅ Root README updated")


def main():
    day = get_day_number()
    themes = pick_themes(day)
    print(f"\n{'='*60}")
    print(f"🚀 Daily Generator — Day {day:02d} | {datetime.now().strftime('%Y-%m-%d')}")
    print(f"  Theme 1: {themes[0]}")
    print(f"  Theme 2: {themes[1]}")
    print('='*60)

    saved = []
    for i, theme in enumerate(themes, 1):
        project = generate_project(theme, day, i)
        folder = save_project(project, day, i)
        saved.append({"title": project["title"], "description": project["description"], "folder": folder})

    if len(saved) == 2:
        update_readme(day, saved)

    print(f"\n🎉 Done! 2 projects generated for Day {day:02d}")
    print(f"🔗 https://github.com/nithyala/ai-agent-projects")


if __name__ == "__main__":
    main()
